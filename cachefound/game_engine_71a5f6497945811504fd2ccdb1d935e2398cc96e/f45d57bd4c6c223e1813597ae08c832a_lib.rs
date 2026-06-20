ÿ3pub mod render;

use std::time::{Duration, Instant};

use game_engine_ecs::world::World;
use game_engine_shaders_types::packet::{GpuCirclePacket, GpuTrianglePacket};
use game_engine_shaders_types::shapes::ColorRGBA;
use game_engine_shaders_types::vec2;
use winit::{
    event_loop::{ControlFlow, EventLoop},
    platform::pump_events::EventLoopExtPumpEvents,
};

use crate::render::storage::RenderPacket;
use crate::render::{RenderManager, packet::RenderPacketContents};

// Our main application struct
pub struct GameApp {
    render_manager: RenderManager,
    world: World,
    x: (f32, f32),
    dir: bool,
    last_packet: RenderPacket,
}

impl Default for GameApp {
    fn default() -> Self {
        Self {
            render_manager: RenderManager::new(),
            world: World::new(),
            x: (-0.5, 0.5),
            dir: true,
            last_packet: RenderPacket::new(),
        }
    }
}

impl GameApp {
    pub fn run(&mut self) -> GameEngineResult<()> {
        let mut event_loop = EventLoop::new()?;

        // Continuously runs the event loop, ideal for games.
        event_loop.set_control_flow(ControlFlow::Poll);
        let mut time_last_update = Instant::now();

        let mut time_fps_report = Instant::now();
        let mut frame_count = 0u32;

        self.update()?;
        self.update_render_world()?;
        loop {
            if time_last_update.elapsed() >= Duration::from_millis(200) {
                self.update()?;
                self.update_render_world()?;
                time_last_update = Instant::now();
            }
            if time_fps_report.elapsed() >= Duration::from_secs(2) {
                let fps = frame_count as f64 / time_fps_report.elapsed().as_secs_f64();
                log::info!("FPS: {fps:.2}");
                frame_count = 0;
                time_fps_report = Instant::now();
            }
            match event_loop
                .pump_app_events(Some(Duration::from_millis(5)), &mut self.render_manager)
            {
                winit::platform::pump_events::PumpStatus::Continue => (),
                winit::platform::pump_events::PumpStatus::Exit(_) => break,
            };
            self.render_manager.window()?.request_redraw();
            frame_count += 1;
        }
        Ok(())
    }

    fn update(&mut self) -> GameEngineResult<()> {
        // We use x.0 as a global "time/phase" for our wave animation
        if self.dir {
            self.x.0 += 0.1; // Increment phase
            if self.x.0 >= 2.0 {
                // self.dir = false;
            }
        } else {
            self.x.0 -= 0.1;
            if self.x.0 <= -2.0 {
                // self.dir = true;
            }
        }

        Ok(())
    }

    fn update_render_world(&mut self) -> GameEngineResult<()> {
        let mut packet = RenderPacket::new();
        self.build_scene(&mut packet)?;

        self.last_packet = packet.clone();
        self.render_manager.push_snapshot(packet);

        Ok(())
    }

    /// Build the scene using the fluent builder API.
    fn build_scene(&mut self, packet: &mut RenderPacket) -> GameEngineResult<()> {
        if self.render_manager.window().is_err() {
            return Ok(());
        }

        let size = self.render_manager.window()?.inner_size();
        let w = size.width as f32;
        let h = size.height as f32;
        let phase = self.x.0;

        // --- Stroked triangles ---
        let rows = 50;
        let cols = 50;
        let spacing_x = w / cols as f32;
        let spacing_y = h / rows as f32;
        let tri_size = 15.0;

        for r in 0..rows {
            for c in 0..cols {
                let cx = c as f32 * spacing_x + (phase + r as f32 * 0.5).sin() * 20.0;
                let cy = r as f32 * spacing_y + (phase + c as f32 * 0.5).cos() * 20.0;
                let angle = phase * 2.0 + (r + c) as f32;

                let rotate = |px: f32, py: f32| {
                    vec2(
                        cx + (px - cx) * angle.cos() - (py - cy) * angle.sin(),
                        cy + (px - cx) * angle.sin() + (py - cy) * angle.cos(),
                    )
                };
                let p1 = rotate(cx, cy - tri_size);
                let p2 = rotate(cx - tri_size, cy + tri_size);
                let p3 = rotate(cx + tri_size, cy + tri_size);

                let rc = c as f32 / cols as f32;
                let gc = phase.sin() * 0.5 + 0.5;
                let bc = r as f32 / rows as f32;

                packet
                    .entity()
                    .triangle(
                        GpuTrianglePacket::new(p1, p2, p3)
                            .with_color(ColorRGBA::new(rc, gc, bc, 1.0)),
                    )
                    .stroke(ColorRGBA::new(1.0 - rc, 1.0 - gc, 1.0 - bc, 1.0), 2.0)
                    .spawn();
            }
        }

        // --- Stroked circles ---
        let c_rows = 20;
        let c_cols = 20;
        let c_spacing_x = w / c_cols as f32;
        let c_spacing_y = h / c_rows as f32;

        for r in 0..c_rows {
            for c in 0..c_cols {
                let cx = c as f32 * c_spacing_x
                    + c_spacing_x * 0.5
                    + (phase + r as f32 * 0.5).sin() * 15.0;
                let cy = r as f32 * c_spacing_y
                    + c_spacing_y * 0.5
                    + (phase + c as f32 * 0.5).cos() * 15.0;

                let rc = c as f32 / c_cols as f32;
                let gc = phase.sin() * 0.5 + 0.5;
                let bc = r as f32 / c_rows as f32;

                packet
                    .entity()
                    .circle(
                        GpuCirclePacket::new(vec2(cx, cy), 12.0)
                            .with_color(ColorRGBA::new(rc, gc, bc, 1.0)),
                    )
                    .stroke(ColorRGBA::new(1.0 - rc, 1.0 - gc, 1.0 - bc, 1.0), 2.0)
                    .spawn();
            }
        }

        // Mark all as newly spawned on first frame
        if self.last_packet.entity_heads.is_empty() {
            let tri_count = packet.triangles.data.len() as u32;
            packet.triangles.newly_spawned_indices = (0..tri_count).collect();
            let circle_count = packet.circles.data.len() as u32;
            packet.circles.newly_spawned_indices = (0..circle_count).collect();
            let stroke_count = packet.strokes.data.len() as u32;
            packet.strokes.newly_spawned_indices = (0..stroke_count).collect();
        }

        Ok(())
    }

    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }
}

type GameEngineResult<T> = Result<T, Box<dyn std::error::Error>>;
€ *cascade08€ *cascade08‘‘£ *cascade08£¤¤« *cascade08«¬¬­ *cascade08­®®± *cascade08
±³ ³À *cascade08ÀÃÃÄ *cascade08ÄÅÅÆ *cascade08ÆÈÈÉ *cascade08ÉÊÊË *cascade08ËÌÌÍ *cascade08ÍÒÒÓ *cascade08Ó××Ø *cascade08Øééì *cascade08ìüüÿ *cascade08ÿ„„Ë *cascade08ËÑÑÒ *cascade08ÒÓÓÔ *cascade08ÔÙÙÜ *cascade08Üá*cascade08áÿ *cascade08ÿÓ *cascade08
ÓÕ ÕÙ *cascade08ÙÚ*cascade08Úí *cascade08
íï ï« *cascade08«»»¼ *cascade08¼ÂÂÅ *cascade08ÅÆÆ½ *cascade08½¿ *cascade08¿À *cascade08ÀÁÁÂ *cascade08ÂÃÃÆ *cascade08ÆÈÈÉ *cascade08ÉÊÊË *cascade08ËÎÎĞ *cascade08ĞÑÑÒ *cascade08
ÒÚ ÚÅ *cascade08ÅÇÇÚ *cascade08ÚÛ *cascade08Ûİİ¥  *cascade08¥ ª ª »  *cascade08» ¾ ¾ Ğ  *cascade08Ğ Ñ Ñ Ó  *cascade08
Ó Ú  Ú Û  *cascade08Û Ü Ü Ş  *cascade08Ş à à ç  *cascade08ç ë ë í  *cascade08í î î ğ  *cascade08ğ ñ ñ ! *cascade08!‚! *cascade08‚!…!…!‡! *cascade08‡!ˆ!ˆ!‹! *cascade08‹!!!—! *cascade08
—! !  !¤! *cascade08¤!¥!¥!¦! *cascade08¦!ª!ª!¬! *cascade08¬!®!®!¯! *cascade08¯!°!°!Á! *cascade08Á!Ã! *cascade08Ã!Ä!Ä!È! *cascade08
È!Í! Í!Î! *cascade08Î!Ï!Ï!Ñ! *cascade08
Ñ!Ö! Ö!Ø! *cascade08Ø!Ú!Ú!Û! *cascade08Û!İ!İ!Ş! *cascade08Ş!ß!ß!à! *cascade08
à!ç! ç!è! *cascade08è!ì!ì!í! *cascade08í!ğ!ğ!–# *cascade08–#Ğ#Ğ#Ú# *cascade08Ú#Ş#Ş#­$ *cascade08­$±$±$İ$ *cascade08İ$Ş$Ş$ß$ *cascade08ß$à$à$æ$ *cascade08æ$õ$õ$‡% *cascade08‡%‰%‰%¾% *cascade08
¾%Ã% Ã%Ä% *cascade08
Ä%ç% ç%è% *cascade08
è%š& š&›& *cascade08
›&«& «&¬& *cascade08
¬&´& ´&¸& *cascade08¸&¹&¹&É& *cascade08
É&ã& ã&ó& *cascade08ó&õ&õ&' *cascade08''' ' *cascade08 '¢'¢'Ä' *cascade08Ä'Æ'Æ'æ' *cascade08æ'è'è'( *cascade08
(ƒ( ƒ(’( *cascade08’(”(”(²( *cascade08
²(¶( ¶(¾( *cascade08¾(¿(¿(À( *cascade08À(Á(Á(Â( *cascade08Â(Æ(Æ(Ç( *cascade08
Ç(Û( Û(ı( *cascade08
ı() )—) *cascade08—)˜)˜)œ) *cascade08œ)))¨) *cascade08¨)©)©)ª) *cascade08ª)³)³)Ä) *cascade08Ä)Æ)*cascade08Æ)Ç) *cascade08Ç)È)È)É) *cascade08É)Ê) *cascade08
Ê)Ò) Ò)Ó) *cascade08Ó)Ô)Ô)Õ) *cascade08Õ)Ö)Ö)×) *cascade08×)Ú)Ú)ë) *cascade08ë)í) *cascade08í)ï) *cascade08ï)ğ)*cascade08ğ)ñ) *cascade08ñ)õ)õ)÷) *cascade08
÷)ù) ù)ú) *cascade08ú)ÿ) *cascade08ÿ)„*„*…* *cascade08…*‹*‹*Œ* *cascade08Œ** *cascade08**** *cascade08*‘*‘*’* *cascade08’*˜*˜*½* *cascade08½*¿*¿*¢+ *cascade08¢+¤+¤+Â+ *cascade08
Â+Æ+ Æ+Ç+ *cascade08Ç+È+È+Ù+ *cascade08Ù+Ú+*cascade08Ú+İ+ *cascade08
İ+å+ å+æ+ *cascade08æ+é+ *cascade08é+ü+ *cascade08
ü+ÿ+ ÿ+€, *cascade08€,,,˜, *cascade08˜,œ,*cascade08œ, , ,¡, *cascade08¡,¢,¢,¤, *cascade08¤,«,«,±, *cascade08±,³,³,´, *cascade08´,¶,¶,·, *cascade08·,¸,¸,», *cascade08»,½,½,¿, *cascade08¿,À, *cascade08À,Á,Á,Ã, *cascade08Ã,Å, *cascade08Å,É,*cascade08É,İ, *cascade08İ,á,á,ü, *cascade08ü,–-–-£- *cascade08£-¹-
¹-º- 
º-Õ- Õ-Ö- *cascade08Ö-Ù-Ù-Ú- *cascade08Ú-İ-İ-Ş- *cascade08Ş-ß-ß-à- *cascade08à-ã-ã-ñ- *cascade08ñ-ò-ò-ô- *cascade08ô-ù-ù-‰. *cascade08
‰.”. ”.». *cascade08».¼.¼.½. *cascade08½.¾.¾.Ç. *cascade08
Ç.Ï. Ï.Ğ. *cascade08Ğ.Ñ.Ñ.Ö. *cascade08Ö.×.×.€/ *cascade08
€/ƒ/ ƒ/„/ *cascade08
„/‹/ ‹/Œ/ *cascade08
Œ// /‘/ *cascade08‘/•/•/ª/ *cascade08
ª/°/ °/±/ *cascade08±/³/³/¶/ *cascade08
¶/»/ »/½/ *cascade08½/Ã/Ã/Å/ *cascade08
Å/Ï/ Ï/Ñ/ *cascade08Ñ/Ø/Ø/í/ *cascade08
í/ğ/ ğ/ò/ *cascade08ò/ô/ô/ö/ *cascade08ö/÷/÷/û/ *cascade08û/ü/ü/ı/ *cascade08
ı/‚0 ‚0ƒ0 *cascade08ƒ0…0…0Œ0 *cascade08
Œ00 00 *cascade08
0•0 •0–0 *cascade08–0—0—0œ0 *cascade08œ000Ÿ0 *cascade08Ÿ0¡0¡0£0 *cascade08£0¦0¦0µ0 *cascade08
µ0Ä0 Ä0Å0 *cascade08
Å0å0 å0æ0 *cascade08æ0é0é0„1 *cascade08„1†1†1‡1 *cascade08‡1ˆ1ˆ1‰1 *cascade08
‰1Œ1 Œ11 *cascade081—1—1™1 *cascade08™1 1 1§1 *cascade08
§1¯1 ¯1°1 *cascade08°1³1³1´1 *cascade08´1µ1µ1Æ1 *cascade08
Æ1Ì1 Ì1Í1 *cascade08
Í1Ø1 Ø1Ù1 *cascade08
Ù1İ1 İ1Ş1 *cascade08Ş1ß1ß1à1 *cascade08à1á1á1â1 *cascade08â1ã1ã1ä1 *cascade08ä1å1å1æ1 *cascade08æ1ç1ç1è1 *cascade08è1é1é1ë1 *cascade08ë1î1î1ï1 *cascade08ï1ğ1ğ1ó1 *cascade08ó1õ1õ1ö1 *cascade08ö1û1û1ƒ2 *cascade08ƒ2‡2‡22 *cascade08222‘2 *cascade08‘2“2“2–2 *cascade08
–2¤2 ¤2¥2 *cascade08¥2«2«2®2 *cascade08
®2¹2 ¹2º2 *cascade08º2Ã2Ã2Å2 *cascade08Å2É2É2ÿ3 *cascade08"(71a5f6497945811504fd2ccdb1d935e2398cc96e2=file:///home/jackc/Documents/code/rust/game_engine/src/lib.rs:2file:///home/jackc/Documents/code/rust/game_engine