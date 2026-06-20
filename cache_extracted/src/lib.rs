pub mod render;

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
