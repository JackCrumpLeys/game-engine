pub mod render;

use std::time::{Duration, Instant};

use game_engine_ecs::world::World;
use game_engine_shaders_types::vec2;
use game_engine_shaders_types::{packet::GpuTrianglePacket, shapes::ColorRGBA};
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
                log::info!("FPS: {:.2}", fps);
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
        if let Err(_) = self.render_manager.window() {
            return Ok(());
        }

        let size = self.render_manager.window()?.inner_size();
        let w = size.width as f32;
        let h = size.height as f32;
        let phase = self.x.0;

        let mut triangles = Vec::new();

        // --- STRESS TEST CONFIG ---
        let rows = 500;
        let cols = 500;
        let spacing_x = w / cols as f32;
        let spacing_y = h / rows as f32;
        let tri_size = 5.0;

        for r in 0..rows {
            for c in 0..cols {
                // Calculate base position
                let base_x = c as f32 * spacing_x;
                let base_y = r as f32 * spacing_y;

                // Add some "organic" movement based on phase and grid position
                let offset_x = (phase + (r as f32 * 0.5)).sin() * 20.0;
                let offset_y = (phase + (c as f32 * 0.5)).cos() * 20.0;

                let cx = base_x + offset_x;
                let cy = base_y + offset_y;

                // Create a rotating triangle
                let angle = phase * 2.0 + (r + c) as f32;
                let p1 = vec2(cx, cy - tri_size); // Top
                let p2 = vec2(cx - tri_size, cy + tri_size); // Bottom Left
                let p3 = vec2(cx + tri_size, cy + tri_size); // Bottom Right

                // Shift color based on position and time
                let r_col = (c as f32 / cols as f32);
                let g_col = (phase.sin() * 0.5 + 0.5);
                let b_col = (r as f32 / rows as f32);

                triangles.push(
                    GpuTrianglePacket::new(p1, p2, p3)
                        .with_color(ColorRGBA::new(r_col, g_col, b_col, 1.0))
                        .with_thickness(1.0)
                        .with_stroke_color(ColorRGBA::new(
                            1.0 - r_col,
                            1.0 - g_col,
                            1.0 - b_col,
                            1.0,
                        )),
                );
            }
        }

        // --- Packet Management ---
        let mut packet = RenderPacket::new();
        let count = triangles.len() as u32;

        // Force a resize/rebuild if the count changed or if it's the first frame
        if self.last_packet.triangles.data.len() != triangles.len() {
            let indices: Vec<u32> = (0..count).collect();
            packet.triangles = RenderPacketContents::new_all_data(triangles, indices);
            log::info!("Stress test: Rendering {} triangles", count);
        } else {
            // Update existing data to avoid re-allocating the Index buffer
            packet.triangles = self.last_packet.triangles.clone_data();
            for (i, tri) in triangles.into_iter().enumerate() {
                packet.triangles.data[i] = tri;
            }
        }

        self.render_manager.push_snapshot(packet.clone());
        self.last_packet = packet;

        Ok(())
    }

    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }
}

type GameEngineResult<T> = Result<T, Box<dyn std::error::Error>>;
