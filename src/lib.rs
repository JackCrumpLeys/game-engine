pub mod render;

use std::time::{Duration, Instant};

use game_engine_ecs::world::World;
use game_engine_shaders_types::packet::GpuTrianglePacket;
use game_engine_shaders_types::vec2;
use winit::{
    event_loop::{ControlFlow, EventLoop},
    platform::pump_events::EventLoopExtPumpEvents,
};

use crate::render::{
    RenderManager,
    packet::{RenderPacket, RenderPacketContents},
};

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
            if time_last_update.elapsed() >= Duration::from_millis(1800) {
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
            self.x.0 += 0.4; // Increment phase
            if self.x.0 >= 2.0 {
                // self.dir = false;
            }
        } else {
            self.x.0 -= 0.4;
            if self.x.0 <= -2.0 {
                // self.dir = true;
            }
        }

        Ok(())
    }

    fn update_render_world(&mut self) -> GameEngineResult<()> {
        let mut triangles = Vec::new();
        let count = 1_500_000; // Let's draw 15 triangles
        let global_phase = self.x.0;

        for i in 0..count {
            // 1. Calculate a base X position for each triangle (spread -0.8 to 0.8)
            let x_base = -0.8 + (i as f32 * (1.6 / (count - 1) as f32));

            // 2. Calculate a unique Y offset for each using a sine wave
            // We use the index `i` to offset the wave so they don't move in perfect unison
            let y_offset = (global_phase + (i as f32 * 0.6)).sin() * 0.4;

            // 3. Make them grow and shrink slightly for extra test coverage
            let size = 0.05 + ((global_phase + i as f32).cos().abs() * 0.05);

            triangles.push(GpuTrianglePacket {
                vertices: [
                    vec2(x_base - size, y_offset - size), // Bottom Left
                    vec2(x_base, y_offset + size),        // Top Middle
                    vec2(x_base + size, y_offset - size), // Bottom Right
                ],
            });
        }

        // Create the packet with all 15 triangles
        let mut packet = RenderPacket::new();
        packet.triangles = self.last_packet.triangles.clone_data();
        if packet.triangles.data.is_empty() {
            // First time setup: insert all triangles
            packet.triangles =
                RenderPacketContents::new_all_data(triangles, (0..count as u32).collect());
        } else {
            // Update existing triangles
            for (i, triangle) in triangles.into_iter().enumerate() {
                packet.triangles.data[i] = triangle;
            }
        }

        // Push to the SnapshotPair for interpolation
        self.render_manager.push_snapshot(packet.clone());
        self.last_packet = packet;

        Ok(())
    }

    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }
}

type GameEngineResult<T> = Result<T, Box<dyn std::error::Error>>;
