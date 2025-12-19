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
}

impl Default for GameApp {
    fn default() -> Self {
        Self {
            render_manager: RenderManager::new(),
            world: World::new(),
            x: (-0.5, 0.5),
            dir: true,
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
            if time_last_update.elapsed() >= Duration::from_millis(500) {
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
        // Update game logic here

        if self.dir {
            self.x.0 += 0.1;
            if self.x.0 >= 0.5 {
                self.dir = false;
            }

            self.x.1 -= 0.1;
            if self.x.1 <= -0.5 {
                self.dir = false;
            }
        } else {
            self.x.0 -= 0.1;
            if self.x.0 <= -0.5 {
                self.dir = true;
            }

            self.x.1 += 0.1;
            if self.x.1 >= 0.5 {
                self.dir = true;
            }
        }

        Ok(())
    }

    fn update_render_world(&mut self) -> GameEngineResult<()> {
        let mut packet = RenderPacket::new();
        packet.triangles = RenderPacketContents::new_all_data(vec![GpuTrianglePacket {
            vertices: [vec2(self.x.0, -0.5), vec2(0.0, 0.5), vec2(self.x.1, -0.5)],
        }]);
        self.render_manager.push_snapshot(packet);

        Ok(())
    }

    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }
}

type GameEngineResult<T> = Result<T, Box<dyn std::error::Error>>;
