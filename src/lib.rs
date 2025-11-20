pub mod ecs;
pub mod render;

use std::time::Duration;

use ecs::World;
use winit::{
    event_loop::{ControlFlow, EventLoop},
    platform::pump_events::EventLoopExtPumpEvents,
};

use crate::render::{
    RenderManager,
    components::{Position, Renderable},
    packet::{RenderPacket, VulVertex},
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
        let mut i = 0;
        loop {
            i += 1;
            if i % 20 == 0 {
                self.update()?;
                self.update_render_world()?;
            }
            match event_loop
                .pump_app_events(Some(Duration::from_millis(5)), &mut self.render_manager)
            {
                winit::platform::pump_events::PumpStatus::Continue => (),
                winit::platform::pump_events::PumpStatus::Exit(_) => break,
            };
            self.render_manager.window()?.request_redraw();
        }
        Ok(())
    }

    fn update(&mut self) -> GameEngineResult<()> {
        // Update game logic here

        if self.dir {
            self.x.0 += 0.01;
            if self.x.0 >= 0.5 {
                self.dir = false;
            }

            self.x.1 -= 0.01;
            if self.x.1 <= -0.5 {
                self.dir = false;
            }
        } else {
            self.x.0 -= 0.01;
            if self.x.0 <= -0.5 {
                self.dir = true;
            }

            self.x.1 += 0.01;
            if self.x.1 >= 0.5 {
                self.dir = true;
            }
        }

        Ok(())
    }

    fn update_render_world(&mut self) -> GameEngineResult<()> {
        self.render_manager
            .push_snapshot(RenderPacket::new().with_vertex_buffer([
                VulVertex::new(self.x.0, -0.5),
                VulVertex::new(0.0, 0.5),
                VulVertex::new(self.x.1, -0.5),
            ]));

        Ok(())
    }

    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }
}

type GameEngineResult<T> = Result<T, Box<dyn std::error::Error>>;
