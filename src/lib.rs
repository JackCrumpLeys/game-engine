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
};

// Our main application struct
pub struct GameApp {
    render_manager: RenderManager,
    world: World,
}

impl Default for GameApp {
    fn default() -> Self {
        Self {
            render_manager: RenderManager::new(),
            world: World::new(),
        }
    }
}

impl GameApp {
    pub fn run(&mut self) {
        let mut event_loop = EventLoop::new().unwrap();

        // Continuously runs the event loop, ideal for games.
        event_loop.set_control_flow(ControlFlow::Poll);

        loop {
            event_loop.pump_app_events(Some(Duration::from_millis(5)), &mut self.render_manager);
            self.update();
            self.update_render_world();
        }
    }

    fn update(&mut self) {
        // Update game logic here
    }

    fn update_render_world(&mut self) {
        let renderables = self.world.query::<Renderable>();
        let positions = self.world.query::<Position>();

        let render_world = self.render_manager.world_mut();
        render_world.clear();
        let mut old_entity_map = std::collections::HashMap::new();

        for renderable in renderables.iter().cloned() {
            if !old_entity_map.contains_key(&renderable.0) {
                let e = render_world.spawn_entity();
                old_entity_map.insert(renderable.0, e);
            }
            render_world.add_component(
                *old_entity_map.get(&renderable.0).unwrap(),
                renderable.1.clone(),
            );
        }
        for position in renderables.iter().cloned() {
            if !old_entity_map.contains_key(&position.0) {
                let e = render_world.spawn_entity();
                old_entity_map.insert(position.0, e);
            }
            render_world.add_component(
                *old_entity_map.get(&position.0).unwrap(),
                position.1.clone(),
            );
        }
    }

    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }
}
