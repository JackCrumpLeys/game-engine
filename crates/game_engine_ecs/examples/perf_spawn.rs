#![allow(dead_code)]
use std::hint::black_box;

use game_engine_derive::Component;
use game_engine_ecs::world::World;

#[derive(Debug, Clone, Copy, Default, Component)]
struct Position {
    x: f32,
    y: f32,
}

#[derive(Debug, Clone, Copy, Default, Component)]
struct Velocity {
    x: f32,
    y: f32,
}

#[derive(Debug, Clone, Copy, Default, Component)]
struct Rotation {
    radians: f32,
}

#[derive(Debug, Clone, Copy, Default, Component)]
struct TransformMatrix {
    data: [f32; 16],
}

#[derive(Debug, Clone, Copy, Default, Component)]
struct Health {
    val: f32,
}

#[derive(Debug, Clone, Copy, Default, Component)]
struct Regen {
    rate: f32,
}

#[derive(Debug, Clone, Copy, Default, Component)]
struct Poisoned; // Marker component

const ENTITY_COUNT: usize = 1_000_000;
const ITERATIONS: usize = 100;
fn main() {
    let mut world = World::new();
    for _ in 0..ITERATIONS {
        for i in 0..ENTITY_COUNT {
            let pos = Position {
                x: i as f32,
                y: 0.0,
            };
            let vel = Velocity { x: 1.0, y: 1.0 };
            let rot = Rotation { radians: 0.0 };
            let hp = Health { val: 100.0 };

            // Fragment data to stress archetype creation
            if i.is_multiple_of(2) {
                world.spawn_deferred((
                    pos,
                    vel,
                    rot,
                    hp,
                    Regen { rate: 1.0 },
                    TransformMatrix::default(),
                ));
            } else {
                world.spawn_deferred((pos, vel, rot, hp, Poisoned, TransformMatrix::default()));
            }
        }
        world.flush();
    }
    black_box(world);
}
