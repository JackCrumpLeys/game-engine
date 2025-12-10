#![allow(dead_code)]
use std::collections::HashMap;
use std::ops::DerefMut;

use game_engine_derive::Component;
use game_engine_ecs::prelude::*;
use game_engine_ecs::query::Mut;
use game_engine_ecs::system::{Local, System, UnsafeWorldCell};

use game_engine_ecs::world::World;

struct App {
    world: World,
    systems: Vec<Box<dyn System>>,
}

impl App {
    fn new() -> Self {
        Self {
            world: World::new(),
            systems: Vec::new(),
        }
    }

    /// The "Clean API" target.
    /// We want to call this without `::<...>` type hints.
    fn add_system<M, S>(&mut self, system: S)
    where
        S: IntoSystem<M>,
    {
        let mut sys = system.into_system();
        sys.init(&mut self.world);
        self.systems.push(Box::new(sys));
    }

    fn run(&mut self) {
        let cell = UnsafeWorldCell::new(&mut self.world);
        for system in &mut self.systems {
            unsafe {
                system.run(&cell);
            }
        }
    }
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

#[derive(Debug, Clone, Copy, Component)]
struct LinkedTo(Entity);

#[derive(Debug, Clone, Copy, Default, Component)]
struct Position {
    x: f32,
    y: f32,
}

/// Unsafe way to do nested mutable queries
/// an abstraction over unsafe code that allows nested mutabilty would be in a real implementation
#[allow(mutable_transmutes)] // We would use unsafecell in a real implementation
fn unsafe_sys_a(mut pos: Query<(Entity, &mut Position)>, mut linked: Query<&LinkedTo>) {
    let entities: Vec<Entity> = pos.iter().map(|(e, _)| e).collect();
    // Mut<&mut T> implements DerefMut<Target=&mut T> and is used for change tracking.
    let map: HashMap<Entity, Mut<Position>> = pos.iter().collect();

    for entity in entities.into_iter() {
        let e = map.get(&entity).expect("Something went horrably wrong");
        // swindle the compiler into letting us have a mutable reference
        // SAFETY: We will make sure that we explicitly do not create multiple mutable references
        // to the same data.
        let e: &mut Mut<Position> = unsafe { std::mem::transmute(e) };
        let e: &mut Position = e.deref_mut();

        // Do somthing with our mutable reference
        e.x += 1.0;
        e.y += 1.0;
        if let Some(linked_to) = linked.get(entity) {
            // We could have UB here if linked_to.0 == entity
            assert!(linked_to.0 != entity, "Entity cannot link to itself");
            if let Some(linked_pos) = map.get(&linked_to.0) {
                // swindle the compiler again
                // SAFETY: We have already asserted that linked_to.0 != entity
                let linked_pos: &mut Mut<Position> = unsafe { std::mem::transmute(linked_pos) };
                let linked_pos: &mut Position = linked_pos.deref_mut();

                linked_pos.x += 1.0;
                linked_pos.y += 1.0;
            }
        }
    }
}

#[derive(Debug, Clone, Copy)] // Resources/Messages must be Clone/Copy/Send/Sync
struct MoveLinkEvent {
    target: Entity,
    dx: f32,
    dy: f32,
}

/// System A:
/// 1. Moves the current entity.
/// 2. If linked, queues a message to move the target.
fn sys_movement_logic(
    mut query: Query<(&mut Position, &LinkedTo)>,
    mut events: MessageWriter<MoveLinkEvent>,
    mut local: Local<u32>,
) {
    *local += 1;

    query.for_each(|(mut pos, link)| {
        // 1. Move Self
        pos.x += *local as f32;
        pos.y += *local as f32;

        // 2. Queue movement for the linked entity
        // Note: We don't need to check for self-linking here for safety.
        // If an entity links to itself, it will simply move, then get a message
        // to move again in the next system. No UB.
        events.write(MoveLinkEvent {
            target: link.0,
            dx: *local as f32,
            dy: *local as f32,
        });
    });
}

/// System B:
/// 1. Reads messages.
/// 2. Safely borrows the specific entity requested to apply the move.
fn sys_apply_link_effects(
    mut events: MessageReader<MoveLinkEvent>,
    mut query: Query<&mut Position>,
) {
    // We can iterate the messages safely.
    for event in events.iter() {
        // query.get() handles the locking logic internally.
        // If the entity is alive and has a Position, we get a guard.
        if let Some(mut pos) = query.get(event.target) {
            pos.x += event.dx;
            pos.y += event.dy;
        }
    }
}

// safe

fn main() {
    let mut app = App::new();
    let ecs_world = &mut app.world;

    for i in 0..10 {
        let pos = Position {
            x: i as f32,
            y: 0.0,
        };
        let vel = Velocity { x: 1.0, y: 1.0 };
        let rot = Rotation { radians: 0.0 };
        let hp = Health { val: 100.0 };
        if i % 2 == 0 {
            ecs_world.spawn((
                pos,
                vel,
                rot,
                hp,
                Regen { rate: 1.0 },
                TransformMatrix::default(),
            ));
        } else {
            ecs_world.spawn((pos, vel, rot, hp, Poisoned, TransformMatrix::default()));
        }
    }
    // gen some entities linked to Entity 0
    for i in 10..15 {
        let pos = Position {
            x: i as f32,
            y: 0.0,
        };
        let linked = LinkedTo(Entity::new(0, 0));
        ecs_world.spawn((pos, linked));
    }

    app.add_system(unsafe_sys_a);
    app.add_system(sys_movement_logic);
    app.add_system(sys_apply_link_effects);

    app.run();
}
