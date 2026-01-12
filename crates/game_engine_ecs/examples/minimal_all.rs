/*
    minimal_all.rs

    Demonstrates features of game_engine_ecs:
    - Component registration & storage
    - Entity Spawning (Immediate, Deferred, Batch)
    - Resources (Mutable/Immutable)
    - Systems (Function conversion, State injection)
    - System Params (Res, ResMut, Query, Local, Commands)
    - Queries (Filters, Iteration, Random Access)
    - Change Detection
    - Command Flushing
*/

use game_engine_derive::Component;
use game_engine_ecs::prelude::*;
use game_engine_ecs::query::{Changed, With, Without};
use game_engine_ecs::system::command::Command;
use game_engine_ecs::system::{Local, System, UnsafeWorldCell};
use std::ops::Deref;

// --- 1. COMPONENTS ---
// Simple POD data
#[derive(Debug, Clone, Copy, PartialEq, Component)]
struct Position {
    x: f32,
    y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Component)]
struct Velocity {
    x: f32,
    y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Component)]
struct Health(f32);

// Marker Component (Zero Sized Type)
#[derive(Debug, Clone, Copy, Component)]
struct Player;

// --- 2. RESOURCES ---
// Global state accessible by systems
#[derive(Debug, Default)]
struct GameStats {
    frame_count: u32,
    entities_spawned: u32,
}

// --- 3. SYSTEMS ---

/// System: Uses Commands to spawn entities later (Deferred).
/// Demonstrates: Commands, ResMut
fn spawner_system(mut commands: Command, mut stats: ResMut<GameStats>) {
    if stats.frame_count == 0 {
        println!("[System: Spawner] First frame! Queueing spawn of Player via Commands...");

        // Spawn a Player deferred
        commands.spawn((
            Player,
            Position { x: 0.0, y: 0.0 },
            Velocity { x: 1.0, y: 1.0 },
            Health(100.0),
        ));

        stats.entities_spawned += 1;
    }
}

/// System: Moves entities.
/// Demonstrates: Query iteration, Mutable components
fn movement_system(mut query: Query<(&mut Position, &Velocity)>) {
    println!("[System: Movement] Updating positions...");
    for (mut pos, vel) in query.iter() {
        pos.x += vel.x;
        pos.y += vel.y;
    }
}

/// System: Prints positions ONLY if they changed.
/// Demonstrates: Change Detection (Filters)
fn logger_system(mut query: Query<(Entity, &Position), Changed<Position>>) {
    for (e, pos) in query.iter() {
        println!("  -> Entity {:?} moved to {:?}", e, pos);
    }
}

/// System: Damages entities using Local state to track time/interval.
/// Demonstrates: Local<T>, Query Filters (Without), Random Access (via get)
fn damage_system(
    mut timer: Local<u32>,
    mut query: Query<(Entity, &mut Health), Without<Player>>, // Don't hurt the player
    mut all_query: Query<&Position>, // Secondary query for random access lookups
) {
    *timer += 1;
    if *timer % 2 != 0 {
        return;
    } // Only run every other frame

    println!("[System: Damage] Applying environmental damage to non-players...");

    for (e, mut health) in query.iter() {
        health.0 -= 5.0;

        // Random access demo: Look up position of the entity we are damaging
        // just to prove we can inspect other components of the same entity via a different query
        if let Some(pos) = all_query.get(e) {
            println!(
                "     Hurt Entity {:?} at {:?} (Health: {})",
                e,
                pos.deref(),
                health.0
            );
        }
    }
}

/// System: Despawns dead entities.
/// Demonstrates: Commands (Despawn)
fn cleanup_system(mut commands: Command, mut query: Query<(Entity, &Health)>) {
    for (e, health) in query.iter() {
        if health.0 <= 0.0 {
            println!("[System: Cleanup] Entity {:?} died. Despawning.", e);
            commands.despawn(e);
        }
    }
}

// --- 4. MANUAL SCHEDULER HELPER ---
// Since the `Schedule` struct isn't fully implemented in the provided files,
// we build a tiny wrapper to store and run systems sequentially.
struct ManualExecutor {
    systems: Vec<Box<dyn System>>,
}

impl ManualExecutor {
    fn new() -> Self {
        Self {
            systems: Vec::new(),
        }
    }

    fn add<M, S: IntoSystem<M>>(&mut self, system: S, world: &mut World) {
        let mut sys = system.into_system();
        sys.init(world);
        self.systems.push(Box::new(sys));
    }

    fn run_frame(&mut self, world: &mut World) {
        // 1. Update Access (Dynamic queries need to know about new archetypes)
        for sys in &mut self.systems {
            sys.update_access(world);
        }

        // 2. Run Systems
        let cell = UnsafeWorldCell::new(world);
        for sys in &mut self.systems {
            unsafe {
                sys.run(&cell);
            }
        }

        // 3. Increment Tick (Crucial for Change Detection)
        world.increment_tick();

        // 4. Flush Commands (Spawn/Despawn happens here)
        world.flush();
    }
}

// --- 5. MAIN EXECUTION ---

fn main() {
    println!("=== Starting Minimal ECS Demo ===");

    let mut world = World::new();
    let mut executor = ManualExecutor::new();

    // --- SETUP RESOURCES ---
    world.resources_mut().insert(GameStats::default());

    // --- REGISTER SYSTEMS ---
    // Note: We don't use the turbofish syntax thanks to `impl_system_function` macro magic
    executor.add(spawner_system, &mut world);
    executor.add(movement_system, &mut world);
    executor.add(logger_system, &mut world);
    executor.add(damage_system, &mut world);
    executor.add(cleanup_system, &mut world);

    // --- BATCH SPAWN (Feature Demo) ---
    println!("--- Initializing: Batch Spawning 10 Enemies ---");
    let mut enemies = Vec::new();
    for i in 0..10 {
        enemies.push((
            Position {
                x: i as f32 * 10.0,
                y: 0.0,
            },
            Velocity { x: 0.0, y: -1.0 }, // Move down
            Health(15.0),                 // Will die in 3 damage ticks (6 frames)
        ));
    }
    // Directly spawns into archetypes (highly optimized)
    world.spawn_batch(enemies);
    world.flush(); // Ensure they are ready

    // --- MAIN LOOP ---
    // Run for 10 frames to demonstrate lifecycle
    for i in 1..=10 {
        println!("\n--- FRAME {} ---", i);

        // Update Frame Count Resource
        {
            let mut stats = world.resources_mut().get_mut::<GameStats>().unwrap();
            stats.frame_count = i;
        }

        // Run Logic
        executor.run_frame(&mut world);

        // Verification logic for the demo output
        let stats = world.resources().get::<GameStats>().unwrap();
        let entity_count = world.entities().len();
        println!(
            "End of Frame Status: [Entities: {}] [Spawned Total: {}]",
            entity_count, stats.entities_spawned
        );
    }

    println!("\n=== Demo Complete ===");
}
