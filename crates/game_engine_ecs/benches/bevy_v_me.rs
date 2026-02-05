use std::hint::black_box;

use bevy_ecs::prelude::{
    Component as BevyComponent, Entity as BevyEntity, Query as BevyQuery, World as BevyWorld,
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use game_engine_derive::Component;
use game_engine_ecs::prelude::*;
// --- Components for Your Engine ---

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
struct Marker;

// --- Components for Bevy ---
#[derive(BevyComponent, Default, Clone, Copy)]
struct BPosition {
    x: f32,
    y: f32,
}
#[derive(BevyComponent, Default, Clone, Copy)]
struct BVelocity {
    x: f32,
    y: f32,
}
#[derive(BevyComponent, Default, Clone, Copy)]
struct BMarker;

const ENTITY_COUNT_1M: usize = 1_000_000;
const ENTITY_COUNT_FRAG: usize = 10_000;
const CORE_COUNT: usize = 24;

/// 1. MASS SPAWNING (1,000,000 entities)
fn bench_mass_spawn(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mass Spawning (1M)");

    group.bench_function("My Engine: spawn loop", |b| {
        b.iter(|| {
            let mut world = World::new();
            for _ in 0..ENTITY_COUNT_1M {
                world.spawn_deferred((Position { x: 1.0, y: 1.0 }, Velocity { x: 1.0, y: 1.0 }));
            }
            world.flush();
        });
    });

    group.bench_function("Bevy: spawn loop", |b| {
        b.iter(|| {
            let mut world = BevyWorld::new();
            for _ in 0..ENTITY_COUNT_1M {
                world.spawn((BPosition { x: 1.0, y: 1.0 }, BVelocity { x: 1.0, y: 1.0 }));
            }
            world.flush();
        });
    });

    group.finish();
}

/// 2. SYSTEM ITERATION (1,000,000 entities - pos += vel)
/// Using `for_each` as requested.
fn bench_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("System Iteration (1M)");

    // Setup My Engine
    let mut my_world = World::new();

    (0..ENTITY_COUNT_1M).for_each(|_| {
        my_world.spawn_deferred((Position::default(), Velocity { x: 1.0, y: 1.0 }));
    });

    let mut query_state = my_world.query::<(&mut Position, &Velocity), ()>();

    group.bench_function("My Engine: for_each", |b| {
        b.iter(|| {
            query_state.for_each_mut(|(mut pos, vel)| {
                pos.x += vel.x;
                pos.y += vel.y;
                black_box(pos);
            });
        });
    });

    // Setup Bevy
    let mut bevy_world = BevyWorld::new();
    for _ in 0..ENTITY_COUNT_1M {
        bevy_world.spawn((BPosition::default(), BVelocity { x: 1.0, y: 1.0 }));
    }
    let mut bevy_query = bevy_world.query::<(&mut BPosition, &BVelocity)>();

    group.bench_function("Bevy: for_each", |b| {
        b.iter(|| {
            bevy_query
                .iter_mut(&mut bevy_world)
                .for_each(|(mut pos, vel)| {
                    pos.x += vel.x;
                    pos.y += vel.y;
                    black_box(pos);
                });
        });
    });

    group.finish();
}

/// 3. FRAGMENTATION (10k spawn, 5k delete, 2k add component)
fn bench_fragmentation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Fragmentation (Move/Add/Remove)");

    group.bench_function("My Engine: Frag", |b| {
        b.iter(|| {
            let mut world = World::new();
            // 1. Spawn 10k
            let entities: Vec<_> = (0..ENTITY_COUNT_FRAG)
                .map(|_| world.spawn_deferred((Position::default(), Velocity::default())))
                .collect();

            // 2. Delete 5k (every second)
            for i in (0..ENTITY_COUNT_FRAG).step_by(2) {
                world.despawn(entities[i]);
            }

            // 3. Add component to 2k of the remaining
            for i in (1..4000).step_by(2) {
                world.insert_component(entities[i], Marker);
            }

            // 4. Tick (flush structural changes)
            world.flush();

            // 5. Run a query to verify table traversal
            let mut q = world.query::<&Position, ()>();
            q.for_each_mut(|pos| {
                black_box(pos);
            });
        });
    });

    group.bench_function("Bevy: Frag", |b| {
        b.iter(|| {
            let mut world = BevyWorld::new();
            // 1. Spawn 10k
            let entities: Vec<_> = (0..ENTITY_COUNT_FRAG)
                .map(|_| {
                    world
                        .spawn((BPosition::default(), BVelocity::default()))
                        .id()
                })
                .collect();

            // 2. Delete 5k
            for i in (0..ENTITY_COUNT_FRAG).step_by(2) {
                world.despawn(entities[i]);
            }

            // 3. Add component to 2k
            for i in (1..4000).step_by(2) {
                world.entity_mut(entities[i]).insert(BMarker);
            }

            // 4. Bevy flush happens automatically on next query or command application
            let mut bevy_query = world.query::<&BPosition>();
            bevy_query.iter(&world).for_each(|pos| {
                black_box(pos);
            });
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_mass_spawn,
    bench_iteration,
    bench_fragmentation
);
fn main() {
    let pool = rayon::ThreadPoolBuilder::new()
        .thread_name(|i| format!("ecs-worker-{}", i))
        .build()
        .unwrap();

    pool.install(|| {
        benches();
        criterion::Criterion::default()
            .configure_from_args()
            .final_summary();
    });
}
