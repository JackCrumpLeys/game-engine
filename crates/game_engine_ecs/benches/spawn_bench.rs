#![allow(dead_code)]
use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use game_engine_derive::Component;
use game_engine_ecs::prelude::*;

// ============================================================================
// YOUR ENGINE COMPONENTS
// ============================================================================

#[derive(Debug, Clone, Copy, Default, Component)]
struct Position {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Debug, Clone, Copy, Default, Component)]
struct Velocity {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Debug, Clone, Copy, Default, Component)]
struct DataBlock {
    _pad: [u64; 8],
}

// ============================================================================
// BEVY COMPONENTS
// ============================================================================

#[derive(bevy_ecs::component::Component, Default, Clone, Copy)]
struct BevyPos {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(bevy_ecs::component::Component, Default, Clone, Copy)]
struct BevyVel {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(bevy_ecs::component::Component, Default, Clone, Copy)]
struct BevyData {
    _pad: [u64; 8],
}

// ============================================================================
// BENCHMARKS
// ============================================================================

fn spawn_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Entity Spawning (10,000 Entities)");
    group.sample_size(20);
    let count = 10_000;

    // ------------------------------------------------------------------------
    // 1. STANDARD LOOP (Immediate)
    // ------------------------------------------------------------------------

    group.bench_function("1a. My Engine: Loop spawn", |b| {
        b.iter_batched(
            World::new,
            |mut world| {
                for i in 0..count {
                    world.spawn((
                        Position {
                            x: i as f32,
                            ..Default::default()
                        },
                        Velocity::default(),
                        DataBlock::default(),
                    ));
                }
                black_box(world);
            },
            BatchSize::LargeInput,
        );
    });

    group.bench_function("1b. Bevy: Loop spawn", |b| {
        b.iter_batched(
            bevy_ecs::world::World::new,
            |mut world| {
                for i in 0..count {
                    world.spawn((
                        BevyPos {
                            x: i as f32,
                            ..Default::default()
                        },
                        BevyVel::default(),
                        BevyData::default(),
                    ));
                }
                black_box(world);
            },
            BatchSize::LargeInput,
        );
    });

    // ------------------------------------------------------------------------
    // 2. DEFERRED / COMMANDS
    // ------------------------------------------------------------------------

    group.bench_function("2a. My Engine: spawn_deferred + flush", |b| {
        b.iter_batched(
            World::new,
            |mut world| {
                for i in 0..count {
                    world.spawn_deferred((
                        Position {
                            x: i as f32,
                            ..Default::default()
                        },
                        Velocity::default(),
                        DataBlock::default(),
                    ));
                }
                world.flush();
                black_box(world);
            },
            BatchSize::LargeInput,
        );
    });

    group.bench_function("2b. Bevy: Commands spawn + apply", |b| {
        b.iter_batched(
            bevy_ecs::world::World::new,
            |mut world| {
                let mut queue = bevy_ecs::world::CommandQueue::default();
                let mut commands = bevy_ecs::system::Commands::new(&mut queue, &world);

                for i in 0..count {
                    commands.spawn((
                        BevyPos {
                            x: i as f32,
                            ..Default::default()
                        },
                        BevyVel::default(),
                        BevyData::default(),
                    ));
                }
                // Apply the command queue (equivalent to your flush)
                queue.apply(&mut world);
                black_box(world);
            },
            BatchSize::LargeInput,
        );
    });

    // ------------------------------------------------------------------------
    // 3. BATCH SPAWN (Vec<Bundle>)
    // ------------------------------------------------------------------------

    group.bench_function("3a. My Engine: spawn_batch", |b| {
        b.iter_batched(
            || {
                let mut bundles = Vec::with_capacity(count);
                for i in 0..count {
                    bundles.push((
                        Position {
                            x: i as f32,
                            ..Default::default()
                        },
                        Velocity::default(),
                        DataBlock::default(),
                    ));
                }
                (World::new(), bundles)
            },
            |(mut world, bundles)| {
                world.spawn_batch(bundles);
                black_box(world);
            },
            BatchSize::LargeInput,
        );
    });

    group.bench_function("3b. Bevy: spawn_batch", |b| {
        b.iter_batched(
            || {
                let mut bundles = Vec::with_capacity(count);
                for i in 0..count {
                    bundles.push((
                        BevyPos {
                            x: i as f32,
                            ..Default::default()
                        },
                        BevyVel::default(),
                        BevyData::default(),
                    ));
                }
                (bevy_ecs::world::World::new(), bundles)
            },
            |(mut world, bundles)| {
                world.spawn_batch(bundles);
                black_box(world);
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

// ============================================================================
// ALLOCATION STRATEGY BENCHMARK
// ============================================================================

fn allocation_strategy(c: &mut Criterion) {
    let mut group = c.benchmark_group("Allocator Strategy (Fragmentation)");
    let count = 10_000;

    group.bench_function("My Engine: Interleaved Deferred", |b| {
        b.iter_batched(
            World::new,
            |mut world| {
                for i in 0..count {
                    if i % 2 == 0 {
                        world.spawn_deferred((Position::default(),));
                    } else {
                        world.spawn_deferred((Position::default(), Velocity::default()));
                    }
                }
                world.flush();
            },
            BatchSize::LargeInput,
        );
    });

    // Bevy Command Queue Fragmentation
    group.bench_function("Bevy: Interleaved Commands", |b| {
        b.iter_batched(
            bevy_ecs::world::World::new,
            |mut world| {
                let mut queue = bevy_ecs::world::CommandQueue::default();
                let mut commands = bevy_ecs::system::Commands::new(&mut queue, &world);

                for i in 0..count {
                    if i % 2 == 0 {
                        commands.spawn((BevyPos::default(),));
                    } else {
                        commands.spawn((BevyPos::default(), BevyVel::default()));
                    }
                }
                queue.apply(&mut world);
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

criterion_group!(benches, spawn_benchmarks, allocation_strategy);
criterion_main!(benches);
