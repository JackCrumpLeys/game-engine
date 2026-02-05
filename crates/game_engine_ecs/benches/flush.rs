use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use bevy_ecs::{
    component::Component as BevyComponent, schedule::Schedule, system::Commands,
    world::CommandQueue,
};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use game_engine_derive::Component;
use game_engine_ecs::{prelude::*, system::UnsafeWorldCell};
use game_engine_utils::graph::EntryUnknown; // Replace with your crate name

#[derive(Component, Default, Clone, Copy, Debug, BevyComponent)]
struct Pos {
    x: f32,
    y: f32,
}

fn bench_my_engine(c: &mut Criterion) {
    let mut world = World::new();

    // --- 1. System Overhead (Querying 10,000 entities) ---
    for _ in 0..10_000 {
        world.spawn(Pos { x: 1.0, y: 1.0 });
    }
    world.flush();

    fn my_system(query: Query<&Pos>) {
        black_box(query);
    }

    let mut sys = my_system.into_system();
    sys.init(&mut world);
    let cell = UnsafeWorldCell::new(&mut world);

    c.bench_function("my_engine_system_overhead_10k", |b| {
        b.iter(|| {
            let result = unsafe { sys.run(&cell) };
            black_box(result);
        })
    });

    // --- 2. Deferred Spawn + Flush (The "Sync Point") ---
    c.bench_function("my_engine_spawn_100_deferred_plus_flush", |b| {
        b.iter_custom(|iters| {
            let mut tot = Duration::default();

            for _ in 0..iters {
                for i in 0..100 {
                    world.spawn_deferred(Pos {
                        x: i as f32,
                        y: 0.0,
                    });
                }
                let now = Instant::now();
                world.flush();
                tot += now.elapsed();
            }
            tot
        })
    });
    c.bench_function("my_engine_zero_cost_flush", |b| {
        b.iter(|| {
            world.flush();
        })
    });
}

fn bench_bevy(c: &mut Criterion) {
    let mut world = bevy_ecs::world::World::new();

    // --- 1. System Overhead (Querying 10,000 entities) ---
    for _ in 0..10_000 {
        world.spawn(Pos { x: 1.0, y: 1.0 });
    }

    fn bevy_system(query: bevy_ecs::prelude::Query<&Pos>) {
        black_box(query);
    }

    let mut schedule = Schedule::default();
    schedule.add_systems(black_box(bevy_system));

    c.bench_function("bevy_system_overhead_10k", |b| {
        b.iter(|| {
            schedule.run(&mut world);
        })
    });

    // --- 2. Commands + Flush (The "Sync Point") ---
    // Bevy's Commands write to a CommandQueue, which is applied to the world.
    c.bench_function("bevy_spawn_100_commands_plus_apply", |b| {
        b.iter_custom(|iters| {
            let mut tot = Duration::default();

            for _ in 0..iters {
                {
                    let mut commands = world.commands();
                    for i in 0..100 {
                        commands.spawn(Pos {
                            x: i as f32,
                            y: 0.0,
                        });
                    }
                }
                let now = Instant::now();
                world.flush();
                tot += now.elapsed();
            }
            tot
        })
    });
}
criterion_group!(benches, bench_my_engine, bench_bevy);
criterion_main!(benches);
