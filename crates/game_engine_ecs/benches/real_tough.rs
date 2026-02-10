use core_affinity::{get_core_ids, set_for_current};
use criterion::{Criterion, criterion_group, criterion_main};
use game_engine_derive::Component;
use game_engine_ecs::prelude::*;
use rayon::prelude::*;
use std::hint::black_box;

// --- Components ---
// We use arrays to ensure struct sizes are significant enough
// to cause cache pressure.

#[derive(Debug, Clone, Copy, Default, Component)]
struct CompA {
    data: [f32; 16], // 64 bytes (cache line)
}

#[derive(Debug, Clone, Copy, Default, Component)]
struct CompB {
    data: [f32; 16],
}

#[derive(Debug, Clone, Copy, Default, Component)]
struct CompC {
    data: [f32; 16],
}

#[derive(Debug, Clone, Copy, Default, Component)]
struct CompD {
    data: [f32; 16],
}

#[derive(Debug, Clone, Copy, Default, Component)]
struct Physics {
    pos: [f32; 3],
    vel: [f32; 3],
}

// --- Resource for Rayon output ---
#[derive(Default)]
struct RayonResults {
    checksum: f64,
}

// --- Systems ---

/// Heavy Math Loop:
/// Performs a useless but expensive calculation (approximating matrix mul).
fn expensive_op(data: &mut [f32; 16]) {
    for i in 0..4 {
        for j in 0..4 {
            let idx = i * 4 + j;
            data[idx] = (data[idx] + 1.0).sin() * (data[idx] * 0.5).cos();
        }
    }
}

fn sys_heavy_a(mut query: Query<&mut CompA>) {
    query.for_each_mut(|mut a| {
        expensive_op(&mut a.data);
    });
}

fn sys_heavy_b(mut query: Query<&mut CompB>) {
    query.for_each_mut(|mut b| {
        expensive_op(&mut b.data);
    });
}

fn sys_heavy_c(mut query: Query<&mut CompC>) {
    query.for_each_mut(|mut c| {
        expensive_op(&mut c.data);
    });
}

fn sys_heavy_d(mut query: Query<&mut CompD>) {
    query.for_each_mut(|mut d| {
        expensive_op(&mut d.data);
    });
}

/// A generic physics system that has to jump across ALL archetypes
/// because nearly all entities have Physics.
fn sys_physics(mut query: Query<&mut Physics>) {
    query.for_each_mut(|mut p| {
        p.pos[0] += p.vel[0];
        p.pos[1] += p.vel[1];
        p.pos[2] += p.vel[2];
    });
}

/// The Rayon System:
/// Reads data, collects it (allocating memory), and uses a thread pool
/// to crunch numbers, then writes to a resource.
/// This puts memory pressure and thread scheduler pressure on the ECS.
fn sys_rayon_crunch(query: Query<&Physics>, mut res: ResMut<RayonResults>) {
    // 1. Collect data (simulate a snapshot)
    let positions: Vec<[f32; 3]> = query.iter().map(|p| p.pos).collect();

    // 2. Heavy parallel reduce
    let sum: f64 = positions
        .par_iter()
        .map(|p| (p[0] * p[1] * p[2]) as f64)
        .sum();

    res.checksum = sum;
}

// --- Setup Function ---

fn setup_world(entity_count: usize) -> (World, Schedule) {
    let mut world = World::new();
    let mut schedule = Schedule::default();

    // Register all components to ensure IDs are stable
    world.registry.register::<CompA>();
    world.registry.register::<CompB>();
    world.registry.register::<CompC>();
    world.registry.register::<CompD>();
    world.registry.register::<Physics>();
    world.resources_mut().insert(RayonResults::default());

    // Add systems
    // A, B, C, D modify disjoint data, so they should run in parallel.
    // Physics touches everything, so it forces a synchronization point.
    // Rayon Crunch reads Physics, so it can run with A,B,C,D but not Physics.
    schedule.add_systems(
        (
            (sys_heavy_a, sys_heavy_b, sys_heavy_c, sys_heavy_d), // Parallel Batch 1
            sys_rayon_crunch, // Can technically run with the above if logic allows
            sys_physics.after(sys_rayon_crunch), // Sync Point
        ),
        &mut world,
    );

    for i in 0..entity_count {
        let e = world.spawn(());

        // Base Physics for everyone
        world.insert_component(e, Physics::default());

        // Randomized components based on index
        if i % 2 == 0 {
            world.insert_component(e, CompA::default());
        }
        if i % 3 == 0 {
            world.insert_component(e, CompB::default());
        }
        if i % 4 == 0 {
            world.insert_component(e, CompC::default());
        }
        if i % 5 == 0 {
            world.insert_component(e, CompD::default());
        }
    }

    (world, schedule)
}

fn bench_torture(c: &mut Criterion) {
    #[cfg(feature = "tracy")]
    let _client = tracy_client::Client::start();

    #[cfg(feature = "tracy")]
    tracy_client::register_demangler!();

    let mut group = c.benchmark_group("ECS_Torture");
    group.sample_size(20); // Low sample size because this is heavy

    // 100,000 entities is the sweet spot where CPU cache is exhausted
    // and algorithmic complexity (O(N)) dominates overhead.
    let count = 100_000;

    group.bench_function("schedule_run_100k_fragmented", |b| {
        let (mut world, mut schedule) = setup_world(count);

        let core_ids = get_core_ids().unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .thread_name(|i| format!("ecs-worker-{}", i))
            .start_handler(move |id| {
                // Pin each worker thread to its corresponding core ID
                set_for_current(core_ids[id % core_ids.len()]);
            })
            .build()
            .unwrap();

        schedule.build_graph(&world);

        pool.install(|| {
            b.iter(|| {
                schedule.update(black_box(&mut world));
            })
        })
    });

    group.finish();
}

criterion_group!(benches, bench_torture);
criterion_main!(benches);
