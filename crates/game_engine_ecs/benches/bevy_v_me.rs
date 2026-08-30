use std::hint::black_box;

use bevy_ecs::prelude::{
    Component as BevyComponent, Entity as BevyEntity, Query as BevyQuery, World as BevyWorld,
};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use game_engine_derive::Component;
use game_engine_ecs::prelude::*;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator};
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
const THREADS: usize = 1;

/// 1. MASS SPAWNING (1,000,000 entities)
fn bench_mass_spawn(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mass Spawning (1M)");

    group.bench_function("My Engine: spawn + flush", |b| {
        b.iter_batched(
            World::new,
            |mut world| {
                world.system_param_with::<Commands, _, _>(|commands| {
                    for _ in 0..ENTITY_COUNT_1M {
                        commands.spawn((Position { x: 1.0, y: 1.0 }, Velocity { x: 1.0, y: 1.0 }));
                    }
                });
                world.flush();
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("Bevy: spawn + flush", |b| {
        b.iter_batched(
            BevyWorld::new,
            |mut world| {
                let mut co = world.commands();
                for _ in 0..ENTITY_COUNT_1M {
                    co.spawn((BPosition { x: 1.0, y: 1.0 }, BVelocity { x: 1.0, y: 1.0 }));
                }
                world.flush();
            },
            BatchSize::SmallInput,
        );
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

    my_world.flush();

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
    black_box(my_world);

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
// --- Heavy Computation Helper ---
// This function will force the CPU to do some complex math for every spawn,
// based on the total global index of the spawn.
#[inline]
fn compute_heavy_data(idx: usize) -> [f32; 4] {
    let x = (idx as f32) + 1.0; // +1.0 to avoid ln(0) or division by zero issues
    [
        (x * 0.123).sin() * (x * 0.456).cos(), // Trig mix
        (x * 0.789).sqrt(),                    // Square root
        x.ln_1p(),                             // Natural log
        x.powf(1.1) % 100.0,                   // Power and modulo
    ]
}

// --- Interleaved Archetypes Macro (With Heavy Data Computation) ---

macro_rules! bench_interleaved_archetypes {
    ( $( ($my_comp:ident, $bevy_comp:ident) ),* ) => {

        // 1. Generate the 100 Data components for My Engine
        $(
            #[derive(Debug, Clone, Copy, Component)]
            struct $my_comp {
                data: [f32; 4],
            }
        )*

        // 2. Generate the 100 Data components for Bevy
        $(
            #[derive(BevyComponent, Clone, Copy)]
            struct $bevy_comp {
                data: [f32; 4],
            }
        )*

        // 3. Define the Benchmark Function
        fn bench_100_archetypes_interleaved(c: &mut Criterion) {
            let mut group = c.benchmark_group("Interleaved Archetypes (100x100)");

            group.bench_function("My Engine: Interleaved 100x100", |b| {
                b.iter_batched(
                    World::new,
                    |mut world| {
                        world.system_param_with::<Commands, _, _>(|commands| {
                            let mut spawn_idx = 0;
                            for _ in 0..100 {
                                $(
                                    let data = compute_heavy_data(spawn_idx);
                                    commands.spawn((Position::default(), $my_comp { data }));
                                    spawn_idx += 1;
                                )*
                            }
                        });
                        world.flush();
                    },
                    BatchSize::SmallInput,
                );
            });

            group.bench_function("Bevy: Interleaved 100x100", |b| {
                b.iter_batched(
                    BevyWorld::new,
                    |mut world| {
                        let mut spawn_idx = 0;
                        for _ in 0..100 {
                            $(
                                let data = compute_heavy_data(spawn_idx);
                                world.commands().spawn((BPosition::default(), $bevy_comp { data }));
                                spawn_idx += 1;
                            )*
                        }
                        world.flush();
                    },
                    BatchSize::SmallInput,
                );
            });

            group.finish();
        }
    };
}

// Invoke the macro with 100 pairs of components (C0-C99 and BC0-BC99)
bench_interleaved_archetypes!(
    (C0, BC0),
    (C1, BC1),
    (C2, BC2),
    (C3, BC3),
    (C4, BC4),
    (C5, BC5),
    (C6, BC6),
    (C7, BC7),
    (C8, BC8),
    (C9, BC9),
    (C10, BC10),
    (C11, BC11),
    (C12, BC12),
    (C13, BC13),
    (C14, BC14),
    (C15, BC15),
    (C16, BC16),
    (C17, BC17),
    (C18, BC18),
    (C19, BC19),
    (C20, BC20),
    (C21, BC21),
    (C22, BC22),
    (C23, BC23),
    (C24, BC24),
    (C25, BC25),
    (C26, BC26),
    (C27, BC27),
    (C28, BC28),
    (C29, BC29),
    (C30, BC30),
    (C31, BC31),
    (C32, BC32),
    (C33, BC33),
    (C34, BC34),
    (C35, BC35),
    (C36, BC36),
    (C37, BC37),
    (C38, BC38),
    (C39, BC39),
    (C40, BC40),
    (C41, BC41),
    (C42, BC42),
    (C43, BC43),
    (C44, BC44),
    (C45, BC45),
    (C46, BC46),
    (C47, BC47),
    (C48, BC48),
    (C49, BC49),
    (C50, BC50),
    (C51, BC51),
    (C52, BC52),
    (C53, BC53),
    (C54, BC54),
    (C55, BC55),
    (C56, BC56),
    (C57, BC57),
    (C58, BC58),
    (C59, BC59),
    (C60, BC60),
    (C61, BC61),
    (C62, BC62),
    (C63, BC63),
    (C64, BC64),
    (C65, BC65),
    (C66, BC66),
    (C67, BC67),
    (C68, BC68),
    (C69, BC69),
    (C70, BC70),
    (C71, BC71),
    (C72, BC72),
    (C73, BC73),
    (C74, BC74),
    (C75, BC75),
    (C76, BC76),
    (C77, BC77),
    (C78, BC78),
    (C79, BC79),
    (C80, BC80),
    (C81, BC81),
    (C82, BC82),
    (C83, BC83),
    (C84, BC84),
    (C85, BC85),
    (C86, BC86),
    (C87, BC87),
    (C88, BC88),
    (C89, BC89),
    (C90, BC90),
    (C91, BC91),
    (C92, BC92),
    (C93, BC93),
    (C94, BC94),
    (C95, BC95),
    (C96, BC96),
    (C97, BC97),
    (C98, BC98),
    (C99, BC99)
);
criterion_group!(
    benches,
    bench_mass_spawn,
    bench_iteration,
    bench_fragmentation,
    bench_100_archetypes_interleaved,
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
