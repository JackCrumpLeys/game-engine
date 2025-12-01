#![allow(dead_code)]
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use game_engine_ecs::query::{QueryInner, QueryState, With};
use game_engine_ecs::world::World;

// ============================================================================
// COMPONENTS
// ============================================================================

// 1. Standard "Hot" Components (16 bytes)
#[derive(Debug, Clone, Copy)]
struct Position {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Debug, Clone, Copy)]
struct Velocity {
    dx: f32,
    dy: f32,
    dz: f32,
}

// 2. Heavy Math Component (64 bytes) - Simulates a Transform Matrix
#[derive(Debug, Clone, Copy)]
struct TransformMatrix {
    m: [f32; 16],
}

impl Default for TransformMatrix {
    fn default() -> Self {
        Self { m: [1.0; 16] } // Identity-ish
    }
}

// 3. "Cold" / Bloat Component (4096 bytes)
// Used to test Cache Locality. In AoS, this destroys performance.
// In ECS (SoA), this should be skipped entirely.
#[derive(Debug, Clone, Copy)]
struct MeshData {
    _bloat: [u8; 4096],
}

// 4. Tags for Fragmentation / Filtering
struct TagA;
struct TagB;
struct TagC;
struct TagD;

// ============================================================================
// SCENARIOS
// ============================================================================

fn heavy_compute_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scenario 1: Heavy Compute (Matrix Mul)");
    // This tests if the ECS overhead is negligible when the CPU is busy.

    let count = 50_000;
    let mut world = World::new();

    for i in 0..count {
        world.spawn((
            Position {
                x: i as f32,
                y: 0.0,
                z: 0.0,
            },
            TransformMatrix::default(),
        ));
    }

    // Reference: Raw Vectors
    let mut raw_pos: Vec<Position> = (0..count)
        .map(|i| Position {
            x: i as f32,
            y: 0.0,
            z: 0.0,
        })
        .collect();
    let raw_mat: Vec<TransformMatrix> = (0..count).map(|_| TransformMatrix::default()).collect();

    // ECS Benchmark
    group.bench_function("ECS", |b| {
        let mut query = QueryState::<(&mut Position, &TransformMatrix)>::new(&mut world.registry);
        b.iter(|| {
            query_iteration(&mut world, &mut query);
        });
    });

    // Reference Benchmark
    group.bench_function("Raw SoA", |b| {
        b.iter(|| {
            for (pos, mat) in raw_pos.iter_mut().zip(raw_mat.iter()) {
                pos.x = pos.x * mat.m[0] + pos.y * mat.m[4] + pos.z * mat.m[8] + mat.m[12];
                pos.y = pos.x * mat.m[1] + pos.y * mat.m[5] + pos.z * mat.m[9] + mat.m[13];
                pos.z = pos.x * mat.m[2] + pos.y * mat.m[6] + pos.z * mat.m[10] + mat.m[14];
                black_box(pos);
            }
        })
    });

    group.finish();
}

fn archetype_fragmentation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scenario 2: Archetype Fragmentation");
    // Tests iteration speed when entities are split across 16 different tables.
    // This stresses the Outer Loop of your iterator.

    let count = 20_000;
    let mut world = World::new();

    for i in 0..count {
        let pos = Position {
            x: i as f32,
            y: 0.0,
            z: 0.0,
        };
        let vel = Velocity {
            dx: 1.0,
            dy: 0.0,
            dz: 0.0,
        };

        // Spawn in 16 different archetypes based on modulo
        match i % 16 {
            0 => {
                world.spawn((pos, vel));
            }
            1 => {
                world.spawn((pos, vel, TagA));
            }
            2 => {
                world.spawn((pos, vel, TagB));
            }
            3 => {
                world.spawn((pos, vel, TagA, TagB));
            }
            4 => {
                world.spawn((pos, vel, TagC));
            }
            5 => {
                world.spawn((pos, vel, TagA, TagC));
            }
            6 => {
                world.spawn((pos, vel, TagB, TagC));
            }
            7 => {
                world.spawn((pos, vel, TagA, TagB, TagC));
            }
            8 => {
                world.spawn((pos, vel, TagD));
            }
            9 => {
                world.spawn((pos, vel, TagA, TagD));
            }
            10 => {
                world.spawn((pos, vel, TagB, TagD));
            }
            11 => {
                world.spawn((pos, vel, TagA, TagB, TagD));
            }
            12 => {
                world.spawn((pos, vel, TagC, TagD));
            }
            13 => {
                world.spawn((pos, vel, TagA, TagC, TagD));
            }
            14 => {
                world.spawn((pos, vel, TagB, TagC, TagD));
            }
            15 => {
                world.spawn((pos, vel, TagA, TagB, TagC, TagD));
            }
            _ => unreachable!(),
        }
    }

    group.bench_function("ECS: Fragmented Iteration", |b| {
        let mut query = QueryState::<(&mut Position, &Velocity)>::new(&mut world.registry);
        b.iter(|| {
            query.for_each(&mut world, |(mut pos, vel)| {
                pos.x += vel.dx;
                black_box(pos);
            });
        });
    });

    group.finish();
}

fn cache_locality_soa_vs_aos(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scenario 3: Cache Locality (The Bloat Test)");
    // This is where ECS (SoA) shines. We iterate Position, but the entity
    // also has a 4KB MeshData component attached.
    // ECS should ignore the 4KB blob. AoS (Vec<(Pos, Mesh)>) has to load it into cache.

    let count = 10_000;
    let mut world = World::new();

    for i in 0..count {
        world.spawn((
            Position {
                x: i as f32,
                y: 0.0,
                z: 0.0,
            },
            MeshData { _bloat: [0; 4096] },
        ));
    }

    // AoS setup
    let mut aos_data: Vec<(Position, MeshData)> = (0..count)
        .map(|i| {
            (
                Position {
                    x: i as f32,
                    y: 0.0,
                    z: 0.0,
                },
                MeshData { _bloat: [0; 4096] },
            )
        })
        .collect();

    group.bench_function("ECS (SoA): Skip Bloat", |b| {
        // We ONLY request Position. The MeshData should be in a separate array.
        let mut query = QueryState::<&mut Position>::new(&mut world.registry);
        b.iter(|| {
            query.for_each(&mut world, |mut pos| {
                pos.x += 1.0;
                black_box(pos);
            });
        });
    });

    group.bench_function("Reference (AoS): Hit Bloat", |b| {
        b.iter(|| {
            // CPU must skip over 4096 bytes for every 12 bytes of data read.
            // This is a cache disaster.
            for (pos, _) in aos_data.iter_mut() {
                pos.x += 1.0;
                black_box(pos);
            }
        });
    });

    group.finish();
}

fn filtering_logic(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scenario 4: Filtering (With/Without)");

    let count = 20_000;
    let mut world = World::new();

    // 50% match rate
    for i in 0..count {
        let pos = Position {
            x: i as f32,
            y: 0.0,
            z: 0.0,
        };
        if i % 2 == 0 {
            world.spawn((pos, TagA)); // Match
        } else {
            world.spawn((pos, TagB)); // No Match
        }
    }

    group.bench_function("ECS: With<TagA>", |b| {
        // Should only iterate 10,000 entities
        let mut query = QueryState::<&mut Position, With<TagA>>::new(&mut world.registry);
        b.iter(|| {
            query.for_each(&mut world, |mut pos| {
                pos.x += 1.0;
                black_box(pos);
            });
        });
    });

    group.finish();
}

#[inline(never)]
fn query_iteration(
    world: &mut World,
    query: &mut QueryState<(&'static mut Position, &'static TransformMatrix)>,
) {
    query.for_each(world, |(mut pos, mat)| {
        // heavy math: Matrix multiplication simulation
        pos.x = pos.x * mat.m[0] + pos.y * mat.m[4] + pos.z * mat.m[8] + mat.m[12];
        pos.y = pos.x * mat.m[1] + pos.y * mat.m[5] + pos.z * mat.m[9] + mat.m[13];
        pos.z = pos.x * mat.m[2] + pos.y * mat.m[6] + pos.z * mat.m[10] + mat.m[14];
        black_box(pos);
    });
}

criterion_group!(
    benches,
    heavy_compute_simd,
    archetype_fragmentation,
    cache_locality_soa_vs_aos,
    filtering_logic
);
criterion_main!(benches);
