use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use game_engine_derive::Component;
use game_engine_ecs::query::Mut;
use std::hint::black_box;

// Adjust this crate name to match your Cargo.toml name
use game_engine_ecs::prelude::*;
use game_engine_ecs::system::{System, UnsafeWorldCell};

// ============================================================================
// COMPONENTS
// ============================================================================

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

// ============================================================================
// ECS SYSTEMS
// ============================================================================

// System 1: Movement (Pos += Vel)
fn sys_movement(mut query: game_engine_ecs::system::Query<(&mut Position, &Velocity)>) {
    query.for_each(|(mut pos, vel)| {
        pos.x += vel.x;
        pos.y += vel.y;
    });
}

// System 2: Boundaries (Clamp Position)
fn sys_boundaries(mut query: game_engine_ecs::system::Query<&mut Position>) {
    query.for_each(|mut pos| {
        pos.x = pos.x.clamp(-1000.0, 1000.0);
        pos.y = pos.y.clamp(-1000.0, 1000.0);
    });
}

// System 3: Rotation (Spin)
fn sys_rotate(mut query: game_engine_ecs::system::Query<&mut Rotation>) {
    query.for_each(|mut rot| {
        rot.radians += 0.01;
    });
}

// System 4: Transform (Heavy-ish Math: Matrix from Pos + Rot)
fn sys_update_transform(mut query: Query<(&Position, &Rotation, &mut TransformMatrix)>) {
    query.for_each(
        |(pos, rot, mut mat): (&Position, &Rotation, Mut<'_, TransformMatrix>)| {
            let c = rot.radians.cos();
            let s = rot.radians.sin();
            // 2D Rotation Matrix logic dumped into 4x4 array
            mat.data[0] = c;
            mat.data[1] = -s;
            mat.data[4] = s;
            mat.data[5] = c;
            mat.data[12] = pos.x;
            mat.data[13] = pos.y;
            // Keep CPU busy
            black_box(mat);
        },
    );
}

// System 5: Life Logic (Split logic: Regen adds, Poison removes)
// This tests query filtering capabilities of ECS
fn sys_life_logic(
    mut q_regen: game_engine_ecs::system::Query<(&mut Health, &Regen)>,
    mut q_poison: game_engine_ecs::system::Query<
        &mut Health,
        game_engine_ecs::query::With<Poisoned>,
    >,
) {
    // 5a. Apply Regen
    q_regen.for_each(|(mut hp, regen)| {
        hp.val += regen.rate;
    });

    // 5b. Apply Poison
    q_poison.for_each(|mut hp| {
        hp.val -= 1.0;
    });
}

// ============================================================================
// VEC COUNTERPARTS
// ============================================================================

// Struct of Arrays container for the Vec benchmark
struct VecWorld {
    pos: Vec<Position>,
    vel: Vec<Velocity>,
    rot: Vec<Rotation>,
    mat: Vec<TransformMatrix>,
    hp: Vec<Health>,
    // Sparse data handling for Vecs (simulating component optionality)
    regen: Vec<Option<Regen>>,
    is_poisoned: Vec<bool>,
}

impl VecWorld {
    fn new() -> Self {
        Self {
            pos: Vec::new(),
            vel: Vec::new(),
            rot: Vec::new(),
            mat: Vec::new(),
            hp: Vec::new(),
            regen: Vec::new(),
            is_poisoned: Vec::new(),
        }
    }

    fn push_entity(
        &mut self,
        pos: Position,
        vel: Velocity,
        rot: Rotation,
        hp: Health,
        regen: Option<Regen>,
        poisoned: bool,
    ) {
        self.pos.push(pos);
        self.vel.push(vel);
        self.rot.push(rot);
        self.mat.push(TransformMatrix::default());
        self.hp.push(hp);
        self.regen.push(regen);
        self.is_poisoned.push(poisoned);
    }
}

// Vec System 1
fn vec_sys_movement(pos: &mut [Position], vel: &[Velocity]) {
    for (p, v) in pos.iter_mut().zip(vel.iter()) {
        p.x += v.x;
        p.y += v.y;
    }
}

// Vec System 2
fn vec_sys_boundaries(pos: &mut [Position]) {
    for p in pos.iter_mut() {
        p.x = p.x.clamp(-1000.0, 1000.0);
        p.y = p.y.clamp(-1000.0, 1000.0);
    }
}

// Vec System 3
fn vec_sys_rotate(rot: &mut [Rotation]) {
    for r in rot.iter_mut() {
        r.radians += 0.01;
    }
}

// Vec System 4
fn vec_sys_update_transform(pos: &[Position], rot: &[Rotation], mat: &mut [TransformMatrix]) {
    for ((p, r), m) in std::iter::zip(pos, rot).zip(mat) {
        let c = r.radians.cos();
        let s = r.radians.sin();
        m.data[0] = c;
        m.data[1] = -s;
        m.data[4] = s;
        m.data[5] = c;
        m.data[12] = p.x;
        m.data[13] = p.y;
        black_box(m);
    }
}

// Vec System 5
fn vec_sys_life_logic(hp: &mut [Health], regen: &[Option<Regen>], poisoned: &[bool]) {
    // This demonstrates the weakness of pure Vecs: Iterating sparse data or flags requires checking every index
    // unless you maintain separate lists (which effectively reinvents archetypes).
    for i in 0..hp.len() {
        if let Some(r) = &regen[i] {
            hp[i].val += r.rate;
        }
        if poisoned[i] {
            hp[i].val -= 1.0;
        }
    }
}

// ============================================================================
// BENCHMARKS
// ============================================================================

const ENTITY_COUNT: usize = 20_000;

fn bench_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Allocation (Spawn 20k)");

    group.bench_function("ECS Spawn", |b| {
        b.iter_batched(
            World::new,
            |mut world| {
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
                        world.spawn_deferred((
                            pos,
                            vel,
                            rot,
                            hp,
                            Poisoned,
                            TransformMatrix::default(),
                        ));
                    }
                }
                world.flush();
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("Vec Push", |b| {
        b.iter(|| {
            let mut world = VecWorld::new();
            // Pre-reserve to be fair (ECS archetypes usually grow, but Vecs can reserve)
            // Even without reserve, Vec is usually faster at raw insert.
            world.pos.reserve(ENTITY_COUNT);

            for i in 0..ENTITY_COUNT {
                let pos = Position {
                    x: i as f32,
                    y: 0.0,
                };
                let vel = Velocity { x: 1.0, y: 1.0 };
                let rot = Rotation { radians: 0.0 };
                let hp = Health { val: 100.0 };

                if i.is_multiple_of(2) {
                    world.push_entity(pos, vel, rot, hp, Some(Regen { rate: 1.0 }), false);
                } else {
                    world.push_entity(pos, vel, rot, hp, None, true);
                }
            }
            black_box(world);
        })
    });

    group.finish();
}
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

fn bench_systems_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("Systems Execution (5 Systems, 20k Ents)");

    // --- SETUP ECS ---
    let mut ecs_world = World::new();
    for i in 0..ENTITY_COUNT {
        let pos = Position {
            x: i as f32,
            y: 0.0,
        };
        let vel = Velocity { x: 1.0, y: 1.0 };
        let rot = Rotation { radians: 0.0 };
        let hp = Health { val: 100.0 };
        if i.is_multiple_of(2) {
            ecs_world.spawn_deferred((
                pos,
                vel,
                rot,
                hp,
                Regen { rate: 1.0 },
                TransformMatrix::default(),
            ));
        } else {
            ecs_world.spawn_deferred((pos, vel, rot, hp, Poisoned, TransformMatrix::default()));
        }
    }
    ecs_world.flush();

    let mut app = App::new();

    app.world = ecs_world;

    app.add_system(sys_movement);
    app.add_system(sys_boundaries);
    app.add_system(sys_rotate);
    app.add_system(sys_update_transform);
    app.add_system(sys_life_logic);

    // --- SETUP VEC ---
    let mut vec_world = VecWorld::new();
    vec_world.pos.reserve(ENTITY_COUNT);
    for i in 0..ENTITY_COUNT {
        let pos = Position {
            x: i as f32,
            y: 0.0,
        };
        let vel = Velocity { x: 1.0, y: 1.0 };
        let rot = Rotation { radians: 0.0 };
        let hp = Health { val: 100.0 };
        if i.is_multiple_of(2) {
            vec_world.push_entity(pos, vel, rot, hp, Some(Regen { rate: 1.0 }), false);
        } else {
            vec_world.push_entity(pos, vel, rot, hp, None, true);
        }
    }

    // --- RUN BENCH ---

    group.bench_function("ECS Systems", |b| {
        b.iter(|| {
            app.run();
        });
    });

    group.bench_function("Vec Systems", |b| {
        b.iter(|| {
            vec_sys_movement(&mut vec_world.pos, &vec_world.vel);
            vec_sys_boundaries(&mut vec_world.pos);
            vec_sys_rotate(&mut vec_world.rot);
            vec_sys_update_transform(&vec_world.pos, &vec_world.rot, &mut vec_world.mat);
            vec_sys_life_logic(&mut vec_world.hp, &vec_world.regen, &vec_world.is_poisoned);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_allocation, bench_systems_execution);
criterion_main!(benches);
