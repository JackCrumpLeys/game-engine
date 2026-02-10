use core_affinity::get_core_ids;
use core_affinity::set_for_current;
use game_engine_utils::graph::DirectedGraph;
use game_engine_utils::graph::DirectedGraphOperations;
use game_engine_utils::graph::NodeIndex;
use game_engine_utils::{DynEq, DynHash};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

use crate::prelude::{IntoSystem, System, World};
use crate::system::{SystemAccess, UnsafeWorldCell};
use std::any::TypeId;
use std::marker::PhantomData;

// ============================================================================
// Labels, Sets, and System Identification
// ============================================================================

/// A marker trait for grouping systems.
///
/// Implement this with `#[derive(SystemSet)]` on an enum or struct.
pub trait SystemSet: DynHash + Debug + Send + Sync + 'static {
    /// Clone the `SystemSet` as a boxed trait object.
    fn dyn_clone(&self) -> Box<dyn SystemSet>;
}

impl PartialEq for dyn SystemSet {
    fn eq(&self, other: &Self) -> bool {
        self.dyn_eq(other.as_dyn_eq())
    }
}

impl Eq for dyn SystemSet {}

impl Hash for dyn SystemSet {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dyn_hash(state);
    }
}

impl Clone for Box<dyn SystemSet> {
    fn clone(&self) -> Self {
        self.dyn_clone()
    }
}

/// A special `SystemSet` that implicitly wraps a single system.
///
/// This is the key to making `system_a.before(system_b)` work. The scheduler
/// treats `system_b` as `SystemTypeSet<Type_Of_System_B>`.
pub struct SystemTypeSet<T: 'static>(PhantomData<fn() -> T>);

impl<T: 'static> SystemTypeSet<T> {
    pub(crate) fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T> Debug for SystemTypeSet<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("SystemTypeSet")
            .field(&std::any::type_name::<T>())
            .finish()
    }
}

impl<T> Hash for SystemTypeSet<T> {
    fn hash<H: Hasher>(&self, _state: &mut H) {
        // all systems of a given type are the same
    }
}
impl<T> Clone for SystemTypeSet<T> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<T> Copy for SystemTypeSet<T> {}

impl<T> PartialEq for SystemTypeSet<T> {
    #[inline]
    fn eq(&self, _other: &Self) -> bool {
        // all systems of a given type are the same
        true
    }
}

impl<T> Eq for SystemTypeSet<T> {}

impl<T> SystemSet for SystemTypeSet<T> {
    fn dyn_clone(&self) -> Box<dyn SystemSet> {
        Box::new(*self)
    }
}

/// A trait that converts a function or a `SystemSet` into its canonical Set representation.
pub trait IntoSystemSet<Marker> {
    type Set: SystemSet;
    fn into_set(&self) -> Self::Set;
}

// Blanket implementation for functions/closures that are systems.
impl<S, Marker> IntoSystemSet<Marker> for S
where
    S: IntoSystem<Marker>,
{
    // The set is a unique type based on the function's final system type.
    type Set = SystemTypeSet<S::System>;
    fn into_set(&self) -> Self::Set {
        SystemTypeSet::new()
    }
}

/// A unique, opaque identifier for a system instance within the schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SystemId(pub usize);

/// A unique identifier for any node in the dependency graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeId {
    System(SystemId),
    Set(TypeId),
}

// ============================================================================
// Configuration & Fluent API
// ============================================================================

pub enum DependencyKind {
    Before,
    After,
}

type BoxedSystemSet = Box<dyn SystemSet>;

pub struct Dependency {
    pub kind: DependencyKind,
    pub target: BoxedSystemSet,
}

/// Holds metadata about a system's placement in the schedule before the graph is built.
pub struct SystemConfig {
    pub(crate) system: Box<dyn System>,
    pub(crate) dependencies: Vec<Dependency>,
    pub(crate) sets: Vec<BoxedSystemSet>,
}

impl SystemConfig {
    pub fn new(system: Box<dyn System>, self_set: BoxedSystemSet) -> Self {
        Self {
            system,
            dependencies: Vec::new(),
            sets: vec![self_set],
        }
    }

    /// Run this system before the target system or set.
    pub fn before<Marker>(mut self, target: &impl IntoSystemSet<Marker>) -> Self {
        self.dependencies.push(Dependency {
            kind: DependencyKind::Before,
            target: target.into_set().dyn_clone(),
        });
        self
    }

    /// Run this system after the target system or set.
    pub fn after<Marker>(mut self, target: &impl IntoSystemSet<Marker>) -> Self {
        self.dependencies.push(Dependency {
            kind: DependencyKind::After,
            target: target.into_set().dyn_clone(),
        });
        self
    }

    /// Add this system to a specific group (a `SystemSet`).
    pub fn in_set<Marker>(mut self, target: &impl IntoSystemSet<Marker>) -> Self {
        self.sets.push(Box::new(target.into_set()));
        self
    }

    /// Get the system
    pub fn system(&self) -> &dyn System {
        self.system.as_ref()
    }
}

/// A trait that allows various types (like functions or tuples of systems)
/// to be converted into a collection of `SystemConfig`.
pub trait IntoSystemConfigs<Marker> {
    fn into_configs(self) -> Vec<SystemConfig>;

    fn before<TargetMarker>(self, target: impl IntoSystemSet<TargetMarker>) -> Vec<SystemConfig>
    where
        Self: Sized,
    {
        self.into_configs()
            .into_iter()
            .map(|config| config.before(&target))
            .collect()
    }

    fn after<TargetMarker>(self, target: impl IntoSystemSet<TargetMarker>) -> Vec<SystemConfig>
    where
        Self: Sized,
    {
        self.into_configs()
            .into_iter()
            .map(|config| config.after(&target))
            .collect()
    }

    fn in_set<TargetMarker>(self, target: impl IntoSystemSet<TargetMarker>) -> Vec<SystemConfig>
    where
        Self: Sized,
    {
        self.into_configs()
            .into_iter()
            .map(|config| config.in_set(&target))
            .collect()
    }
}

// Base case: A single system (or a function that can become one).
impl<S, Marker> IntoSystemConfigs<Marker> for S
where
    S: IntoSystem<Marker>,
{
    fn into_configs(self) -> Vec<SystemConfig> {
        let system_set = self.into_set().dyn_clone();
        vec![SystemConfig::new(Box::new(self.into_system()), system_set)]
    }
}

// Base case: An already-configured system.
impl IntoSystemConfigs<()> for SystemConfig {
    fn into_configs(self) -> Vec<SystemConfig> {
        vec![self]
    }
}

impl IntoSystemConfigs<()> for Vec<SystemConfig> {
    fn into_configs(self) -> Vec<SystemConfig> {
        self
    }
}

// macro produces for m!(A,0, B,1) => impl for (A, B) (nums unused)
//impl<A, B> IntoSystemConfigs<()> for (A, B)
// where
//     A: IntoSystemConfigs<()>,
//     B: IntoSystemConfigs<()>,
// {
//     fn into_configs(self) -> Vec<SystemConfig> {
//         #[allow(non_snake_case)]
//         let (A, B) = self;
//         vec![A.into_configs(), B.into_configs()]
//             .into_iter()
//             .flatten()
//             .collect()
//     }
// }

macro_rules! impl_tuple_system_configs {
    ($($name:ident $num: tt),*) => {
        paste::paste! {
            impl<$($name),*, $( [<$name Marker>] ),*> IntoSystemConfigs<($( [<$name Marker>] ),*)> for ($($name ,)*)
            where
                $($name: IntoSystemConfigs<[<$name Marker>]>,)*
            {
                fn into_configs(self) -> Vec<SystemConfig> {
                    #[allow(non_snake_case)]
                    let ($($name,)*) = self;
                    vec![$($name.into_configs(),)*]
                        .into_iter()
                        .flatten()
                        .collect()
                }
            }
        }
    };
}

impl_tuple_system_configs!(A 0,B 1);
impl_all_tuples!(@recurse impl_tuple_system_configs,(A 0,B 1),C 2,D 3,E 4,F 5,G 6,H 7,I 8,J 9,K 10,L 11,M 12,N 13,O 14,P 15,Q 16,R 17,S 18,T 19,U 20,V 21,W 22,X 23,Y 24,Z 25);

// ============================================================================
// 3. The Scheduler
// ============================================================================

pub struct Schedule {
    configs: Vec<SystemConfig>,

    /// Execution plan: Batches of system indices that can run in parallel.
    batches: Vec<Vec<usize>>,
    needs_rebuild: bool,
    last_seen_arch_count: usize,

    mask: ExecutedMask, // Replaces Vec<bool>
}

pub struct ExecutedMask {
    phase: bool,
    storage: Vec<bool>,
}

impl ExecutedMask {
    pub fn new() -> Self {
        Self {
            phase: false,
            storage: Vec::new(),
        }
    }

    /// Resets the mask for a new frame in O(1)
    pub fn reset(&mut self) {
        self.phase = !self.phase;
    }

    /// Returns true if the system at index i has been executed this frame
    pub fn is_executed(&mut self, i: usize) -> bool {
        if i >= self.storage.len() {
            self.storage.resize(i + 1, self.phase);
        }
        self.storage[i] != self.phase
    }

    /// Marks the system at index i as executed
    pub fn mark_executed(&mut self, i: usize) {
        if i >= self.storage.len() {
            self.storage.resize(i + 1, self.phase);
        }
        self.storage[i] = !self.phase;
    }
}

impl Default for Schedule {
    fn default() -> Self {
        Self::new()
    }
}

impl Schedule {
    pub fn new() -> Self {
        Self {
            configs: Vec::new(),
            // set_members: HashMap::new(),
            batches: Vec::new(),
            needs_rebuild: true,
            last_seen_arch_count: 0,
            mask: ExecutedMask::new(),
        }
    }

    /// Add one or more systems to the schedule.
    ///
    /// ## Example
    /// ```ignore
    /// schedule.add_systems((
    ///     system_a,
    ///     system_b.after(system_a),
    ///     (system_c, system_d).in_set(MySet),
    ///     &mut world
    /// ));
    /// ```
    pub fn add_systems<M>(&mut self, configs: impl IntoSystemConfigs<M>, world: &mut World) {
        let new_configs = configs.into_configs();
        for config in new_configs.into_iter() {
            let mut config = config;
            config.system.init(world);
            self.configs.push(config);
        }
        self.needs_rebuild = true;
    }

    fn update_archetype(&mut self, world: &World) {
        if world.archetypes.len() != self.last_seen_arch_count {
            self.last_seen_arch_count = world.archetypes.len();
            self.needs_rebuild = true;
        }
    }

    pub fn build_graph(&mut self, world: &World) {
        self.update_archetype(world);

        // Update system access patterns for the current world state
        for config in &mut self.configs {
            config.system.update_access(world);
        }

        let num_systems = self.configs.len();
        let mut graph = DirectedGraph::<usize>::new();

        // We add nodes where NodeData is the index into self.configs.
        // Because DirectedGraph uses a simple Vec and NodeIndex(usize),
        // node_indices[i] will effectively be NodeIndex(i).
        for i in 0..num_systems {
            graph.add_node(i);
        }

        // 2. Resolve explicit dependencies
        for (i, config) in self.configs.iter().enumerate() {
            for dep in &config.dependencies {
                // Check every other system to see if it satisfies the dependency target
                for j in 0..num_systems {
                    let matches = self.configs[j].sets.iter().any(|set| **set == *dep.target);

                    if matches {
                        match dep.kind {
                            DependencyKind::Before => {
                                // System i must run before System j
                                graph.add_edge(NodeIndex(i), NodeIndex(j), ());
                            }
                            DependencyKind::After => {
                                // System j must run before System i
                                graph.add_edge(NodeIndex(j), NodeIndex(i), ());
                            }
                        }
                    }
                }
            }
        }

        // This ensures that two systems touching the same data are never in the same batch.
        for i in 0..num_systems {
            for j in (i + 1)..num_systems {
                let access_a = self.configs[i].system().access();
                let access_b = self.configs[j].system().access();

                if systems_conflict(access_a, access_b) {
                    graph.add_edge(NodeIndex(i), NodeIndex(j), ());
                }
            }
        }

        match graph.phased_topological_sort() {
            Ok(phases) => {
                // Convert NodeIndex batches back to usize batches for self.batches
                self.batches = phases
                    .into_iter()
                    .map(|phase| phase.into_iter().map(|node_idx| node_idx.0).collect())
                    .collect();
            }
            Err(cycles) => {
                // In a real engine, you'd format this to show the names of the systems
                // involved in the cycle.
                panic!("Dependency cycle detected in Schedule: {:?}", cycles);
            }
        }

        self.needs_rebuild = false;
    }

    /// Executes one full frame of the schedule.
    pub fn update(&mut self, world: &mut World) {
        #[cfg(feature = "tracy")]
        let _trace = {
            let trace = tracy_client::span!("Update");
            trace.emit_color(0xFFFFFF);
            trace
        };

        self.update_archetype(world);

        if self.needs_rebuild {
            self.build_graph(world);
        }

        let mut batch_ptr = 0;
        while batch_ptr < self.batches.len() {
            world.increment_tick(); // Increment change tick.

            // Get unexecuted systems from current batch
            let batch: Vec<usize> = self.batches[batch_ptr]
                .iter()
                .filter(|&&idx| !self.mask.is_executed(idx))
                .copied()
                .collect();

            if batch.is_empty() {
                batch_ptr += 1;
                continue;
            }

            // Execute
            self.run_batch_parallel(world, batch_ptr);

            // Mark as done
            for &idx in &batch {
                self.mask.mark_executed(idx);
            }

            // Flush side effects
            world.flush();

            self.update_archetype(world);

            if self.needs_rebuild {
                let mut changed = false;
                for config in &mut self.configs {
                    if config.system.update_access(world) {
                        changed = true;
                    }
                }

                if changed {
                    self.build_graph(world);
                    batch_ptr = 0; // Restart iterator
                    continue;
                }
            }
            batch_ptr += 1;
        }
        self.mask.reset();

        world
            .thread_local
            .iter_mut()
            .for_each(|t| t.entity_allocator.inc_tick());
        world.increment_frame();

        #[cfg(feature = "tracy")]
        let _frame = tracy_client::frame_mark();
    }

    /// Executes foever or a number of times
    pub fn run(&mut self, world: &mut World, amount: Option<u64>) {
        let core_ids = get_core_ids().unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .thread_name(|i| format!("ecs-worker-{}", i))
            .start_handler(move |id| {
                set_for_current(core_ids[id % core_ids.len()]);
            })
            .build()
            .unwrap();

        pool.install(|| {
            let mut i = 0;
            while amount.is_none_or(|a| {
                let res = i < a;
                i += 1;
                res
            }) {
                self.update(world);
            }
        });
    }

    fn run_batch_parallel(&mut self, world: &mut World, batch: usize) {
        let batch = &self.batches[batch];

        if batch.len() <= 1 {
            for idx in batch {
                unsafe {
                    let system = &mut self.configs[*idx].system;
                    system.run(&UnsafeWorldCell::new(world));
                }
            }
            return;
        }

        let world_cell = &UnsafeWorldCell::new(world);
        rayon::scope(|s| {
            for idx in batch {
                let config = &mut self.configs[*idx] as *mut SystemConfig;
                let system = unsafe { &mut (&mut *config).system };
                s.spawn(move |_| unsafe { system.run(world_cell) });
            }
        });
    }
}

/// Helper to check if two systems have conflicting access patterns.
fn systems_conflict(access_a: &SystemAccess, access_b: &SystemAccess) -> bool {
    // 1. Check Resource Conflicts (Write-Read or Write-Write)
    for res_a in &access_a.resources_write {
        if access_b.resources_read.contains(res_a) || access_b.resources_write.contains(res_a) {
            return true;
        }
    }
    for res_b in &access_b.resources_write {
        if access_a.resources_read.contains(res_b) {
            return true;
        }
    }

    // 2. Check Component Column Conflicts
    // We iterate archetypes that both systems have evaluated access for.
    let common_len = access_a.col.len().min(access_b.col.len());
    for i in 0..common_len {
        if access_a.col[i].conflicts(&access_b.col[i]) {
            return true;
        }
    }

    false
}

macro_rules! impl_into_type_system_set {
    ($($name:ident),*) => {
        $(
            impl IntoSystemSet<()> for $name {
                type Set = SystemTypeSet<$name>;

                fn into_set(&self) -> Self::Set {
                    SystemTypeSet::new()
                }
            }
        )*
    };
}

#[cfg(test)]
mod scheduler_tests {
    use game_engine_derive::Component;

    use super::*;
    use crate::{
        prelude::*,
        system::{Local, command::Command},
    };
    use std::{
        hint::black_box,
        sync::{Arc, Mutex},
        time::Instant,
    };

    // --- MOCK DATA ---
    #[derive(Component, Debug, Default, Clone, Copy)]
    struct A;
    #[derive(Component, Debug, Default, Clone, Copy)]
    struct B;

    #[derive(Default)]
    struct ExecutionOrder(Arc<Mutex<Vec<&'static str>>>);

    #[derive(Default)]
    struct Counter(u32);

    fn sys_1(order: Res<ExecutionOrder>) {
        order.0.lock().unwrap().push("sys_1");
    }
    fn sys_2(order: Res<ExecutionOrder>) {
        order.0.lock().unwrap().push("sys_2");
    }
    fn sys_3(order: Res<ExecutionOrder>) {
        order.0.lock().unwrap().push("sys_3");
    }

    #[test]
    fn test_explicit_ordering_before_after() {
        let mut world = World::new();
        world.resources_mut().insert(ExecutionOrder::default());

        let mut schedule = Schedule::new();
        // 2 before 1, 3 after 1 => Expected 2, 1, 3
        schedule.add_systems((sys_1, sys_2.before(sys_1), sys_3.after(sys_1)), &mut world);

        schedule.update(&mut world);

        let order = world.resources().get::<ExecutionOrder>().unwrap();
        let results = order.0.lock().unwrap();
        assert_eq!(*results, vec!["sys_2", "sys_1", "sys_3"]);
    }

    #[test]
    fn test_resource_conflict_separation() {
        let mut world = World::new();
        world.resources_mut().insert(Counter(0));

        // System A writes, System B reads. They MUST be in different batches.
        fn writer(mut c: ResMut<Counter>) {
            c.0 += 1;
        }
        fn reader(_c: Res<Counter>) { /* read only */
        }

        let mut schedule = Schedule::new();
        schedule.add_systems((writer, reader), &mut world);

        // Build graph to inspect batches
        schedule.build_graph(&world);

        // Since there is a RW conflict on Counter, batches must be size 1
        // (Assuming internal conflict logic triggers correctly)
        assert_eq!(schedule.batches.len(), 2);
    }

    #[test]
    fn test_component_conflict_separation() {
        let mut world = World::new();
        world.spawn(A); // Create an archetype with A

        fn sys_a(mut _q: Query<&mut A>) {}
        fn sys_b(_q: Query<&A>) {}

        let mut schedule = Schedule::new();
        schedule.add_systems((sys_a, sys_b), &mut world);

        schedule.build_graph(&world);

        // Mut conflict on Component A means they cannot run in parallel
        assert_eq!(schedule.batches.len(), 2);
    }

    #[test]
    fn test_parallel_independent_systems() {
        let mut world = World::new();

        // These systems touch completely different data
        fn sys_a(_q: Query<&A>) {}
        fn sys_b(_q: Query<&B>) {}

        let mut schedule = Schedule::new();
        schedule.add_systems((sys_a, sys_b), &mut world);

        schedule.build_graph(&world);

        // Should be able to run in a single batch
        assert_eq!(schedule.batches.len(), 1);
        assert_eq!(schedule.batches[0].len(), 2);
    }

    #[test]
    fn basic_speed_test() {
        let mut world = World::new();

        // System 1 spawns an entity.
        // System 2 queries for that entity.
        // Because System 2 is explicitly "after" System 1,
        // the scheduler must flush the Command buffer between them.

        fn spawner(mut cmd: Command) {
            cmd.spawn(100i32);
        }

        fn checker(query: Query<&i32>, mut c: Local<usize>) {
            *c += query.iter().count();
            black_box(c);
        }

        let mut schedule = Schedule::new();
        schedule.add_systems((spawner, checker.after(spawner)), &mut world);
        let now = Instant::now();
        const TICKS: u64 = 10_000_000;
        schedule.run(&mut world, Some(TICKS));
        let time_taken = now.elapsed();
        println!(
            "{:?}. {:.2}TPS, Average tick time: {:?}",
            time_taken,
            TICKS as f64 / time_taken.as_secs_f64(),
            time_taken / TICKS as u32,
        );
        panic!();
    }

    #[test]
    fn heavy_stress_test() {
        let mut world = World::new();

        // Components
        #[derive(Component, Debug, Default, Clone, Copy)]
        struct Pos {
            x: f32,
            y: f32,
        }
        #[derive(Component, Debug, Default, Clone, Copy)]
        struct Vel {
            x: f32,
            y: f32,
        }
        #[derive(Component, Debug, Default, Clone, Copy)]
        struct Tag;

        // System 1: Updates Velocity (Simple)
        fn update_velocity(mut query: Query<&mut Vel>) {
            query.for_each_mut(|mut vel| {
                vel.x += 0.1;
                vel.y += 0.1;
            });
        }

        // System 2: Applies Velocity to Position (Read/Write)
        fn apply_velocity(mut query: Query<(&mut Pos, &Vel)>) {
            query.for_each_mut(|(mut pos, vel)| {
                pos.x += vel.x;
                pos.y += vel.y;
            });
        }

        // System 3: Structural Churn (Spawn/Despawn)
        // This forces the scheduler to re-evaluate the world every tick
        fn churn(mut cmd: Command, query: Query<Entity, With<Tag>>) {
            // Despawn all "Tag" entities
            query.for_each(|e| {
                cmd.despawn(e);
            });
            // Spawn 100 new ones to replace them
            for _ in 0..100 {
                cmd.spawn((Pos { x: 0., y: 0. }, Vel { x: 1., y: 1. }, Tag));
            }
        }

        // Setup: 100,000 base entities
        for _ in 0..100_000 {
            world.spawn((Pos { x: 0., y: 0. }, Vel { x: 1., y: 1. }));
        }

        let checked = world.spawn((Pos { x: 0., y: 0. }, Vel { x: 1., y: 1. }));

        let mut schedule = Schedule::new();
        // Order: Update Vel -> Apply Vel -> Churn
        schedule.add_systems(
            (
                update_velocity,
                apply_velocity.after(update_velocity),
                churn.after(apply_velocity),
            ),
            &mut world,
        );

        let now = Instant::now();
        const TICKS: u64 = 1000;
        schedule.run(&mut world, Some(TICKS));
        let time_taken = now.elapsed();

        println!(
            "Total: {:?}, Avg tick: {:?}",
            time_taken,
            time_taken / TICKS as u32
        );
        println!(
            "Proof: {:?}",
            *world.query::<&Pos, ()>().get(checked).unwrap()
        );

        panic!();
    }

    #[test]
    fn test_system_sets_ordering() {
        let mut world = World::new();
        world.resources_mut().insert(ExecutionOrder::default());

        #[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
        struct PhysicsSet;

        impl_into_type_system_set!(PhysicsSet);

        let mut schedule = Schedule::new();

        // Group 1 and 2 into Physics, make 3 run after the whole set
        schedule.add_systems(
            ((sys_1, sys_2).in_set(PhysicsSet), sys_3.after(PhysicsSet)),
            &mut world,
        );

        schedule.update(&mut world);

        let order = world.resources().get::<ExecutionOrder>().unwrap();
        let results = order.0.lock().unwrap();

        // sys_3 must be last
        assert_eq!(results[2], "sys_3");
        // sys_1 and sys_2 can be in any order in the first batch (0 and 1)
        assert!(results.contains(&"sys_1"));
        assert!(results.contains(&"sys_2"));
    }

    #[test]
    #[should_panic(expected = "Dependency cycle detected")]
    fn test_cycle_detection() {
        let mut world = World::new();
        let mut schedule = Schedule::new();
        world.resources_mut().insert(ExecutionOrder::default());

        // A -> B -> A
        schedule.add_systems((sys_1.before(sys_2), sys_2.before(sys_1)), &mut world);

        schedule.update(&mut world); // This should panic
    }

    #[test]
    fn test_flush_between_batches() {
        let mut world = World::new();

        // System 1 spawns an entity.
        // System 2 queries for that entity.
        // Because System 2 is explicitly "after" System 1,
        // the scheduler must flush the Command buffer between them.

        fn spawner(mut cmd: Command) {
            cmd.spawn(A);
        }

        fn checker(query: Query<&A>, order: Res<ExecutionOrder>) {
            if query.iter().count() > 0 {
                order.0.lock().unwrap().push("found_a");
            }
        }

        world.resources_mut().insert(ExecutionOrder::default());

        let mut schedule = Schedule::new();
        schedule.add_systems((spawner, checker.after(spawner)), &mut world);

        schedule.update(&mut world);

        let order = world.resources().get::<ExecutionOrder>().unwrap();
        let results = order.0.lock().unwrap();
        assert_eq!(
            *results,
            vec!["found_a"],
            "System 2 did not see entity from System 1 flush"
        );
    }

    #[test]
    fn test_archetype_update_rebuilds_graph() {
        let mut world = World::new();

        fn sys_a(mut _q: Query<&mut A>) {}
        fn sys_b(_q: Query<&A>) {}

        let mut schedule = Schedule::new();
        schedule.add_systems((sys_a, sys_b), &mut world);

        // First run: World is empty. No archetypes. Conflict check might not see A.
        schedule.update(&mut world);

        // Now we spawn an entity with A. The scheduler should detect a new archetype
        // and rebuild the graph on the next update to ensure sys_a and sys_b don't collide.
        world.spawn(A);

        schedule.update(&mut world);

        // Check if graph was marked for rebuild (needs_rebuild is set false at end of build_graph)
        assert!(!schedule.needs_rebuild);
        // If conflict detection is working, they should be in separate batches now that A exists
        assert_eq!(schedule.batches.len(), 2);
    }
}
