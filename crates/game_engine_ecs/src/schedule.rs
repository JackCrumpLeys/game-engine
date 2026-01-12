use game_engine_utils::graph::DirectedGraph;
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
    pub fn new(system: Box<dyn System>) -> Self {
        Self {
            system,
            dependencies: Vec::new(),
            sets: Vec::new(),
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
        vec![SystemConfig::new(Box::new(self.into_system()))]
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
        impl<$($name),*> IntoSystemConfigs<()> for ($($name,)*)
        where
            $($name: IntoSystemConfigs<()>,)*
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
    };
}

auto_impl_all_tuples!(impl_tuple_system_configs);

// ============================================================================
// 3. The Scheduler
// ============================================================================

pub struct Schedule {
    configs: Vec<SystemConfig>,

    /// Execution plan: Batches of system indices that can run in parallel.
    batches: Vec<Vec<usize>>,
    needs_rebuild: bool,
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
        }
    }

    /// Add one or more systems to the schedule.
    ///
    /// ## Example
    /// ```ignore
    /// schedule.add_systems((
    ///     system_a,
    ///     system_b.after(system_a),
    ///     (system_c, system_d).in_set(MySet)
    /// ));
    /// ```
    pub fn add_systems<M>(&mut self, configs: impl IntoSystemConfigs<M>) {
        let new_configs = configs.into_configs();
        for config in new_configs {
            self.configs.push(config);
        }
        self.needs_rebuild = true;
    }

    /// Builds the dependency graph and computes the execution batches.
    /// This is the most complex part of the scheduler.
    fn build_graph(&mut self, world: &World) {
        //    - Add a node for every system in `self.systems`.
        //    - For each dependency (A.before(B)), if B is a set, add an edge from A to every system in B.
        //    - Iterate `self.configs` and add edges for every `.before()` and `.after()` dependency.
        //    - For every pair of systems (A, B) that don't already have an edge between them:
        //      - If `systems_conflict(A.access(), B.access())` is true:
        //        - Add an edge A -> B (based on insertion order to be deterministic).
        //    - Check for Cycles: Use the graph library to detect cycles. Panic with a helpful error message.
        //    - Perform a topological sort on the graph.
        //    - Group the sorted nodes into parallel batches. A system goes into the current batch
        //      if all its dependencies are in previous batches AND it doesn't conflict with
        //      any other system already in the current batch.

        let mut graph = DirectedGraph::<&SystemConfig>::new();

        for node in &self.configs {
            graph.add_node(node);
        }

        self.needs_rebuild = false;
    }

    /// Executes one full frame of the schedule.
    pub fn run(&mut self, world: &mut World) {
        // 1. Check if the world's archetypes have changed, which requires updating system access.
        let access_changed = false;
        // for sys in &mut self.systems {
        //     if sys.update_access(world) {
        //         access_changed = true;
        //     }
        // }

        // 2. Rebuild the graph if systems were added or access patterns changed.
        if self.needs_rebuild || access_changed {
            self.build_graph(world);
        }

        // 3. Prepare for execution.
        world.increment_tick();
        let world_cell = UnsafeWorldCell::new(world);

        // 4. Run the batches.
        for batch in self.batches.clone() {
            self.run_batch_parallel(&world_cell, &batch);

            unsafe {
                world_cell.world_mut().flush();
            }
        }
    }

    /// Executes a single batch of systems in parallel.
    fn run_batch_parallel(&mut self, world: &UnsafeWorldCell, batch: &[usize]) {
        // Use a thread pool like Rayon to execute the systems.
        // The `System` trait needs to be `Send + Sync` for this.
        //
        // SAFETY: The `build_graph` logic guarantees that all systems in this batch
        // have disjoint `SystemAccess`, so they cannot cause data races or panics
        // related to your `AtomicBorrow` mechanism.
        todo!("Parallel execution logic using Rayon's `scope` or `par_iter`");
    }
}

// ============================================================================
// 5. Logical Utilities
// ============================================================================

/// Determines if two systems conflict based on their resource and component access.
fn systems_conflict(access_a: &SystemAccess, access_b: &SystemAccess) -> bool {
    // 1. Check for resource conflicts (Write vs Read/Write).
    // 2. Iterate through archetypes and check for component conflicts using `ColumnBorrowChecker::conflicts`.
    todo!()
}

// ============================================================================
// 6. Example Usage (for conceptual understanding)
// ============================================================================
