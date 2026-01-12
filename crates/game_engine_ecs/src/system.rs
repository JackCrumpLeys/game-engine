pub mod command;
pub mod function;

use crate::archetype::ArchetypeId;
use crate::borrow::ColumnBorrowChecker;
use crate::prelude::{Res, ResMut, Resource};
use crate::query::{Filter, GetGuard, QueryInner, QueryIter, QueryToken, View};
use crate::world::World;
use std::any::TypeId;
use std::collections::HashSet;
use std::ops::{Deref, DerefMut};

/// Unsafe access to the world.
/// This exists to allow splitting the borrow of World.
pub struct UnsafeWorldCell<'w> {
    world: *mut World,
    _marker: std::marker::PhantomData<&'w mut World>,
}

impl<'w> UnsafeWorldCell<'w> {
    pub fn new(world: &'w mut World) -> Self {
        Self {
            world: world as *mut World,
            _marker: std::marker::PhantomData,
        }
    }

    /// # Safety
    /// Caller must ensure no other mutable references to World exist,
    /// OR that this usage does not conflict with other active usages.
    pub unsafe fn world_mut(&self) -> &'w mut World {
        unsafe { &mut *self.world }
    }

    /// Returns a reference to the world of this [`UnsafeWorldCell`].
    ///
    /// # Safety
    ///
    /// Caller must ensure no conflicts arise from having multiple
    /// references to the World.
    pub unsafe fn world(&self) -> &'w World {
        unsafe { &*self.world }
    }
}

// SAFETY: We promise to verify disjoint access via the Scheduler before sending this across threads.
unsafe impl<'w> Send for UnsafeWorldCell<'w> {}
unsafe impl<'w> Sync for UnsafeWorldCell<'w> {}

/// Metadata about what a system accesses.
#[derive(Clone)]
pub struct SystemAccess {
    pub col: Vec<ColumnBorrowChecker>, // must be the same length as world.archetypes
    pub resources_read: HashSet<std::any::TypeId>,
    pub resources_write: HashSet<std::any::TypeId>,
}

impl SystemAccess {
    pub fn new(world: &World) -> Self {
        Self {
            col: vec![ColumnBorrowChecker::new(); world.archetypes.len()],
            resources_read: HashSet::new(),
            resources_write: HashSet::new(),
        }
    }

    pub fn new_empty() -> Self {
        Self {
            col: Vec::new(),
            resources_read: HashSet::new(),
            resources_write: HashSet::new(),
        }
    }

    pub fn read_resource<T: Resource>(&mut self) {
        self.resources_read.insert(std::any::TypeId::of::<T>());
    }

    pub fn write_resource<T: Resource>(&mut self) {
        self.resources_write.insert(std::any::TypeId::of::<T>());
    }
}

pub trait System: Send + Sync + 'static {
    /// Initialize the system.
    /// Registers access, allocates query caches, etc.
    /// Must be called before `run`.
    fn init(&mut self, world: &mut World);

    /// Runs the system.
    /// # Safety
    /// Scheduler must ensure no conflicts.
    unsafe fn run(&mut self, world: &UnsafeWorldCell);

    /// Returns the access requirements.
    /// Valid only after `init` is called.
    fn access(&self) -> &SystemAccess;

    /// Updates dynamic access (archetypes).
    /// Returns true if access changed.
    fn update_access(&mut self, world: &World) -> bool;
}

/// A parameter that can be passed into a system function.
pub trait SystemParam {
    /// The state cached between runs (e.g. ComponentIds, ArchetypeCache, or Local<T>)
    type State: Send + Sync + 'static;

    /// The item passed to the function (e.g. Query<'w, ...>)
    type Item<'w>;

    /// Does the accsess depend on world state (eg: adding/removing archetypes)?
    const DYNAMIC: bool = false;

    /// Called once when the system is initialized.
    /// Populates 'access' for the scheduler.
    fn init_state(world: &mut World, access: &mut SystemAccess) -> Self::State;

    /// Called every frame to fetch the parameter.
    ///
    /// # Safety
    ///
    /// The scheduler ensures 'access' is respected, so this is safe to call
    /// if the scheduler logic is correct.
    unsafe fn get_param<'w>(
        state: &'w mut Self::State,
        world: &'w UnsafeWorldCell<'w>,
    ) -> Self::Item<'w>;

    /// Called every frame IF `DYNAMIC` is true.
    /// returns true if the access has changed.
    /// Becouse Archetypes cannot be removed we assume accses is strictly increasing.
    fn update_access(_state: &mut Self::State, _world: &World, _access: &mut SystemAccess) -> bool {
        false
    }
}

/// Local<T> Provides a way to store data local to a system.
/// eg: Local<f32> will provide a f32 that is unique to the system instance,
/// mutable, and persists across system runs.
#[derive(Debug)]
pub struct Local<'w, T: Send + Sync + Default> {
    inner: &'w mut T,
}

impl<'w, T: Send + Sync + Default> Deref for Local<'w, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

impl<'w, T: Send + Sync + Default> DerefMut for Local<'w, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner
    }
}

impl<T: Send + Sync + Default + 'static> SystemParam for Local<'_, T> {
    type State = T;
    type Item<'w> = Local<'w, T>;

    fn init_state(_world: &mut World, _access: &mut SystemAccess) -> Self::State {
        T::default()
    }

    unsafe fn get_param<'w>(
        state: &'w mut Self::State,
        _world: &UnsafeWorldCell<'w>,
    ) -> Self::Item<'w> {
        Local { inner: state }
    }
}

/// Res<T> provides immutable access to a Resource of type T.
impl<T: Send + Sync + 'static> SystemParam for Res<'_, T> {
    type State = ();
    type Item<'w> = Res<'w, T>;

    fn init_state(_world: &mut World, access: &mut SystemAccess) -> Self::State {
        access.resources_read.insert(TypeId::of::<T>());
    }

    unsafe fn get_param<'w>(
        _state: &'w mut Self::State,
        world: &UnsafeWorldCell<'w>,
    ) -> Self::Item<'w> {
        let world = unsafe { world.world() };
        world
            .resources()
            .get::<T>()
            .unwrap_or_else(|| panic!("Resource of type {} not found", std::any::type_name::<T>()))
    }
}

/// ResMut<T> provides mutable access to a Resource of type T.
impl<T: Send + Sync + 'static> SystemParam for ResMut<'_, T> {
    type State = ();
    type Item<'w> = ResMut<'w, T>;

    fn init_state(_world: &mut World, access: &mut SystemAccess) -> Self::State {
        access.resources_write.insert(TypeId::of::<T>());
    }

    unsafe fn get_param<'w>(
        _state: &'w mut Self::State,
        world: &UnsafeWorldCell<'w>,
    ) -> Self::Item<'w> {
        // Safety: The scheduler ensures mutable access of this specific resource is safe.
        let world = unsafe { world.world_mut() };
        world
            .resources_mut()
            .get_mut::<T>()
            .unwrap_or_else(|| panic!("Resource of type {} not found", std::any::type_name::<T>()))
    }
}

pub struct Query<'w, Q: QueryToken, F: Filter = ()> {
    // Maps Q -> Q::Persistent (The static data struct)
    query: &'w mut QueryInner<Q::Persistent, F::Persistent>,
    world: &'w UnsafeWorldCell<'w>,
    tick: u32,
}

impl<Q: QueryToken, F: Filter> SystemParam for Query<'_, Q, F> {
    /// Query_inner, Last known archetype idx, World tick of last system run
    type State = (QueryInner<Q::Persistent, F::Persistent>, ArchetypeId, u32);
    type Item<'w> = Query<'w, Q, F>;

    const DYNAMIC: bool = true;

    fn init_state(world: &mut World, access: &mut SystemAccess) -> Self::State {
        // Use the factory method to create the static Inner from dynamic types
        let mut query = QueryInner::new::<Q, F>(&mut world.registry);

        for (id, col) in query.granular_borrow_check(world, ArchetypeId(0)) {
            access.col[id.0].extend(&col);
        }

        (
            query,
            ArchetypeId(world.archetypes.len()),
            world.tick().wrapping_sub(1),
        )
    }

    unsafe fn get_param<'w>(
        state: &'w mut Self::State,
        world: &'w UnsafeWorldCell<'w>,
    ) -> Self::Item<'w> {
        let res = Query {
            query: &mut state.0,
            world,
            tick: state.2,
        };
        // SAFETY: The scheduler ensures that the tick is static during this system run.
        state.2 = unsafe { world.world().tick() };
        res
    }

    fn update_access(state: &mut Self::State, world: &World, access: &mut SystemAccess) -> bool {
        let mut changed = false;
        if *state.0.last_updated_arch_idx() != world.archetypes.len() {
            for (id, col) in state.0.granular_borrow_check(world, state.1) {
                changed = if let Some(system_col) = access.col.get_mut(id.0) {
                    system_col.extend(&col)
                } else {
                    // New archetype added since last frame, extend the access vector
                    access.col.push(col);
                    true
                };
            }
        }
        state.1 = ArchetypeId(world.archetypes.len());
        changed
    }
}

impl<'w, Q: QueryToken, F: Filter> Query<'w, Q, F> {
    /// Returns an iterator over the query results.
    /// If you are doing a for loop over this, consider using
    /// for_each instead to have a more optimized hot loop iteration.
    pub fn iter(&mut self) -> QueryIter<'_, Q::View<'_>, F> {
        // Saftety at this point the column borrow checks have been done by the scheduler.
        debug_assert!(
            unsafe { self.world.world().archetypes.len() } == self.query.last_updated_arch_idx().0
        );
        // Explicitly pass the View and Filter types to the generic iter method
        self.query.iter::<Q::View<'_>, F>(self.world, self.tick)
    }

    /// runs a closure for each item in the query.
    pub fn for_each<'a, Func>(&'a mut self, func: Func)
    where
        Func: FnMut(<Q::View<'a> as View<'a>>::Item),
    {
        // Saftety at this point the column borrow checks have been done by the scheduler.
        let world = unsafe { self.world.world_mut() };
        debug_assert!(world.archetypes.len() == self.query.last_updated_arch_idx().0);
        self.query
            .for_each::<Q::View<'a>, F, Func>(self.world, func, self.tick)
    }

    /// get a specific entity's query item, if it exists.
    pub fn get(
        &mut self,
        entity: crate::prelude::Entity,
    ) -> Option<GetGuard<'_, <Q::View<'_> as View<'_>>::Item>> {
        let world = unsafe { self.world.world_mut() };
        debug_assert!(world.archetypes.len() == self.query.last_updated_arch_idx().0);
        // debug_assert!(world.archetypes.len() == self.query.last_updated_arch_idx().0 as usize);
        self.query
            .get::<Q::View<'_>, F>(self.world, entity, self.tick)
    }
}

// Macro to implement SystemParam for tuples (A, B, ...)
macro_rules! impl_system_param_tuple {
    ($($name:ident $_:tt),*) => {
        impl<$($name: SystemParam),*> SystemParam for ($($name,)*) {
            type State = ($($name::State,)*);
            type Item<'w> = ($($name::Item<'w>,)*);

            // If ANY param is dynamic, the tuple is dynamic
            const DYNAMIC: bool = $($name::DYNAMIC)||*;

            fn init_state(world: &mut World, access: &mut SystemAccess) -> Self::State {
                (
                    $($name::init_state(world, access),)*
                )
            }

            unsafe fn get_param<'w>(state: &'w mut Self::State, world: &'w UnsafeWorldCell<'w>) -> Self::Item<'w> {
                #[allow(non_snake_case)]
                let ($($name,)*) = state;
                unsafe {
                    (
                        $($name::get_param($name, world),)*
                    )
                }
            }

            fn update_access(
                state: &mut Self::State,
                world: &World,
                access: &mut SystemAccess,
            ) -> bool {
                #[allow(non_snake_case)]
                let ($($name,)*) = state;
                $(
                     $name::update_access($name, world, access)
                )||*
            }
        }
    };
}

auto_impl_all_tuples!(impl_system_param_tuple);
