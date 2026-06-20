pub mod command;
pub mod function;

use game_engine_utils::NoOpHash;
use smallvec::SmallVec;

use crate::archetype::ArchetypeId;
use crate::borrow::ColumnBorrowChecker;
use crate::prelude::{Res, ResMut, Resource};
use crate::query::{
    Filter, GetGuard, QueryInner, QueryIter, QueryToken, ReadOnly, SkipFilter, View,
};
use crate::resource::{ResourceId, TLRes, TLResMut};
use crate::world::{ChangeTick, World};
use std::any::TypeId;
use std::collections::HashSet;
use std::ops::{Deref, DerefMut};

/// Unsafe access to the world.
/// This exists to allow splitting the borrow of World.
#[derive(Clone)]
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

pub type TypeIdSet = HashSet<TypeId, NoOpHash>;

/// Metadata about what a system accesses.
#[derive(Clone, Debug)]
pub struct SystemAccess {
    pub col: Vec<ColumnBorrowChecker>, // must be the same length as world.archetypes
    pub resources_read: TypeIdSet,
    pub resources_write: TypeIdSet,
}

impl SystemAccess {
    pub fn new(world: &World) -> Self {
        Self {
            col: vec![ColumnBorrowChecker::new(); world.archetypes.len()],
            resources_read: TypeIdSet::default(),
            resources_write: TypeIdSet::default(),
        }
    }

    pub fn new_empty() -> Self {
        Self {
            col: Vec::new(),
            resources_read: TypeIdSet::default(),
            resources_write: TypeIdSet::default(),
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

pub struct Single;
pub struct Many;

/// A parameter that can be passed into a system function.
pub trait SystemParam {
    type Marker: Send + Sync + 'static = Single;
    /// The state cached between runs
    type State: Send + Sync + 'static;

    /// The item passed to the function
    ///
    /// 'w for the world
    /// 's for the systyem
    type Item<'w, 's>;

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
    unsafe fn get_param<'w, 's>(
        state: &'w mut Self::State,
        world: &'w UnsafeWorldCell<'w>,
    ) -> Self::Item<'w, 's>;

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
    type Item<'w, 's> = Local<'w, T>;

    fn init_state(_world: &mut World, _access: &mut SystemAccess) -> Self::State {
        T::default()
    }

    unsafe fn get_param<'w, 's>(
        state: &'w mut Self::State,
        _world: &UnsafeWorldCell<'w>,
    ) -> Self::Item<'w, 's> {
        Local { inner: state }
    }
}

/// Res<T> provides immutable access to a Resource of type T.
impl<T: Send + Sync + 'static> SystemParam for Res<'_, T> {
    type State = ResourceId;
    type Item<'w, 's> = Res<'w, T>;

    fn init_state(world: &mut World, access: &mut SystemAccess) -> Self::State {
        access.resources_read.insert(TypeId::of::<T>());

        world.resources().get_id::<T>().unwrap_or_else(|| {
            panic!(
                "Resource of type {} not found. Insert before adding systems using it.",
                std::any::type_name::<T>()
            )
        })
    }

    unsafe fn get_param<'w, 's>(
        res_id: &'w mut Self::State,
        world: &UnsafeWorldCell<'w>,
    ) -> Self::Item<'w, 's> {
        let world = unsafe { world.world() };
        world
            .resources()
            .get_from_id(*res_id)
            .unwrap_or_else(|| panic!("Resource of type {} not found", std::any::type_name::<T>()))
    }
}

/// ResMut<T> provides mutable access to a Resource of type T.
impl<T: Send + Sync + 'static> SystemParam for ResMut<'_, T> {
    type State = ResourceId;
    type Item<'w, 's> = ResMut<'w, T>;

    fn init_state(world: &mut World, access: &mut SystemAccess) -> Self::State {
        access.resources_write.insert(TypeId::of::<T>());

        world.resources().get_id::<T>().unwrap_or_else(|| {
            panic!(
                "Resource of type {} not found. Insert before adding systems using it.",
                std::any::type_name::<T>()
            )
        })
    }

    unsafe fn get_param<'w, 's>(
        res_id: &'w mut Self::State,
        world: &UnsafeWorldCell<'w>,
    ) -> Self::Item<'w, 's> {
        // Safety: The scheduler ensures mutable access of this specific resource is safe.
        let world = unsafe { world.world_mut() };

        world
            .resources_mut()
            .get_mut_from_id(*res_id)
            .unwrap_or_else(|| panic!("Resource of type {} not found", std::any::type_name::<T>()))
    }
}

/// TLRes<T> provides immutable access to a thread-local Resource of type T.
impl<T: Send + Sync + 'static> SystemParam for TLRes<'_, T> {
    type State = ();
    type Item<'w, 's> = TLRes<'w, T>;
    const DYNAMIC: bool = false;

    fn init_state(_world: &mut World, _access: &mut SystemAccess) -> Self::State {}

    unsafe fn get_param<'w, 's>(
        _state: &'w mut Self::State,
        world: &'w UnsafeWorldCell<'w>,
    ) -> Self::Item<'w, 's> {
        let world = unsafe { world.world() };
        let res = world
            .thread_local_res()
            .get::<T>()
            .unwrap_or_else(|| panic!("ThreadLocalResource of type {} not found. Did you insert it into thread_local_res_mut()?", std::any::type_name::<T>()));
        TLRes { value: res }
    }
}

/// TLResMut<T> provides mutable access to a thread-local Resource of type T.
impl<T: Send + Sync + 'static> SystemParam for TLResMut<'_, T> {
    type State = ();
    type Item<'w, 's> = TLResMut<'w, T>;
    const DYNAMIC: bool = false;

    fn init_state(_world: &mut World, _access: &mut SystemAccess) -> Self::State {}

    unsafe fn get_param<'w, 's>(
        _state: &'w mut Self::State,
        world: &'w UnsafeWorldCell<'w>,
    ) -> Self::Item<'w, 's> {
        // Safety: Thread-local res mut access is safe because ecs_thread_id() ensures isolation.
        let world = unsafe { world.world() };
        // We use world_mut() to satisfy borrow checker, but physically thread_local_res_mut() doesn't need global &mut World if we use internal mutations, but for now world.thread_local_res_mut() uses &self because WorldThreadLocalStore uses UnsafeCell internally.
        // Actually, world.thread_local_res_mut() takes &self!
        // So we can just use `world()` instead of `world_mut()`.

        // Wait, TLResMut needs `value: &'w mut T`.
        // Let's get the reference:
        // `thread_local_res_mut(&self) -> &mut ThreadLocalResources`. But wait, does thread_local_res_mut return a borrow corresponding to `'w`?
        // `thread_local_res_mut(&'a self) -> &'a mut ThreadLocalResources`.

        let res = world
            .thread_local_res_mut()
            .get_mut::<T>()
            .unwrap_or_else(|| panic!("ThreadLocalResource of type {} not found. Did you insert it into thread_local_res_mut()?", std::any::type_name::<T>()));
        TLResMut { value: res }
    }
}

pub struct Query<'w, Q: QueryToken, F: Filter = ()> {
    // Now uses a shared reference because QueryInner uses RefCell interior mutability
    query: &'w QueryInner<Q::State, Q::Persistent, F::Persistent>,
    world: &'w UnsafeWorldCell<'w>,
    tick: ChangeTick,
}

unsafe impl<'w, 's, Q: QueryToken, F: Filter> Send for Query<'w, Q, F> {}

unsafe impl<'w, 's, Q: QueryToken, F: Filter> Sync for Query<'w, Q, F> {}

// pub(crate) struct QueryLocalCache<'w, Q: QueryToken, F: Filter> {
//     entry: SmallVec<Option<usize>, 32>,
//     fetches: SmallVec<<Q::View<'w> as View<'w, Q::State>>::Fetch, 16>,
//     skips: SmallVec<Option<F::SkipFilter<'w>>, 16>,
// }

// impl<'s, Q: QueryToken, F: Filter> QueryLocalCache<'s, Q, F> {
//     // fn get_or_set_using<
//     //     G: Fn() -> Option<(<Q::View<'s> as View<'s>>::Fetch, Option<F::SkipFilter<'s>>)>,
//     // >(
//     //     &mut self,
//     //     arch: ArchetypeId,
//     //     get: G,
//     // ) -> Option<(
//     //     &<Q::View<'s> as View<'s>>::Fetch,
//     //     &Option<F::SkipFilter<'s>>,
//     // )> {
//     //     if let Some(idx) = self.entry.get(arch.0).unwrap_or(&None) {
//     //         if let Some(fetch) = self.fetches.get(*idx) {
//     //             return Some((fetch, self.skips.get(*idx).unwrap_or(&None)));
//     //         } else {
//     //             if let Some((fetch, skip)) = g() {
//     //                 if !self.fetches.len() < idx {
//     //                     self.fetches.resi
//     //                 }
//     //                 self.fetches[idx] = fetch;
//     //                 self.skips[idx] = skip;
//     //                 return Some((fetch, skip));
//     //             }
//     //         }
//     //     }
//     // }

//     pub fn matches_or_compute<G: Fn() -> bool>(&mut self, arch: ArchetypeId, get: G) -> bool {
//         if let Some(o) = self.entry.get(arch.0) {
//             match o {
//                 Some(_) => true,
//                 None => false,
//             }
//         } else if get() {
//             if self.entry.len() < arch.0 {
//                 self.entry.resize(arch.0 + 1, None);
//             }
//             self.entry[arch.0] = Some(self.fetches.len());
//             true
//         } else {
//             false
//         }
//     }

//     pub fn given_matches_fetch_or_compute<
//         'a,
//         G: Fn() -> <Q::View<'s> as View<'s, Q::State>>::Fetch,
//     >(
//         &'a mut self,
//         arch: ArchetypeId,
//         get: G,
//     ) -> &'a mut <Q::View<'s> as View<'s, >>::Fetch {
//         self.matches_or_compute(arch, || true);
//         let idx = &mut self.entry[arch.0].unwrap();
//         if self.fetches.len() > *idx + 1 {
//             &mut self.fetches[*idx]
//         } else {
//             if *idx != self.fetches.len() {
//                 *idx = self.fetches.len();
//             }

//             self.fetches.push(get());
//             &mut self.fetches[*idx]
//         }
//     }

//     pub fn given_matches_skip_or_compute<'a, G: Fn() -> Option<F::SkipFilter<'s>>>(
//         &'a mut self,
//         arch: ArchetypeId,
//         get: G,
//     ) -> &'a mut Option<F::SkipFilter<'s>> {
//         self.matches_or_compute(arch, || true);
//         let idx = &mut self.entry[arch.0].unwrap();
//         if self.skips.len() > *idx + 1 {
//             &mut self.skips[*idx]
//         } else {
//             if *idx != self.skips.len() {
//                 *idx = self.skips.len();
//             }

//             self.skips.push(get());
//             &mut self.skips[*idx]
//         }
//     }
// }

impl<'w, 's, Q: QueryToken, F: Filter> Query<'w, Q, F> {
    pub fn tick(&self) -> ChangeTick {
        self.tick
    }

    /// Returns an iterator over the query results.
    /// Requires &mut self for queries containing &mut T.
    pub fn iter_mut(&mut self) -> QueryIter<'_, Q::State, Q::View<'_>, F> {
        // Safety: at this point the column borrow checks have been done by the scheduler.
        self.query.iter::<Q::View<'_>, F>(self.world, self.tick)
    }

    pub fn for_each_mut_since<'a, Func>(&'a mut self, tick: ChangeTick, func: Func)
    where
        Func: FnMut(<Q::View<'a> as View<'a, Q::State>>::Item) + Send + Sync,
    {
        self.query
            .for_each::<Q::View<'a>, F, Func>(self.world, func, tick)
    }

    /// Runs a closure for each item in the query.
    /// Requires &mut self for queries containing &mut T.
    pub fn for_each_mut<'a, Func>(&'a mut self, func: Func)
    where
        Func: FnMut(<Q::View<'a> as View<'a, Q::State>>::Item) + Send + Sync,
    {
        self.query
            .for_each::<Q::View<'a>, F, Func>(self.world, func, self.tick)
    }

    /// Runs a closure for each item in the query.
    /// Requires &mut self for queries containing &mut T.
    pub fn for_each_par_mut<'a, Func>(&'a mut self, func: Func)
    where
        Func: FnMut(<Q::View<'a> as View<'a, Q::State>>::Item) + Send + Sync + Clone,
    {
        self.query
            .for_each_par::<Q::View<'a>, F, Func>(self.world, func, self.tick)
    }

    /// Get a specific entity's query item, if it exists.
    /// Requires &mut self for queries containing &mut T.
    pub fn get_mut(
        &mut self,
        entity: crate::entity::Entity,
    ) -> Option<<Q::View<'w> as View<'w, Q::State>>::Item> {
        // Safety: We are in system
        unsafe {
            self.query
                .get_unchecked::<Q::View<'w>, F>(self.world, entity, self.tick)
        }
    }
}

/// Methods available when the query only contains read-only data (e.g., &T, Entity).
impl<'w, 's, Q: QueryToken + ReadOnly, F: Filter> Query<'w, Q, F> {
    pub fn iter(&self) -> QueryIter<'_, Q::State, Q::View<'_>, F> {
        self.query.iter::<Q::View<'_>, F>(self.world, self.tick)
    }

    pub fn for_each<'a, Func>(&'a self, func: Func)
    where
        Func: FnMut(<Q::View<'a> as View<'a, Q::State>>::Item) + Send + Sync,
    {
        self.query
            .for_each::<Q::View<'a>, F, Func>(self.world, func, self.tick)
    }

    pub fn for_each_par<'a, Func>(&'a self, func: Func)
    where
        Func: FnMut(<Q::View<'a> as View<'a, Q::State>>::Item) + Send + Sync + Clone,
    {
        self.query
            .for_each_par::<Q::View<'a>, F, Func>(self.world, func, self.tick)
    }

    pub fn get(
        &self,
        entity: crate::entity::Entity,
    ) -> Option<<Q::View<'w> as View<'w, Q::State>>::Item> {
        // Safety: We are in system
        unsafe {
            self.query
                .get_unchecked::<Q::View<'w>, F>(self.world, entity, self.tick)
        }
    }
}

impl<Q: QueryToken, F: Filter> SystemParam for Query<'_, Q, F> {
    type State = (
        QueryInner<Q::State, Q::Persistent, F::Persistent>,
        ArchetypeId,
        ChangeTick,
    );
    type Item<'w, 's> = Query<'w, Q, F>;

    const DYNAMIC: bool = true;

    fn init_state(world: &mut World, access: &mut SystemAccess) -> Self::State {
        let mut query = QueryInner::new::<Q, F>(&mut world.registry);

        for (id, col) in query.granular_borrow_check(world, ArchetypeId(0)) {
            access.col[id.0].extend(&col);
        }

        query.update_archetype_cache(world);

        (
            query,
            ArchetypeId(world.archetypes.len()),
            world.tick().0.wrapping_sub(1).into(),
        )
    }

    unsafe fn get_param<'w, 's>(
        state: &'w mut Self::State,
        world: &'w UnsafeWorldCell<'w>,
    ) -> Self::Item<'w, 's> {
        state.0.update_archetype_cache(unsafe { world.world() });

        let res = Query {
            query: &state.0,
            world,
            tick: state.2,
        };
        state.2 = unsafe { world.world().tick() };
        res
    }

    fn update_access(state: &mut Self::State, world: &World, access: &mut SystemAccess) -> bool {
        let mut changed = false;
        if state.0.last_updated_arch_idx().0 != world.archetypes.len() {
            for (id, col) in state.0.granular_borrow_check(world, state.1) {
                if let Some(system_col) = access.col.get_mut(id.0) {
                    if system_col.extend(&col) {
                        changed = true;
                    }
                } else {
                    access.col.push(col);
                    changed = true;
                };
            }
        }

        state.1 = ArchetypeId(world.archetypes.len());
        changed
    }
}

// Macro to implement SystemParam for tuples (A, B, ...)
macro_rules! impl_system_param_tuple {
    ($($name:ident $_:tt),*) => {
        impl<$($name: SystemParam),*> SystemParam for ($($name,)*) {
            type Marker = Many;
            type State = ($($name::State,)*);
            type Item<'w, 's> = ($($name::Item<'w, 's>,)*);

            // If ANY param is dynamic, the tuple is dynamic
            const DYNAMIC: bool = $($name::DYNAMIC)||*;

            fn init_state(world: &mut World, access: &mut SystemAccess) -> Self::State {
                (
                    $($name::init_state(world, access),)*
                )
            }

            unsafe fn get_param<'w, 's>(state: &'w mut Self::State, world: &'w UnsafeWorldCell<'w>) -> Self::Item<'w, 's> {
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
