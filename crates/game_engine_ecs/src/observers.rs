use crate::resource::TypeIdMap;
use smallvec::SmallVec;
use std::any::Any;
use std::ptr;
use std::{
    any::{TypeId, type_name},
    marker::PhantomData,
    ops::Deref,
};

use crate::{
    component::ComponentMask,
    entity::Entity,
    prelude::Component,
    query::{Mut, QueryToken},
    system::{Single, SystemAccess, SystemParam, UnsafeWorldCell},
    world::{ChangeTick, EntityLocation, World},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ObserverId(pub usize);

pub struct ObserverStore {
    pub(crate) observers: Vec<Box<dyn ObserverSystem>>,
    pub(crate) map: TypeIdMap<SmallVec<ObserverId, 16>>,
}

impl ObserverStore {
    pub(crate) fn get(&self, idx: ObserverId) -> Option<&Box<dyn ObserverSystem + 'static>> {
        self.observers.get(idx.0)
    }

    pub(crate) fn get_mut(
        &mut self,
        idx: ObserverId,
    ) -> Option<&mut Box<dyn ObserverSystem + 'static>> {
        self.observers.get_mut(idx.0)
    }
}

pub struct ObserverExecutionQueue {
    pub(crate) waiting: SmallVec<SmallVec<ObserverExecutionContext, 4>, 64>, // should be the size of world.observers.observers
    pub(crate) arena: Vec<u8>, // The data that the observer executer
    pub(crate) drop_shims: Vec<(usize, DropShim)>,
}

impl Drop for ObserverExecutionQueue {
    fn drop(&mut self) {
        // If we drop the queue without flushing, we must run destructors
        // for the data sitting in the byte buffer.
        self.drop_arena();
    }
}

type DropShim = unsafe fn(*mut u8);

pub struct ObserverExecutionContext {
    /// Where is the data in the store?
    pub(crate) data_offset: usize,
    /// What entity is this observer for?
    pub(crate) entity: Entity,
}

impl ObserverExecutionQueue {
    pub(crate) fn fire<E: Any>(&mut self, world: &World, data: E, entity: Entity) {
        let Some(fired) = world.observers.map.get(&TypeId::of::<E>()) else {
            return;
        };
        let mask = match world.entity_location(entity) {
            Some(loc) => {
                world
                    .archetypes
                    .get(*loc.archetype_id())
                    .expect("Invalid EntityLocation on world")
                    .component_mask
            }
            // Its either in thread local buffer or invalid
            None => {
                let Some(res) = world
                    .thread_local()
                    .insert_buffer
                    .entity_to_mask
                    .get(entity.index() as usize)
                else {
                    return;
                };
                *res
            }
        };
        /// # Safety
        /// `ptr` must point to `E`.
        unsafe fn drop_shim_impl<E>(ptr: *mut u8) {
            let in_d = unsafe { ptr::read_unaligned(ptr as *mut E) };
            drop(in_d);
        }

        let size = std::mem::size_of::<E>();

        self.arena.reserve(size);

        // Get the write location
        let data_offset = self.arena.len();
        let ptr = unsafe { self.arena.as_mut_ptr().add(data_offset) };

        unsafe { ptr::write_unaligned(ptr as *mut E, data) };

        unsafe { self.arena.set_len(data_offset + size) };

        self.drop_shims.push((data_offset, drop_shim_impl::<E>));

        for fired in fired {
            if self.waiting.len() <= fired.0 {
                self.waiting.resize_with(fired.0 + 1, SmallVec::new);
            }
            if world
                .observers
                .get(*fired)
                .expect("bad id.")
                .can_run_on(world, mask)
            {
                unsafe { self.waiting.get_unchecked_mut(fired.0) }.push(ObserverExecutionContext {
                    data_offset,
                    entity,
                });
            }
        }
    }

    pub(crate) fn drop_arena(&mut self) {
        for i in &self.drop_shims {
            unsafe {
                let ptr = self.arena.as_mut_ptr().add(i.0);
                (i.1)(ptr);
            }
        }
    }
}

/// A superset of `SystemParam`.
/// All `SystemParam`s are `ObserverSystemParam`s, but not vice-versa.
pub trait ObserverSystemParam {
    type Marker: Send + Sync + 'static = Single;
    type State: Send + Sync + 'static;
    type Item<'w, 's>;

    /// Does the accsess depend on world state (eg: adding/removing archetypes)?
    const DYNAMIC: bool = false;

    fn local_borrows() -> ObserverLocalBorrow;

    /// Must call some time afeter init_state
    fn can_run_on<'w>(state: &'w Self::State, world: &'w World, target_mask: ComponentMask)
    -> bool;

    /// Registers access.
    fn init_state(world: &mut World, access: &mut SystemAccess) -> Self::State;

    /// The execution-time fetch.
    /// Note the extra `ctx` argument compared to `SystemParam`.
    ///
    /// # Safety
    ///
    /// The scheduler ensures 'access' is respected, so this is safe to call
    /// if the scheduler logic is correct.
    unsafe fn get_param<'w, 's>(
        state: &'w mut Self::State,
        world: &'w UnsafeWorldCell<'w>,
        target: EntityLocation,
    ) -> Self::Item<'w, 's>;

    /// Called every frame that archetypes change IF `DYNAMIC` is true.
    fn update_access(_state: &mut Self::State, _world: &World, _access: &mut SystemAccess) {}
}

impl<P: SystemParam<Marker = Single>> ObserverSystemParam for P {
    type Marker = P::Marker;
    type State = P::State;
    type Item<'w, 's> = P::Item<'w, 's>;

    /// Does the accsess depend on world state (eg: adding/removing archetypes)?
    const DYNAMIC: bool = P::DYNAMIC;

    fn local_borrows() -> ObserverLocalBorrow {
        ObserverLocalBorrow::None
    }

    fn init_state(world: &mut World, access: &mut SystemAccess) -> Self::State {
        P::init_state(world, access)
    }

    unsafe fn get_param<'w, 's>(
        state: &'w mut Self::State,
        world: &'w UnsafeWorldCell<'w>,
        _target: EntityLocation,
    ) -> Self::Item<'w, 's> {
        unsafe { P::get_param(state, world) }
    }

    fn can_run_on<'w>(_state: &'w Self::State, _world: &'w World, _target: ComponentMask) -> bool {
        true
    }

    fn update_access(state: &mut Self::State, world: &World, access: &mut SystemAccess) {
        P::update_access(state, world, access);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ObserverLocalBorrow {
    None,
    Imm,
    Mut,
}

impl ObserverLocalBorrow {
    fn combine(self, other: Self) -> Self {
        use ObserverLocalBorrow::*;
        match (self, other) {
            (None, _) => other,
            (_, None) => self,
            (Imm, Imm) => Imm,
            (_, Mut) => Mut,
            (Mut, _) => Mut,
        }
    }

    /// Get the conflicts of running on the same entity
    fn local_conflicts(self, other: Self) -> bool {
        use ObserverLocalBorrow::*;
        match (self, other) {
            (None, _) => false,
            (_, None) => false,
            (Imm, Imm) => false,
            (Imm, Mut) => true,
            (Mut, Imm) => true,
            (Mut, Mut) => true,
        }
    }
}

pub struct Get<'a, T: QueryToken> {
    val: T::View<'a>,
}

impl<'a, T: QueryToken> std::ops::Deref for Get<'a, T> {
    type Target = T::View<'a>;
    fn deref(&self) -> &Self::Target {
        &self.val
    }
}

impl<'a, T: QueryToken> std::ops::DerefMut for Get<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.val
    }
}

impl<T: Component> ObserverSystemParam for Get<'_, &'_ T> {
    type State = ();
    type Item<'w, 's> = Get<'w, &'w T>;

    fn init_state(world: &mut World, access: &mut SystemAccess) -> Self::State {
        world.registry.register::<T>();
    }

    /// Only safe for a given entity if `can_run_on` is called and is true
    unsafe fn get_param<'w, 's>(
        _state: &'w mut Self::State,
        world: &'w UnsafeWorldCell<'w>,
        entity_location: EntityLocation,
    ) -> Self::Item<'w, 's> {
        let world = unsafe { world.world() };
        let val = unsafe {
            world.archetypes[entity_location.archetype_id()].columns[T::get_id().0]
                .as_ref()
                .unwrap()
                .get_unsafe(entity_location.row())
        };
        Get { val }
    }

    fn local_borrows() -> ObserverLocalBorrow {
        ObserverLocalBorrow::Imm
    }

    fn can_run_on<'w>(_state: &'w Self::State, world: &'w World, target: ComponentMask) -> bool {
        target.has_id(&T::get_id())
    }
}

impl<T: Component> ObserverSystemParam for Get<'_, &'_ mut T> {
    type State = ();
    type Item<'w, 's> = Get<'w, &'w mut T>;

    fn init_state(world: &mut World, access: &mut SystemAccess) -> Self::State {
        world.registry.register::<T>();
    }

    /// Only safe for a given entity if `can_run_on` is called and is true
    unsafe fn get_param<'w, 's>(
        _state: &'w mut Self::State,
        world: &'w UnsafeWorldCell<'w>,
        entity_location: EntityLocation,
    ) -> Self::Item<'w, 's> {
        let world = unsafe { world.world_mut() };
        let current_tick = world.tick();
        let column = unsafe {
            world.archetypes[entity_location.archetype_id()].columns[T::get_id().0]
                .as_mut()
                .unwrap_unchecked()
        };
        let val = Mut {
            current_tick,
            tick_ptr: unsafe {
                column
                    .changed_ticks_mut()
                    .get_unchecked_mut(entity_location.row())
            } as *mut ChangeTick,
            value: unsafe { column.get_mut_unsafe(entity_location.row()) },
        };

        Get { val }
    }

    fn local_borrows() -> ObserverLocalBorrow {
        ObserverLocalBorrow::Mut
    }

    fn can_run_on<'w>(state: &'w Self::State, world: &'w World, target: ComponentMask) -> bool {
        target.has_id(&T::get_id())
    }
}

macro_rules! impl_observer_system_param_tuple {
    ($($name:ident $num:tt),*) => {
            impl<$($name: ObserverSystemParam<Marker=Single>),*, > ObserverSystemParam for ($($name,)*) {
                type State = ($($name::State,)*);
                type Item<'w, 's> = ($($name::Item<'w, 's>,)*);

                const DYNAMIC: bool = $($name::DYNAMIC)||*;


                fn local_borrows() -> ObserverLocalBorrow {
                    ObserverLocalBorrow::None
                        $(.combine($name::local_borrows()))*
                }

                /// Must call some time afeter init_state
                fn can_run_on<'w>(state: &'w Self::State, world: &'w World, target: ComponentMask)
                -> bool {
                    $($name::can_run_on(&state.$num, world, target))&&*
                }

                fn init_state(world: &mut World, access: &mut SystemAccess) -> Self::State {
                    (
                        $($name::init_state(world, access),)*
                    )
                }

                unsafe fn get_param<'w, 's>(state: &'w mut Self::State, world: &'w UnsafeWorldCell<'w>, loc: EntityLocation) -> Self::Item<'w, 's> {
                    #[allow(non_snake_case)]
                    let ($($name,)*) = state;
                    unsafe {
                        (
                            $($name::get_param($name, world, loc)),*,
                        )
                    }
                }

                fn update_access(
                    state: &mut Self::State,
                    world: &World,
                    access: &mut SystemAccess,
                ) {
                    #[allow(non_snake_case)]
                    let ($($name,)*) = state;
                    $(
                         $name::update_access($name, world, access);
                    )*
                }
            }
    };
}

auto_impl_all_tuples!(impl_observer_system_param_tuple);

pub trait TriggerEvent: Send + Sync + 'static {}
impl<T: Send + Sync + 'static> TriggerEvent for T {}

/// A wrapper around an event fired on a specific entity.
///
/// The lifetime `'a` is tied to the flush cycle of the observer queue.
/// The data `E` is stored in a linear allocator (bump/arena) and is immutable.
pub struct Trigger<'a, E: TriggerEvent> {
    pub entity: Entity,
    pub event: &'a E,
}

impl<'a, E: TriggerEvent> Deref for Trigger<'a, E> {
    type Target = E;
    fn deref(&self) -> &Self::Target {
        self.event
    }
}

/// A system that runs in response to a specific event `E`.
pub trait ObserverSystem: Send + Sync + 'static {
    /// Initialize system state (local variables, access tracking).
    fn init(&mut self, world: &mut World);

    /// Runs the observer.
    ///
    /// # Safety
    /// - `event_ptr` must point to a valid instance of `E`. Get expected typeid from `event`
    /// - `world` must be capable of handling the access declared in `update_access`.
    /// - The caller ensures the event data stays valid for the duration of the call.
    unsafe fn run(
        &mut self,
        world: &UnsafeWorldCell,
        entity: Entity,
        loc: EntityLocation,
        event_ptr: *const u8,
    );

    fn name(&self) -> &'static str {
        type_name::<Self>()
    }

    /// Get the type id this observer runs in response to
    fn event(&self) -> TypeId;

    /// Updates access information (e.g. if archetypes changed).
    fn update_access(&mut self, world: &World) -> bool;

    /// Returns the access requirements for this system.
    fn access(&self) -> &SystemAccess;

    /// Must call some time afeter init_state
    fn can_run_on(&self, world: &World, target_mask: ComponentMask) -> bool;
}

/// The concrete struct that holds a function and turns it into an ObserverSystem.
pub struct ObserverFunctionSystem<
    E: TriggerEvent,
    Func: Send,
    Params: ObserverSystemParam + 'static,
    Marker: 'static,
> {
    func: Func,
    state: Option<Params::State>,
    access: SystemAccess,
    // Marker to track types during compilation
    _marker: PhantomData<fn() -> (E, Params, Marker)>,
}

/// Trait to convert a function into an ObserverSystem.
pub trait IntoObserverSystem<E: TriggerEvent, Marker, Params> {
    type System: ObserverSystem;
    fn into_observer_system(self) -> Self::System;
}

/// Helper trait to handle the varargs logic for function execution.
pub trait ObserverSystemParamFunction<E: TriggerEvent, Marker>: Send + Sync + 'static {
    type Params: ObserverSystemParam;

    fn run(
        &mut self,
        trigger: Trigger<E>,
        state: &mut <Self::Params as ObserverSystemParam>::State,
        world: &UnsafeWorldCell,
        entity: EntityLocation,
    );
}

impl<E: TriggerEvent, Func, Params, Marker> IntoObserverSystem<E, Marker, Params> for Func
where
    Params: ObserverSystemParam + 'static,
    Marker: 'static,
    Func: ObserverSystemParamFunction<E, Marker, Params = Params>,
{
    type System = ObserverFunctionSystem<E, Func, Params, Marker>;

    fn into_observer_system(self) -> Self::System {
        ObserverFunctionSystem {
            func: self,
            state: None,
            access: SystemAccess::new_empty(),
            _marker: PhantomData,
        }
    }
}

impl<E: TriggerEvent, Func, Params, Marker> ObserverSystem
    for ObserverFunctionSystem<E, Func, Params, Marker>
where
    E: 'static,
    Params: ObserverSystemParam,
    Params: ObserverSystemParam + 'static,
    Marker: 'static,
    Func: ObserverSystemParamFunction<E, Marker, Params = Params>,
{
    fn init(&mut self, world: &mut World) {
        self.state = Some(Params::init_state(world, &mut self.access));
    }

    fn update_access(&mut self, world: &World) -> bool {
        if Params::DYNAMIC {
            let state = self.state.as_mut().unwrap();
            Params::update_access(state, world, &mut self.access);
            true
        } else {
            false
        }
    }

    fn access(&self) -> &SystemAccess {
        &self.access
    }

    /// Must check can_run_on
    unsafe fn run(
        &mut self,
        world: &UnsafeWorldCell,
        entity: Entity,
        loc: EntityLocation,
        event_ptr: *const u8,
    ) {
        // SAFETY: Caller guarantees ptr is valid E.
        let event = unsafe { &*(event_ptr as *const E) };

        let trigger = Trigger { entity, event };

        // SAFETY: State is initialized in `init`.
        let state = unsafe { self.state.as_mut().unwrap_unchecked() };

        self.func.run(trigger, state, world, loc);
    }

    /// Must call some time afeter init_state
    fn can_run_on(&self, world: &World, target: ComponentMask) -> bool {
        Params::can_run_on(
            unsafe { self.state.as_ref().unwrap_unchecked() },
            world,
            target,
        )
    }

    fn event(&self) -> TypeId {
        TypeId::of::<E>()
    }
}

macro_rules! impl_observer_system_function {
    ($($param:ident $_:tt),*) => {
        #[allow(non_snake_case)]
        impl<Event, Func, $($param),*> ObserverSystemParamFunction<Event, fn(Trigger<Event>, $($param),*)> for Func
        where
            Event: TriggerEvent,
            Func: for<'a, 's> FnMut(Trigger<Event>, $($param::Item<'a, 's>),*) +
                  FnMut(Trigger<Event>, $($param),*) +
                  Send + Sync + 'static,
            $($param: ObserverSystemParam<Marker=Single> + 'static),*
        {
            type Params = ($($param,)*);

            fn run(
                &mut self,
                trigger: Trigger<Event>,
                state: &mut <Self::Params as ObserverSystemParam>::State,
                world: &UnsafeWorldCell,
                location: EntityLocation,
            ) {
                let ($($param,)*) = state;

                unsafe {
                    (self)(
                        trigger,
                        $($param::get_param($param, world, location)),*
                    )
                }
            }
        }
    };
}

impl<Event, Func> ObserverSystemParamFunction<Event, fn(Trigger<Event>)> for Func
where
    Event: TriggerEvent,
    Func: for<'a> FnMut(Trigger<Event>) + Send + Sync + 'static,
{
    type Params = ();

    fn run(
        &mut self,
        trigger: Trigger<Event>,
        state: &mut <Self::Params as ObserverSystemParam>::State,
        world: &UnsafeWorldCell,
        location: EntityLocation,
    ) {
        unsafe { (self)(trigger) }
    }
}

impl ObserverSystemParam for () {
    type Marker = Single;
    type State = ();
    type Item<'w, 's> = ();

    /// Does the accsess depend on world state (eg: adding/removing archetypes)?
    const DYNAMIC: bool = false;

    fn local_borrows() -> ObserverLocalBorrow {
        ObserverLocalBorrow::None
    }

    fn init_state(_world: &mut World, _access: &mut SystemAccess) -> Self::State {}

    unsafe fn get_param<'w, 's>(
        _state: &'w mut Self::State,
        _world: &'w UnsafeWorldCell<'w>,
        _target: EntityLocation,
    ) -> Self::Item<'w, 's> {
    }

    fn can_run_on<'w>(_state: &'w Self::State, _world: &'w World, _target: ComponentMask) -> bool {
        true
    }

    fn update_access(state: &mut Self::State, world: &World, access: &mut SystemAccess) {}
}

auto_impl_all_tuples!(impl_observer_system_function);

#[cfg(test)]
mod tests {

    use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
    use std::{mem, sync::Once};

    static INIT: Once = Once::new();

    fn setup_all() {
        // Initialization code here. This runs once before any test in this file.
        INIT.call_once(|| {
            println!("Running global setup for the test binary!");

            #[cfg(feature = "tracy")]
            let _client = tracy_client::Client::start();

            #[cfg(feature = "tracy")]
            tracy_client::register_demangler!();
        });
    }
    use game_engine_derive::Component;
    use rand::seq::SliceRandom;

    use super::*;
    use crate::prelude::*;
    use crate::system::command::Commands;
    use std::{
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        time::Instant,
    };

    // --- Mock Components & Events ---

    #[derive(Debug, Component)]
    struct Health(i32);

    #[derive(Debug, Component)]
    struct Name(String);

    struct DamageEvent {
        amount: i32,
    }

    struct HealEvent {
        amount: i32,
    }

    struct PingEvent;

    // --- Helpers ---

    /// Helper to get a Command object to run test logic.
    /// In a real app, this is passed by the scheduler.
    fn with_command(world: &mut World, f: impl FnOnce(Commands)) {
        let mut cmd = Commands::new(world);
        f(cmd);
    }

    #[test]
    fn test_basic_observer_trigger() {
        let mut world = World::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        // 1. Register Observer
        world.add_observer(move |_trigger: Trigger<PingEvent>| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        // 2. Spawn Entity
        let mut entity = Entity::new(0, 0);
        with_command(&mut world, |cmd| {
            entity = cmd.spawn(Name("Test".to_string()));
        });
        world.flush();

        // 3. Trigger Event
        with_command(&mut world, |cmd| {
            cmd.trigger(entity, PingEvent);
        });

        assert_eq!(
            counter.load(Ordering::SeqCst),
            0,
            "Observer should not run before flush"
        );

        // 4. Flush to execute observers
        world.flush();

        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "Observer should have run once"
        );
    }

    #[test]
    fn test_observer_component_access() {
        let mut world = World::new();
        let result = Arc::new(AtomicUsize::new(0));
        let result_clone = result.clone();

        // Observer reads Health and adds it to the atomic counter
        world.add_observer(move |_trigger: Trigger<PingEvent>, hp: Get<&Health>| {
            result_clone.store(hp.0 as usize, Ordering::SeqCst);
        });

        let mut entity = Entity::new(0, 0);
        with_command(&mut world, |cmd| {
            entity = cmd.spawn(Health(42));
        });
        world.flush();

        with_command(&mut world, |cmd| {
            cmd.trigger(entity, PingEvent);
        });
        world.flush();

        assert_eq!(result.load(Ordering::SeqCst), 42);
    }

    #[test]
    fn test_observer_component_mutation() {
        let mut world = World::new();

        // Observer reduces health by damage amount
        world.add_observer(|trigger: Trigger<DamageEvent>, mut hp: Get<&mut Health>| {
            hp.0 -= trigger.amount;
        });

        let mut entity = Entity::new(0, 0);
        with_command(&mut world, |cmd| {
            entity = cmd.spawn(Health(100));
        });
        world.flush();

        with_command(&mut world, |cmd| {
            cmd.trigger(entity, DamageEvent { amount: 30 });
        });
        world.flush();

        // Verify result
        let q = world.query::<&Health, ()>();
        let final_hp = q.get(entity).unwrap();
        assert_eq!(final_hp.0, 70);
    }

    #[test]
    fn test_observer_filtering() {
        let mut world = World::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        // Observer requires 'Health' component to run
        world.add_observer(move |_trigger: Trigger<PingEvent>, _hp: Get<&Health>| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        let mut healthy = Entity::new(0, 0);
        let mut ghost = Entity::new(0, 0);

        with_command(&mut world, |cmd| {
            healthy = cmd.spawn(Health(10));
            ghost = cmd.spawn(Name("No Health".to_string()));
        });
        world.flush();

        with_command(&mut world, |cmd| {
            cmd.trigger(healthy, PingEvent);
            cmd.trigger(ghost, PingEvent);
        });
        world.flush();

        // Should only run for the entity that has Health.
        // The entity 'ghost' lacks Health, so `can_run_on` should return false (or query fails safely),
        // preventing the execution count from reaching 2.
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_chained_logic() {
        let mut world = World::new();

        // 1. Damage reduces health
        world.add_observer(|trigger: Trigger<DamageEvent>, mut hp: Get<&mut Health>| {
            hp.0 -= trigger.amount;
        });

        // 2. Heal increases health
        world.add_observer(|trigger: Trigger<HealEvent>, mut hp: Get<&mut Health>| {
            hp.0 += trigger.amount;
        });

        let mut entity = Entity::new(0, 0);
        with_command(&mut world, |cmd| {
            entity = cmd.spawn(Health(50));
        });
        world.flush();

        // Fire events in specific order (commands are processed in order)
        with_command(&mut world, |cmd| {
            cmd.trigger(entity, DamageEvent { amount: 20 }); // 50 - 20 = 30
            cmd.trigger(entity, HealEvent { amount: 10 }); // 30 + 10 = 40
        });
        world.flush();

        let q = world.query::<&Health, ()>();
        let hp = q.get(entity).unwrap();
        assert_eq!(hp.0, 40);
    }

    #[test]
    fn test_multiple_observers_same_event() {
        let mut world = World::new();
        let counter = Arc::new(AtomicUsize::new(0));

        let c1 = counter.clone();
        world.add_observer(move |_t: Trigger<PingEvent>| {
            c1.fetch_add(1, Ordering::Relaxed);
        });

        let c2 = counter.clone();
        world.add_observer(move |_t: Trigger<PingEvent>| {
            c2.fetch_add(2, Ordering::Relaxed);
        });

        let mut e = Entity::new(0, 0);
        with_command(&mut world, |cmd| {
            e = cmd.spawn(());
        });
        world.flush();

        with_command(&mut world, |cmd| {
            cmd.trigger(e, PingEvent);
        });
        world.flush();

        // 1 + 2 = 3
        assert_eq!(counter.load(Ordering::Relaxed), 3);
    }

    #[test]
    #[ignore]
    fn stress_test_observer_throughput() {
        let mut world = World::new();

        // --- Components ---
        #[derive(Component, Debug)]
        struct Health(f32);

        #[derive(Component, Debug)]
        struct HitStats {
            hits_received: u32,
        }

        // --- Event ---
        struct DamageEvent {
            amount: f32,
        }

        // --- Observer Setup ---
        // This is the hot path. For every event, the engine must:
        // 1. Identify the entity.
        // 2. check if it has Health and HitStats.
        // 3. Borrow them mutably (checking for conflicts).
        // 4. Run this closure.
        world.add_observer(
            |trigger: Trigger<DamageEvent>,
             mut hp: Get<&mut Health>,
             mut stats: Get<&mut HitStats>| {
                hp.0 -= trigger.amount;
                stats.hits_received += 1;
            },
        );

        // --- Population Phase ---
        const ENTITY_COUNT: usize = 100_000_000;
        const ROUNDS: usize = 1;

        let mut entities = Vec::with_capacity(ENTITY_COUNT);

        with_command(&mut world, |mut cmd| {
            for _ in 0..ENTITY_COUNT {
                entities.push(cmd.spawn((Health(10000.0), HitStats { hits_received: 0 })));
            }
        });
        world.flush(); // Commit spawns

        entities.shuffle(&mut rand::rng());
        println!("Spawned {} entities. Starting bombardment...", ENTITY_COUNT);

        // --- Execution Phase ---
        let start_time = Instant::now();

        for _round in 0..ROUNDS {
            with_command(&mut world, |cmd| {
                rayon::scope(|s| {
                    for i in 0..12 {
                        let cmd = &cmd;
                        let entities = &entities;
                        s.spawn(move |_| {
                            let start_idx = (entities.len() / 24) * i;
                            for e in entities[start_idx..start_idx + (entities.len() / 24)].iter() {
                                cmd.trigger(*e, DamageEvent { amount: 1.0 });
                            }
                        })
                    }
                })
            });

            // The engine must process 100k events here
            world.flush();
        }

        let elapsed = start_time.elapsed();
        let total_events = ENTITY_COUNT * ROUNDS;

        // --- Verification ---
        // Ensure the logic actually ran and wasn't optimized away
        let q = world.query::<&HitStats, ()>();
        let sample = q.get(entities[0]).unwrap();
        assert_eq!(
            sample.hits_received as usize, ROUNDS,
            "Observer logic failed to execute correctly"
        );

        // --- Reporting ---
        println!("==================================================");
        println!("OBSERVER STRESS TEST RESULTS");
        println!("==================================================");
        println!("Total Entities:      {}", ENTITY_COUNT);
        println!("Total Rounds:        {}", ROUNDS);
        println!("Total Events:        {}", total_events);
        println!("Time Taken:          {:.4?}", elapsed);

        let events_per_sec = total_events as f64 / elapsed.as_secs_f64();
        println!(
            "Throughput:          {:.2} Million Events/sec",
            events_per_sec / 1_000_000.0
        );
        println!(
            "Avg Time per Event:  {:.2} ns",
            (elapsed.as_nanos() as f64 / total_events as f64)
        );
        println!("==================================================");

        // Force failure to see stdout
        panic!("Test finished successfully. View output above.");
    }

    #[cfg(test)]
    mod bevy_comparison {
        use bevy_ecs::prelude::*;
        use bevy_ecs::system::SystemState;
        use rand::seq::SliceRandom;
        use std::time::Instant;

        #[derive(Component)]
        struct Health(f32);

        #[derive(Component)]
        struct HitStats {
            hits_received: u32,
        }

        #[derive(EntityEvent)]
        struct DamageEvent {
            amount: f32,
            entity: Entity,
        }

        #[test]
        #[ignore]
        fn bevy_command_throughput() {
            let mut world = World::new();

            // --- Observer Setup ---
            let o = Observer::new(
                |trigger: On<DamageEvent>, mut query: Query<(&mut Health, &mut HitStats)>| {
                    #[cfg(feature = "tracy")]
                    let _spawn = tracy_client::span!("DAMAGE");
                    if let Ok((mut hp, mut stats)) = query.get_mut(trigger.entity) {
                        hp.0 -= trigger.event().amount;
                        stats.hits_received += 1;
                    }
                },
            );

            // --- Population ---
            const ENTITY_COUNT: usize = 100_000;
            const ROUNDS: usize = 100;

            let mut entities = Vec::with_capacity(ENTITY_COUNT);
            for _ in 0..ENTITY_COUNT {
                entities.push({
                    world
                        .spawn((Health(10000.0), HitStats { hits_received: 0 }))
                        .id()
                });
            }

            world.spawn(o);

            println!(
                "Bevy: Spawned {} entities. Starting bombardment via Commands...",
                ENTITY_COUNT
            );

            // entities.shuffle(&mut rand::rng());

            // We need SystemState to get Commands in a test environment easily
            let mut system_state: SystemState<Commands> = SystemState::new(&mut world);

            let start_time = Instant::now();

            for _ in 0..ROUNDS {
                // 1. Queue Phase
                // Note: This mimics your `cmd.trigger`
                let mut commands = world.commands();
                for &entity in &entities {
                    commands.trigger(DamageEvent {
                        amount: 1.0,
                        entity,
                    });
                }

                {
                    #[cfg(feature = "tracy")]
                    let _span = tracy_client::span!("Bevy Flush", 100);
                    world.flush();
                }
            }

            let elapsed = start_time.elapsed();
            let total_events = ENTITY_COUNT * ROUNDS;

            println!("==================================================");
            println!("BEVY COMMAND -> OBSERVER RESULTS");
            println!("==================================================");
            println!("Total Events:        {}", total_events);
            println!("Time Taken:          {:.4?}", elapsed);

            let events_per_sec = total_events as f64 / elapsed.as_secs_f64();
            println!(
                "Throughput:          {:.2} Million Events/sec",
                events_per_sec / 1_000_000.0
            );
            println!(
                "Avg Time per Event:  {:.2} ns",
                (elapsed.as_nanos() as f64 / total_events as f64)
            );
            println!("==================================================");

            panic!();
        }
    }
}
