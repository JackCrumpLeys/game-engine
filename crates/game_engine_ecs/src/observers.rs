use std::{marker::PhantomData, ops::Deref};

use rustc_hash::FxHashSet;

use crate::{
    archetype::ArchetypeId,
    entity::Entity,
    prelude::Component,
    query::{Mut, QueryToken, ReadOnly},
    system::{Single, SystemAccess, SystemParam, UnsafeWorldCell},
    world::{ChangeTick, EntityLocation, World},
};

/// A superset of `SystemParam`.
/// All `SystemParam`s are `ObserverSystemParam`s, but not vice-versa.
pub trait ObserverSystemParam {
    type Marker: Send + Sync + 'static = Single;
    type State: Send + Sync + 'static;
    type Item<'w>;

    /// Does the accsess depend on world state (eg: adding/removing archetypes)?
    const DYNAMIC: bool = false;

    fn local_borrows() -> ObserverLocalBorrow;

    /// Must call some time afeter init_state
    fn can_run_on<'w>(state: &'w Self::State, world: &'w World, target: EntityLocation) -> bool;

    /// Registers access.
    fn init_state(world: &mut World, access: &mut SystemAccess) -> Self::State;

    /// The execution-time fetch.
    /// Note the extra `ctx` argument compared to `SystemParam`.
    ///
    /// # Safety
    ///
    /// The scheduler ensures 'access' is respected, so this is safe to call
    /// if the scheduler logic is correct.
    unsafe fn get_param<'w>(
        state: &'w mut Self::State,
        world: &'w UnsafeWorldCell<'w>,
        target: EntityLocation,
    ) -> Self::Item<'w>;

    /// Called every frame that archetypes change IF `DYNAMIC` is true.
    fn update_access(_state: &mut Self::State, _world: &World, _access: &mut SystemAccess) {}
}

impl<P: SystemParam<Marker = Single>> ObserverSystemParam for P {
    type Marker = P::Marker;
    type State = P::State;
    type Item<'w> = P::Item<'w>;

    /// Does the accsess depend on world state (eg: adding/removing archetypes)?
    const DYNAMIC: bool = P::DYNAMIC;

    fn local_borrows() -> ObserverLocalBorrow {
        ObserverLocalBorrow::None
    }

    fn init_state(world: &mut World, access: &mut SystemAccess) -> Self::State {
        P::init_state(world, access)
    }

    unsafe fn get_param<'w>(
        state: &'w mut Self::State,
        world: &'w UnsafeWorldCell<'w>,
        _target: EntityLocation,
    ) -> Self::Item<'w> {
        unsafe { P::get_param(state, world) }
    }

    fn can_run_on<'w>(_state: &'w Self::State, _world: &'w World, _target: EntityLocation) -> bool {
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

impl<T: Component> ObserverSystemParam for Get<'_, &'_ T> {
    type State = ();
    type Item<'w> = Get<'w, &'w T>;

    fn init_state(world: &mut World, access: &mut SystemAccess) -> Self::State {
        world.registry.register::<T>();
    }

    /// Only safe for a given entity if `can_run_on` is called and is true
    unsafe fn get_param<'w>(
        _state: &'w mut Self::State,
        world: &'w UnsafeWorldCell<'w>,
        entity_location: EntityLocation,
    ) -> Self::Item<'w> {
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

    fn can_run_on<'w>(_state: &'w Self::State, world: &'w World, target: EntityLocation) -> bool {
        world.archetypes[target.archetype_id()]
            .component_mask
            .has_id(&T::get_id())
    }
}

impl<T: Component> ObserverSystemParam for Get<'_, &'_ mut T> {
    type State = ();
    type Item<'w> = Get<'w, &'w mut T>;

    fn init_state(world: &mut World, access: &mut SystemAccess) -> Self::State {
        world.registry.register::<T>();
    }

    /// Only safe for a given entity if `can_run_on` is called and is true
    unsafe fn get_param<'w>(
        _state: &'w mut Self::State,
        world: &'w UnsafeWorldCell<'w>,
        entity_location: EntityLocation,
    ) -> Self::Item<'w> {
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

    fn can_run_on<'w>(state: &'w Self::State, world: &'w World, target: EntityLocation) -> bool {
        world.archetypes[target.archetype_id()]
            .component_mask
            .has_id(&T::get_id())
    }
}

macro_rules! impl_observer_system_param_tuple {
    ($($name:ident $num:tt),*) => {
            impl<$($name: ObserverSystemParam<Marker=Single>),*, > ObserverSystemParam for ($($name,)*) {
                type State = ($($name::State,)*);
                type Item<'w> = ($($name::Item<'w>,)*);

                const DYNAMIC: bool = $($name::DYNAMIC)||*;


                fn local_borrows() -> ObserverLocalBorrow {
                    ObserverLocalBorrow::None
                        $(.combine($name::local_borrows()))*
                }

                /// Must call some time afeter init_state
                fn can_run_on<'w>(state: &'w Self::State, world: &'w World, target: EntityLocation)
                -> bool {
                    $($name::can_run_on(&state.$num, world, target))&&*
                }

                fn init_state(world: &mut World, access: &mut SystemAccess) -> Self::State {
                    (
                        $($name::init_state(world, access),)*
                    )
                }

                unsafe fn get_param<'w>(state: &'w mut Self::State, world: &'w UnsafeWorldCell<'w>, loc: EntityLocation) -> Self::Item<'w> {
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

trait TriggerEvent: Send + Sync + 'static {}

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
    /// - `event_ptr` must point to a valid instance of `E`.
    /// - `world` must be capable of handling the access declared in `update_access`.
    /// - The caller ensures the event data stays valid for the duration of the call.
    unsafe fn run(
        &mut self,
        world: &UnsafeWorldCell,
        entity: Entity,
        loc: EntityLocation,
        event_ptr: *const u8,
    );

    /// Updates access information (e.g. if archetypes changed).
    fn update_access(&mut self, world: &World) -> bool;

    /// Returns the access requirements for this system.
    fn access(&self) -> &SystemAccess;

    /// Must call some time afeter init_state
    fn can_run_on<'w>(&self, world: &'w World, target: EntityLocation) -> bool;
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
    _marker: PhantomData<(E, Params, Marker)>,
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
    Params: ObserverSystemParam + Send + Sync + 'static,
    Marker: Send + Sync + 'static,
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
    Params: ObserverSystemParam + Send + Sync + 'static,
    Marker: Send + Sync + 'static,
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
    fn can_run_on<'w>(&self, world: &'w World, target: EntityLocation) -> bool {
        Params::can_run_on(
            unsafe { self.state.as_ref().unwrap_unchecked() },
            world,
            target,
        )
    }
}

macro_rules! impl_observer_system_function {
    ($($param:ident $_:tt),*) => {
        #[allow(non_snake_case)]
        impl<Event, Func, $($param),*> ObserverSystemParamFunction<Event, fn(Trigger<Event>, $($param),*)> for Func
        where
            Event: TriggerEvent,
            Func: for<'a> FnMut(Trigger<Event>, $($param::Item<'a>),*) +
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

auto_impl_all_tuples!(impl_observer_system_function);
