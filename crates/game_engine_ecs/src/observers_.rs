use rustc_hash::FxHashSet;

use crate::{
    archetype::ArchetypeId,
    entity::Entity,
    prelude::Component,
    query::{Mut, QueryToken, ReadOnly},
    system::{SystemAccess, SystemParam, UnsafeWorldCell},
    world::{ChangeTick, EntityLocation, World},
};

/// A superset of `SystemParam`.
/// All `SystemParam`s are `ObserverSystemParam`s, but not vice-versa.
pub trait ObserverSystemParam<Marker> {
    type State: Send + Sync + 'static;
    type Item<'w>;

    /// Does the accsess depend on world state (eg: adding/removing archetypes)?
    const DYNAMIC: bool = false;

    fn local_borrows() -> ObserverLocalBorrow;

    /// Must call some time afeter init_state
    fn can_run_on<'w>(state: &'w mut Self::State, world: &'w World, target: EntityLocation)
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
    unsafe fn get_param<'w>(
        state: &'w mut Self::State,
        world: &'w UnsafeWorldCell<'w>,
        target: EntityLocation,
    ) -> Self::Item<'w>;

    /// Called every frame that archetypes change IF `DYNAMIC` is true.
    fn update_access(_state: &mut Self::State, _world: &World, _access: &mut SystemAccess) {}
}

pub struct Super;

impl<P: SystemParam> ObserverSystemParam<Super> for P {
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

    fn can_run_on<'w>(
        _state: &'w mut Self::State,
        _world: &'w World,
        _target: EntityLocation,
    ) -> bool {
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

impl<T: Component> ObserverSystemParam<()> for Get<'_, &'_ T> {
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

    fn can_run_on<'w>(
        state: &'w mut Self::State,
        world: &'w World,
        target: EntityLocation,
    ) -> bool {
        world.archetypes[target.archetype_id()]
            .component_mask
            .has_id(&T::get_id())
    }
}

impl<T: Component> ObserverSystemParam<()> for Get<'_, &'_ mut T> {
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

    fn can_run_on<'w>(
        state: &'w mut Self::State,
        world: &'w World,
        target: EntityLocation,
    ) -> bool {
        world.archetypes[target.archetype_id()]
            .component_mask
            .has_id(&T::get_id())
    }
}

macro_rules! impl_observer_system_param_tuple {
    ($($name:ident $num:tt),*) => {
        paste::paste! {
            impl<$( [<$name Marker>] ),*, $($name: ObserverSystemParam<[<$name Marker>]>),*, > ObserverSystemParam<($( [<$name Marker>] ),*,)> for ($($name,)*) {
                type State = ($($name::State,)*);
                type Item<'w> = ($($name::Item<'w>,)*);

                const DYNAMIC: bool = $($name::DYNAMIC)||*;


                fn local_borrows() -> ObserverLocalBorrow {
                    ObserverLocalBorrow::None
                        $(.combine($name::local_borrows()))*
                }

                /// Must call some time afeter init_state
                fn can_run_on<'w>(state: &'w mut Self::State, world: &'w World, target: EntityLocation)
                -> bool {
                    $($name::can_run_on(&mut state.$num, world, target))&&*
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
        }
    };
}

auto_impl_all_tuples!(impl_observer_system_param_tuple);
