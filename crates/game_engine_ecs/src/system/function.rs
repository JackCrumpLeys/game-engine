use crate::system::{System, SystemAccess, SystemParam, UnsafeWorldCell};
use crate::world::World;
use std::marker::PhantomData;

/// The structure that wraps a user function to make it a System.
pub struct FunctionSystem<Func, Params, Marker>
where
    Params: SystemParam + 'static,
    Marker: 'static,
{
    func: Func,
    state: Option<Params::State>,
    access: SystemAccess,
    #[allow(dead_code)]
    name: String, // TODO this will be used for debugging/profiling later
    _marker: PhantomData<Marker>,
}

// SAFETY: FunctionSystem is Send/Sync if the Function is.
unsafe impl<Func, Params, Marker> Send for FunctionSystem<Func, Params, Marker>
where
    Params: SystemParam + 'static,
    Marker: 'static,
    Func: Send,
{
}
unsafe impl<Func, Params, Marker> Sync for FunctionSystem<Func, Params, Marker>
where
    Params: SystemParam + 'static,
    Marker: 'static,
    Func: Sync,
{
}

// ----------------------------------------------------------------------------
// 1. The System Implementation
// ----------------------------------------------------------------------------

impl<Func, Params, Marker> System for FunctionSystem<Func, Params, Marker>
where
    Func: SystemParamFunction<Marker, Param = Params>,
    Params: SystemParam + 'static,
    Marker: 'static,
{
    fn init(&mut self, world: &mut World) {
        self.access = SystemAccess::new(world);
        self.state = Some(Params::init_state(world, &mut self.access));
    }

    unsafe fn run(&mut self, world: UnsafeWorldCell) {
        let state = self.state.as_mut().expect("System not initialized");
        let item = unsafe { Params::get_param(state, world) };
        self.func.run(item);
    }

    fn update_access(&mut self, world: &World) -> bool {
        if Params::DYNAMIC {
            let state = self.state.as_mut().unwrap();
            Params::update_access(state, world, &mut self.access)
        } else {
            false
        }
    }

    fn access(&self) -> &SystemAccess {
        &self.access
    }
}

// ----------------------------------------------------------------------------
// 2. IntoSystem Trait
// ----------------------------------------------------------------------------

pub trait IntoSystem<Marker> {
    type System: System;
    fn into_system(self) -> Self::System;
}

impl<Func, Marker> IntoSystem<Marker> for Func
where
    Marker: 'static,
    Func: SystemParamFunction<Marker>,
{
    type System = FunctionSystem<Func, Func::Param, Marker>;

    fn into_system(self) -> Self::System {
        FunctionSystem {
            func: self,
            state: None,
            access: SystemAccess::new_empty(),
            name: std::any::type_name::<Func>().to_string(),
            _marker: PhantomData,
        }
    }
}

// ----------------------------------------------------------------------------
// 3. SystemParamFunction Trait & Macro
// ----------------------------------------------------------------------------

pub trait SystemParamFunction<Marker>: Send + Sync + 'static {
    type Param: SystemParam;
    fn run<'w>(&mut self, item: <Self::Param as SystemParam>::Item<'w>);
}

// Implementation for single parameter (fn(P))
impl<Func, P> SystemParamFunction<fn(P)> for Func
where
    P: SystemParam + 'static,
    Func: for<'a> FnMut(P::Item<'a>)
        +
        // DISAMBIGUATION: Func must also be callable with P directly.
        // Since functions are contravariant, fn(Query<'a>) implements Fn(Query<'static>).
        // This forces P to be the 'static version of the param.
        FnMut(P)
        + Send
        + Sync
        + 'static,
{
    type Param = P;

    fn run<'w>(&mut self, item: P::Item<'w>) {
        (self)(item)
    }
}

// Macro for tuples
macro_rules! impl_system_function {
    ($($param:ident $idx:tt),*) => {
        impl<Func, $($param),*> SystemParamFunction<fn($($param),*)> for Func
        where
            $($param: SystemParam + 'static),*,
            Func: for<'a> FnMut($($param::Item<'a>),*) +
                  // DISAMBIGUATION: Force P to be static by requiring the func to accept the static P
                  FnMut($($param),*) +
                  Send + Sync + 'static,
        {
            type Param = ($($param,)*);

            fn run<'w>(&mut self, item: <Self::Param as SystemParam>::Item<'w>) {
                #[allow(non_snake_case)]
                let ($($param,)*) = item;
                (self)($($param),*)
            }
        }
    };
}

impl_all_tuples!(
    impl_system_function,
    A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12
);

// ----------------------------------------------------------------------------
// 4. Tests
// ----------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use crate::system::{Query, UnsafeWorldCell};

    // ========================================================================
    // MOCK DATA
    // ========================================================================

    #[derive(Debug, PartialEq, Default)]
    struct Position {
        x: f32,
        y: f32,
    }

    #[derive(Debug, PartialEq, Default)]
    struct Velocity {
        x: f32,
        y: f32,
    }

    #[derive(Debug, Default, PartialEq)]
    struct FrameCount(u32);

    // ========================================================================
    // MOCK APP / SCHEDULER
    // ========================================================================

    /// A mock application structure to verify API ergonomics.
    /// This simulates how a user would add systems to the engine.
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
                    system.run(cell);
                }
            }
        }
    }

    // ========================================================================
    // SYSTEMS
    // ========================================================================

    // Note: We use 'static lifetimes in signatures.
    // The `SystemParamFunction` impl projects these to specific lifetimes at runtime.

    fn movement_system(mut query: Query<(&mut Position, &Velocity)>) {
        for (mut pos, vel) in query.iter() {
            pos.x += vel.x;
            pos.y += vel.y;
        }
    }

    fn frame_counter_system(mut count: ResMut<FrameCount>) {
        count.0 += 1;
    }

    fn complex_system(count: Res<FrameCount>, mut query: Query<&mut Position>) {
        // Only run if frame > 0
        if count.0 > 0 {
            for mut pos in query.iter() {
                pos.x += 100.0;
            }
        }
    }

    // ========================================================================
    // TESTS
    // ========================================================================

    #[test]
    fn test_system_initialization_and_run() {
        let mut world = World::new();
        world.spawn((Position { x: 0.0, y: 0.0 }, Velocity { x: 1.0, y: 2.0 }));

        // 1. manual IntoSystem conversion
        let mut system = movement_system.into_system();

        // 2. Init
        system.init(&mut world);

        // 3. Run
        unsafe {
            system.run(UnsafeWorldCell::new(&mut world));
        }

        // 4. Verify Side Effects via standard QueryState
        let mut q = crate::query::QueryState::<&Position>::new(&mut world.registry);
        let pos = q.iter(&mut world).next().unwrap();

        assert_eq!(pos.x, 1.0);
        assert_eq!(pos.y, 2.0);
    }

    #[test]
    fn test_resources_and_tuples() {
        let mut world = World::new();
        world.resources_mut().insert(FrameCount(0));

        let mut system = frame_counter_system.into_system();
        system.init(&mut world);

        unsafe {
            system.run(UnsafeWorldCell::new(&mut world));
            system.run(UnsafeWorldCell::new(&mut world));
        }

        let res = world.resources().get::<FrameCount>().unwrap();
        assert_eq!(res.0, 2);
    }

    #[test]
    fn test_clean_api_ergonomics() {
        // This is the critical test for the compiler inference changes.
        // If this compiles, the `FnMut(P) + FnMut(P::Item)` bounds are working.
        let mut app = App::new();

        // Setup Data
        app.world
            .spawn((Position { x: 0.0, y: 0.0 }, Velocity { x: 1.0, y: 0.0 }));
        app.world.resources_mut().insert(FrameCount(0));

        // Add Systems - No Turbofish! <><
        app.add_system(movement_system); // Query
        app.add_system(frame_counter_system); // ResMut
        app.add_system(complex_system); // (Res, Query)

        assert_eq!(app.systems.len(), 3);

        // Frame 1
        app.run();

        // Check Position (Movement ran once, Complex didn't run because frame was 0 at start of frame)
        // Note: complex_system reads frame count. Frame count is updated AFTER complex system in the list?
        // Actually order matters here.
        // systems: [movement, frame, complex]
        // 1. movement: x=1.0
        // 2. frame: count=1
        // 3. complex: read count (1). x += 100.0 -> x=101.0
        {
            let res = app.world.resources().get::<FrameCount>().unwrap();
            assert_eq!(res.0, 1);
        }

        let mut q = crate::query::QueryState::<&Position>::new(&mut app.world.registry);
        let pos = q.iter(&mut app.world).next().unwrap();
        assert_eq!(pos.x, 101.0);
    }

    #[test]
    fn test_system_names() {
        let s1 = movement_system.into_system();
        let s2 = frame_counter_system.into_system();

        // We expect the fully qualified path, or at least the function name
        assert!(s1.name.contains("movement_system"));
        assert!(s2.name.contains("frame_counter_system"));
    }
}
