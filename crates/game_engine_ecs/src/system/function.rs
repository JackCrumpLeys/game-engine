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

    unsafe fn run(&mut self, world: &UnsafeWorldCell) {
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
mod function_tests {
    use super::*;
    use crate::message::MessageQueue;
    use crate::prelude::*;
    use crate::query::Changed;
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

    impl_component!(Position, Velocity, FrameCount);

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
                // SAFETY: The world cell is valid for the duration of the run.
                system.update_access(unsafe { cell.world() });
                unsafe {
                    system.run(&cell);
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
        for (mut pos, vel) in query.iter_mut() {
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
            for mut pos in query.iter_mut() {
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
            system.run(&UnsafeWorldCell::new(&mut world));
        }

        // 4. Verify Side Effects via standard QueryState
        let q = world.query::<&Position, ()>();
        let pos = q.iter().next().unwrap();

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
            system.run(&UnsafeWorldCell::new(&mut world));
            system.run(&UnsafeWorldCell::new(&mut world));
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
        // systems: [movement, frame, complex]
        // 1. movement: x=1.0
        // 2. frame: count=1
        // 3. complex: read count (1). x += 100.0 -> x=101.0
        {
            let res = app.world.resources().get::<FrameCount>().unwrap();
            assert_eq!(res.0, 1);
        }

        let q = app.world.query::<&Position, ()>();
        let pos = q.iter().next().unwrap();
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
    // ========================================================================
    // CHANGED FILTER TESTS
    // ========================================================================

    fn changed_tracker_system(
        query: Query<&Position, Changed<Position>>,
        mut count: ResMut<FrameCount>,
    ) {
        for _ in query.iter() {
            count.0 += 1;
        }
    }

    #[test]
    fn test_system_change_detection() {
        let mut app = App::new();
        // Use FrameCount as a "Changed Counter" for this test
        app.world.resources_mut().insert(FrameCount(0));

        app.add_system(changed_tracker_system);

        // 1. Spawn Entity (Tick 1)
        // Note: World starts at Tick 1 by default
        let e = app.world.spawn((Position { x: 0.0, y: 0.0 },));
        app.world.flush();

        // Run Frame 1
        // Component Tick: 1 (spawn)
        // System Last Run: 0 (init default)
        // Filter: 1 > 0 -> Matches
        app.run();

        assert_eq!(
            app.world.resources().get::<FrameCount>().unwrap().0,
            1,
            "Spawn should trigger changed detection"
        );

        // 2. Increment Tick (Tick 2)
        // App::run doesn't auto-increment tick in this mock, so we do it manually
        app.world.increment_tick();

        // Run Frame 2 (No mutations)
        // Component Tick: 1
        // System Last Run: 1 (from Frame 1)
        // Filter: 1 > 1 -> Fails
        app.run();

        assert_eq!(
            app.world.resources().get::<FrameCount>().unwrap().0,
            1,
            "No changes should be detected on static entity"
        );

        // 3. Mutate Entity (Tick 3)
        app.world.increment_tick();

        {
            // Mutate manually via direct world access to simulate another system
            let mut q = app.world.query::<&mut Position, ()>();
            let mut pos = q.get_mut(e).unwrap();
            pos.x += 1.0;
            // Guard drop updates component tick to 3
        }

        // Run Frame 3
        // Component Tick: 3
        // System Last Run: 2
        // Filter: 3 > 2 -> Matches
        app.run();

        assert_eq!(
            app.world.resources().get::<FrameCount>().unwrap().0,
            2,
            "Mutation should trigger changed detection"
        );
    }

    #[test]
    fn test_changed_filter_with_multiple_entities() {
        let mut app = App::new();
        app.world.resources_mut().insert(FrameCount(0));
        app.add_system(changed_tracker_system);

        let _e1 = app.world.spawn((Position::default(),));
        app.world.flush();

        // Frame 1: e1 detected
        app.run();
        assert_eq!(app.world.resources().get::<FrameCount>().unwrap().0, 1);

        app.world.increment_tick();

        // Spawn e2 (Tick 2)
        let _e2 = app.world.spawn((Position::default(),));
        app.world.flush();

        // Frame 2: e1 (old) ignored, e2 (new) detected
        app.run();
        assert_eq!(
            app.world.resources().get::<FrameCount>().unwrap().0,
            2,
            "Should detect only the new entity"
        );
    }

    // ========================================================================
    // MOCK DATA
    // ========================================================================

    #[derive(Debug, Clone, PartialEq)]
    struct DamageEvent {
        amount: u32,
    }

    // A resource to track what the receiver system actually saw
    #[derive(Debug, Default)]
    struct ReceivedLog {
        total_damage: u32,
        event_count: u32,
    }

    impl_component!(ReceivedLog); // Treat as resource, impl not strictly necessary but good practice

    // ========================================================================
    // SYSTEMS
    // ========================================================================

    fn sender_system_a(mut events: MessageWriter<DamageEvent>) {
        events.write(DamageEvent { amount: 10 });
    }

    fn sender_system_b(mut events: MessageWriter<DamageEvent>) {
        events.write(DamageEvent { amount: 20 });
        events.write(DamageEvent { amount: 30 });
    }

    fn receiver_system(mut events: MessageReader<DamageEvent>, mut log: ResMut<ReceivedLog>) {
        for event in events.iter() {
            log.total_damage += event.amount;
            log.event_count += 1;
        }
    }

    // ========================================================================
    // TESTS
    // ========================================================================

    #[test]
    fn test_same_frame_communication() {
        let mut world = World::new();
        world.resources_mut().insert(ReceivedLog::default());

        // Initialize Systems
        let mut sender = sender_system_a.into_system();
        let mut receiver = receiver_system.into_system();

        sender.init(&mut world);
        receiver.init(&mut world);

        // --- FRAME 1 ---
        // 1. Run Sender
        unsafe {
            sender.run(&UnsafeWorldCell::new(&mut world));
        }

        // 2. Run Receiver (Same Frame)
        // expected behavior: The MessageReader implementation iterates `front` + `back`.
        // Since we haven't swapped, the message is in `back`. The reader SHOULD see it.
        unsafe {
            receiver.run(&UnsafeWorldCell::new(&mut world));
        }

        let log = world.resources().get::<ReceivedLog>().unwrap();
        assert_eq!(log.total_damage, 10);
        assert_eq!(log.event_count, 1);
    }

    #[test]
    fn test_cross_frame_communication() {
        let mut world = World::new();
        world.resources_mut().insert(ReceivedLog::default());

        let mut sender = sender_system_a.into_system();
        let mut receiver = receiver_system.into_system();

        sender.init(&mut world);
        receiver.init(&mut world);

        // --- FRAME 1 ---
        // Run Sender only
        unsafe {
            sender.run(&UnsafeWorldCell::new(&mut world));
        }

        // Validate receiver hasn't run
        {
            let log = world.resources().get::<ReceivedLog>().unwrap();
            assert_eq!(log.event_count, 0);
        }

        // --- SIMULATE END OF FRAME ---
        // Manually swap buffers on the resource
        {
            let mut queue = world
                .resources_mut()
                .get_mut::<MessageQueue<DamageEvent>>()
                .unwrap();
            queue.swap_buffers();
        }

        // --- FRAME 2 ---
        // Run Receiver
        unsafe {
            receiver.run(&UnsafeWorldCell::new(&mut world));
        }

        let log = world.resources().get::<ReceivedLog>().unwrap();
        assert_eq!(
            log.total_damage, 10,
            "Message should persist across frame swap"
        );
        assert_eq!(log.event_count, 1);
    }

    #[test]
    fn test_reader_cursor_progress() {
        let mut world = World::new();
        world.resources_mut().insert(ReceivedLog::default());

        let mut sender = sender_system_a.into_system(); // Sends 10
        let mut receiver = receiver_system.into_system();

        sender.init(&mut world);
        receiver.init(&mut world);

        // --- BATCH 1 ---
        unsafe {
            sender.run(&UnsafeWorldCell::new(&mut world));
        }
        unsafe {
            receiver.run(&UnsafeWorldCell::new(&mut world));
        }

        {
            let log = world.resources().get::<ReceivedLog>().unwrap();
            assert_eq!(log.total_damage, 10);
        }

        // --- BATCH 2 ---
        // Run sender again. Receiver runs again.
        // Receiver should ONLY process the new message, not re-process the old one.
        unsafe {
            sender.run(&UnsafeWorldCell::new(&mut world));
        }
        unsafe {
            receiver.run(&UnsafeWorldCell::new(&mut world));
        }

        let log = world.resources().get::<ReceivedLog>().unwrap();
        // 10 (batch 1) + 10 (batch 2) = 20
        // If it re-read batch 1, it would be 30.
        assert_eq!(log.total_damage, 20);
        assert_eq!(log.event_count, 2);
    }

    #[test]
    fn test_independent_readers() {
        // Create a second log type for a second system
        #[derive(Debug, Default)]
        struct SpyLog {
            count: u32,
        }
        impl_component!(SpyLog);

        fn spy_system(mut events: MessageReader<DamageEvent>, mut log: ResMut<SpyLog>) {
            for _ in events.iter() {
                log.count += 1;
            }
        }

        let mut world = World::new();
        world.resources_mut().insert(ReceivedLog::default());
        world.resources_mut().insert(SpyLog::default());

        let mut sender = sender_system_b.into_system(); // Sends 2 events (20, 30)
        let mut receiver_1 = receiver_system.into_system();
        let mut receiver_2 = spy_system.into_system();

        sender.init(&mut world);
        receiver_1.init(&mut world);
        receiver_2.init(&mut world);

        // --- RUN ---
        unsafe {
            sender.run(&UnsafeWorldCell::new(&mut world));
        } // writes 2 events

        // Both systems read the SAME events
        unsafe {
            receiver_1.run(&UnsafeWorldCell::new(&mut world));
        }
        unsafe {
            receiver_2.run(&UnsafeWorldCell::new(&mut world));
        }

        let log1 = world.resources().get::<ReceivedLog>().unwrap();
        let log2 = world.resources().get::<SpyLog>().unwrap();

        assert_eq!(log1.event_count, 2);
        assert_eq!(log1.total_damage, 50); // 20 + 30

        assert_eq!(
            log2.count, 2,
            "Second system should independently read the messages"
        );
    }

    #[test]
    fn test_accumulation_and_buffer_swaps() {
        let mut world = World::new();
        world.resources_mut().insert(ReceivedLog::default());

        let mut sender = sender_system_a.into_system(); // Sends 10
        let mut receiver = receiver_system.into_system();

        sender.init(&mut world);
        receiver.init(&mut world);

        // 1. Frame 1: Write
        unsafe {
            sender.run(&UnsafeWorldCell::new(&mut world));
        }

        // 2. Swap Buffers
        {
            let mut queue = world
                .resources_mut()
                .get_mut::<MessageQueue<DamageEvent>>()
                .unwrap();
            queue.swap_buffers();
        }

        // 3. Frame 2: Write MORE
        unsafe {
            sender.run(&UnsafeWorldCell::new(&mut world));
        }

        // 4. Receiver runs now.
        // It should see Frame 1 (now in Front Buffer) AND Frame 2 (now in Back Buffer)
        unsafe {
            receiver.run(&UnsafeWorldCell::new(&mut world));
        }

        let log = world.resources().get::<ReceivedLog>().unwrap();
        assert_eq!(log.event_count, 2);
        assert_eq!(log.total_damage, 20);
    }
}
