use std::{marker::PhantomData, ptr};

use crate::{
    bundle::Bundle,
    entity::Entity,
    prelude::{Component, Resource},
    system::{SystemAccess, SystemParam, UnsafeWorldCell},
    threading::FromWorldThread,
    world::{CommandThreadLocalContext, World},
};

type ExecuteShim = unsafe fn(*mut u8, &mut World);
type DropShim = unsafe fn(*mut u8);

pub struct ThreadLocalCommandQueue {
    store: Vec<u8>,
    ptrs: Vec<CommandExecutablePtr>,
}

pub struct CommandExecutablePtr {
    /// Where is the data in the store?
    data_offset: usize,
    /// What we will do with that data?
    execute_shim: ExecuteShim,
    /// If the world is dropped before flush, how to clean up?
    drop_shim: DropShim,
}

impl Drop for ThreadLocalCommandQueue {
    fn drop(&mut self) {
        // If we drop the queue without flushing, we must run destructors
        // for the data sitting in the byte buffer.
        for cmd in &self.ptrs {
            unsafe {
                let ptr = self.store.as_mut_ptr().add(cmd.data_offset);
                (cmd.drop_shim)(ptr);
            }
        }
    }
}

pub trait CommandExecutable: Sized {
    /// What the user gets back immediately (e.g., Entity ID).
    type Output;

    /// The state saved to the queue for the flush phase.
    /// Usually `Self`, but can be `()` if everything was done immediately.
    type Storage: Send + 'static;

    /// Do we requier a call to `execute` during flush?
    const IS_DEFERRED: bool = true;

    /// Runs immediately on the current thread.
    /// Has exclusive access to thread-local resources (Entity Allocator, Insert Buffer).
    fn immediate(self, context: &mut CommandThreadLocalContext) -> (Self::Storage, Self::Output);

    /// Runs during `World::flush`.
    /// Has exclusive access to the whole World.
    fn execute(storage: Self::Storage, world: &mut World);
}

pub struct SpawnCommand<B>
where
    B: Bundle,
{
    pub bundle: B,
}

impl<B> CommandExecutable for SpawnCommand<B>
where
    B: crate::bundle::Bundle,
{
    type Output = Entity;
    type Storage = ();
    const IS_DEFERRED: bool = false;

    fn execute(_: (), _world: &mut World) {}

    fn immediate(self, thread_local: &mut CommandThreadLocalContext) -> ((), Entity) {
        let e = thread_local.entity_allocator_mut().alloc();
        thread_local
            .insert_buffer_mut()
            .insert_bundle(e, self.bundle);
        ((), e)
    }
}

pub struct DespawnCommand(pub Entity);

impl CommandExecutable for DespawnCommand {
    type Output = ();
    type Storage = Entity;
    const IS_DEFERRED: bool = true;

    fn immediate(self, _: &mut CommandThreadLocalContext) -> (Entity, ()) {
        (self.0, ())
    }

    fn execute(entity: Entity, world: &mut World) {
        world.despawn(entity);
    }
}

pub struct ClosureCommand<F>(pub F);

impl<F> CommandExecutable for ClosureCommand<F>
where
    F: FnOnce(&mut World) + Send + 'static,
{
    type Output = ();
    type Storage = F; // Store the closure itself
    const IS_DEFERRED: bool = true;

    fn immediate(self, _: &mut CommandThreadLocalContext) -> (F, ()) {
        (self.0, ())
    }

    fn execute(f: F, world: &mut World) {
        (f)(world);
    }
}

pub struct InsertComponentCommand<C>
where
    C: Component,
{
    pub entity: Entity,
    pub component: C,
}

impl<C> CommandExecutable for InsertComponentCommand<C>
where
    C: Component + Send + 'static,
{
    type Output = ();
    type Storage = (Entity, C);
    const IS_DEFERRED: bool = true;

    fn immediate(self, _: &mut CommandThreadLocalContext) -> ((Entity, C), ()) {
        ((self.entity, self.component), ())
    }

    fn execute(data: (Entity, C), world: &mut World) {
        let (entity, component) = data;
        world.insert_component(entity, component);
    }
}

pub struct RemoveComponentCommand<C>
where
    C: Component,
{
    pub entity: Entity,
    _marker: PhantomData<C>,
}

impl<C> CommandExecutable for RemoveComponentCommand<C>
where
    C: Component + Send + 'static,
{
    type Output = ();
    type Storage = Entity;
    const IS_DEFERRED: bool = true;

    fn immediate(self, _: &mut CommandThreadLocalContext) -> (Entity, ()) {
        (self.entity, ())
    }

    fn execute(entity: Entity, world: &mut World) {
        world.remove_component::<C>(entity);
    }
}

pub struct InsertResourceCommand<R>
where
    R: Resource,
{
    pub resource: R,
}

impl<R> CommandExecutable for InsertResourceCommand<R>
where
    R: Resource + Send + 'static,
{
    type Output = ();
    type Storage = R;
    const IS_DEFERRED: bool = true;

    fn immediate(self, _: &mut CommandThreadLocalContext) -> (R, ()) {
        (self.resource, ())
    }

    fn execute(resource: R, world: &mut World) {
        world.resources_mut().insert(resource);
    }
}

impl FromWorldThread for ThreadLocalCommandQueue {
    fn new_thread_local(_thread_id: usize, _world: &World) -> ThreadLocalCommandQueue {
        ThreadLocalCommandQueue::new()
    }
}

impl Default for ThreadLocalCommandQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl ThreadLocalCommandQueue {
    pub fn new() -> Self {
        Self {
            store: Vec::new(),
            ptrs: Vec::new(),
        }
    }

    unsafe fn push<C: CommandExecutable>(
        &mut self,
        command: C,
        thread_local: &mut CommandThreadLocalContext,
    ) -> C::Output {
        let (immediate_data_internal, immediate_data_external) = command.immediate(thread_local);
        if C::IS_DEFERRED {
            /// # Safety
            /// `ptr` must point to `Self::ImmediateData`.
            unsafe fn execute_shim_impl<C: CommandExecutable>(ptr: *mut u8, world: &mut World) {
                // We assume ptr was created via Box::into_raw
                let in_d = unsafe { ptr::read_unaligned(ptr as *mut C::Storage) };

                C::execute(in_d, world);
            }
            /// # Safety
            /// `ptr` must point to `Self::ImmediateData`.
            unsafe fn drop_shim_impl<C: CommandExecutable>(ptr: *mut u8) {
                let in_d = unsafe { ptr::read_unaligned(ptr as *mut C::Storage) };
                drop(in_d);
            }

            let size = std::mem::size_of::<C::Storage>();

            self.store.reserve(size);

            // Get the write location
            let data_offset = self.store.len();
            let ptr = unsafe { self.store.as_mut_ptr().add(data_offset) };

            unsafe { ptr::write_unaligned(ptr as *mut C::Storage, immediate_data_internal) };

            unsafe { self.store.set_len(data_offset + size) };

            self.ptrs.push(CommandExecutablePtr {
                data_offset,
                execute_shim: execute_shim_impl::<C>,
                drop_shim: drop_shim_impl::<C>,
            });
        }
        immediate_data_external
    }

    /// Execute all commands and clear the buffer
    pub fn apply(&mut self, world: &mut World) {
        // Iterate over commands
        for cmd in &self.ptrs {
            unsafe {
                // Reconstruct the pointer into the buffer
                let ptr = self.store.as_mut_ptr().add(cmd.data_offset);
                // Call the shim
                (cmd.execute_shim)(ptr, world);
            }
        }

        self.ptrs.clear();
        self.store.clear();
    }
}

#[derive(Clone, Copy)]
pub struct Command<'w> {
    // We hold the world to allow lazy init via FromWorldThread
    world: &'w UnsafeWorldCell<'w>,
}

// SAFETY:
// 1. &World is Send+Sync.
// 2. We only mutate thread-local data specific to the executing thread.
unsafe impl Send for Command<'_> {}
unsafe impl Sync for Command<'_> {}

impl SystemParam for Command<'_> {
    type State = ();
    type Item<'w> = Command<'w>;

    // Commands are deferred, so they don't impact archetype access or conflict with other systems
    // during the execution phase.
    fn init_state(_world: &mut World, _access: &mut SystemAccess) -> Self::State {}

    unsafe fn get_param<'w>(
        _state: &'w mut Self::State,
        world: &'w UnsafeWorldCell<'w>,
    ) -> Self::Item<'w> {
        Command { world }
    }
}

impl<'w> Command<'w> {
    /// Pushes a generic command to the thread-local queue.
    pub fn push<T: CommandExecutable + Send + 'static>(&mut self, command: T) -> T::Output {
        // SAFETY:
        // 1. Systems run on specific threads managed by the scheduler.
        // 2. `WorldThreadLocalStore` guarantees that `get_mut()` (or equivalent) returns
        //    a reference unique to the current thread.
        // 3. Because `Command` is obtained via `SystemParam`, we are inside a valid system run.
        let thread_local = unsafe { self.world.world().thread_local_mut() };
        let data = &mut CommandThreadLocalContext {
            entity_allocator: &mut thread_local.entity_allocator,
            insert_buffer: &mut thread_local.insert_buffer,
        };
        unsafe { thread_local.command_queue.push(command, data) }
    }

    /// Spawns a new entity with the given bundle.
    /// This action is deferred until the world is flushed..
    pub fn spawn<B: Bundle + Send + 'static>(&mut self, bundle: B) -> Entity {
        self.push(SpawnCommand { bundle })
    }

    /// Despawns the given entity.
    /// This action is deferred until the world is flushed.
    pub fn despawn(&mut self, entity: Entity) {
        self.push(DespawnCommand(entity))
    }

    /// Inserts a component into an existing entity.
    /// This action is deferred until the world is flushed.\
    pub fn insert_component<C: Component>(&mut self, entity: Entity, component: C) {
        self.push(InsertComponentCommand { entity, component })
    }

    /// Removes a component from an existing entity.
    /// This action is deferred until the world is flushed.
    pub fn remove_component<C: Component>(&mut self, entity: Entity) {
        self.push(RemoveComponentCommand::<C> {
            entity,
            _marker: PhantomData,
        })
    }

    /// Executes a custom closure on the world during flush.
    pub fn execute<F>(&mut self, f: F)
    where
        F: FnOnce(&mut World) + Send + 'static,
    {
        self.push(ClosureCommand(f))
    }
}

#[cfg(test)]
mod test_commands {
    use std::ops::Deref;

    use super::*;
    use crate::prelude::*;
    use crate::query::QueryState;
    use crate::world::World;

    // --- Mock Data ---
    #[derive(Debug, PartialEq, Default, Clone, Copy)]
    struct Position {
        x: f32,
        y: f32,
    }
    impl_component!(Position);

    #[derive(Debug, Default)]
    struct Score(u32);
    // Treat as resource

    // --- Helper to run a system once ---
    fn run_system<S, M>(world: &mut World, system: S)
    where
        S: IntoSystem<M>,
    {
        let mut sys = system.into_system();
        sys.init(world);
        unsafe {
            sys.run(&UnsafeWorldCell::new(world));
        }
    }

    #[test]
    fn test_spawn_command_optimization() {
        let mut world = World::new();

        // System that spawns an entity
        fn spawn_sys(mut commands: Command) {
            let e = commands.spawn((Position { x: 10.0, y: 20.0 },));

            // Crucial Check: The Entity ID is correct
            commands.execute(move |world| {
                assert!(world.entities().is_alive(e));
                let query = QueryState::<&Position>::new(world);
                assert_eq!(
                    *query.get(e).unwrap().deref(),
                    &Position { x: 10.0, y: 20.0 }
                );
            });
        }

        run_system(&mut world, spawn_sys);

        // Pre-Flush:
        // The entity ID is allocated in the thread-local allocator,
        // but it is not yet "alive" in the main world index or archetype storage.
        {
            let query = crate::query::QueryState::<&Position>::new(&mut world);
            assert_eq!(query.iter().count(), 0);
        }

        // Flush
        world.flush();

        // Post-Flush:
        // The insert buffer should have been drained and the entity created.
        {
            let query = crate::query::QueryState::<&Position>::new(&mut world);
            let pos = query.iter().next().expect("Entity should exist now");
            assert_eq!(pos.x, 10.0);
        }
    }

    #[test]
    fn test_despawn_command_deferred() {
        let mut world = World::new();
        let e = world.spawn((Position { x: 0.0, y: 0.0 },));
        world.flush(); // Ensure E exists

        fn despawn_sys(mut commands: Command, query: Query<Entity>) {
            for e in query.iter() {
                commands.despawn(e);
            }
        }

        run_system(&mut world, despawn_sys);

        // Pre-Flush: Entity should STILL be alive.
        // Despawn is IS_DEFERRED = true, so it sits in the queue.
        assert!(world.entities().is_alive(e));
        assert!(world.entity_location(e).is_some());

        world.flush();

        // Post-Flush: Entity should be dead.
        assert!(!world.entities().is_alive(e));
        assert!(world.entity_location(e).is_none());
    }

    #[test]
    fn test_closure_command_execution() {
        let mut world = World::new();
        world.resources_mut().insert(Score(0));

        fn closure_sys(mut commands: Command) {
            // Queue a closure to modify the resource
            commands.execute(|world| {
                let mut score = world.resources_mut().get_mut::<Score>().unwrap();
                score.0 += 100;
            });
        }

        run_system(&mut world, closure_sys);

        // Pre-Flush: No change.
        assert_eq!(world.resources().get::<Score>().unwrap().0, 0);

        world.flush();

        // Post-Flush: Closure ran.
        assert_eq!(world.resources().get::<Score>().unwrap().0, 100);
    }

    #[test]
    fn test_mixed_commands_ordering() {
        // This tests that we can mix immediate and deferred commands securely.
        let mut world = World::new();

        fn mixed_sys(mut commands: Command) {
            // 1. Immediate: Spawns to insert_buffer
            let e = commands.spawn((Position { x: 1.0, y: 1.0 },));

            // 2. Deferred: Closure
            commands.execute(move |world| {
                // By the time this runs (flush), 'e' should exist in the world
                // insert buffers flush before command queues

                // Allocator update -> Insert Buffer -> Command Queue
                if world.entities().is_alive(e) {
                    let mut res = world.resources_mut().get_mut_or_insert_default::<Score>();
                    res.0 = 999;
                }
            });
        }

        run_system(&mut world, mixed_sys);
        world.flush();

        // Verify the closure saw the entity as alive
        let score = world.resources().get::<Score>();
        assert!(score.is_some(), "Closure should have run");
        assert_eq!(
            score.unwrap().0,
            999,
            "Entity should have been alive during closure execution"
        );
    }

    #[test]
    fn test_custom_command_implementation() {
        // User-defined command
        struct DoubleScore;
        impl CommandExecutable for DoubleScore {
            type Output = ();
            type Storage = (); // No state to save
            const IS_DEFERRED: bool = true;

            fn immediate(self, _ctx: &mut CommandThreadLocalContext) -> ((), ()) {
                // Logic could go here if thread-local
                ((), ())
            }

            fn execute(_: (), world: &mut World) {
                if let Some(mut score) = world.resources_mut().get_mut::<Score>() {
                    score.0 *= 2;
                }
            }
        }

        let mut world = World::new();
        world.resources_mut().insert(Score(10));

        fn custom_cmd_sys(mut commands: Command) {
            commands.push(DoubleScore);
        }

        run_system(&mut world, custom_cmd_sys);
        assert_eq!(world.resources().get::<Score>().unwrap().0, 10);

        world.flush();
        assert_eq!(world.resources().get::<Score>().unwrap().0, 20);
    }
}
