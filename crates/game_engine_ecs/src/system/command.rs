use std::{any::Any, array, marker::PhantomData, ptr};

use crate::{
    bundle::Bundle,
    component::{ComponentMask, ComponentMeta, MAX_COMPONENTS},
    entity::Entity,
    observers::ObserverExecutionQueue,
    prelude::{Component, ComponentId, Resource},
    resource::ThreadLocalResources,
    storage::TypeErasedSequence,
    system::{SystemAccess, SystemParam, UnsafeWorldCell},
    thread_entity_allocator::LocalThreadEntityAllocator,
    threading::FromWorldThread,
    world::World,
};

type ExecuteShim = unsafe fn(*mut u8, &mut World);
type DropShim = unsafe fn(*mut u8);

pub struct ThreadLocalCommandQueue {
    store: Vec<u8>,
    ptrs: Vec<CommandExecutablePtr>,
}

/// Special view of thread-local data not involving the command queue.
pub struct CommandThreadLocalContext<'a> {
    pub entity_allocator: &'a mut LocalThreadEntityAllocator,
    pub insert_buffer: &'a mut EntityInsertBuffer,
    pub observer_queue: &'a mut ObserverExecutionQueue,
    pub thread_local_resorces: &'a mut ThreadLocalResources,
}

#[derive(Default)]
pub struct ToBeDespawned {
    pub to_be_despawned: Vec<Entity>,
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
    type Storage: Send + 'static;

    /// Do we requier a call to `execute` during flush?
    const IS_DEFERRED: bool = true;

    /// Runs immediately on the current thread.
    /// Has exclusive access to thread-local resources.
    fn immediate(
        self,
        context: &mut CommandThreadLocalContext,
        world: &World,
    ) -> (Self::Storage, Self::Output);

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

    fn immediate(
        self,
        thread_local: &mut CommandThreadLocalContext,
        _world: &World,
    ) -> ((), Entity) {
        let e = thread_local.entity_allocator_mut().alloc();
        thread_local
            .insert_buffer_mut()
            .insert_bundle(e, self.bundle);
        ((), e)
    }
}

#[derive(Default)]
pub struct EntityInsertBuffer {
    /// A map of entity -> mask/archetype they are in.
    /// Useful for other thread local ops to get data
    /// about entities that are not spawned yet
    pub(crate) entity_to_mask: Vec<ComponentMask>,
    /// The entity ids to add, grouped by archetype
    pub(crate) ids: Vec<Vec<Entity>>,

    /// Mask per archetype
    pub(crate) masks: Vec<ComponentMask>,

    pub(crate) comp_ids: Vec<Vec<ComponentId>>,
    pub(crate) comp_metas: Vec<Vec<ComponentMeta>>,

    /// The sparse columns for each archetype's buffered entities.
    pub(crate) bundle_cols: Vec<Box<[Option<TypeErasedSequence>; MAX_COMPONENTS]>>,
}

impl EntityInsertBuffer {
    /// Resets the entity buffers but KEEPS the allocated sparse arrays.
    pub fn clear_entities(&mut self) {
        // We only clear the entity IDs.
        // The sparse arrays (bundle_cols) are cleared/reset during the flush process
        // to ensure slots remain valid (Some) for the next frame.
        for ids in &mut self.ids {
            ids.clear();
        }
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        // We can just check the first archetype list or maintain a dirty flag
        self.ids.iter().all(|v| v.is_empty())
    }

    fn get_or_create_archetype_buffer<B: Bundle>(&mut self, mask: ComponentMask) -> usize {
        match self.masks.binary_search(&mask) {
            Ok(index) => index,
            Err(index) => {
                self.masks.insert(index, mask);
                self.ids.insert(index, Vec::new());
                self.comp_ids.insert(index, Vec::new());
                self.comp_metas.insert(index, Vec::new());

                // Allocate the sparse array
                let mut sparse_cols: Box<[Option<TypeErasedSequence>; MAX_COMPONENTS]> =
                    Box::new(array::from_fn(|_| None));

                // We collect here only once per archetype creation
                let ids: Vec<ComponentId> = B::component_ids().collect();
                let metas: Vec<ComponentMeta> = B::component_metas().collect();

                let mut sorted_components: Vec<_> = ids.iter().zip(metas.iter()).collect();
                sorted_components.sort_unstable_by_key(|(id, _)| **id);

                let comp_ids_cache = &mut self.comp_ids[index];
                let comp_metas_cache = &mut self.comp_metas[index];

                for (&id, &meta) in sorted_components {
                    comp_ids_cache.push(id);
                    comp_metas_cache.push(meta);
                    sparse_cols[id.0] = Some(TypeErasedSequence::new(&meta));
                }

                self.bundle_cols.insert(index, sparse_cols);
                index
            }
        }
    }

    pub fn insert_bundle<B: Bundle>(&mut self, entity: Entity, bundle: B) {
        let mask = ComponentMask::from_ids(B::component_ids());

        let arch_index = self.get_or_create_archetype_buffer::<B>(mask);
        let bundle_cols = &mut self.bundle_cols[arch_index];

        // SAFETY:
        // 1. `bundle_cols` is guaranteed to have `Some` at indices for `B::component_ids()`.
        //    This is ensured by `get_or_create_archetype_buffer` AND `flush_into` (which replenishes slots).
        unsafe {
            bundle.put(bundle_cols.as_mut_slice());
        }

        if self.entity_to_mask.len() <= entity.index() as usize {
            self.entity_to_mask
                .resize(entity.index() as usize + 1, ComponentMask::new());
        }
        *unsafe {
            self.entity_to_mask
                .get_unchecked_mut(entity.index() as usize)
        } = mask;
        self.ids[arch_index].push(entity);
    }
}

impl FromWorldThread for EntityInsertBuffer {
    fn new_thread_local(_thread_id: usize, _world: &World) -> Self {
        Self::default()
    }
}

pub struct TriggerCommand<E: Any> {
    pub event: E,
    pub entity: Entity,
}

impl<E> CommandExecutable for TriggerCommand<E>
where
    E: Any,
{
    type Output = ();
    type Storage = ();
    const IS_DEFERRED: bool = false;

    fn execute(_: (), _world: &mut World) {}

    fn immediate(self, thread_local: &mut CommandThreadLocalContext, world: &World) -> ((), ()) {
        thread_local
            .observer_queue
            .fire(world, self.event, self.entity);
        ((), ())
    }
}

pub struct DespawnCommand(pub Entity);

impl CommandExecutable for DespawnCommand {
    type Output = ();
    type Storage = ();
    const IS_DEFERRED: bool = true;

    fn immediate(self, thread_local: &mut CommandThreadLocalContext, _world: &World) -> ((), ()) {
        thread_local
            .thread_local_resorces
            .get_or_insert_default::<ToBeDespawned>()
            .to_be_despawned
            .push(self.0);
        ((), ())
    }

    fn execute(_: (), _world: &mut World) {}
}

pub struct ClosureCommand<F>(pub F);

impl<F> CommandExecutable for ClosureCommand<F>
where
    F: FnOnce(&mut World) + Send + 'static,
{
    type Output = ();
    type Storage = F; // Store the closure itself
    const IS_DEFERRED: bool = true;

    fn immediate(self, _: &mut CommandThreadLocalContext, _world: &World) -> (F, ()) {
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

    fn immediate(self, _: &mut CommandThreadLocalContext, _world: &World) -> ((Entity, C), ()) {
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

    fn immediate(self, _: &mut CommandThreadLocalContext, _world: &World) -> (Entity, ()) {
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

    fn immediate(self, _: &mut CommandThreadLocalContext, _world: &World) -> (R, ()) {
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
        world: &World,
    ) -> C::Output {
        let (immediate_data_internal, immediate_data_external) =
            command.immediate(thread_local, world);
        if C::IS_DEFERRED {
            /// # Safety
            /// `ptr` must point to `Self::ImmediateData`.
            unsafe fn execute_shim_impl<C: CommandExecutable>(ptr: *mut u8, world: &mut World) {
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
        #[cfg(feature = "tracy")]
        let _trace = {
            let trace = tracy_client::span!("Apply commands.");
            trace.emit_color(0xFFAA00);
            trace
        };

        // "Steal" the pointers.
        let ptrs = std::mem::take(&mut self.ptrs);

        // Iterate the local vector.
        for cmd in &ptrs {
            unsafe {
                let ptr = self.store.as_mut_ptr().add(cmd.data_offset);

                // Call the shim
                (cmd.execute_shim)(ptr, world);
            }
        }

        self.store.clear();
    }
}

pub struct Commands<'w> {
    // We hold the world to allow lazy init via FromWorldThread
    world: UnsafeWorldCell<'w>,
}

impl<'w> Clone for Commands<'w> {
    fn clone(&self) -> Self {
        Self {
            world: self.world.clone(),
        }
    }
}

// SAFETY:
// 1. &World is Send+Sync.
// 2. We only mutate thread-local data specific to the executing thread.
unsafe impl Send for Commands<'_> {}
unsafe impl Sync for Commands<'_> {}

impl SystemParam for Commands<'_> {
    type State = ();
    type Item<'w, 's> = Commands<'w>;

    // Commands are deferred, so they don't impact archetype access or conflict with other systems
    // during the execution phase.
    fn init_state(_world: &mut World, _access: &mut SystemAccess) -> Self::State {}

    unsafe fn get_param<'w, 's>(
        _state: &'w mut Self::State,
        world: &'w UnsafeWorldCell<'w>,
    ) -> Self::Item<'w, 's> {
        Commands {
            world: world.clone(),
        }
    }
}

impl<'w> Commands<'w> {
    pub fn new(world: &'w mut World) -> Self {
        Self {
            world: UnsafeWorldCell::new(world),
        }
    }
    /// Pushes a generic command to the thread-local queue.
    pub fn push<T: CommandExecutable>(&self, command: T) -> T::Output {
        // SAFETY:
        // 1. Systems run on specific threads managed by the scheduler.
        // 2. `WorldThreadLocalStore` guarantees that `get_mut()` (or equivalent) returns
        //    a reference unique to the current thread.
        // 3. Because `Command` is obtained via `SystemParam`, we are inside a valid system run.
        let thread_local = unsafe { self.world.world().thread_local_mut() };
        let data = &mut CommandThreadLocalContext {
            entity_allocator: &mut thread_local.entity_allocator,
            insert_buffer: &mut thread_local.insert_buffer,
            observer_queue: &mut thread_local.observer_queue,
            thread_local_resorces: &mut thread_local.thread_local_resorces,
        };
        unsafe {
            thread_local
                .command_queue
                .push(command, data, self.world.world())
        }
    }

    /// Spawns a new entity with the given bundle.
    /// This action is deferred until the world is flushed..
    pub fn spawn<B: Bundle + Send + 'static>(&self, bundle: B) -> Entity {
        self.push(SpawnCommand { bundle })
    }

    /// Fires an event on an entity triggering any observers listening to it
    /// This action is deferred until the world is flushed.
    pub fn trigger<E: Any>(&self, entity: Entity, event: E) {
        self.push(TriggerCommand { event, entity })
    }

    /// Despawns the given entity.
    /// This action is deferred until the world is flushed.
    pub fn despawn(&self, entity: Entity) {
        self.push(DespawnCommand(entity))
    }

    /// Inserts a component into an existing entity.
    /// This action is deferred until the world is flushed.\
    pub fn insert_component<C: Component>(&self, entity: Entity, component: C) {
        self.push(InsertComponentCommand { entity, component })
    }

    /// Removes a component from an existing entity.
    /// This action is deferred until the world is flushed.
    pub fn remove_component<C: Component>(&self, entity: Entity) {
        self.push(RemoveComponentCommand::<C> {
            entity,
            _marker: PhantomData,
        })
    }

    /// Executes a custom closure on the world during flush.
    pub fn execute<F>(&self, f: F)
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
        fn spawn_sys(mut commands: Commands) {
            let e = commands.spawn((Position { x: 10.0, y: 20.0 },));

            // Crucial Check: The Entity ID is correct
            commands.execute(move |world| {
                assert!(world.entities().is_alive(e));
                let query = world.query::<&Position, ()>();
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
            let query = world.query::<&Position, ()>();
            assert_eq!(query.iter().count(), 0);
        }

        // Flush
        world.flush();

        // Post-Flush:
        // The insert buffer should have been drained and the entity created.
        {
            let query = world.query::<&Position, ()>();
            let pos = query.iter().next().expect("Entity should exist now");
            assert_eq!(pos.x, 10.0);
        }
    }

    #[test]
    fn test_despawn_command_deferred() {
        let mut world = World::new();
        let e = world.spawn((Position { x: 0.0, y: 0.0 },));
        world.flush(); // Ensure E exists

        fn despawn_sys(mut commands: Commands, query: Query<Entity>) {
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

        fn closure_sys(mut commands: Commands) {
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

        fn mixed_sys(mut commands: Commands) {
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

            fn immediate(self, _ctx: &mut CommandThreadLocalContext, _world: &World) -> ((), ()) {
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

        fn custom_cmd_sys(mut commands: Commands) {
            commands.push(DoubleScore);
        }

        run_system(&mut world, custom_cmd_sys);
        assert_eq!(world.resources().get::<Score>().unwrap().0, 10);

        world.flush();
        assert_eq!(world.resources().get::<Score>().unwrap().0, 20);
    }
}
