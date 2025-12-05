use crate::archetype::{Archetype, ArchetypeId};
use crate::borrow::{AtomicBorrow, ColumnBorrowChecker};
use crate::component::{Component, ComponentId, ComponentMask, ComponentRegistry};
use crate::entity::Entity;
use crate::world::World;

use std::marker::PhantomData;
use std::ptr::NonNull;

// ============================================================================
// Internal Machinery (Fetch & View)
// ============================================================================

/// Holds raw pointers to columns for a specific Archetype iteration.
/// 'Current Indec' starts at 0
/// 'cirrent index' exeeding length is undefined behavior.
pub trait Fetch<'a> {
    type Item;
    /// Advances the fetch and returns the next item.
    /// current index is advanced by 1.
    /// # Safety
    /// Caller must ensure current index < length.
    unsafe fn next(&mut self) -> Self::Item;
    /// Get the item at given index. Don't advance.
    /// # Safety
    /// Caller must ensure index < length.
    unsafe fn get(&mut self, index: usize) -> Self::Item;
    /// Skips the given number of items.
    /// Current index is advanced by count.
    /// # Safety
    /// Caller must ensure current index + count <= length.
    unsafe fn skip(&mut self, count: usize) {
        for _ in 0..count {
            let _ = unsafe { self.next() };
        }
    }
}

/// A View defines how to access data from an Archetype for a specific lifetime 'a.
pub trait View<'a>: Sized {
    type Item;
    type Fetch: Fetch<'a, Item = Self::Item>;

    /// Creates a Fetch for a specific Archetype.
    /// Returns None if the archetype is missing required components.
    fn create_fetch(
        archetype: &'a mut Archetype,
        registry: &ComponentRegistry,
        tick: u32,
    ) -> Option<Self::Fetch>;

    /// Borrow the columns needed for the view.
    fn borrow_columns(registry: &ComponentRegistry, borrow_checker: &mut ColumnBorrowChecker);
}

// --- Read Implementation ---
pub struct ReadFetch<T> {
    ptr: NonNull<T>,
}

impl<'a, T: 'static> Fetch<'a> for ReadFetch<T> {
    type Item = &'a T;

    #[inline(always)]
    unsafe fn next(&mut self) -> Self::Item {
        unsafe {
            let ret = self.ptr.as_ref();
            self.ptr = NonNull::new_unchecked(self.ptr.as_ptr().add(1));
            ret
        }
    }

    #[inline(always)]
    unsafe fn get(&mut self, index: usize) -> Self::Item {
        unsafe { &*self.ptr.as_ptr().add(index) }
    }

    #[inline(always)]
    unsafe fn skip(&mut self, count: usize) {
        unsafe {
            self.ptr = NonNull::new_unchecked(self.ptr.as_ptr().add(count));
        }
    }
}

// ============================================================================
// Public Interface (QueryToken)
//    This is the bridge between Static Types (Query<(&A, &B)>) and Dynamic Views.
// ============================================================================

/// A marker trait for types that describe a query.
/// Implemented for &'static T, &'static mut T, and tuples.
/// Bridge between the lifetime-bound View and the static QueryData.
pub trait QueryToken {
    /// The actual View type constructed during iteration with lifetime 'a.
    type View<'a>: View<'a>;
    /// The static metadata type.
    type Persistent: QueryData;
}

// ============================================================================
// Token and static impls
// ============================================================================

impl<T: Component> QueryToken for &T {
    type View<'a> = &'a T;
    type Persistent = ReadComponent<T>;
}

#[derive(Debug)]
pub struct ReadComponent<T>(PhantomData<T>);

impl<T: Component> QueryData for ReadComponent<T> {
    fn populate_ids(registry: &mut ComponentRegistry, out: &mut Vec<ComponentId>) {
        out.push(registry.register::<T>());
    }

    fn borrow_columns(registry: &ComponentRegistry, checker: &mut ColumnBorrowChecker) {
        checker.borrow(registry.get_id::<T>().expect("Component not registered"));
    }
}

impl<T: Component> QueryToken for &mut T {
    type View<'a> = Mut<'a, T>;
    type Persistent = WriteComponent<T>;
}

#[derive(Debug)]
pub struct WriteComponent<T>(PhantomData<T>);

impl<T: Component> QueryData for WriteComponent<T> {
    fn populate_ids(registry: &mut ComponentRegistry, out: &mut Vec<ComponentId>) {
        out.push(registry.register::<T>());
    }

    fn borrow_columns(registry: &ComponentRegistry, checker: &mut ColumnBorrowChecker) {
        checker.borrow_mut(registry.get_id::<T>().expect("Component not registered"));
    }
}

// --- Write Implementation ---
pub struct WriteFetch<'a, T> {
    ptr: NonNull<T>,
    ticks: NonNull<u32>,
    current_tick: u32,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T: 'static> Fetch<'a> for WriteFetch<'a, T> {
    type Item = Mut<'a, T>;

    #[inline(always)]
    unsafe fn next(&mut self) -> Self::Item {
        unsafe {
            let ret = &mut *self.ptr.as_ptr();
            self.ptr = NonNull::new_unchecked(self.ptr.as_ptr().add(1));

            // mark as mutated this tick
            // *self.ticks.as_ptr() = self.current_tick;
            // update tick pointer to point to next
            self.ticks = NonNull::new_unchecked(self.ticks.as_ptr().add(1));

            Mut {
                value: ret,
                tick_ptr: self.ticks.as_ptr().sub(1),
                current_tick: self.current_tick,
            }
        }
    }

    #[inline(always)]
    unsafe fn get(&mut self, index: usize) -> Self::Item {
        unsafe {
            let ret = &mut *self.ptr.as_ptr().add(index);

            // mark as mutated this tick
            // *self.ticks.as_ptr().add(index) = self.current_tick;

            Mut {
                value: ret,
                tick_ptr: self.ticks.as_ptr().add(index),
                current_tick: self.current_tick,
            }
        }
    }
    #[inline(always)]
    unsafe fn skip(&mut self, count: usize) {
        unsafe {
            self.ptr = NonNull::new_unchecked(self.ptr.as_ptr().add(count));
            self.ticks = NonNull::new_unchecked(self.ticks.as_ptr().add(count));
        }
    }
}

impl<'a, T: Component> View<'a> for Mut<'a, T> {
    type Item = Mut<'a, T>;
    type Fetch = WriteFetch<'a, T>;

    fn create_fetch(
        archetype: &'a mut Archetype,
        registry: &ComponentRegistry,
        tick: u32,
    ) -> Option<Self::Fetch> {
        let id = registry.get_id::<T>()?;
        let column = archetype.column_mut(&id)?;

        // SAFETY: We have exclusive access to the column due to the mutable borrow.
        // Caller must use borrow_columns to first borrow the column mutably.
        // The column's layout must match T as it was registered.
        // The Column Must make sure that the ptr is valid for T.
        // And the ticks pointer must be pointing to a valid u32 array. parrallel to data.
        let ptr = column.get_ptr(0) as *mut T;
        let ticks_ptr = column.get_ticks_ptr();
        Some(WriteFetch {
            ptr: unsafe { NonNull::new_unchecked(ptr) },
            current_tick: tick,
            ticks: unsafe { NonNull::new_unchecked(ticks_ptr) },
            _marker: PhantomData,
        })
    }

    fn borrow_columns(registry: &ComponentRegistry, borrow_checker: &mut ColumnBorrowChecker) {
        borrow_checker.borrow_mut(registry.get_id::<T>().expect("Component not registered"));
    }
}

impl<'a, T: Component> View<'a> for &'a T {
    type Item = &'a T;
    type Fetch = ReadFetch<T>;

    fn create_fetch(
        archetype: &'a mut Archetype,
        registry: &ComponentRegistry,
        _tick: u32,
    ) -> Option<Self::Fetch> {
        let id = registry.get_id::<T>()?;
        let column = archetype.column_mut(&id)?;

        let ptr = column.get_ptr(0) as *mut T;
        // SAFETY: We have a shared borrow to the column.
        // column's layout must match T as it was registered.
        Some(ReadFetch {
            ptr: unsafe { NonNull::new_unchecked(ptr) },
        })
    }

    fn borrow_columns(registry: &ComponentRegistry, borrow_checker: &mut ColumnBorrowChecker) {
        borrow_checker.borrow(registry.get_id::<T>().expect("Component not registered"));
    }
}
// The wrapper yielded by the iterator
pub struct Mut<'a, T> {
    value: &'a mut T,
    tick_ptr: *mut u32,
    current_tick: u32,
}

impl<'a, T> std::ops::Deref for Mut<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value
    }
}

impl<'a, T> std::ops::DerefMut for Mut<'a, T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        // ONLY update the tick when DerefMut is actually called
        // SAFETY: tick_ptr is valid and points to a u32.
        // This is guaranteed by the WriteFetch struct.
        unsafe { *self.tick_ptr = self.current_tick };
        self.value
    }
}

/// Trait for the static metadata required to manage a query (IDs, Locks).
/// This MUST be 'static.
pub trait QueryData: 'static + Send + Sync {
    fn populate_ids(registry: &mut ComponentRegistry, out: &mut Vec<ComponentId>);
    fn borrow_columns(registry: &ComponentRegistry, borrow_checker: &mut ColumnBorrowChecker);
}

impl QueryToken for Entity {
    type View<'a> = Entity;
    type Persistent = EntityData;
}

pub struct EntityData;
impl QueryData for EntityData {
    fn populate_ids(_: &mut ComponentRegistry, _: &mut Vec<ComponentId>) {}
    fn borrow_columns(_: &ComponentRegistry, _: &mut ColumnBorrowChecker) {}
}

impl<'a> View<'a> for Entity {
    type Item = Entity;
    type Fetch = EntityFetch<'a>;

    fn create_fetch(
        archetype: &'a mut Archetype,
        _registry: &ComponentRegistry,
        _tick: u32,
    ) -> Option<Self::Fetch> {
        Some(EntityFetch {
            entities: archetype.entities(),
            current: 0,
        })
    }

    fn borrow_columns(_registry: &ComponentRegistry, _borrow_checker: &mut ColumnBorrowChecker) {
        // No components to borrow
    }
}

pub struct EntityFetch<'a> {
    entities: &'a [Entity],
    current: usize,
}

impl<'a> Fetch<'a> for EntityFetch<'a> {
    type Item = Entity;

    unsafe fn next(&mut self) -> Self::Item {
        let ent = self.entities[self.current];
        self.current += 1;
        ent
    }

    unsafe fn get(&mut self, index: usize) -> Self::Item {
        self.entities[index]
    }

    unsafe fn skip(&mut self, count: usize) {
        self.current += count;
    }
}

// --- Impl for &'static mut T ---

// ============================================================================
// The QueryInner Struct (Stores Static Data & State)
// ============================================================================

/// `QueryInner` acts as the persistent state for a system query.
///
/// It holds:
/// 1. **Static Metadata**: Component IDs and Masks required for Views (`QD`) and Filters (`FD`).
/// 2. **State**: A cache of matching `ArchetypeId`s to avoid scanning all archetypes every frame.
/// 3. **Safety**: A `ColumnBorrowChecker` to ensure this query doesn't violate aliasing rules
///    (e.g., mutable access to the same component twice).
///
/// This struct is `Send + Sync` because it only holds metadata, not actual component references.
#[derive(Debug)]
pub struct QueryInner<QD: QueryData, FD: FilterData = ()> {
    cached_archetypes: Vec<ArchetypeId>,
    last_updated_arch_idx: ArchetypeId,
    /// The tick at which this query was last run. Used for change detection filters.
    last_query_tick: u32,
    /// Helper to track which columns are borrowed during iteration.
    borrow_checker: ColumnBorrowChecker,
    /// Mask of components required by the View (e.g., `&Position`, `&mut Velocity`).
    view_required: ComponentMask,
    /// Mask of components required by the Filter (e.g., `With<Player>`).
    filter_required: ComponentMask,
    /// Mask of components explicitly excluded (e.g., `Without<Static>`).
    filter_excluded: ComponentMask,
    _marker: PhantomData<(QD, FD)>,
}

// SAFETY: QueryInner only holds integers (IDs) and bitmasks. It does not hold
// references to World data.
unsafe impl<QD: QueryData, FD: FilterData> Send for QueryInner<QD, FD> {}
unsafe impl<QD: QueryData, FD: FilterData> Sync for QueryInner<QD, FD> {}

impl<QD: QueryData, FD: FilterData> QueryInner<QD, FD> {
    /// Factory method that takes Dynamic tokens (`Q`, `F`) to extract their
    /// Static counterparts (`QD`, `FD`) and initialize the struct.
    pub fn new<Q: QueryToken<Persistent = QD>, F: Filter<Persistent = FD>>(
        registry: &mut ComponentRegistry,
    ) -> Self {
        let mut view_required = Vec::new();
        QD::populate_ids(registry, &mut view_required);

        let mut filter_required = Vec::new();
        let mut filter_excluded = Vec::new();
        FD::populate_requirements(registry, &mut filter_required, &mut filter_excluded);

        // 1. Calculate View borrows (e.g., &mut Position)
        // 2. Calculate Filter borrows (e.g., Changed<Velocity> requires reading ticks)
        let mut filter_borrow_checker = ColumnBorrowChecker::new();
        let mut borrow_checker = ColumnBorrowChecker::new();
        QD::borrow_columns(registry, &mut borrow_checker);
        FD::borrow_columns(registry, &mut filter_borrow_checker);

        // 3. Merge them. Allows both View and Filter to borrow their columns (even if
        //    overlapping). Thsi is safe because View and Filter never run concurrently.
        borrow_checker.overlay(&filter_borrow_checker);

        Self {
            borrow_checker,
            cached_archetypes: Vec::new(),
            last_updated_arch_idx: ArchetypeId(0),
            view_required: ComponentMask::from_ids(&view_required),
            filter_required: ComponentMask::from_ids(&filter_required),
            filter_excluded: ComponentMask::from_ids(&filter_excluded),
            last_query_tick: 0,
            _marker: PhantomData,
        }
    }

    /// Scans new archetypes in the World since the last update and adds matches to the cache.
    fn update_archetype_cache(&mut self, world: &World) {
        if self.last_updated_arch_idx.0 == world.archetypes.len() {
            return;
        }
        for arch in world.archetypes.since(self.last_updated_arch_idx) {
            if self.check_archetype(arch) {
                self.cached_archetypes.push(arch.id);
            }
        }
        self.last_updated_arch_idx.0 = world.archetypes.len();
    }

    /// Checks if a specific archetype matches the View and Filter requirements.
    fn check_archetype(&self, arch: &Archetype) -> bool {
        // Must have all components requested by the View (e.g., &Position)
        if !arch.component_mask.contains_all(&self.view_required) {
            return false;
        }
        // Must have all components requested by With<T>
        if !arch.component_mask.contains_all(&self.filter_required) {
            return false;
        }
        // Must NOT have any components requested by Without<T>
        if arch.component_mask.intersects(&self.filter_excluded) {
            return false;
        }
        true
    }

    /// Validates that the query's borrow requirements do not conflict internally.
    /// Also applies these borrows to the matching archetypes in the world to ensure
    /// runtime safety against other simultaneous queries (if we were multi-threaded without a scheduler).
    pub(crate) fn borrow_check(&mut self, world: &mut World) -> bool {
        for &arch_id in &self.cached_archetypes {
            let arch = &mut world.archetypes[arch_id];
            if !self.borrow_checker.apply_borrow(arch) {
                return false;
            }
        }
        true
    }

    /// Returns a list of borrows needed by this query. Used by the `Scheduler` to
    /// determine which systems can run in parallel.
    pub(crate) fn granular_borrow_check(
        &mut self,
        world: &World,
        start_from: ArchetypeId,
    ) -> Vec<(ArchetypeId, ColumnBorrowChecker)> {
        self.update_archetype_cache(world);

        self.cached_archetypes
            .iter()
            .filter(|&&id| id >= start_from)
            .copied()
            .map(|arch_id| {
                let arch = &world.archetypes[arch_id];
                (arch_id, self.borrow_checker.for_archetype(arch))
            })
            .collect()
    }

    /// Creates an iterator over the query results.
    ///
    /// # Generic Parameters
    /// * `V`: The Dynamic View type (e.g., `&'a Position`).
    /// * `F`: The Dynamic Filter type (e.g., `Changed<Velocity>`).
    pub fn iter<'a, V: View<'a>, F: Filter<Persistent = FD>>(
        &'a mut self,
        world: &'a mut World,
    ) -> QueryIter<'a, V, F> {
        self.update_archetype_cache(world);

        // Runtime Borrow Check: Locks the columns in the archetypes.
        // If this fails, it means another iterator is holding a conflicting lock.
        if !self.borrow_check(world) {
            panic!("Query borrow conflict detected");
        }

        let tick = world.tick();
        self.last_query_tick = tick;

        QueryIter {
            world,
            archetype_ids: &self.cached_archetypes,
            current_arch_idx: 0,
            borrow_checker: &mut self.borrow_checker,
            current_fetch: None,
            current_skip_filter: None,
            current_len: 0,
            current_row: 0,
            last_query_tick: tick,
        }
    }

    /// Iterates deeply, running a closure on every match.
    /// This is often faster than `iter()` because it avoids strict iterator state management
    /// and compiler optimization barriers.
    pub fn for_each<'a, V: View<'a>, F: Filter<Persistent = FD>, Func>(
        &'a mut self,
        world: &'a mut World,
        mut func: Func,
    ) where
        Func: FnMut(V::Item),
    {
        self.update_archetype_cache(world);
        if !self.borrow_check(world) {
            panic!("Query borrow conflict detected");
        }

        // Capture tick before update for filter logic
        let query_tick = self.last_query_tick;

        for &arch_id in &self.cached_archetypes {
            // SAFETY: We iterate distinct archetypes sequentially.
            // We cast `*mut Archetype` to create two mutable references (`arch`, `arch2`).
            //
            // WHY IS THIS SAFE?
            // 1. `arch` is used for the View (reading/writing component data).
            // 2. `arch2` is used for the Filter (reading component ticks/data).
            //
            // The `borrow_check` above guarantees that the columns requested by View and Filter
            // do not overlap mutably (e.g. View writes Pos, Filter reads Pos is forbidden).
            // The `AtomicBorrow` locks inside the columns ensure runtime enforcement.
            let arch = unsafe { &mut *(&mut world.archetypes[arch_id] as *mut Archetype) };
            let arch2 = unsafe { &mut *(&mut world.archetypes[arch_id] as *mut Archetype) };

            let len = arch.len();
            if len == 0 {
                continue;
            }

            // Create Filter (e.g., checks change ticks)
            let mut current_skip_filter = F::create_skip_filter(arch2, &world.registry, query_tick);

            // Create Fetch (pointers to data columns)
            if let Some(mut fetch) = V::create_fetch(arch, &world.registry, world.tick()) {
                if let Some(skip_filter) = &mut current_skip_filter {
                    for _ in 0..len {
                        // Check filter first
                        if skip_filter.should_skip() {
                            // SAFETY: `skip(1)` simply advances the pointers. We are in bounds (0..len).
                            unsafe { fetch.skip(1) };
                            continue;
                        }
                        // SAFETY: `next()` reads/writes data. We are in bounds.
                        unsafe {
                            func(fetch.next());
                        }
                    }
                } else {
                    // Optimized loop without filter checks
                    for _ in 0..len {
                        unsafe {
                            func(fetch.next());
                        }
                    }
                }
            }
        }

        self.last_query_tick = world.tick();
        self.release_borrows(world);
    }

    /// Fetches a specific entity's data if it matches the query.
    pub fn get<'a, V: View<'a>, F: Filter<Persistent = FD>>(
        &'a mut self,
        world: &'a mut World,
        entity: Entity,
    ) -> Option<GetGuard<'a, V::Item>> {
        let location = world.entity_location(entity)?;

        let arch_id = location.archetype_id();
        let arch = unsafe { &mut *(&mut world.archetypes[arch_id] as *mut Archetype) };

        // 2. Check if Entity's Archetype matches Query masks
        if !self.check_archetype(arch) {
            return None;
        }

        // 3. Local Borrow Check (just for this specific access)
        let arch2 = unsafe { &mut *(&mut world.archetypes[arch_id] as *mut Archetype) };
        let mut borrow_checker = ColumnBorrowChecker::new();
        QD::borrow_columns(&world.registry, &mut borrow_checker);
        FD::borrow_columns(&world.registry, &mut borrow_checker);

        if !borrow_checker.apply_borrow(arch) {
            panic!("Query borrow conflict detected");
        }

        // 4. Create Fetch pointing to the start of the archetype
        let mut fetch = V::create_fetch(arch, &world.registry, self.last_query_tick)?;

        // 5. Retrieve the atomic borrow states so `GetGuard` can release them later
        let raw_borrows = borrow_checker.get_raw_borrows(arch2);

        // 6. Get the specific row data
        // SAFETY: `location.row()` comes from `world.entity_index` which is kept in sync with archetypes.
        unsafe { Some(GetGuard::new(fetch.get(location.row()), raw_borrows)) }
    }

    /// Releases locks on all cached archetypes. Called when Iterators/Guards drop.
    fn release_borrows(&mut self, world: &mut World) {
        for &arch_id in &self.cached_archetypes {
            let arch = unsafe { &mut *(&mut world.archetypes[arch_id] as *mut Archetype) };
            self.borrow_checker.release_borrow(arch);
        }
    }

    pub(crate) fn last_updated_arch_idx(&self) -> ArchetypeId {
        self.last_updated_arch_idx
    }
}

// ============================================================================
// GetGuard (RAII for Single Entity Access)
// ============================================================================

/// An RAII wrapper returned by `Query::get`.
/// It provides access to the query item (e.g. `(&mut Position, &Velocity)`).
/// When dropped, it releases the runtime locks on the component columns.
pub struct GetGuard<'a, T> {
    inner: T,
    /// List of atomic borrows to release on drop.
    /// Format: (Pointer to AtomicBorrow, is_mutable)
    borrows: Vec<(&'a AtomicBorrow, bool)>,
}

impl<'a, T> Drop for GetGuard<'a, T> {
    fn drop(&mut self) {
        // Release all locks acquired for this specific access
        for (borrow, is_mut) in &self.borrows {
            if *is_mut {
                borrow.release_mut();
            } else {
                borrow.release();
            }
        }
    }
}

impl<T> std::ops::Deref for GetGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.inner
    }
}

impl<T> std::ops::DerefMut for GetGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.inner
    }
}

impl<'a, T> GetGuard<'a, T> {
    fn new(inner: T, borrows: Vec<(&'a AtomicBorrow, bool)>) -> Self {
        Self { inner, borrows }
    }
}

// ============================================================================
// The Query Iterator
// ============================================================================

/// An iterator that walks through matching archetypes and returns component data.
/// It holds the locks on the columns for the duration of its lifetime.
pub struct QueryIter<'a, V: View<'a>, F: Filter> {
    /// Reference to the world to access archetypes.
    world: &'a mut World,
    /// List of archetypes that match the query requirements.
    archetype_ids: &'a [ArchetypeId],
    /// Index of the archetype we are currently iterating.
    current_arch_idx: usize,
    /// The tick used for filter change detection.
    last_query_tick: u32,
    /// Reference to the checker in `QueryInner` to release borrows on drop.
    borrow_checker: &'a mut ColumnBorrowChecker,

    /// The Fetch object for the current archetype. Holds raw pointers to columns.
    current_fetch: Option<V::Fetch>,
    /// The Filter object for the current archetype.
    current_skip_filter: Option<F::SkipFilter<'a>>,
    /// Length of the current archetype.
    current_len: usize,
    /// Current row index within the archetype.
    current_row: usize,
}

impl<'a, V: View<'a>, F: Filter> Iterator for QueryIter<'a, V, F> {
    type Item = V::Item;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // 1. Try to pull from the current archetype's fetch
            if let Some(fetch) = &mut self.current_fetch
                && self.current_row < self.current_len
            {
                self.current_row += 1;

                // Apply Filter Logic
                if let Some(skip_filter) = &mut self.current_skip_filter
                    && skip_filter.should_skip()
                {
                    // Skip this entity.
                    // SAFETY: `fetch.skip(1)` simply advances the pointers.
                    // We checked `current_row < current_len`, so this is safe.
                    unsafe { fetch.skip(1) };
                    continue;
                }

                // Return Item
                // SAFETY:
                // 1. `fetch` holds valid pointers to columns for this archetype.
                // 2. We checked bounds `current_row < current_len`.
                // 3. The `AtomicBorrow` locks prevent data races.
                return Some(unsafe { fetch.next() });
            }

            // 2. Current archetype exhausted or not set. Move to next.
            if self.current_arch_idx >= self.archetype_ids.len() {
                return None;
            }

            let arch_id = self.archetype_ids[self.current_arch_idx];
            self.current_arch_idx += 1;

            // SAFETY: See comment in `QueryInner::for_each`.
            // We are splitting the archetype pointer to allow disjoint column access
            // between the View (V) and the Filter (F).
            // Aliasing is guaranteed safe by `QueryInner::borrow_check`.
            let arch = unsafe { &mut *(&mut self.world.archetypes[arch_id] as *mut Archetype) };
            let arch2 = unsafe { &mut *(&mut self.world.archetypes[arch_id] as *mut Archetype) };

            if arch.len() == 0 {
                continue;
            }

            // Setup state for new archetype
            self.current_len = arch.len();
            self.current_row = 0;
            self.current_fetch = V::create_fetch(arch, &self.world.registry, self.world.tick());
            self.current_skip_filter =
                F::create_skip_filter(arch2, &self.world.registry, self.last_query_tick);
        }
    }
}

impl<'a, V: View<'a>, F: Filter> Drop for QueryIter<'a, V, F> {
    fn drop(&mut self) {
        // Critical: Release the runtime locks on all archetypes we touched.
        for &arch_id in self.archetype_ids {
            let arch = unsafe { &mut *(&mut self.world.archetypes[arch_id] as *mut Archetype) };
            self.borrow_checker.release_borrow(arch);
        }
        self.borrow_checker.clear();
    }
}
// ============================================================================
// Filters
// ============================================================================

// Static data for filters (IDs, Borrows)
pub trait FilterData: 'static + Send + Sync {
    fn populate_requirements(
        registry: &mut ComponentRegistry,
        required: &mut Vec<ComponentId>,
        excluded: &mut Vec<ComponentId>,
    );
    fn borrow_columns(_registry: &ComponentRegistry, _checker: &mut ColumnBorrowChecker) {}
}

// Dynamic behavior for filters (Skip logic)
pub trait Filter {
    type Persistent: FilterData;
    type SkipFilter<'a>: SkipFilter = ();

    fn create_skip_filter<'a>(
        _archetype: &'a mut Archetype,
        _registry: &ComponentRegistry,
        _tick: u32,
    ) -> Option<Self::SkipFilter<'a>> {
        None
    }
}

pub trait SkipFilter {
    fn should_skip(&mut self) -> bool;
    fn skip_rows(&mut self, count: usize) {
        for _ in 0..count {
            let _ = self.should_skip();
        }
    }
}

impl SkipFilter for () {
    fn should_skip(&mut self) -> bool {
        false
    }
}

// Unit ()
impl FilterData for () {
    fn populate_requirements(
        _: &mut ComponentRegistry,
        _: &mut Vec<ComponentId>,
        _: &mut Vec<ComponentId>,
    ) {
    }
    fn borrow_columns(_: &ComponentRegistry, _: &mut ColumnBorrowChecker) {}
}
impl Filter for () {
    type Persistent = ();
    type SkipFilter<'a> = ();
    fn create_skip_filter<'a>(
        _: &'a mut Archetype,
        _: &ComponentRegistry,
        _: u32,
    ) -> Option<Self::SkipFilter<'a>> {
        None
    }
}

// With<T>
pub struct WithData<T>(PhantomData<T>);
impl<T: Component> FilterData for WithData<T> {
    fn populate_requirements(
        registry: &mut ComponentRegistry,
        required: &mut Vec<ComponentId>,
        _: &mut Vec<ComponentId>,
    ) {
        required.push(registry.register::<T>());
    }
    fn borrow_columns(_: &ComponentRegistry, _: &mut ColumnBorrowChecker) {}
}

pub struct With<T>(PhantomData<T>);
impl<T: Component> Filter for With<T> {
    type Persistent = WithData<T>;
    type SkipFilter<'a> = ();
    fn create_skip_filter(_: &mut Archetype, _: &ComponentRegistry, _: u32) -> Option<()> {
        None
    }
}

// Without<T>
pub struct WithoutData<T>(PhantomData<T>);
impl<T: Component> FilterData for WithoutData<T> {
    fn populate_requirements(
        registry: &mut ComponentRegistry,
        _: &mut Vec<ComponentId>,
        excluded: &mut Vec<ComponentId>,
    ) {
        excluded.push(registry.register::<T>());
    }
    fn borrow_columns(_: &ComponentRegistry, _: &mut ColumnBorrowChecker) {}
}

pub struct Without<T>(PhantomData<T>);
impl<T: Component> Filter for Without<T> {
    type Persistent = WithoutData<T>;
    type SkipFilter<'a> = ();
    fn create_skip_filter(_: &mut Archetype, _: &ComponentRegistry, _: u32) -> Option<()> {
        None
    }
}

// AND<T, U>: Combines two filters with AND logic
pub struct And<T, U>(PhantomData<(T, U)>);
impl<T: Filter, U: Filter> Filter for And<T, U> {
    type SkipFilter<'a> = AndSkip<T::SkipFilter<'a>, U::SkipFilter<'a>>;
    type Persistent = And<T::Persistent, U::Persistent>;

    #[inline(always)]
    fn create_skip_filter<'a>(
        archetype: &'a mut Archetype,
        registry: &ComponentRegistry,
        tick: u32,
    ) -> Option<Self::SkipFilter<'a>> {
        // safety: The filters SHOULD check that they obey borrow rules. using
        // AtomicBorrow.

        let archetype_2 = unsafe { &mut *(&mut *archetype as *mut Archetype) };

        let t_filter = T::create_skip_filter(archetype, registry, tick)?;
        let u_filter = U::create_skip_filter(archetype_2, registry, tick)?;
        Some(AndSkip(t_filter, u_filter))
    }
}

impl<T: FilterData, U: FilterData> FilterData for And<T, U> {
    fn populate_requirements(
        registry: &mut ComponentRegistry,
        required: &mut Vec<ComponentId>,
        excluded: &mut Vec<ComponentId>,
    ) {
        T::populate_requirements(registry, required, excluded);
        U::populate_requirements(registry, required, excluded);
    }
    fn borrow_columns(registry: &ComponentRegistry, checker: &mut ColumnBorrowChecker) {
        T::borrow_columns(registry, checker);
        U::borrow_columns(registry, checker);
    }
}

pub struct AndSkip<T: SkipFilter, U: SkipFilter>(T, U);

impl<T: SkipFilter, U: SkipFilter> SkipFilter for AndSkip<T, U> {
    fn should_skip(&mut self) -> bool {
        self.0.should_skip() || self.1.should_skip()
    }

    fn skip_rows(&mut self, count: usize) {
        self.0.skip_rows(count);
        self.1.skip_rows(count);
    }
}

pub struct ChangedData<T>(PhantomData<T>);
impl<T: Component> FilterData for ChangedData<T> {
    fn populate_requirements(
        registry: &mut ComponentRegistry,
        required: &mut Vec<ComponentId>,
        _: &mut Vec<ComponentId>,
    ) {
        required.push(registry.register::<T>());
    }
    fn borrow_columns(registry: &ComponentRegistry, checker: &mut ColumnBorrowChecker) {
        checker.borrow(registry.get_id::<T>().expect("Component not registered"));
    }
}

pub struct ChangedSkipFilter<T: Component> {
    change_tick: u32,
    changed_ptr: *const u32,
    _marker: PhantomData<T>,
}
impl<T: Component> SkipFilter for ChangedSkipFilter<T> {
    #[inline(always)]
    fn should_skip(&mut self) -> bool {
        let last_changed = unsafe { *self.changed_ptr };
        unsafe { self.changed_ptr = self.changed_ptr.add(1) };
        last_changed < self.change_tick
    }
}

pub struct Changed<T>(PhantomData<T>);
impl<T: Component> Filter for Changed<T> {
    type Persistent = ChangedData<T>;
    type SkipFilter<'a> = ChangedSkipFilter<T>;

    fn create_skip_filter<'a>(
        archetype: &'a mut Archetype,
        registry: &ComponentRegistry,
        tick: u32,
    ) -> Option<Self::SkipFilter<'a>> {
        let id = registry.get_id::<T>()?;
        let column = archetype.column_mut(&id)?;
        let ptr = column.get_ticks_ptr();
        Some(ChangedSkipFilter {
            change_tick: tick,
            changed_ptr: ptr,
            _marker: PhantomData,
        })
    }
}

// --- Tuple Implementation (Recursion) ---

macro_rules! impl_query_tuples {
    ($($name:ident $num:tt),*) => {
        // Fetch & View (Already covered, but Fetch Tuple is needed)
        impl<'a, $($name),*> Fetch<'a> for ($($name,)*)
        where $($name: Fetch<'a>),*
        {
            type Item = ($($name::Item,)*);
            #[inline(always)]
            unsafe fn next(&mut self) -> Self::Item {
                unsafe { ( $( self.$num.next(), )* ) }
            }
            #[inline(always)]
            unsafe fn get(&mut self, index: usize) -> Self::Item {
                unsafe { ( $( self.$num.get(index), )* ) }
            }
             #[inline(always)]
            unsafe fn skip(&mut self, count: usize) {
                unsafe { $( self.$num.skip(count); )* }
            }
        }

        impl<'a, $($name: View<'a>),*> View<'a> for ($($name,)*) {
            type Item = ($($name::Item,)*);
            type Fetch = ($($name::Fetch,)*);
            fn create_fetch(arch: &'a mut Archetype, reg: &ComponentRegistry, tick: u32) -> Option<Self::Fetch> {
                let ptr = arch as *mut Archetype;
                Some(( $( $name::create_fetch(unsafe { &mut *ptr }, reg, tick)?, )* ))
            }

            fn borrow_columns(registry: &ComponentRegistry, borrow_checker: &mut ColumnBorrowChecker) {
                $( $name::borrow_columns(registry, borrow_checker); )*
            }
        }

        // QueryData Tuple
        impl<$($name: QueryData),*> QueryData for ($($name,)*) {
             fn populate_ids(registry: &mut ComponentRegistry, out: &mut Vec<ComponentId>) {
                $( $name::populate_ids(registry, out); )*
            }
            fn borrow_columns(registry: &ComponentRegistry, borrow_checker: &mut ColumnBorrowChecker) {
                $( $name::borrow_columns(registry, borrow_checker); )*
            }
        }

        // QueryToken Tuple (The Bridge)
        impl<$($name: QueryToken),*> QueryToken for ($($name,)*) {
            type View<'a> = ($($name::View<'a>,)*);
            type Persistent = ($($name::Persistent,)*);
        }

        // FilterData Tuple
        impl<$($name: FilterData),*> FilterData for ($($name,)*) {
             fn populate_requirements(reg: &mut ComponentRegistry, req: &mut Vec<ComponentId>, exc: &mut Vec<ComponentId>) {
                $( $name::populate_requirements(reg, req, exc); )*
            }
            fn borrow_columns(reg: &ComponentRegistry, chk: &mut ColumnBorrowChecker) {
                $( $name::borrow_columns(reg, chk); )*
            }
        }

        // Filter Tuple
        impl<$($name: Filter),*> Filter for ($($name,)*) {
            type Persistent = ($($name::Persistent,)*);
            type SkipFilter<'a> = ($(Option<$name::SkipFilter<'a>>,)*);

            fn create_skip_filter<'a>(arch: &'a mut Archetype, reg: &ComponentRegistry, tick: u32) -> Option<Self::SkipFilter<'a>> {
                 let ptr = arch as *mut Archetype;
                 Some(( $( $name::create_skip_filter(unsafe { &mut *ptr }, reg, tick), )* ))
            }
        }

        // SkipFilter for Tuple Options
        impl<'a, $($name),*> SkipFilter for ($(Option<$name>,)*)
        where $($name: SkipFilter),*
        {
            #[inline(always)]
            fn should_skip(&mut self) -> bool {
                $(
                    if let Some(filter) = &mut self.$num {
                        if filter.should_skip() { return true; }
                    }
                )*
                false
            }
        }

    }
}

impl_all_tuples!(
    impl_query_tuples, A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13);

// ============================================================================
// Test
// ============================================================================

/// A high-level wrapper around `QueryInner` that remembers the Dynamic types (Q, F).
/// Useful for manual querying outside of systems (e.g., tests, scripts).
pub struct QueryState<Q: QueryToken, F: Filter = ()> {
    inner: QueryInner<Q::Persistent, F::Persistent>,
    _marker: std::marker::PhantomData<(Q, F)>,
}

impl<Q: QueryToken, F: Filter> QueryState<Q, F> {
    pub fn new(registry: &mut crate::component::ComponentRegistry) -> Self {
        Self {
            inner: QueryInner::new::<Q, F>(registry),
            _marker: std::marker::PhantomData,
        }
    }

    /// Iterates without needing explicit type annotations.
    pub fn iter<'w>(
        &'w mut self,
        world: &'w mut crate::world::World,
    ) -> QueryIter<'w, Q::View<'w>, F> {
        self.inner.iter::<Q::View<'w>, F>(world)
    }

    pub fn for_each<'w, Func>(&'w mut self, world: &'w mut World, func: Func)
    where
        Func: FnMut(<Q::View<'w> as View<'w>>::Item),
    {
        self.inner.for_each::<Q::View<'w>, F, Func>(world, func)
    }

    pub fn get<'w>(
        &'w mut self,
        world: &'w mut crate::world::World,
        entity: Entity,
    ) -> Option<GetGuard<'w, <Q::View<'w> as View<'w>>::Item>> {
        self.inner.get::<Q::View<'w>, F>(world, entity)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    // Assuming Component, Entity, World are reachable
    use crate::world::World;

    // ========================================================================
    // TEST HELPERS & COMPONENTS
    // ========================================================================

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Pos {
        x: f32,
        y: f32,
    }

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Vel {
        x: f32,
        y: f32,
    }

    #[derive(Debug, PartialEq)]
    struct Name(String);

    // Marker Components
    #[derive(Debug)]
    struct Player;
    #[derive(Debug)]
    struct Enemy;
    #[derive(Debug)]
    struct Dead;

    impl_component!(Pos, Vel, Name, Player, Enemy, Dead);

    // ------------------------------------------------------------------------
    // Helper: QueryState Wrapper
    // This allows us to write `query.iter(&mut world)` without explicit types
    // ------------------------------------------------------------------------
    #[derive(Debug)]
    pub struct QueryState<Q: QueryToken, F: Filter = ()> {
        inner: QueryInner<Q::Persistent, F::Persistent>,
        _marker: std::marker::PhantomData<(Q, F)>,
    }

    impl<Q: QueryToken, F: Filter> QueryState<Q, F> {
        pub fn new(world: &mut World) -> Self {
            Self {
                inner: QueryInner::new::<Q, F>(&mut world.registry),
                _marker: std::marker::PhantomData,
            }
        }

        pub fn iter<'w>(&'w mut self, world: &'w mut World) -> QueryIter<'w, Q::View<'w>, F> {
            self.inner.iter::<Q::View<'w>, F>(world)
        }

        pub fn get<'w>(
            &'w mut self,
            world: &'w mut World,
            entity: Entity,
        ) -> Option<GetGuard<'w, <Q::View<'w> as View<'w>>::Item>> {
            self.inner.get::<Q::View<'w>, F>(world, entity)
        }
    }

    // ========================================================================
    // 1. BASIC ITERATION & DATA ACCESS
    // ========================================================================

    #[test]
    fn test_basic_iteration() {
        let mut world = World::new();

        // Archetype 1: Pos, Vel
        world.spawn((Pos { x: 0.0, y: 0.0 }, Vel { x: 1.0, y: 1.0 }));
        // Archetype 2: Pos (No Vel) - Should be skipped
        world.spawn((Pos { x: 10.0, y: 10.0 },));
        // Archetype 3: Pos, Vel, Player
        world.spawn((Pos { x: 5.0, y: 5.0 }, Vel { x: 0.0, y: 1.0 }, Player));

        let mut query = QueryState::<(&Pos, &Vel)>::new(&mut world);

        let mut count = 0;
        for (pos, vel) in query.iter(&mut world) {
            // Check data logic
            if pos.x == 0.0 {
                assert_eq!(vel.x, 1.0);
            }
            if pos.x == 5.0 {
                assert_eq!(vel.y, 1.0);
            }
            count += 1;
        }

        assert_eq!(count, 2, "Should match Arch 1 and Arch 3, skip Arch 2");
    }

    #[test]
    fn test_mutation() {
        let mut world = World::new();
        world.spawn((Pos { x: 0.0, y: 0.0 }, Vel { x: 1.0, y: 2.0 }));

        let mut query = QueryState::<(&mut Pos, &Vel)>::new(&mut world);

        // Apply Velocity
        for (mut pos, vel) in query.iter(&mut world) {
            pos.x += vel.x;
            pos.y += vel.y;
        }

        // Verify result with immutable query
        let mut check = QueryState::<&Pos>::new(&mut world);
        let pos = check.iter(&mut world).next().unwrap();

        assert_eq!(pos.x, 1.0);
        assert_eq!(pos.y, 2.0);
    }

    #[test]
    fn test_random_access_get() {
        let mut world = World::new();
        let e1 = world.spawn((10u32, 5.0f32)); // Matches
        let e2 = world.spawn((20u32,)); // Misses (missing f32)

        let mut query = QueryState::<(&mut u32, &f32)>::new(&mut world);

        // Test Match
        if let Some(mut guard) = query.get(&mut world, e1) {
            let (val, _) = &mut *guard;
            **val += 100;
        } else {
            panic!("Entity 1 should match");
        }

        // Verify mutation
        if let Some(guard) = query.get(&mut world, e1) {
            assert_eq!(*guard.0, 110);
        }

        // Test Mismatch
        assert!(
            query.get(&mut world, e2).is_none(),
            "Entity 2 missing component"
        );

        // Test Despawned
        assert!(world.despawn(e1));
        assert!(query.get(&mut world, e1).is_none(), "Entity 1 is dead");
    }

    // ========================================================================
    // 2. FILTERS (With, Without, And)
    // ========================================================================

    #[test]
    fn test_filters_with_without() {
        let mut world = World::new();

        // A: Player
        world.spawn((Player, Pos { x: 1.0, y: 0.0 }));
        // B: Enemy
        world.spawn((Enemy, Pos { x: 2.0, y: 0.0 }));
        // C: Player + Dead
        world.spawn((Player, Dead, Pos { x: 3.0, y: 0.0 }));

        // Query: Get Pos for Players who are NOT Dead
        let mut query = QueryState::<&Pos, (With<Player>, Without<Dead>)>::new(&mut world);

        let results: Vec<f32> = query.iter(&mut world).map(|p| p.x).collect();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0], 1.0);
    }

    #[test]
    fn test_complex_and_filters() {
        let mut world = World::new();
        // Matches: Has Pos, Vel, Player. Does NOT have Dead.
        world.spawn((Pos { x: 0., y: 0. }, Vel { x: 0., y: 0. }, Player));
        // Fails: Missing Player
        world.spawn((Pos { x: 0., y: 0. }, Vel { x: 0., y: 0. }));
        // Fails: Has Dead
        world.spawn((Pos { x: 0., y: 0. }, Vel { x: 0., y: 0. }, Player, Dead));

        // Logic: (With<Player> AND Without<Dead>)
        // Note: Tuple filters imply AND logic.
        let mut query = QueryState::<(&Pos, &Vel), (With<Player>, Without<Dead>)>::new(&mut world);

        assert_eq!(query.iter(&mut world).count(), 1);
    }

    // ========================================================================
    // 3. CHANGE DETECTION
    // ========================================================================

    // ========================================================================
    // 4. STRUCTURAL DYNAMICS
    // ========================================================================

    #[test]
    fn test_archetype_fragmentation() {
        let mut world = World::new();
        let mut query = QueryState::<&Pos>::new(&mut world);

        // 1. Spawn Arch A
        world.spawn((Pos { x: 1.0, y: 0.0 },));
        assert_eq!(query.iter(&mut world).count(), 1);

        // 2. Spawn Arch B (Added component)
        world.spawn((Pos { x: 2.0, y: 0.0 }, Vel { x: 0.0, y: 0.0 }));
        // Query must update cache internally
        assert_eq!(query.iter(&mut world).count(), 2);

        // 3. Spawn Arch C (Different component)
        world.spawn((Pos { x: 3.0, y: 0.0 }, Name("Bob".into())));
        assert_eq!(query.iter(&mut world).count(), 3);
    }

    #[test]
    fn test_scattered_memory() {
        // Ensure iteration works across disparate archetypes
        let mut world = World::new();
        world.spawn((1u32,)); // Match
        world.spawn((1.0f32,)); // Miss
        world.spawn((2u32, true)); // Match
        world.spawn((3u32, "Str")); // Match

        let mut query = QueryState::<&u32>::new(&mut world);
        let sum: u32 = query.iter(&mut world).sum();

        assert_eq!(sum, 1 + 2 + 3);
    }

    // ========================================================================
    // 5. SAFETY & BORROW RULES
    // ========================================================================

    #[test]
    #[should_panic]
    fn test_panic_mut_mut_aliasing() {
        let mut world = World::new();
        world.spawn((Pos { x: 0., y: 0. },));

        // Cannot borrow &mut Pos and &mut Pos
        let mut query = QueryState::<(&mut Pos, &mut Pos)>::new(&mut world);
        query.iter(&mut world);
    }

    #[test]
    #[should_panic]
    fn test_panic_mut_ref_aliasing() {
        let mut world = World::new();
        world.spawn((Pos { x: 0., y: 0. },));

        // Cannot borrow &mut Pos and &Pos
        let mut query = QueryState::<(&mut Pos, &Pos)>::new(&mut world);
        query.iter(&mut world);
    }

    #[test]
    fn test_safe_ref_ref_aliasing() {
        let mut world = World::new();
        world.spawn((Pos { x: 10., y: 10. },));

        // Shared access is fine
        let mut query = QueryState::<(&Pos, &Pos)>::new(&mut world);
        for (p1, p2) in query.iter(&mut world) {
            assert_eq!(p1, p2);
        }
    }

    // ========================================================================
    // 6. ZST & EMPTY STRUCTS
    // ========================================================================
    #[test]
    fn test_zst_component() {
        let mut world = World::new();
        world.spawn((10u32, Dead));
        world.spawn((20u32,));

        let mut query = QueryState::<(&u32, &Dead)>::new(&mut world);
        let count = query.iter(&mut world).count();
        assert_eq!(count, 1);
    }
}
