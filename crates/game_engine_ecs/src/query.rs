use crate::archetype::{Archetype, ArchetypeId};
use crate::borrow::AtomicBorrow;
use crate::component::{Component, ComponentId, ComponentRegistry};
use crate::entity::Entity;
use crate::world::World;

use std::marker::PhantomData;
use std::ptr::NonNull;

// ============================================================================
// Internal Machinery (Fetch & View)
//    These operate on specific lifetimes 'a during iteration.
// ============================================================================

/// Holds raw pointers to columns for a specific Archetype iteration.
pub trait Fetch<'a> {
    type Item;
    /// Advances the fetch and returns the next item.
    /// # Safety
    /// Caller must ensure current index < length.
    unsafe fn next(&mut self) -> Self::Item;
    /// Get the item at given index. Don't advance.
    /// # Safety
    /// Caller must ensure index < length.
    unsafe fn get(&mut self, index: usize) -> Self::Item;
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
}

// --- Read Implementation ---
pub struct ReadFetch<'a, T> {
    ptr: NonNull<T>,
    borrow: &'a AtomicBorrow,
}

impl<'a, T> Drop for ReadFetch<'a, T> {
    fn drop(&mut self) {
        self.borrow.release();
    }
}

impl<'a, T: 'static> Fetch<'a> for ReadFetch<'a, T> {
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
}

// --- Write Implementation ---
pub struct WriteFetch<'a, T> {
    ptr: NonNull<T>,
    borrow: &'a AtomicBorrow,
    ticks: NonNull<u32>,
    current_tick: u32,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T> Drop for WriteFetch<'a, T> {
    fn drop(&mut self) {
        self.borrow.release_mut();
    }
}

impl<'a, T: 'static> Fetch<'a> for WriteFetch<'a, T> {
    type Item = &'a mut T;

    #[inline(always)]
    unsafe fn next(&mut self) -> Self::Item {
        unsafe {
            let ret = &mut *self.ptr.as_ptr();
            self.ptr = NonNull::new_unchecked(self.ptr.as_ptr().add(1));

            // mark as mutated this tick
            *self.ticks.as_ptr() = self.current_tick;
            // update tick pointer to point to next
            self.ticks = NonNull::new_unchecked(self.ticks.as_ptr().add(1));

            ret
        }
    }

    #[inline(always)]
    unsafe fn get(&mut self, index: usize) -> Self::Item {
        unsafe {
            let ret = &mut *self.ptr.as_ptr().add(index);

            // mark as mutated this tick
            *self.ticks.as_ptr().add(index) = self.current_tick;

            ret
        }
    }
}

// ============================================================================
// Public Interface (QueryToken)
//    This is the bridge between Static Types (Query<(&A, &B)>) and Dynamic Views.
// ============================================================================

/// A marker trait for types that describe a query.
/// Implemented for &'static T, &'static mut T, and tuples.
pub trait QueryToken: 'static {
    /// The actual View type constructed during iteration with lifetime 'a.
    type View<'a>: View<'a>;

    fn populate_ids(registry: &mut ComponentRegistry, out: &mut Vec<ComponentId>);
}

// --- Impl for &'static T ---
impl<T: Component> QueryToken for &'static T {
    type View<'a> = &'a T;

    fn populate_ids(registry: &mut ComponentRegistry, out: &mut Vec<ComponentId>) {
        out.push(registry.register::<T>());
    }
}

impl<'a, T: Component> View<'a> for &'a T {
    type Item = &'a T;
    type Fetch = ReadFetch<'a, T>;

    fn create_fetch(
        archetype: &'a mut Archetype,
        registry: &ComponentRegistry,
        _tick: u32,
    ) -> Option<Self::Fetch> {
        let id = registry.get_id::<T>()?;
        let column = archetype.column(id)?;

        if !column.borrow_state().borrow() {
            panic!(
                "Immutable borrow failed for {}, you have a mutable borrow somwhere else",
                std::any::type_name::<T>()
            );
        }

        let ptr = column.get_ptr(0) as *mut T;
        Some(ReadFetch {
            ptr: unsafe { NonNull::new_unchecked(ptr) },
            borrow: column.borrow_state(),
        })
    }
}

// --- Impl for &'static mut T ---
impl<T: Component> QueryToken for &'static mut T {
    type View<'a> = &'a mut T;

    fn populate_ids(registry: &mut ComponentRegistry, out: &mut Vec<ComponentId>) {
        out.push(registry.register::<T>());
    }
}

impl<'a, T: Component> View<'a> for &'a mut T {
    type Item = &'a mut T;
    type Fetch = WriteFetch<'a, T>;

    fn create_fetch(
        archetype: &'a mut Archetype,
        registry: &ComponentRegistry,
        tick: u32,
    ) -> Option<Self::Fetch> {
        let id = registry.get_id::<T>()?;
        let column = archetype.column(id)?;

        if !column.borrow_state().borrow_mut() {
            panic!("Mutable borrow failed for {}", std::any::type_name::<T>());
        }

        // SAFETY: We have exclusive access to the column due to the mutable borrow.
        // The column's layout must match T as it was registered.
        // The Column Must make sure that the ptr is valid for T.
        // And the ticks pointer must be pointing to a valid u32 array. parrallel to data.
        let ptr = column.get_ptr(0) as *mut T;
        let ticks_ptr = column.get_ticks_ptr();
        Some(WriteFetch {
            ptr: unsafe { NonNull::new_unchecked(ptr) },
            borrow: column.borrow_state(),
            current_tick: tick,
            ticks: unsafe { NonNull::new_unchecked(ticks_ptr) },
            _marker: PhantomData,
        })
    }
}

// ============================================================================
// The Query Struct
// ============================================================================

pub struct Query<Q: QueryToken, F: Filter = ()> {
    cached_archetypes: Vec<ArchetypeId>,

    // IDs needed for the View (Q)
    view_required: Vec<ComponentId>,

    // IDs needed for the Filter (F)
    filter_required: Vec<ComponentId>,
    filter_excluded: Vec<ComponentId>,

    _marker: PhantomData<(Q, F)>,
}

impl<Q: QueryToken, F: Filter> Query<Q, F> {
    pub fn new(registry: &mut ComponentRegistry) -> Self {
        let mut view_required = Vec::new();
        Q::populate_ids(registry, &mut view_required);
        view_required.sort_unstable();

        let mut filter_required = Vec::new();
        let mut filter_excluded = Vec::new();
        F::populate_requirements(registry, &mut filter_required, &mut filter_excluded);
        filter_required.sort_unstable();
        filter_excluded.sort_unstable();

        Self {
            cached_archetypes: Vec::new(),
            view_required,
            filter_required,
            filter_excluded,
            _marker: PhantomData,
        }
    }

    fn update_archetype_cache(&mut self, world: &World) {
        self.cached_archetypes.clear();

        for arch in &world.archetypes {
            // 1. Check View Requirements (MUST have these to access data)
            let has_view = self
                .view_required
                .iter()
                .all(|req| arch.component_ids.binary_search(req).is_ok());
            if !has_view {
                continue;
            }

            // 2. Check Filter Requirements (With<T>)
            let has_filter = self
                .filter_required
                .iter()
                .all(|req| arch.component_ids.binary_search(req).is_ok());
            if !has_filter {
                continue;
            }

            // 3. Check Filter Exclusions (Without<T>)
            let has_excluded = self
                .filter_excluded
                .iter()
                .any(|req| arch.component_ids.binary_search(req).is_ok());
            if has_excluded {
                continue;
            }

            self.cached_archetypes.push(arch.id);
        }
    }

    pub fn iter<'a>(&'a mut self, world: &'a mut World) -> QueryIter<'a, Q::View<'a>> {
        self.update_archetype_cache(world); // TODO: Optimize to only update when archetypes change

        QueryIter {
            world,
            archetype_ids: &self.cached_archetypes,
            current_arch_idx: 0,
            current_fetch: None,
            current_len: 0,
            current_row: 0,
        }
    }

    /// Get specific entity's query view, if it matches.
    /// Returns None if the entity does not exist or does not match the query.
    pub fn get<'a>(
        &'a mut self,
        world: &'a mut World,
        entity: Entity,
    ) -> Option<<Q::View<'a> as View<'a>>::Item> {
        // 1. O(1) Lookup
        let location = world.entity_location(entity)?;
        let arch_id = location.archetype_id();

        // 2. O(1) access to archetype (Vec index)
        let arch = unsafe { &mut *(&mut world.archetypes[arch_id.0 as usize] as *mut Archetype) };

        // 3. Check Match (Local check only)
        // We do NOT use self.cached_archetypes here.
        // We check if THIS archetype has the required components.
        // This makes it truly O(1) relative to world size.

        // Helper closure to check a list against the archetype
        let matches = |reqs: &[ComponentId]| {
            reqs.iter()
                .all(|req| arch.component_ids.binary_search(req).is_ok())
        };

        if !matches(&self.view_required) {
            return None;
        }
        if !matches(&self.filter_required) {
            return None;
        }

        let has_excluded = self
            .filter_excluded
            .iter()
            .any(|req| arch.component_ids.binary_search(req).is_ok());
        if has_excluded {
            return None;
        }

        // 4. Fetch
        let mut fetch = Q::View::create_fetch(arch, &world.registry, world.tick())?;
        unsafe { Some(fetch.get(location.row())) }
    }
}

// ============================================================================
// The Iterator
// ============================================================================

pub struct QueryIter<'a, V: View<'a>> {
    world: &'a mut World,
    archetype_ids: &'a [ArchetypeId],
    current_arch_idx: usize,

    current_fetch: Option<V::Fetch>,
    current_len: usize,
    current_row: usize,
}

impl<'a, V: View<'a>> Iterator for QueryIter<'a, V> {
    type Item = V::Item;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Case 1: Fetch valid?
            if let Some(fetch) = &mut self.current_fetch
                && self.current_row < self.current_len
            {
                self.current_row += 1;

                // SAFETY: We have verified that current_row < current_len
                return Some(unsafe { fetch.next() });
            }

            // Case 2: We have finished all matched archetypes
            if self.current_arch_idx >= self.archetype_ids.len() {
                return None;
            }

            // Case 3: Setup next archetype
            let arch_id = self.archetype_ids[self.current_arch_idx];
            self.current_arch_idx += 1;

            // Borrow Archetype mutably
            // SAFETY: We access distinct archetypes sequentially.
            // Creating the fetch will borrow columns as needed using the runtime AtomicBorrow.
            // we need to do this unsafe trick to get a mutable reference from a shared one.
            let arch =
                unsafe { &mut *(&mut self.world.archetypes[arch_id.0 as usize] as *mut Archetype) };

            if arch.len() == 0 {
                continue;
            }

            self.current_len = arch.len();
            self.current_row = 0;
            self.current_fetch = V::create_fetch(arch, &self.world.registry, self.world.tick());
        }
    }
}

// ============================================================================
// Filters
// ============================================================================

pub trait Filter: 'static {
    fn populate_requirements(
        registry: &mut ComponentRegistry,
        required: &mut Vec<ComponentId>,
        excluded: &mut Vec<ComponentId>,
    );
}

// Default filter: () -> Matches everything
impl Filter for () {
    fn populate_requirements(
        _: &mut ComponentRegistry,
        _: &mut Vec<ComponentId>,
        _: &mut Vec<ComponentId>,
    ) {
    }
}

// With<T>: Requires T present
pub struct With<T>(PhantomData<T>);
impl<T: Component> Filter for With<T> {
    fn populate_requirements(
        registry: &mut ComponentRegistry,
        required: &mut Vec<ComponentId>,
        _: &mut Vec<ComponentId>,
    ) {
        required.push(registry.register::<T>());
    }
}

// Without<T>: Requires T absent
pub struct Without<T>(PhantomData<T>);
impl<T: Component> Filter for Without<T> {
    fn populate_requirements(
        registry: &mut ComponentRegistry,
        _: &mut Vec<ComponentId>,
        excluded: &mut Vec<ComponentId>,
    ) {
        excluded.push(registry.register::<T>());
    }
}

// AND<T, U>: Combines two filters with AND logic
pub struct And<T, U>(PhantomData<(T, U)>);
impl<T: Filter, U: Filter> Filter for And<T, U> {
    fn populate_requirements(
        registry: &mut ComponentRegistry,
        required: &mut Vec<ComponentId>,
        excluded: &mut Vec<ComponentId>,
    ) {
        T::populate_requirements(registry, required, excluded);
        U::populate_requirements(registry, required, excluded);
    }
}

// --- Tuple Implementation (Recursion) ---

/// Imagine macro parameters, but more like those Russian dolls.
///
/// Calls m!(), m!(A 1), m!(A 1, B 2), and m!(A 1, B 2, C 3) for i.e. (m, A 1, B 2, C 3)
/// where m is any macro, for any number of parameters.
macro_rules! impl_all_tuples {
    // Entry point: We need at least two items (A and B) to start the sequence
    (
        $callback:ident,
        $first_name:ident $first_idx:tt,
        $second_name:ident $second_idx:tt
        $(, $rest_name:ident $rest_idx:tt)*
    ) => {
        // 1. Generate for the first pair (A 0, B 1)
        $callback!($first_name $first_idx, $second_name $second_idx);

        // 2. Start the recursion with the first pair as the "base"
        impl_all_tuples!(
            @recurse
            $callback,
            ($first_name $first_idx, $second_name $second_idx) // Accumulator
            $(, $rest_name $rest_idx)* // Remaining items
        );
    };

    // Recursive Step: We have an accumulator and at least one new item to add
    (
        @recurse
        $callback:ident,
        ($($acc:tt)*), // Everything done so far
        $next_name:ident $next_idx:tt // The next item to process
        $(, $tail_name:ident $tail_idx:tt)* // Items left after this one
    ) => {
        // 1. Generate the callback with the Accumulated items + Next item
        $callback!($($acc)*, $next_name $next_idx);

        // 2. Recurse, moving "Next" into the accumulator
        impl_all_tuples!(
            @recurse
            $callback,
            ($($acc)*, $next_name $next_idx)
            $(, $tail_name $tail_idx)*
        );
    };

    // Base Case: Recursion ends when there are no "next" items left
    (@recurse $callback:ident, ($($acc:tt)*) $(,)?) => {
        // Done.
    };
}
macro_rules! impl_query_token_tuple {
    ($($name:ident $num:tt),*) => {
        impl<'a, $($name),*> Fetch<'a> for ($($name,)*)
        where
            $($name: Fetch<'a>),*

        {
            type Item = ($($name::Item,)*);

            #[inline(always)]
            unsafe fn next(&mut self) -> Self::Item {
                // SAFETY: Caller must ensure we only call this len times.
                unsafe {
                    (
                        $(
                            self.$num.next(),
                        )*
                    )
                }
            }

            #[inline(always)]
            unsafe fn get(&mut self, index: usize) -> Self::Item {
                // SAFETY: Caller must ensure index < length.
                unsafe {
                    (
                        $(
                            self.$num.get(index),
                        )*
                    )
                }
            }
        }

        impl<$($name: QueryToken),*> QueryToken for ($($name,)*) {
            type View<'a> = ($($name::View<'a>,)*);

            fn populate_ids(registry: &mut ComponentRegistry, out: &mut Vec<ComponentId>) {
                $(
                    $name::populate_ids(registry, out);
                )*
            }
        }

        impl<'a, $($name: View<'a>),*> View<'a> for ($($name,)*) {
            type Item = ($($name::Item,)*);
            type Fetch = ($($name::Fetch,)*);

            fn create_fetch(
                archetype: &'a mut Archetype,
                registry: &ComponentRegistry,
                tick: u32,
            ) -> Option<Self::Fetch> {
                // SAFETY: We trick the compiler to allow multiple mutable borrows of archetype.
                // We rely on the fact that each $name accesses disjoint columns (runtime verified by AtomicBorrow).
                let ptr = archetype as *mut Archetype;
                Some((
                    $(
                        $name::create_fetch(unsafe { &mut *ptr }, registry, tick)?,
                    )*
                ))
            }
        }

        impl<$($name: Filter),*> Filter for ($($name,)*) {
            fn populate_requirements(
                registry: &mut ComponentRegistry,
                req: &mut Vec<ComponentId>,
                exc: &mut Vec<ComponentId>
            ) {
                $($name::populate_requirements(registry, req, exc);)*
            }
        }
    }
}

impl_all_tuples!(
    impl_query_token_tuple, A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13);

// ============================================================================
// Test
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::World;

    #[test]
    fn test_query_iteration() {
        let mut world = World::new();

        world.spawn((10u32, 5.0f32));
        world.spawn((20u32, 10.0f32));
        world.spawn((30u32,));

        let mut query = Query::<(&u32, &mut f32)>::new(&mut world.registry);

        let mut count = 0;
        for (pos, vel) in query.iter(&mut world) {
            *vel += 1.0;
            count += 1;
            println!("Pos: {pos}, Vel: {vel}");
        }

        assert_eq!(count, 2);
    }

    #[test]
    #[should_panic]
    fn test_borrow_check() {
        let mut world = World::new();
        world.spawn((10u32,));

        let mut q1 = Query::<(&mut u32, &u32)>::new(&mut world.registry);
        // This should panic because we request &mut u32 and &u32 on the same component
        q1.iter(&mut world).next();
    }

    #[test]
    fn test_filters() {
        let mut world = World::new();

        // Entity 1: [u32, f32]
        world.spawn((10u32, 1.0f32));

        // Entity 2: [u32]
        world.spawn((20u32,));

        // Entity 3: [u32, f32, bool]
        world.spawn((30u32, 2.0f32, true));

        // Query: Give me u32, but ONLY if they don't have a bool (Without<bool>)
        let mut query = Query::<&u32, Without<bool>>::new(&mut world.registry);

        let mut count = 0;
        for _val in query.iter(&mut world) {
            count += 1;
        }

        // Should match E1 (has f32, no bool) and E2 (no bool).
        // Should skip E3 (has bool).
        assert_eq!(count, 2);
    }

    #[derive(Debug, PartialEq)]
    struct Position {
        x: f32,
        y: f32,
    }

    #[derive(Debug, PartialEq)]
    struct Velocity {
        dx: f32,
        dy: f32,
    }

    #[derive(Debug, PartialEq)]
    struct Name(String);

    // Tag components
    struct Player;
    struct Enemy;
    struct Frozen; // Status effect

    // ============================================================================
    // Borrowing Rule Tests
    // ============================================================================

    #[test]
    fn test_shared_borrow_allows_multiple_reads() {
        let mut world = World::new();
        world.spawn((Position { x: 0.0, y: 0.0 },));

        // Requesting immutable access twice to the same component is valid
        let mut query = Query::<(&Position, &Position)>::new(&mut world.registry);

        let mut count = 0;
        for (p1, p2) in query.iter(&mut world) {
            assert_eq!(p1, p2);
            count += 1;
        }
        assert_eq!(count, 1);
    }

    #[test]
    #[should_panic(expected = "Mutable borrow failed")]
    fn test_borrow_rule_mut_mut_conflict() {
        let mut world = World::new();
        world.spawn((Position { x: 0.0, y: 0.0 },));

        // Illegal: Two mutable references to the same component
        let mut query = Query::<(&mut Position, &mut Position)>::new(&mut world.registry);

        // Should panic immediately upon creating the fetch for the first matching archetype
        query.iter(&mut world).next();
    }

    #[test]
    #[should_panic(expected = "Mutable borrow failed")]
    fn test_borrow_rule_ref_mut_conflict() {
        let mut world = World::new();
        world.spawn((Position { x: 0.0, y: 0.0 },));

        // Illegal: Aliasing rules (one mutable, one immutable)
        let mut query = Query::<(&Position, &mut Position)>::new(&mut world.registry);

        query.iter(&mut world).next();
    }

    #[test]
    #[should_panic(expected = "Immutable borrow failed")]
    fn test_borrow_rule_mut_ref_conflict() {
        let mut world = World::new();
        world.spawn((Position { x: 0.0, y: 0.0 },));

        // Illegal: Aliasing rules (flipped order)
        let mut query = Query::<(&mut Position, &Position)>::new(&mut world.registry);

        query.iter(&mut world).next();
    }

    // ============================================================================
    // Complex Filter Tests
    // ============================================================================

    #[test]
    fn test_complex_tuple_filters() {
        let mut world = World::new();

        // Archetype 1: Player, Pos, Vel (Matches)
        world.spawn((
            Player,
            Position { x: 0.0, y: 0.0 },
            Velocity { dx: 1.0, dy: 0.0 },
        ));

        // Archetype 2: Enemy, Pos, Vel (Filtered out by With<Player>)
        world.spawn((
            Enemy,
            Position { x: 10.0, y: 10.0 },
            Velocity { dx: 0.0, dy: 0.0 },
        ));

        // Archetype 3: Player, Pos, Vel, Frozen (Filtered out by Without<Frozen>)
        world.spawn((
            Player,
            Position { x: 0.0, y: 0.0 },
            Velocity { dx: 0.0, dy: 0.0 },
            Frozen,
        ));

        // Archetype 4: Player, Pos (Filtered out, missing Velocity required by View)
        world.spawn((Player, Position { x: 5.0, y: 5.0 }));

        // Logic: Has Position + Velocity AND is a Player AND is NOT Frozen
        // The tuple (With<Player>, Without<Frozen>) acts as an AND gate.
        let mut query = Query::<(&Position, &Velocity), (With<Player>, Without<Frozen>)>::new(
            &mut world.registry,
        );

        let mut count = 0;
        for (_pos, _vel) in query.iter(&mut world) {
            count += 1;
        }

        assert_eq!(count, 1, "Only Archetype 1 should match");
    }

    #[test]
    fn test_disjoint_filter_sets() {
        let mut world = World::new();

        world.spawn((10u32, "A")); // 1. Missing B (f32)
        world.spawn((20u32, 1.0f32)); // 2. Has A, B. No C. MATCH.
        world.spawn((30u32, 2.0f32, true)); // 3. Has A, B. Has C (bool). FAIL.
        world.spawn((40u32, 3.0f32, "D")); // 4. Has A, B. No C. MATCH.

        // Query: Give me u32. Must have f32. Must NOT have bool.
        let mut query = Query::<&u32, (With<f32>, Without<bool>)>::new(&mut world.registry);

        let results: Vec<u32> = query.iter(&mut world).copied().collect();

        // Should match 20 and 40.
        assert_eq!(results.len(), 2);
        assert!(results.contains(&20));
        assert!(results.contains(&40));
    }

    // ============================================================================
    // Custom Data Structures & Mutation
    // ============================================================================

    #[test]
    fn test_mutation_of_complex_structs() {
        let mut world = World::new();

        // Spawn 100 entities with positions
        for i in 0..100 {
            world.spawn((
                Position {
                    x: i as f32,
                    y: 0.0,
                },
                Velocity { dx: 1.0, dy: 1.0 },
            ));
        }

        // System: Update Position based on Velocity
        {
            let mut query = Query::<(&mut Position, &Velocity)>::new(&mut world.registry);
            for (pos, vel) in query.iter(&mut world) {
                pos.x += vel.dx;
                pos.y += vel.dy;
            }
        }

        // Validation
        let mut query = Query::<&Position>::new(&mut world.registry);
        let mut check_count = 0;
        for pos in query.iter(&mut world) {
            // x was 'i', dx was 1.0. So new x should be > 0.0
            assert!(pos.x >= 1.0);
            assert_eq!(pos.y, 1.0);
            check_count += 1;
        }
        assert_eq!(check_count, 100);
    }

    #[test]
    fn test_heap_allocated_components() {
        let mut world = World::new();

        world.spawn((Name("Alice".to_string()), 1u32));
        world.spawn((Name("Bob".to_string()), 2u32));
        world.spawn((3u32,)); // No name

        let mut query = Query::<(&Name, &mut u32)>::new(&mut world.registry);

        let mut found_names = Vec::new();

        for (name, score) in query.iter(&mut world) {
            found_names.push(name.0.clone());
            *score += 10;
        }

        found_names.sort();
        assert_eq!(found_names, vec!["Alice", "Bob"]);

        // Verify mutations persisted
        let mut check_q = Query::<&u32>::new(&mut world.registry);
        let scores: Vec<u32> = check_q.iter(&mut world).cloned().collect();

        // 1 + 10 = 11, 2 + 10 = 12, 3 + 0 = 3
        assert!(scores.contains(&11));
        assert!(scores.contains(&12));
        assert!(scores.contains(&3));
    }

    // ============================================================================
    // Multi-Archetype Iteration Edge Cases
    // ============================================================================

    #[test]
    fn test_scattered_archetypes() {
        let mut world = World::new();

        // Create fragmentation in storage
        world.spawn((1u32, 1.0f32)); // Arch 1
        world.spawn((2u32,)); // Arch 2 (Skipped)
        world.spawn((3u32, 2.0f32, false)); // Arch 3
        world.spawn((4u32, 3.0f32, "dummy")); // Arch 4

        // Query matches Arch 1, 3, and 4 (all have u32 and f32)
        let mut query = Query::<(&u32, &mut f32)>::new(&mut world.registry);

        let mut count = 0;
        let mut sum_ids = 0;

        for (id, _val) in query.iter(&mut world) {
            count += 1;
            sum_ids += *id;
        }

        assert_eq!(count, 3);
        assert_eq!(sum_ids, 1 + 3 + 4);
    }

    // ============================================================================
    // get() Method Tests
    // ============================================================================

    #[test]
    fn test_query_get_method() {
        let mut world = World::new();

        let e1 = world.spawn((10u32, 5.0f32)); // Matches query
        let e2 = world.spawn((20u32,)); // Missing f32, should not match

        let mut query = Query::<(&u32, &mut f32)>::new(&mut world.registry);

        // Test get for matching entity
        if let Some((pos, vel)) = query.get(&mut world, e1) {
            assert_eq!(*pos, 10);
            *vel += 2.0;
            assert_eq!(*vel, 7.0);
        } else {
            panic!("Entity e1 should match the query");
        }

        // Test get for non-matching entity
        assert!(query.get(&mut world, e2).is_none());
    }

    #[test]
    fn test_query_get_nonexistent_entity() {
        let mut world = World::new();

        let e1 = world.spawn((10u32, 5.0f32));

        let mut query = Query::<(&u32, &mut f32)>::new(&mut world.registry);

        // Despawn entity
        world.despawn(e1);

        // Test get for despawned entity
        assert!(query.get(&mut world, e1).is_none());
    }
}
