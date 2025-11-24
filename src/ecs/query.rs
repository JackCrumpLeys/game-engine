use crate::ecs::archetype::{Archetype, ArchetypeId};
use crate::ecs::borrow::AtomicBorrow;
use crate::ecs::component::{Component, ComponentId, ComponentRegistry};
use crate::ecs::storage::Column;
use crate::ecs::world::World;

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
    unsafe fn next(&mut self) -> Self::Item {
        unsafe {
            let ret = self.ptr.as_ref();
            self.ptr = NonNull::new_unchecked(self.ptr.as_ptr().add(1));
            ret
        }
    }
}

// --- Write Implementation ---
pub struct WriteFetch<'a, T> {
    ptr: NonNull<T>,
    borrow: &'a AtomicBorrow,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T> Drop for WriteFetch<'a, T> {
    fn drop(&mut self) {
        self.borrow.release_mut();
    }
}

impl<'a, T: 'static> Fetch<'a> for WriteFetch<'a, T> {
    type Item = &'a mut T;
    unsafe fn next(&mut self) -> Self::Item {
        unsafe {
            let ret = &mut *self.ptr.as_ptr();
            self.ptr = NonNull::new_unchecked(self.ptr.as_ptr().add(1));
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

// Since View<'a> is not implemented for &'a T yet, do it here:
impl<'a, T: Component> View<'a> for &'a T {
    type Item = &'a T;
    type Fetch = ReadFetch<'a, T>;

    fn create_fetch(
        archetype: &'a mut Archetype,
        registry: &ComponentRegistry,
    ) -> Option<Self::Fetch> {
        let id = registry.get_id::<T>()?;
        let column = archetype.column(id)?;

        if !column.borrow_state().borrow() {
            panic!("Immutable borrow failed for {}", std::any::type_name::<T>());
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
    ) -> Option<Self::Fetch> {
        let id = registry.get_id::<T>()?;
        let column = archetype.column(id)?;

        if !column.borrow_state().borrow_mut() {
            panic!("Mutable borrow failed for {}", std::any::type_name::<T>());
        }

        let ptr = column.get_ptr(0) as *mut T;
        Some(WriteFetch {
            ptr: unsafe { NonNull::new_unchecked(ptr) },
            borrow: column.borrow_state(),
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

    pub fn iter<'a>(&'a mut self, world: &'a mut World) -> QueryIter<'a, Q::View<'a>> {
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

        QueryIter {
            world,
            archetype_ids: &self.cached_archetypes,
            current_arch_idx: 0,
            current_fetch: None,
            current_len: 0,
            current_row: 0,
        }
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
            if let Some(fetch) = &mut self.current_fetch {
                if self.current_row < self.current_len {
                    self.current_row += 1;
                    return Some(unsafe { fetch.next() });
                }
            }

            // Case 2: Next Archetype
            if self.current_arch_idx >= self.archetype_ids.len() {
                return None;
            }

            let arch_id = self.archetype_ids[self.current_arch_idx];
            self.current_arch_idx += 1;

            // Borrow Archetype mutably
            // SAFETY: We access distinct archetypes sequentially.
            let arch =
                unsafe { &mut *(&mut self.world.archetypes[arch_id.0 as usize] as *mut Archetype) };

            if arch.len() == 0 {
                continue;
            }

            self.current_len = arch.len();
            self.current_row = 0;
            self.current_fetch = V::create_fetch(arch, &self.world.registry);
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
/// Calls m!(), m!(A), m!(A, B), and m!(A, B, C) for i.e. (m, A, B, C)
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
                unsafe {
                    (
                        $(
                            self.$num.next(),
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
            ) -> Option<Self::Fetch> {
                // SAFETY: We trick the compiler to allow multiple mutable borrows of archetype.
                // We rely on the fact that each $name accesses disjoint columns (runtime verified by AtomicBorrow).
                let ptr = archetype as *mut Archetype;
                Some((
                    $(
                        $name::create_fetch(unsafe { &mut *ptr }, registry)?,
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
    use crate::ecs::world::World;

    #[test]
    fn test_query_iteration() {
        let mut world = World::new();

        world.spawn((10u32, 5.0f32));
        world.spawn((20u32, 10.0f32));
        world.spawn((30u32,));

        // Note the usage of 'static lifetimes in the type definition.
        // This maps to the QueryToken implementation.
        let mut query = Query::<(&u32, &mut f32)>::new(&mut world.registry);

        let mut count = 0;
        for (pos, vel) in query.iter(&mut world) {
            *vel += 1.0;
            count += 1;
            println!("Pos: {}, Vel: {}", pos, vel);
        }

        assert_eq!(count, 2);
    }

    #[test]
    #[should_panic]
    fn test_borrow_check() {
        let mut world = World::new();
        world.spawn((10u32,));

        let mut q1 = Query::<(&'static mut u32, &'static u32)>::new(&mut world.registry);
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
}
