use std::alloc::{self, Layout};
use std::any::type_name;
use std::ptr::{self, NonNull};

use crate::borrow::AtomicBorrow;
use crate::component::ComponentMeta;
use crate::prelude::Component;

const BASE_LEN: usize = 64;

pub struct TypeErasedSequence {
    ptr: NonNull<u8>,
    len: usize,
    capacity: usize,
    layout: Layout,
    drop_fn: unsafe fn(*mut u8),
    #[cfg(debug_assertions)]
    name: &'static str,
    #[cfg(debug_assertions)]
    dbg_fn: unsafe fn(*const u8) -> String,
}
#[cfg(debug_assertions)]
impl std::fmt::Debug for TypeErasedSequence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypeErasedSequence")
            .field("len", &self.len)
            .field("capacity", &self.capacity)
            .field("layout", &self.layout)
            .field("name", &self.name)
            .field("contents", &{
                let mut vec = Vec::new();
                unsafe {
                    let mut current_ptr = self.ptr.as_ptr();
                    for _ in 0..self.len {
                        vec.push((self.dbg_fn)(current_ptr));
                        current_ptr = current_ptr.add(self.layout.size());
                    }
                }
                vec
            })
            .finish()
    }
}

impl TypeErasedSequence {
    pub fn new(meta: &ComponentMeta) -> Self {
        TypeErasedSequence {
            ptr: NonNull::dangling(),
            len: 0,
            capacity: 0,
            layout: meta.layout,
            drop_fn: meta.drop_fn,
            #[cfg(debug_assertions)]
            name: meta.name,
            #[cfg(debug_assertions)]
            dbg_fn: meta.dbg_fn,
        }
    }

    /// Makes a dummy TypeErasedSequence with zero length and capacity.
    /// useful for
    pub fn dummy() -> Self {
        TypeErasedSequence {
            ptr: NonNull::dangling(),
            len: 0,
            capacity: 0,
            layout: Layout::from_size_align(0, 1).unwrap(),
            drop_fn: |_ptr: *mut u8| {},
            #[cfg(debug_assertions)]
            name: "Dummy",
            #[cfg(debug_assertions)]
            dbg_fn: |_ptr: *const u8| "Dummy".to_string(),
        }
    }

    /// Appends an element
    /// # Safety
    /// The type T must match the Layout this TypeErasedSequence was created with.
    pub unsafe fn push<T>(&mut self, value: T) {
        #[cfg(debug_assertions)]
        {
            if self.name.is_empty() {
                self.name = type_name::<T>();
            }
        }
        // 1. Check if (len == capacity) -> grow()
        // 2. Calculate byte offset: len * layout.size()
        // 3. ptr::write the value
        // 4. len += 1
        #[cfg(debug_assertions)]
        debug_assert_eq!(
            std::any::type_name::<T>(),
            self.name,
            "Type mismatch in push"
        );
        debug_assert!(std::mem::size_of::<T>() == self.layout.size());
        debug_assert!(std::mem::align_of::<T>() == self.layout.align());

        if self.len == self.capacity {
            self.grow();
        }

        if self.layout.size() == 0 {
            // For ZSTs, we must still "forget" the value so its Drop doesn't run
            // (though ZSTs usually don't have Drop, they theoretically can)
            std::mem::forget(value);
        } else {
            let byte_offset = self.len * self.layout.size();

            // Safety: Caller guarantees T matches layout.
            unsafe {
                ptr::write(self.ptr.as_ptr().add(byte_offset) as *mut T, value);
            }
        }

        self.len += 1;
    }

    // SAFETY: Caller ensures capacity > len and T matches layout
    #[inline(always)]
    pub unsafe fn push_unchecked<T>(&mut self, value: T) {
        #[cfg(debug_assertions)]
        debug_assert_eq!(
            std::any::type_name::<T>(),
            self.name,
            "Type mismatch in push"
        );
        debug_assert!(std::mem::size_of::<T>() == self.layout.size());
        debug_assert!(std::mem::align_of::<T>() == self.layout.align());

        let byte_offset = self.len * self.layout.size();
        // Use fast pointer arithmetic
        let dest = unsafe { self.ptr.as_ptr().add(byte_offset) as *mut T };
        unsafe { std::ptr::write(dest, value) };
        self.len += 1;
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns a pointer to the element at index.
    /// panics if index is out of bounds.
    pub fn get_ptr(&self, index: usize) -> *mut u8 {
        if index >= self.len {
            panic!("Index out of bounds");
        }

        // SAFETY: We have already checked that index < len.
        unsafe { self.ptr.as_ptr().add(index * self.layout.size()) }
    }

    /// Removes the element at index by swapping it with the last element.
    /// panics if index is out of bounds.
    pub fn swap_remove(&mut self, index: usize) {
        assert!(index < self.len);

        // 1. Calculate pointers
        let size = self.layout.size();
        unsafe {
            let base_ptr = self.ptr.as_ptr();
            let ptr_at_index = base_ptr.add(index * size);
            let ptr_last = base_ptr.add((self.len - 1) * size);

            // 2. Drop the item currently at `index` using our stored function
            (self.drop_fn)(ptr_at_index);

            // 3. If it wasn't the last item, move the last item into this slot
            if index != self.len - 1 {
                ptr::copy_nonoverlapping(ptr_last, ptr_at_index, size);
            }

            // Note: We do NOT drop ptr_last. We moved its bits to ptr_at_index.
            // Ownership has effectively transferred to index.
        }

        self.len -= 1;
    }

    /// Copy data before removing it into other
    /// Removes the element at index by swapping it with the last element.
    /// panics if index is out of bounds.
    pub fn swap_remove_into(&mut self, index: usize, other: &mut TypeErasedSequence) {
        assert!(index < self.len);
        assert_eq!(self.layout.size(), other.layout.size());

        // 1. Calculate pointers
        let size = self.layout.size();
        unsafe {
            let base_ptr = self.ptr.as_ptr();
            let ptr_at_index = base_ptr.add(index * size);
            let ptr_last = base_ptr.add((self.len - 1) * size);

            // 2. Copy the item at `index` into `other`
            other.reserve(1);
            let dest_ptr = other.ptr.as_ptr().add(other.len * size);
            ptr::copy_nonoverlapping(ptr_at_index, dest_ptr, size);
            other.len += 1;

            // 4. If it wasn't the last item, move the last item into this slot
            if index != self.len - 1 {
                ptr::copy_nonoverlapping(ptr_last, ptr_at_index, size);
            }
        }

        self.len -= 1;
    }

    /// Consumes `other`, moving its elements into `self`.
    /// `other` is automatically dropped at the end of this function, freeing its buffer.
    pub fn append(&mut self, mut other: TypeErasedSequence) {
        // 1. Safety Checks
        assert_eq!(self.layout.size(), other.layout.size());

        // If other is empty, just let it drop immediately.
        if other.len == 0 {
            return;
        }

        // 2. Reserve space in self
        self.reserve(other.len);

        // 3. Bitwise Copy (The Move)
        // We copy the bits from the source buffer to our buffer.
        unsafe {
            ptr::copy_nonoverlapping(
                other.ptr.as_ptr(),
                self.ptr.as_ptr().add(self.len * self.layout.size()),
                other.len * self.layout.size(),
            );
        }

        // 4. Update Self Length
        self.len += other.len;

        // 5. Neutering `other` (CRITICAL)
        // We moved the items. We must set other.len to 0 so that when it drops,
        // it does NOT try to run destructors on the items we just stole.
        other.len = 0;
    }

    // Fixed reserve logic from previous answer
    pub fn reserve(&mut self, additional: usize) {
        if self.layout.size() == 0 {
            self.capacity = usize::MAX;
            return;
        }

        if self.capacity - self.len >= additional {
            return;
        }

        let old_capacity = self.capacity;
        let new_capacity = (self.capacity + additional).max(4).next_power_of_two();

        let new_size = new_capacity * self.layout.size();
        let new_layout = Layout::from_size_align(new_size, self.layout.align()).unwrap();

        unsafe {
            let new_ptr = if old_capacity == 0 {
                alloc::alloc(new_layout)
            } else {
                let old_layout =
                    Layout::from_size_align(old_capacity * self.layout.size(), self.layout.align())
                        .unwrap();
                alloc::realloc(self.ptr.as_ptr(), old_layout, new_size)
            };

            if new_ptr.is_null() {
                alloc::handle_alloc_error(new_layout);
            }
            self.ptr = NonNull::new_unchecked(new_ptr);
            self.capacity = new_capacity;
        }
    }

    fn grow(&mut self) {
        // If we have a zero-sized type, we don't need to allocate memory.
        if self.layout.size() == 0 {
            self.capacity = usize::MAX;
            return;
        }

        // Standard vector resizing logic using std::alloc::realloc
        let new_len = if self.capacity == 0 {
            BASE_LEN
        } else {
            self.capacity * 2
        };

        self.reserve(new_len - self.capacity);
    }
}

#[cfg_attr(debug_assertions, derive(Debug))]
pub struct Column {
    inner: TypeErasedSequence,
    borrow_state: AtomicBorrow,
    mutated_ticks: Vec<u32>,
    tick: u32,
}

impl Column {
    pub fn new<T: Component>() -> Self {
        // Hint: Layout::from_size_align(0, layout.align()) is useful for
        // creating a dangling pointer for an empty vector.

        Column {
            inner: TypeErasedSequence::new(&T::meta()),
            borrow_state: AtomicBorrow::new(),
            mutated_ticks: Vec::new(),
            tick: 0,
        }
    }

    pub fn from_meta(meta: &ComponentMeta) -> Self {
        Column {
            inner: TypeErasedSequence::new(meta),
            borrow_state: AtomicBorrow::new(),
            mutated_ticks: Vec::new(),
            tick: 0,
        }
    }

    pub fn borrow_state(&self) -> &AtomicBorrow {
        &self.borrow_state
    }

    /// Appends an element at the given tick.
    /// # Safety
    /// The type T must match the Layout this column was created with.
    /// Appends an element at the given tick.
    pub unsafe fn push<T>(&mut self, value: T) {
        unsafe { self.inner.push(value) };
        self.mutated_ticks.push(self.tick);
    }

    pub fn inner(&self) -> &TypeErasedSequence {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut TypeErasedSequence {
        &mut self.inner
    }

    pub fn push_ticks(&mut self, n: usize) {
        self.mutated_ticks.extend(vec![self.tick; n])
    }

    pub fn layout(&self) -> Layout {
        self.inner.layout
    }

    /// Appends another sequence to this column and updates the change detection ticks.
    /// Consumes the other sequence.
    pub fn append(&mut self, other: TypeErasedSequence) {
        let count = other.len();

        // 1. Move the raw component data
        // TypeErasedSequence::append handles the raw memory copy and updates len
        self.inner.append(other);

        // 2. Synchronize ticks
        // Every entity added via flush counts as "changed" on this tick.
        self.mutated_ticks.reserve(count);
        self.push_ticks(count);
    }

    pub fn set_tick(&mut self, tick: u32) {
        self.tick = tick;
    }

    pub fn len(&self) -> usize {
        self.inner.len
    }

    /// Get a pointer to the mutated ticks array.
    pub fn get_ticks_ptr(&mut self) -> *mut u32 {
        self.mutated_ticks.as_mut_ptr()
    }

    /// Returns a pointer to the element at index.
    /// panics if index is out of bounds.
    pub fn get_ptr(&self, index: usize) -> *mut u8 {
        self.inner.get_ptr(index)
    }

    /// Removes the element at index by swapping it with the last element.
    /// Returns true if an element was actually moved (i.e. we didn't remove the very last one).
    /// panics if index is out of bounds.
    pub fn swap_remove(&mut self, index: usize) {
        debug_assert!(self.len() == self.mutated_ticks.len());

        self.inner.swap_remove(index);
        self.mutated_ticks.swap_remove(index);
    }

    /// Copy data before removing it into other
    /// Removes the element at index by swapping it with the last element.
    /// panics if index is out of bounds.
    pub fn swap_remove_into(&mut self, index: usize, other: &mut Column) {
        debug_assert!(self.len() == self.mutated_ticks.len());
        debug_assert!(self.inner.layout.size() == other.inner.layout.size());

        self.inner.swap_remove_into(index, &mut other.inner);

        other
            .mutated_ticks
            .push(self.mutated_ticks.swap_remove(index));
    }

    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional);
        self.mutated_ticks.reserve(additional);
    }
}

// Crucial: The Drop implementation for the container itself
impl Drop for TypeErasedSequence {
    fn drop(&mut self) {
        // 1. Drop all the elements inside
        if self.len > 0 && self.layout.size() > 0 {
            unsafe {
                let mut current_ptr = self.ptr.as_ptr();
                for _ in 0..self.len {
                    // Call the captured drop function for every element
                    (self.drop_fn)(current_ptr);
                    current_ptr = current_ptr.add(self.layout.size());
                }
            }
        }

        // 2. Free the backing memory
        if self.capacity > 0 && self.layout.size() > 0 {
            unsafe {
                let total_bytes = self.capacity * self.layout.size();
                let layout = Layout::from_size_align(total_bytes, self.layout.align()).unwrap();
                alloc::dealloc(self.ptr.as_ptr(), layout);
            }
        }
    }
}

impl Drop for Column {
    fn drop(&mut self) {
        debug_assert!(self.borrow_state.borrow_mut());
    }
}

#[cfg(test)]
mod storage_tests {
    use crate::prelude::ComponentId;

    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_column_push_and_get() {
        let mut col = Column::new::<u32>();

        unsafe {
            col.push(10u32);
            col.push(20u32);
            col.push(30u32);
        }

        assert_eq!(col.len(), 3);

        // Read back
        let p0 = col.get_ptr(0) as *const u32;
        let p1 = col.get_ptr(1) as *const u32;

        unsafe {
            assert_eq!(*p0, 10);
            assert_eq!(*p1, 20);
        }
    }

    #[test]
    fn test_swap_remove() {
        let mut col = Column::new::<u32>();

        unsafe {
            col.push(10u32); // idx 0
            col.push(20u32); // idx 1
            col.push(30u32); // idx 2

            // Remove index 0. Element 30 should move to index 0.
            col.swap_remove(0);
        }

        assert_eq!(col.len(), 2);
        let p0 = col.get_ptr(0) as *const u32;
        unsafe {
            assert_eq!(*p0, 30);
        }
    }

    // tick tests
    #[test]
    fn test_mutated_ticks() {
        let mut col = Column::new::<u32>();

        unsafe {
            col.set_tick(1);
            col.push(10u32);
            col.set_tick(2);
            col.push(20u32);
            col.set_tick(3);
            col.push(30u32);
        }

        assert_eq!(col.mutated_ticks[..col.len()], vec![1, 2, 3]);

        // Remove index 1
        col.swap_remove(1);

        assert_eq!(col.len(), 2);
        assert_eq!(col.mutated_ticks[..col.len()], vec![1, 3]);
    }

    // ============================================================================
    // TEST HELPERS
    // ============================================================================

    /// A helper that tracks when it is dropped.
    /// Useful for ensuring TypeErasedSequence cleans up correctly.
    #[derive(Debug)]
    struct DropTracker {
        id: usize,
        counter: Arc<AtomicUsize>,
    }

    impl Drop for DropTracker {
        fn drop(&mut self) {
            self.counter.fetch_add(1, Ordering::Relaxed);
        }
    }

    impl_component!(DropTracker);

    // ============================================================================
    // BASIC FUNCTIONALITY
    // ============================================================================

    #[test]
    fn test_push_and_get_primitive() {
        let mut col = Column::new::<u32>();

        unsafe {
            col.push(10u32);
            col.push(20u32);
            col.push(30u32);
        }

        assert_eq!(col.len(), 3);
        assert!(col.inner().capacity >= 3);

        unsafe {
            let p0 = col.get_ptr(0) as *const u32;
            let p1 = col.get_ptr(1) as *const u32;
            let p2 = col.get_ptr(2) as *const u32;

            assert_eq!(*p0, 10);
            assert_eq!(*p1, 20);
            assert_eq!(*p2, 30);
        }
    }

    #[test]
    fn test_swap_remove_logic() {
        let mut col = Column::new::<u32>();

        unsafe {
            col.push(0u32); // idx 0
            col.push(1u32); // idx 1
            col.push(2u32); // idx 2
            col.push(3u32); // idx 3
        }

        // Remove from middle (idx 1).
        // Expected: Last element (3) moves to idx 1.
        col.swap_remove(1);

        assert_eq!(col.len(), 3);

        unsafe {
            assert_eq!(*(col.get_ptr(0) as *const u32), 0);
            assert_eq!(*(col.get_ptr(1) as *const u32), 3); // Moved
            assert_eq!(*(col.get_ptr(2) as *const u32), 2);
        }

        // Remove last element (now idx 2, val 2).
        // Expected: Just a pop.
        col.swap_remove(2);
        assert_eq!(col.len(), 2);
        unsafe {
            assert_eq!(*(col.get_ptr(0) as *const u32), 0);
            assert_eq!(*(col.get_ptr(1) as *const u32), 3);
        }
    }

    // ============================================================================
    // MEMORY SAFETY & LIFECYCLES
    // ============================================================================

    #[test]
    fn test_drop_safety_on_column_drop() {
        let drop_count = Arc::new(AtomicUsize::new(0));

        {
            let mut col = Column::new::<DropTracker>();
            for i in 0..10 {
                unsafe {
                    col.push(DropTracker {
                        id: i,
                        counter: drop_count.clone(),
                    });
                }
            }
            assert_eq!(col.len(), 10);
            assert_eq!(drop_count.load(Ordering::Relaxed), 0);
            // col drops here
        }

        // Ensure all 10 items were dropped exactly once
        assert_eq!(drop_count.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn test_drop_safety_on_swap_remove() {
        let drop_count = Arc::new(AtomicUsize::new(0));
        let mut col = Column::new::<DropTracker>();

        unsafe {
            col.push(DropTracker {
                id: 0,
                counter: drop_count.clone(),
            });
            col.push(DropTracker {
                id: 1,
                counter: drop_count.clone(),
            });
            col.push(DropTracker {
                id: 2,
                counter: drop_count.clone(),
            });
        }

        // Remove index 0. It should drop immediately.
        col.swap_remove(0);

        assert_eq!(drop_count.load(Ordering::Relaxed), 1);
        assert_eq!(col.len(), 2);

        // Verify the remaining items are the correct ones (id 2 should be at 0 now)
        unsafe {
            let p0 = col.get_ptr(0) as *const DropTracker;
            assert_eq!((*p0).id, 2);
        }
    }

    #[test]
    fn test_growth_integrity() {
        // Push enough items to force multiple reallocations
        let mut col = Column::new::<usize>();
        let count = 10_000;

        for i in 0..count {
            unsafe {
                col.push(i);
            }
        }

        assert_eq!(col.len(), count);

        // Verify data integrity after potential moves
        for i in 0..count {
            unsafe {
                let val = *(col.get_ptr(i) as *const usize);
                assert_eq!(val, i);
            }
        }
    }

    // ============================================================================
    // ALIGNMENT & ZST
    // ============================================================================

    #[test]
    fn test_high_alignment() {
        // A struct with 128-byte alignment (e.g., similar to SIMD)
        #[repr(align(128))]
        #[derive(Debug, PartialEq)]
        struct BigAlign(u8);

        impl crate::prelude::Component for BigAlign {
            fn get_id() -> ComponentId {
                ComponentId(0) // Mock ID
            }
        }

        let mut col = Column::new::<BigAlign>();

        unsafe {
            col.push(BigAlign(1));
            col.push(BigAlign(2));
            col.push(BigAlign(3));
        }

        // Force a resize
        for i in 4..100 {
            unsafe {
                col.push(BigAlign(i as u8));
            }
        }

        // Check alignment of pointers
        for i in 0..col.len() {
            let ptr = col.get_ptr(i) as usize;
            assert_eq!(ptr % 128, 0, "Index {i} is misaligned");
        }
    }

    #[test]
    fn test_zero_sized_types() {
        #[derive(Debug)]
        struct Marker;

        impl crate::prelude::Component for Marker {
            fn get_id() -> crate::prelude::ComponentId {
                ComponentId(1)
            }
        }

        let mut col = Column::new::<Marker>();

        unsafe {
            for _ in 0..100 {
                col.push(Marker);
            }
        }

        assert_eq!(col.len(), 100);
        // Capacity logic for ZST varies, but usually is usize::MAX
        // We just ensure accessing it doesn't crash

        let _ = col.get_ptr(50);
        let _ = col.get_ptr(99);

        // Ensure removal works
        col.swap_remove(0);
        assert_eq!(col.len(), 99);
    }

    // ============================================================================
    // TypeErasedSequence TESTS (Raw Memory Logic)
    // ============================================================================

    #[test]
    fn test_sequence_push_and_get() {
        // We use the Meta from u32 to create the sequence
        let meta = u32::meta();
        let mut seq = TypeErasedSequence::new(&meta);

        unsafe {
            seq.push(10u32);
            seq.push(20u32);
            seq.push(30u32);
        }

        assert_eq!(seq.len(), 3);
        assert!(seq.capacity >= 3);

        unsafe {
            let p0 = seq.get_ptr(0) as *const u32;
            let p1 = seq.get_ptr(1) as *const u32;
            let p2 = seq.get_ptr(2) as *const u32;

            assert_eq!(*p0, 10);
            assert_eq!(*p1, 20);
            assert_eq!(*p2, 30);
        }
    }

    #[test]
    fn test_sequence_swap_remove() {
        let meta = u32::meta();
        let mut seq = TypeErasedSequence::new(&meta);

        unsafe {
            seq.push(0u32); // idx 0
            seq.push(1u32); // idx 1
            seq.push(2u32); // idx 2
            seq.push(3u32); // idx 3
        }

        // Remove from middle (idx 1). Last element (3) moves to idx 1.
        seq.swap_remove(1);

        assert_eq!(seq.len(), 3);
        unsafe {
            assert_eq!(*(seq.get_ptr(0) as *const u32), 0);
            assert_eq!(*(seq.get_ptr(1) as *const u32), 3); // Moved here
            assert_eq!(*(seq.get_ptr(2) as *const u32), 2);
        }

        // Remove last element (now idx 2, val 2).
        seq.swap_remove(2);
        assert_eq!(seq.len(), 2);
    }

    #[test]
    fn test_sequence_drop_safety() {
        let drop_count = Arc::new(AtomicUsize::new(0));
        let meta = DropTracker::meta();

        {
            let mut seq = TypeErasedSequence::new(&meta);
            for i in 0..10 {
                unsafe {
                    seq.push(DropTracker {
                        id: i,
                        counter: drop_count.clone(),
                    });
                }
            }
            assert_eq!(seq.len(), 10);
            assert_eq!(drop_count.load(Ordering::Relaxed), 0);
            // seq drops here
        }

        // Ensure all 10 items were dropped exactly once via the internal drop_fn
        assert_eq!(drop_count.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn test_sequence_growth_integrity() {
        let meta = usize::meta();
        let mut seq = TypeErasedSequence::new(&meta);
        let count = 10_000;

        for i in 0..count {
            unsafe {
                seq.push(i);
            }
        }

        assert_eq!(seq.len(), count);

        // Verify data integrity after multiple reallocations
        for i in 0..count {
            unsafe {
                let val = *(seq.get_ptr(i) as *const usize);
                assert_eq!(val, i);
            }
        }
    }

    #[test]
    fn test_sequence_append_ownership() {
        let drop_count = Arc::new(AtomicUsize::new(0));
        let meta = DropTracker::meta();

        let mut dest = TypeErasedSequence::new(&meta);
        unsafe {
            dest.push(DropTracker {
                id: 999,
                counter: drop_count.clone(),
            });
        }

        let mut src = TypeErasedSequence::new(&meta);
        unsafe {
            src.push(DropTracker {
                id: 1,
                counter: drop_count.clone(),
            });
            src.push(DropTracker {
                id: 2,
                counter: drop_count.clone(),
            });
        }

        // The critical test: append consumes src.
        // Src elements should NOT drop. They should move to Dest.
        dest.append(src);

        assert_eq!(drop_count.load(Ordering::Relaxed), 0);
        assert_eq!(dest.len(), 3);

        unsafe {
            let p1 = (*(dest.get_ptr(1) as *const DropTracker)).id;
            assert_eq!(p1, 1);
        }

        // Now drop dest, all 3 should drop
        drop(dest);
        assert_eq!(drop_count.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_sequence_zst() {
        #[derive(Debug)]
        struct Marker;
        impl_component!(Marker);

        let meta = Marker::meta();
        let mut seq = TypeErasedSequence::new(&meta);

        unsafe {
            for _ in 0..100 {
                seq.push(Marker);
            }
        }

        assert_eq!(seq.len(), 100);
        // Ensure swapping and getting pointers doesn't panic or segfault
        seq.swap_remove(0);
        let _ = seq.get_ptr(50);
        assert_eq!(seq.len(), 99);
    }

    #[test]
    fn test_sequence_high_alignment() {
        #[repr(align(128))]
        #[derive(Debug)]
        #[allow(dead_code)]
        struct BigAlign(u8);
        impl_component!(BigAlign);

        let meta = BigAlign::meta();
        let mut seq = TypeErasedSequence::new(&meta);

        unsafe {
            seq.push(BigAlign(1));
            // Force resize
            for i in 0..100 {
                seq.push(BigAlign(i as u8));
            }
        }

        // Verify alignment
        for i in 0..seq.len() {
            let ptr = seq.get_ptr(i) as usize;
            assert_eq!(ptr % 128, 0, "Index {i} is misaligned");
        }
    }
}
