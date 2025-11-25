use std::alloc::{self, Layout};
use std::ptr::{self, NonNull, drop_in_place};

use crate::borrow::AtomicBorrow;

pub struct Column {
    ptr: NonNull<u8>,
    len: usize,
    capacity: usize,
    layout: Layout,
    borrow_state: AtomicBorrow,
    mutated_ticks: Vec<u32>,
}

impl Column {
    pub fn new(layout: Layout) -> Self {
        // Hint: Layout::from_size_align(0, layout.align()) is useful for
        // creating a dangling pointer for an empty vector.

        Column {
            ptr: Layout::from_size_align(0, layout.align())
                .map(|_| NonNull::dangling())
                .unwrap(),
            len: 0,
            capacity: 0,
            layout,
            borrow_state: AtomicBorrow::new(),
            mutated_ticks: Vec::new(),
        }
    }

    pub fn borrow_state(&self) -> &AtomicBorrow {
        &self.borrow_state
    }

    /// Appends an element at the given tick.
    /// # Safety
    /// The type T must match the Layout this column was created with.
    pub unsafe fn push<T>(&mut self, value: T, tick: u32) {
        // 1. Check if (len == capacity) -> grow()
        // 2. Calculate byte offset: len * layout.size()
        // 3. ptr::write the value
        // 4. len += 1
        debug_assert!(std::mem::size_of::<T>() == self.layout.size());

        if self.len == self.capacity {
            self.grow();
        }

        let byte_offset = self.len * self.layout.size();

        // Safety: Caller guarantees T matches layout.
        unsafe {
            ptr::write(self.ptr.as_ptr().add(byte_offset) as *mut T, value);
        }

        if self.mutated_ticks.len() < self.len + 1 {
            self.mutated_ticks.push(tick);
        } else {
            // If we are reusing capacity, overwrite
            self.mutated_ticks[self.len] = tick;
        }

        self.len += 1;
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn get_ticks_ptr(&mut self) -> *mut u32 {
        self.mutated_ticks.as_mut_ptr()
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
    /// Returns true if an element was actually moved (i.e. we didn't remove the very last one).
    /// panics if index is out of bounds.
    pub fn swap_remove(&mut self, index: usize) {
        debug_assert!(self.len == self.mutated_ticks.len());
        assert!(index < self.len);
        // 1. If index is NOT the last element:
        //    Copy bits from (last_index) to (index)
        // 2. len -= 1
        // If we allow Drop types (like String), we MUST call drop_in_place here on the removed item.

        if index != self.len - 1 {
            let last_index = self.len - 1;
            // SAFETY: We have already asserted index < len, so both indices are valid.
            unsafe {
                let src_ptr = self.ptr.as_ptr().add(last_index * self.layout.size());
                let dst_ptr = self.ptr.as_ptr().add(index * self.layout.size());
                drop_in_place(dst_ptr); // drop the pointer we about to overwrite
                ptr::copy_nonoverlapping(src_ptr, dst_ptr, self.layout.size());
            }
        } else {
            // just drop the last element
            unsafe {
                let drop_ptr = self.ptr.as_ptr().add(index * self.layout.size());
                drop_in_place(drop_ptr);
            }
        }

        self.mutated_ticks.swap_remove(index);
        self.len -= 1;
    }

    fn grow(&mut self) {
        // Standard vector resizing logic using std::alloc::realloc
        self.capacity = if self.capacity == 0 {
            4
        } else {
            self.capacity * 2
        };

        // SAFETY: We ensure new_ptr is not null and properly aligned.
        unsafe {
            let new_size = self.capacity * self.layout.size();
            let new_ptr = if self.capacity == 4 {
                alloc::alloc(Layout::from_size_align(new_size, self.layout.align()).unwrap())
            } else {
                alloc::realloc(
                    self.ptr.as_ptr(),
                    Layout::from_size_align(
                        (self.capacity / 2) * self.layout.size(),
                        self.layout.align(),
                    )
                    .unwrap(),
                    new_size,
                )
            };

            if new_ptr.is_null() {
                alloc::handle_alloc_error(
                    Layout::from_size_align(new_size, self.layout.align()).unwrap(),
                );
            }

            self.ptr = NonNull::new_unchecked(new_ptr);
        }
    }
}

impl Drop for Column {
    fn drop(&mut self) {
        if self.capacity == 0 {
            return; // nothing to free
        }

        debug_assert!(self.borrow_state.borrow_mut());

        for i in 0..self.len {
            // SAFETY: We are in Drop, so no other references exist.
            unsafe {
                // destroy each element
                let drop_ptr = self.ptr.as_ptr().add(i * self.layout.size());
                drop_in_place(drop_ptr);
            }
        }

        // SAFETY: We allocated this memory ourselves.
        unsafe {
            alloc::dealloc(
                self.ptr.as_ptr(),
                Layout::from_size_align(self.capacity * self.layout.size(), self.layout.align())
                    .unwrap(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_column_push_and_get() {
        let layout = Layout::new::<u32>();
        let mut col = Column::new(layout);

        unsafe {
            col.push(10u32, 0);
            col.push(20u32, 0);
            col.push(30u32, 0);
        }

        assert_eq!(col.len, 3);

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
        let layout = Layout::new::<u32>();
        let mut col = Column::new(layout);

        unsafe {
            col.push(10u32, 0); // idx 0
            col.push(20u32, 0); // idx 1
            col.push(30u32, 0); // idx 2

            // Remove index 0. Element 30 should move to index 0.
            col.swap_remove(0);
        }

        assert_eq!(col.len, 2);
        let p0 = col.get_ptr(0) as *const u32;
        unsafe {
            assert_eq!(*p0, 30);
        }
    }

    // tick tests
    #[test]
    fn test_mutated_ticks() {
        let layout = Layout::new::<u32>();
        let mut col = Column::new(layout);

        unsafe {
            col.push(10u32, 1);
            col.push(20u32, 2);
            col.push(30u32, 3);
        }

        assert_eq!(col.mutated_ticks[..col.len], vec![1, 2, 3]);

        // Remove index 1
        col.swap_remove(1);

        assert_eq!(col.len, 2);
        assert_eq!(col.mutated_ticks[..col.len], vec![1, 3]);
    }
}
