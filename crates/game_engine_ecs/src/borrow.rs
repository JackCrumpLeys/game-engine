// Thanks to https://github.com/Ralith/hecs/blob/master/src/borrow.rs

use core::sync::atomic::{AtomicUsize, Ordering};

use crate::{archetype::Archetype, component::ComponentMask, prelude::ComponentId};
/// A bit mask used to signal the `AtomicBorrow` has an active mutable borrow.
const UNIQUE_BIT: usize = !(usize::MAX >> 1);

const COUNTER_MASK: usize = usize::MAX >> 1;

/// An atomic integer used to dynamicaly enforce borrowing rules
///
/// The most significant bit is used to track mutable borrow, and the rest is a
/// counter for immutable borrows.
///
/// It has four possible states:
///  - `0b00000000...` the counter isn't mut borrowed, and ready for borrowing
///  - `0b0_______...` the counter isn't mut borrowed, and currently borrowed
///  - `0b10000000...` the counter is mut borrowed
///  - `0b1_______...` the counter is mut borrowed, and some other thread is trying to borrow
pub struct AtomicBorrow(AtomicUsize);

impl Default for AtomicBorrow {
    fn default() -> Self {
        Self::new()
    }
}

impl AtomicBorrow {
    pub const fn new() -> Self {
        Self(AtomicUsize::new(0))
    }

    pub fn borrow(&self) -> bool {
        // Add one to the borrow counter
        let prev_value = self.0.fetch_add(1, Ordering::Acquire);

        // If the previous counter had all of the immutable borrow bits set,
        // the immutable borrow counter overflowed.
        if prev_value & COUNTER_MASK == COUNTER_MASK {
            core::panic!("immutable borrow counter overflowed")
        }

        // If the mutable borrow bit is set, immutable borrow can't occur. Roll back.
        if prev_value & UNIQUE_BIT != 0 {
            self.0.fetch_sub(1, Ordering::Release);
            false
        } else {
            true
        }
    }

    pub fn borrow_mut(&self) -> bool {
        self.0
            .compare_exchange(0, UNIQUE_BIT, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    pub fn release(&self) {
        let value = self.0.fetch_sub(1, Ordering::Release);
        debug_assert!(value != 0, "unbalanced release");
        debug_assert!(value & UNIQUE_BIT == 0, "shared release of unique borrow");
    }

    pub fn release_mut(&self) {
        let value = self.0.fetch_and(!UNIQUE_BIT, Ordering::Release);
        debug_assert_ne!(value & UNIQUE_BIT, 0, "unique release of shared borrow");
    }
}

#[derive(Clone, Debug)]
pub struct ColumnBorrowChecker {
    borrows: ComponentMask,
    mut_borrows: ComponentMask,
}

impl Default for ColumnBorrowChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl ColumnBorrowChecker {
    pub fn new() -> Self {
        Self {
            borrows: ComponentMask::default(),
            mut_borrows: ComponentMask::default(),
        }
    }

    fn try_borrow(&mut self, comp_id: ComponentId, mutable: bool) -> bool {
        match (self.borrows.has(comp_id.0), self.mut_borrows.has(comp_id.0)) {
            (false, false) if mutable => {
                self.borrows.unsafe_set(comp_id.0);
                self.mut_borrows.unsafe_set(comp_id.0);
                true
            }
            (false, false) => {
                self.borrows.unsafe_set(comp_id.0);
                true
            }
            (true, false) => true, // we can have multiple immutable borrows
            _ => false,            // if mutably borrowed, no other borrows allowed
        }
    }

    pub fn conflicts(&self, other: &Self) -> bool {
        if self.mut_borrows.intersects(&other.borrows) {
            return true;
        }
        false
    }

    /// Extend this borrow checker with another borrow checker.
    /// Panics at runtime if there is a conflict.
    pub fn extend(&mut self, other: &ColumnBorrowChecker) {
        if self.conflicts(other) {
            panic!("conflicting borrows detected");
        }

        self.borrows.union(&other.borrows);
        self.mut_borrows.union(&other.mut_borrows);
    }

    /// Create a new ColumnBorrowChecker by retaining only components from a specific archetype.
    pub fn for_archetype(&self, arch: &Archetype) -> ColumnBorrowChecker {
        let mut new_checker = ColumnBorrowChecker::new();
        new_checker.borrows = self.borrows;
        new_checker.mut_borrows = self.mut_borrows;

        new_checker.borrows.intersection(&arch.component_mask);
        new_checker.mut_borrows.intersection(&arch.component_mask);
        new_checker
    }

    /// Overlay another instance representing a second sequentially isolated borrow.
    /// This add any new borrows and upgrade to mut whenere needed.
    /// (Mut) None -> Mut
    /// (imm) None -> imm
    /// (mut) imm -> Mut
    pub fn overlay(&mut self, other: &ColumnBorrowChecker) {
        self.borrows.union(&other.borrows);
        self.mut_borrows.union(&other.mut_borrows);
    }

    pub fn apply_borrow(&self, arch: &Archetype) -> bool {
        for id in &arch.component_ids {
            match (self.borrows.has(id.0), self.mut_borrows.has(id.0)) {
                (true, true) => {
                    // try mut borrow
                    if !arch.borrow_column_mut(id) {
                        return false;
                    }
                }
                (true, false) => {
                    // try imm borrow
                    if !arch.borrow_column(id) {
                        return false;
                    }
                }
                _ => {}
            }
        }
        true
    }

    pub fn release_borrow(&mut self, arch: &Archetype) {
        for id in &arch.component_ids {
            if self.mut_borrows.has(id.0) {
                arch.release_column_mut(*id);
            } else if self.borrows.has(id.0) {
                arch.release_column(*id);
            }
        }
    }

    pub fn clear(&mut self) {
        self.borrows = ComponentMask::default();
        self.mut_borrows = ComponentMask::default();
    }

    /// Extract raw borrows for these columns in the archetype.
    /// Also indicates whether each borrow is mutable.
    pub fn get_raw_borrows<'a>(&mut self, arch: &'a Archetype) -> Vec<(&'a AtomicBorrow, bool)> {
        let mut result = Vec::new();

        for id in &arch.component_ids {
            if self.mut_borrows.has(id.0) {
                if let Some(col) = arch.column(id) {
                    result.push((col.borrow_state(), true));
                }
            } else if self.borrows.has(id.0)
                && let Some(col) = arch.column(id)
            {
                result.push((col.borrow_state(), false));
            }
        }

        result
    }

    /// Attempt to borrow a component column.
    /// panics on conflict.
    pub fn borrow(&mut self, comp_id: ComponentId) {
        if !self.try_borrow(comp_id, false) {
            panic!("conflicting borrow detected for component {comp_id:?}");
        }
    }

    /// attempt to mutably borrow a component column.
    /// panics on conflict.
    pub fn borrow_mut(&mut self, comp_id: ComponentId) {
        if !self.try_borrow(comp_id, true) {
            panic!("conflicting mutable borrow detected for component {comp_id:?}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "immutable borrow counter overflowed")]
    fn test_borrow_counter_overflow() {
        let counter = AtomicBorrow(AtomicUsize::new(COUNTER_MASK));
        counter.borrow();
    }

    #[test]
    #[should_panic(expected = "immutable borrow counter overflowed")]
    fn test_mut_borrow_counter_overflow() {
        let counter = AtomicBorrow(AtomicUsize::new(COUNTER_MASK | UNIQUE_BIT));
        counter.borrow();
    }

    #[test]
    fn test_borrow() {
        let counter = AtomicBorrow::new();
        assert!(counter.borrow());
        assert!(counter.borrow());
        assert!(!counter.borrow_mut());
        counter.release();
        counter.release();

        assert!(counter.borrow_mut());
        assert!(!counter.borrow());
        counter.release_mut();
        assert!(counter.borrow());
    }
}
