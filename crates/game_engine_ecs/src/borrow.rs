// Thanks to https://github.com/Ralith/hecs/blob/master/src/borrow.rs

use core::sync::atomic::{AtomicUsize, Ordering};
use std::collections::HashMap;

use crate::{archetype::Archetype, prelude::ComponentId};
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

#[derive(Clone)]
pub struct ColumnBorrowChecker {
    borrows: HashMap<ComponentId, AccessType>,
}

#[derive(Clone)]
enum AccessType {
    Immutable,
    Mutable,
}

impl Default for ColumnBorrowChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl ColumnBorrowChecker {
    pub fn new() -> Self {
        Self {
            borrows: HashMap::new(),
        }
    }

    fn try_borrow(&mut self, comp_id: ComponentId, mutable: bool) -> bool {
        match self.borrows.get(&comp_id) {
            Some(AccessType::Mutable) => false, // Already mutably borrowed
            Some(AccessType::Immutable) if mutable => false, // Cannot mutably borrow if immutably borrowed
            _ => {
                // No existing borrow or compatible borrow
                self.borrows.insert(
                    comp_id,
                    if mutable {
                        AccessType::Mutable
                    } else {
                        AccessType::Immutable
                    },
                );
                true
            }
        }
    }

    pub fn conflicts(&self, other: &Self) -> bool {
        for (id, my_access) in &self.borrows {
            if let Some(other_access) = other.borrows.get(id) {
                // If either one is Mutable, it's a conflict.
                match (my_access, other_access) {
                    (AccessType::Immutable, AccessType::Immutable) => continue,
                    _ => return true,
                }
            }
        }
        false
    }

    /// Extend this borrow checker with another borrow checker.
    /// Panics at runtime if there is a conflict.
    pub fn extend(&mut self, other: &ColumnBorrowChecker) {
        for (comp_id, access) in &other.borrows {
            match (self.borrows.get(comp_id), access) {
                (Some(AccessType::Mutable), _) => {
                    panic!("Runtime borrow conflict detected. Component ID: {comp_id:?}")
                }
                (_, AccessType::Mutable) => {
                    // Upgrade to mutable
                    self.borrows.insert(*comp_id, AccessType::Mutable);
                }
                (Some(AccessType::Immutable), AccessType::Immutable) => {} // Already immutably borrowed
                (None, AccessType::Immutable) => {
                    self.borrows.insert(*comp_id, AccessType::Immutable);
                }
            }
        }
    }

    /// Create a new ColumnBorrowChecker by retaining only components from a specific archetype.
    pub fn for_archetype(&self, arch: &Archetype) -> ColumnBorrowChecker {
        let mut new_checker = ColumnBorrowChecker::new();
        for comp_id in arch.component_ids.iter() {
            if let Some(access) = self.borrows.get(comp_id) {
                new_checker.borrows.insert(*comp_id, access.clone());
            }
        }
        new_checker
    }

    /// Overlay another instance representing a second sequentially isolated borrow.
    /// This add any new borrows and upgrade to mut whenere needed.
    /// (Mut) None -> Mut
    /// (imm) None -> imm
    /// (mut) imm -> Mut
    pub fn overlay(&mut self, other: &ColumnBorrowChecker) {
        for (comp_id, access) in &other.borrows {
            match (self.borrows.get(comp_id), access) {
                (Some(AccessType::Mutable), _) => {} // Already mutably borrowed
                (_, AccessType::Mutable) => {
                    // Upgrade to mutable
                    self.borrows.insert(*comp_id, AccessType::Mutable);
                }
                (Some(AccessType::Immutable), AccessType::Immutable) => {} // Already immutably borrowed
                (None, AccessType::Immutable) => {
                    self.borrows.insert(*comp_id, AccessType::Immutable);
                }
            }
        }
    }

    pub fn apply_borrow(&self, arch: &Archetype) -> bool {
        for comp_id in arch.component_ids.iter() {
            if let Some(access) = self.borrows.get(comp_id) {
                match access {
                    AccessType::Immutable => {
                        if !arch.borrow_column(*comp_id) {
                            return false;
                        }
                    }
                    AccessType::Mutable => {
                        if !arch.borrow_column_mut(*comp_id) {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    pub fn release_borrow(&mut self, arch: &Archetype) {
        for comp_id in arch.component_ids.iter() {
            if let Some(access) = self.borrows.get(comp_id) {
                match access {
                    AccessType::Immutable => {
                        arch.release_column(*comp_id);
                    }
                    AccessType::Mutable => {
                        arch.release_column_mut(*comp_id);
                    }
                }
            }
        }
    }

    pub fn clear(&mut self) {
        self.borrows.clear();
    }

    pub fn get_raw_borrows<'a>(&mut self, arch: &'a Archetype) -> Vec<(&'a AtomicBorrow, bool)> {
        let mut ret = Vec::new();
        for comp_id in arch.component_ids.iter() {
            if let Some(access) = self.borrows.get(comp_id) {
                match access {
                    AccessType::Immutable => {
                        ret.push((arch.column(*comp_id).unwrap().borrow_state(), false));
                    }
                    AccessType::Mutable => {
                        ret.push((arch.column(*comp_id).unwrap().borrow_state(), true));
                    }
                }
            }
        }
        ret
    }

    /// Attempt to borrow a component column.
    /// panics on conflict.
    pub fn borrow(&mut self, comp_id: ComponentId) {
        if !self.try_borrow(comp_id, false) {
            panic!("Runtime borrow conflict detected") // TODO: better error message
        }
    }

    /// attempt to mutably borrow a component column.
    /// panics on conflict.
    pub fn borrow_mut(&mut self, comp_id: ComponentId) {
        if !self.try_borrow(comp_id, true) {
            panic!("Runtime borrow conflict detected") // TODO: better error message
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
