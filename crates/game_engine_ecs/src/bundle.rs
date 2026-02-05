use crate::component::{Component, ComponentId, ComponentMeta};
use crate::storage::{ComponentStorage, TypeErasedSequence};

pub trait Bundle: 'static + Send + Sync {
    /// Returns an iterator over all unique component IDs in this bundle.
    fn component_ids() -> impl Iterator<Item = ComponentId>;

    /// Returns an iterator over all component metadata in this bundle.
    /// The order MUST match `component_ids()`.
    fn component_metas() -> impl Iterator<Item = ComponentMeta>;

    /// Writes the bundle's data into the given storages.
    ///
    /// # Safety
    /// - `columns` must be a slice of `Option<Storage>` of size `MAX_COMPONENTS`.
    /// - For every `ComponentId` returned by `component_ids()`, the corresponding entry
    ///   in `columns` at `columns[id.0]` must be `Some`.
    /// - The storage `Storage` at `columns[id.0]` must have been created with the correct
    ///   `ComponentMeta` for the component with that `id`.
    unsafe fn put<Storage: ComponentStorage>(self, columns: &mut [Option<Storage>]);
}

pub trait ManyRowBundle {
    type Item;

    /// Writes many instances of this bundle into the TypeErasedSequences.
    ///
    /// # Safety
    /// `seqs` must be sorted by Component ID in the order of the returned `component_ids`.
    unsafe fn put_many(self, seqs: &mut [&mut TypeErasedSequence]);

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// --- Base Case: A single component is a bundle ---
impl<T: Component> Bundle for T {
    #[inline(always)]
    fn component_ids() -> impl Iterator<Item = ComponentId> {
        std::iter::once(T::get_id())
    }

    #[inline(always)]
    fn component_metas() -> impl Iterator<Item = ComponentMeta> {
        std::iter::once(T::meta())
    }

    #[inline(always)]
    unsafe fn put<Storage: ComponentStorage>(self, columns: &mut [Option<Storage>]) {
        // The logic guarantees that the column for this component exists.
        // We use unchecked access for performance.
        let column = unsafe {
            columns
                .get_unchecked_mut(T::get_id().0)
                .as_mut()
                .unwrap_unchecked()
        };
        // The caller guarantees the storage type matches the component type.
        unsafe { column.submit(self) };
    }
}

// --- Recursive Case: Tuples of bundles are bundles ---
macro_rules! impl_recursive_bundle {
    // Note: The `$num` tt is not used here but is required by `auto_impl_all_tuples`.
    ($($name:ident $num:tt),*) => {
        impl<$($name: Bundle),*> Bundle for ($($name,)*) {
            #[inline(always)]
            fn component_ids() -> impl Iterator<Item = ComponentId> {
                std::iter::empty()
                $(
                    .chain($name::component_ids())
                )*
            }

            #[inline(always)]
            fn component_metas() -> impl Iterator<Item = ComponentMeta> {
                std::iter::empty()
                $(
                    .chain($name::component_metas())
                )*
            }

            #[inline(always)]
            #[allow(non_snake_case)]
            unsafe fn put<Storage: ComponentStorage>(self, columns: &mut [Option<Storage>]) {
                let ($($name,)*) = self;
                $(
                    unsafe { $name.put(columns) };
                )*
            }
        }
    };
}

// --- Empty Bundle Implementation ---
impl Bundle for () {
    #[inline(always)]
    fn component_ids() -> impl Iterator<Item = ComponentId> {
        std::iter::empty()
    }

    #[inline(always)]
    fn component_metas() -> impl Iterator<Item = ComponentMeta> {
        std::iter::empty()
    }

    #[inline(always)]
    unsafe fn put<Storage: ComponentStorage>(self, _columns: &mut [Option<Storage>]) {
        // Does nothing
    }
}

// Generate implementations for tuples using the auto macro
auto_impl_all_tuples!(impl_recursive_bundle);

// ==================================================================================
// ManyRowBundle - This is harder to make recursive with the current constraints.
// It's highly optimized for Vecs of concrete component tuples.
// We will keep the old macro for this for now.
// ==================================================================================

// Macro to implement Bundle for tuples (A, B, C...)
macro_rules! impl_many_row_bundle {
    ($($name:ident $num: tt),*) => {
        impl<$($name: Component),*> ManyRowBundle for Vec<($($name,)*)> {
            type Item = ($($name,)*);

            #[inline(always)]
            unsafe fn put_many(mut self, seqs: &mut [&mut TypeErasedSequence]) { unsafe {
                // 1. Setup pointers
                let len = self.len();
                let base_ptr = self.as_ptr();

                // 2. Reserve all columns ONCE
                for i in 0..seqs.len() {
                    seqs[i].reserve(len);
                }

                // 3. Hot Loop (Copy Memory)
                for i in 0..len {
                    let item_ptr = base_ptr.add(i);
                    $(
                        // Read from Vec buffer
                        let field_ptr = std::ptr::addr_of!((*item_ptr).$num);
                        let val = std::ptr::read(field_ptr);

                        // Write to Column (Unchecked, we reserved above)
                        seqs[$num].push_unchecked(val);
                    )*
                }

                // We set len to 0.
                // When `self` drops at the end of this function:
                // - It sees len 0, so it calls destructors on 0 items (Correct, we moved them).
                // - It sees capacity > 0, so it deallocates the backing buffer
                self.set_len(0);
            }}

            #[inline(always)]
            fn len(&self) -> usize {
                self.len()
            }
        }
    }
}

impl ManyRowBundle for Vec<()> {
    type Item = ();
    unsafe fn put_many(self, _s: &mut [&mut TypeErasedSequence]) {}
    fn len(&self) -> usize {
        0
    }
}

// Call the macro
auto_impl_all_tuples!(impl_many_row_bundle);
