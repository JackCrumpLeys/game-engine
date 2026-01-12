use crate::component::{Component, ComponentId, ComponentMask, ComponentMeta};
use crate::storage::TypeErasedSequence;
use std::any::type_name;

pub trait Bundle {
    fn mask() -> ComponentMask;

    /// Returns the list of component IDs in this bundle.
    /// Registers them if not present.
    fn component_ids() -> Vec<ComponentId>;

    /// Returns the metadata for each component in this bundle.
    fn component_metas() -> Vec<ComponentMeta>;

    /// Writes the bundle's data into the given TypeErasedSequences.
    /// # Safety
    /// `columns` and `ids` must be perfectly aligned (index N in ids corresponds to index N in columns).
    /// `ids` must contain ALL ComponentIds provided by this bundle.
    unsafe fn put(self, columns: &mut [&mut TypeErasedSequence], ids: &[ComponentId]);
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

// Macro to implement Bundle for tuples (A, B, C...)
macro_rules! impl_bundle {
    ($($name:ident $num: tt),*) => {
        impl<$($name: Component),*> Bundle for ($($name,)*) {

            #[inline(always)]
            fn mask() -> ComponentMask {
                let mut mask = ComponentMask::new();
                $(
                    mask.set_id(&$name::get_id());
                )*
                mask
            }

            #[inline(always)]
            fn component_ids() -> Vec<ComponentId> {
                // make sure no overlap
                {
                    let mut mask = ComponentMask::new();
                    $(
                        let id = $name::get_id();
                        assert!(!mask.has(id.0), "Duplicate component in bundle: {}", type_name::<$name>());
                        mask.set(id.0);
                    )*
                }
                vec![$(
                    $name::get_id(),
                )*]
            }

            #[inline(always)]
            fn component_metas() -> Vec<ComponentMeta> {
                vec![$(
                    $name::meta(),
                )*]
            }

            #[inline(always)]
            unsafe fn put(
                self,
                columns: &mut [&mut TypeErasedSequence],
                ids: &[ComponentId] // These are the Archetype's sorted IDs
            ) {
                #[allow(non_snake_case)]
                let ($($name,)*) = self;

                unsafe {
                    $(
                        // 1. Find the column index for this component
                        // Since `ids` is sorted (Archetypes are sorted), this is fast.
                        // We use unwrap_unchecked because we know the archetype matches the bundle mask.
                        let id = $name::get_id(); // Or registry lookup if not using static IDs
                        let index = ids.binary_search(&id).unwrap_unchecked();

                        // 2. Write directly to that column
                        columns.get_unchecked_mut(index).push($name);
                    )*
                }
            }
        }


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

// Implement for Unit
impl Bundle for () {
    fn component_ids() -> Vec<ComponentId> {
        vec![]
    }
    fn component_metas() -> Vec<ComponentMeta> {
        vec![]
    }
    unsafe fn put(self, _columns: &mut [&mut TypeErasedSequence], _ids: &[ComponentId]) {}

    fn mask() -> ComponentMask {
        ComponentMask::new()
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
auto_impl_all_tuples!(impl_bundle);
