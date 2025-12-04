use crate::archetype::Archetype;
use crate::component::{Component, ComponentId, ComponentMask, ComponentRegistry};
use crate::storage::TypeErasedSequence;
use std::any::type_name;

pub trait Bundle {
    /// Returns the list of component IDs in this bundle.
    /// Registers them if not present.
    fn component_ids(&self, registry: &mut ComponentRegistry) -> Vec<ComponentId>;

    /// Writes the bundle's data into the given TypeErasedSequences.
    /// # Safety
    /// `seqs` must be sorted by Component ID.
    /// `ids` must be the list of IDs for this bundle in the order of the returned `component_ids`.
    unsafe fn put(self, seqs: Vec<&mut TypeErasedSequence>);
}

pub trait ManyRowBundle {
    type Item;

    /// Writes many instances of this bundle into the TypeErasedSequences.
    ///
    /// # Safety
    /// `seqs` must be sorted by Component ID in the order of the returned `component_ids`.
    unsafe fn put_many(self, seqs: Vec<&mut TypeErasedSequence>);
}

// Macro to implement Bundle for tuples (A, B, C...)
macro_rules! impl_bundle {
    ($($name:ident $num: tt),*) => {
        impl<$($name: Component),*> Bundle for ($($name,)*) {

            #[inline(always)]
            fn component_ids(&self, registry: &mut ComponentRegistry) -> Vec<ComponentId> {
                // make sure no overlap
                #[cfg(debug_assertions)]
                {
                    let mut mask = ComponentMask::new();
                    $(
                        let id = registry.register::<$name>();
                        debug_assert!(!mask.has(id.0), "Duplicate component in bundle: {}", type_name::<$name>());
                        mask.set(id.0);
                    )*
                }
                vec![$(
                    registry.register::<$name>(),
                )*]
            }

            #[inline(always)]
            unsafe fn put(self, mut seqs: Vec<&mut TypeErasedSequence>) {
                #[allow(non_snake_case)]
                let ($($name,)*) = self;

                unsafe {
                    $(
                        seqs[$num].push($name);
                    )*
                }
            }
        }


        impl<$($name: Component),*> ManyRowBundle for Vec<($($name,)*)> {
            type Item = ($($name,)*);

            #[inline(always)]
            unsafe fn put_many(self, mut seqs: Vec<&mut TypeErasedSequence>) {
                let items = std::mem::ManuallyDrop::new(self);
                let base_ptr = items.as_ptr();
                let len = items.len();

                // Calculate tuple size constant
                const COUNT: usize = 0 $(+ { let _ = $num; 1 })*;

                // 2. Reserve
                // (Optional: Unroll this loop if needed, but the compiler usually handles it)
                for i in 0..COUNT {
                    let column = &mut *seqs[i];
                    column.reserve(len);
                }

                unsafe {
                    // 3. Hot Loop
                    for i in 0..len {
                        let item_ptr = base_ptr.add(i);

                        $(
                            let column = &mut *seqs[$num];

                            // Get field pointer: &(tuple).0
                            let field_ptr = std::ptr::addr_of!((*item_ptr).$num);

                            // Move payload
                            let component = std::ptr::read(field_ptr);

                            column.push(component);
                        )*
                    }
                }
            }
        }
    }
}

// Implement for Unit
impl Bundle for () {
    fn component_ids(&self, _r: &mut ComponentRegistry) -> Vec<ComponentId> {
        vec![]
    }
    unsafe fn put(self, _s: Vec<&mut TypeErasedSequence>) {}
}
impl ManyRowBundle for Vec<()> {
    type Item = ();
    unsafe fn put_many(self, _s: Vec<&mut TypeErasedSequence>) {}
}

// Call the macro
impl_all_tuples!(
    impl_bundle, A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13
);
