use crate::archetype::Archetype;
use crate::component::{Component, ComponentId, ComponentRegistry};

pub trait Bundle {
    /// Returns the list of component IDs in this bundle.
    /// Registers them if not present.
    fn component_ids(&self, registry: &mut ComponentRegistry) -> Vec<ComponentId>;

    /// Writes the bundle's data into the archetype at the given row.
    /// # Safety
    /// The archetype must contain columns for all components in this bundle.
    unsafe fn put(self, archetype: &mut Archetype, registry: &ComponentRegistry, tick: u32);
}

// Macro to implement Bundle for tuples (A, B, C...)
macro_rules! impl_bundle {
    ($($name:ident),*) => {
        impl<$($name: Component),*> Bundle for ($($name,)*) {
            fn component_ids(&self, registry: &mut ComponentRegistry) -> Vec<ComponentId> {
                vec![$(registry.register::<$name>()),*]
            }

            #[allow(unused_variables)]
            unsafe fn put(self, archetype: &mut Archetype, registry: &ComponentRegistry, tick: u32) {
                #[allow(non_snake_case)]
                let ($($name,)*) = self;
                unsafe {
                    $(
                        let id = registry.get_id::<$name>().expect("Component ID should exist");
                        let column = archetype.column(id).expect("Column should exist");
                        column.push($name,tick);
                    )*
                }
            }
        }
    }
}

impl Bundle for () {
    fn component_ids(&self, _registry: &mut ComponentRegistry) -> Vec<ComponentId> {
        vec![]
    }

    unsafe fn put(self, _archetype: &mut Archetype, _registry: &ComponentRegistry, _tick: u32) {
        // noop
    }
}

impl_bundle!(A);
impl_bundle!(A, B);
impl_bundle!(A, B, C);
impl_bundle!(A, B, C, D);
impl_bundle!(A, B, C, D, E);
impl_bundle!(A, B, C, D, E, F);
impl_bundle!(A, B, C, D, E, F, G);
impl_bundle!(A, B, C, D, E, F, G, H);
impl_bundle!(A, B, C, D, E, F, G, H, I);
impl_bundle!(A, B, C, D, E, F, G, H, I, J);
impl_bundle!(A, B, C, D, E, F, G, H, I, J, K);
impl_bundle!(A, B, C, D, E, F, G, H, I, J, K, L);
impl_bundle!(A, B, C, D, E, F, G, H, I, J, K, L, M);
impl_bundle!(A, B, C, D, E, F, G, H, I, J, K, L, M, N);
impl_bundle!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O); // enough surely!?
