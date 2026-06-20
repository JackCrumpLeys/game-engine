use crate::component::{Component, ComponentId, ComponentMeta};
use crate::storage::ComponentStorage;

pub trait Bundle: 'static + Send + Sync {
    /// Returns an iterator over component IDs.
    fn component_ids() -> impl Iterator<Item = ComponentId>;

    /// Returns an iterator over component metadata.
    fn component_metas() -> impl Iterator<Item = ComponentMeta>;

    /// Writes data to the provided columns using direct index lookup.
    ///
    /// # Safety
    /// `columns` must be a slice of size `MAX_COMPONENTS`.
    /// `columns[id]` must be `Some` for every component `id` in this bundle.
    unsafe fn put<S: ComponentStorage>(self, columns: &mut [Option<S>]);
}

// Implement for individual Component
impl<T: Component> Bundle for T {
    fn component_ids() -> impl Iterator<Item = ComponentId> {
        std::iter::once(T::get_id())
    }

    fn component_metas() -> impl Iterator<Item = ComponentMeta> {
        std::iter::once(T::meta())
    }

    unsafe fn put<S: ComponentStorage>(self, columns: &mut [Option<S>]) {
        // SAFETY: The caller guarantees that columns for all IDs returned by component_ids() exist.
        // T::get_id() is returned by component_ids().
        unsafe {
            let id = T::get_id();
            let col = columns.get_unchecked_mut(id.0).as_mut().unwrap_unchecked();
            col.submit(self);
        }
    }
}

// Implement for Unit
impl Bundle for () {
    fn component_ids() -> impl Iterator<Item = ComponentId> {
        std::iter::empty()
    }

    fn component_metas() -> impl Iterator<Item = ComponentMeta> {
        std::iter::empty()
    }

    unsafe fn put<S: ComponentStorage>(self, _columns: &mut [Option<S>]) {}
}

// Implement for tuples
macro_rules! impl_bundle_tuple {
    ($($name:ident $num: tt),*) => {
        impl<$($name: Bundle),*> Bundle for ($($name,)*) {
            #[allow(non_snake_case)]
            fn component_ids() -> impl Iterator<Item = ComponentId> {
                std::iter::empty()
                $(
                    .chain($name::component_ids())
                )*
            }

            #[allow(non_snake_case)]
            fn component_metas() -> impl Iterator<Item = ComponentMeta> {
                std::iter::empty()
                $(
                    .chain($name::component_metas())
                )*
            }

            #[allow(non_snake_case)]
            unsafe fn put<Store: ComponentStorage>(self, columns: &mut [Option<Store>]) {
                unsafe {
                    $(
                        self.$num.put(columns);
                    )*
                }
            }
        }
    }
}

// Call the macro (available via #[macro_use] in lib.rs)
auto_impl_all_tuples!(impl_bundle_tuple);
