use game_engine_utils::NoOpHash;

use crate::borrow::AtomicBorrow;
use std::any::{Any, TypeId, type_name};
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

// Marker trait, similar to Component but for globals
pub trait Resource: 'static + Send + Sync {}
impl<T: 'static + Send + Sync> Resource for T {}

pub type TypeIdMap<V> = HashMap<TypeId, V, NoOpHash>;

struct ResourceCell {
    data: UnsafeCell<Box<dyn Any>>,
    borrow: AtomicBorrow,
}

impl ResourceCell {
    fn new(data: Box<dyn Any>) -> Self {
        Self {
            data: UnsafeCell::new(data),
            borrow: AtomicBorrow::new(),
        }
    }
}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct ResourceId(usize);

pub struct Resources {
    map: TypeIdMap<ResourceId>,
    resources: Vec<ResourceCell>,
}

impl Resources {
    pub fn new() -> Self {
        Self {
            map: TypeIdMap::default(),
            resources: Vec::new(),
        }
    }

    /// Inserts a resource of type R.
    /// panics if a resource of this type is already borrowed.
    pub fn insert<R: Resource>(&mut self, resource: R) {
        let idx = self.resources.len();
        self.resources.push(ResourceCell::new(Box::new(resource)));
        self.map.insert(TypeId::of::<R>(), ResourceId(idx));
    }

    pub fn register<R: Resource + Default>(&mut self) {
        self.map.entry(TypeId::of::<R>()).or_insert_with(|| {
            let idx = self.resources.len();
            self.resources
                .push(ResourceCell::new(Box::new(R::default())));
            ResourceId(idx)
        });
    }

    /// Gets an immutable reference to a resource of type R.
    /// panics if the resource is already mutably borrowed.
    pub fn get<R: Resource>(&'_ self) -> Option<Res<'_, R>> {
        // 1. Look up cell
        // 2. Try borrow.borrow() -> Panic if fails (or return None? Bevy panics on contention)
        // 3. Construct Res wrapper

        self.map.get(&TypeId::of::<R>())
            .map(|idx| &self.resources[idx.0])
            .map(|cell| {
                if !cell.borrow.borrow() {
                    panic!("Resource of this type is already mutably borrowed, cannot get immutable reference. {}", type_name::<R>());
                }

                // Safety: We have an immutable borrow, so it's safe to create an immutable reference.
                let value = unsafe { &*cell.data.get() }
                    .downcast_ref::<R>()
                    .expect("Resource type mismatch");

                Res {
                    value,
                    borrow: &cell.borrow,
                }
            }
        )
    }

    /// Gets a mutable reference to a resource of type R.
    /// panics if the resource is already borrowed.
    pub fn get_mut<R: Resource>(&'_ self) -> Option<ResMut<'_, R>> {
        // 1. Look up cell
        // 2. Try borrow.borrow_mut()
        // 3. Construct ResMut wrapper

        self.map.get(&TypeId::of::<R>())
            .map(|idx| &self.resources[idx.0])
            .map(|cell| {
                if !cell.borrow.borrow_mut() {
                    panic!(
                        "Resource of this type is already borrowed, cannot get mutable reference. {}",
                        type_name::<R>()
                    );
                }

                // Safety: We have a mutable borrow, so it's safe to create a mutable reference.
                let value = unsafe { &mut *cell.data.get() }
                    .downcast_mut::<R>()
                    .expect("Resource type mismatch");

                ResMut {
                    value,
                    borrow: &cell.borrow,
                }
            }
        )
    }

    /// Get the id of the given resource if it is registered
    pub fn get_id<R: Resource>(&self) -> Option<ResourceId> {
        self.map.get(&TypeId::of::<R>()).cloned()
    }

    /// Get resource value of R via id immutably
    /// panics if the resource is already borrowed.
    ///
    /// # Safety
    /// - The resource at id but be of type R.
    pub fn get_from_id<R: Resource>(&'_ self, id: ResourceId) -> Option<Res<'_, R>> {
        self.resources.get(id.0)
            .map(|cell| {
                if !cell.borrow.borrow() {
                    panic!("Resource of this type is already mutably borrowed, cannot get immutable reference. {}", type_name::<R>());
                }

                // Safety: We have an immutable borrow, so it's safe to create an immutable reference.
                let value = unsafe { &*cell.data.get() }
                    .downcast_ref::<R>()
                    .expect("Resource type mismatch");

                Res {
                    value,
                    borrow: &cell.borrow,
                }
            }
        )
    }

    /// Get resource value of R via id mutably
    /// panics if the resource is already borrowed.
    ///
    /// # Safety
    /// - The resource at id but be of type R.
    pub fn get_mut_from_id<R: Resource>(&'_ self, id: ResourceId) -> Option<ResMut<'_, R>> {
        self.resources.get(id.0)
            .map(|cell| {
                if !cell.borrow.borrow_mut() {
                    panic!("Resource of this type is already mutably borrowed, cannot get immutable reference. {}", type_name::<R>());
                }

                // Safety: We have an immutable borrow, so it's safe to create an immutable reference.
                let value = unsafe { &mut *cell.data.get() }
                    .downcast_mut::<R>()
                    .expect("Resource type mismatch");

                ResMut {
                    value,
                    borrow: &cell.borrow,
                }
            }
        )
    }

    /// Get or insert default resource of type R.
    pub fn get_or_insert_default<R: Resource + Default>(&mut self) -> Res<'_, R> {
        if !self.map.contains_key(&TypeId::of::<R>()) {
            self.insert(R::default());
        }
        self.get::<R>()
            .expect("Just inserted resource, should be present")
    }

    /// Get or insert default mutable resource of type R.
    pub fn get_mut_or_insert_default<R: Resource + Default>(&mut self) -> ResMut<'_, R> {
        if !self.map.contains_key(&TypeId::of::<R>()) {
            self.insert(R::default());
        }
        self.get_mut::<R>()
            .expect("Just inserted resource, should be present")
    }
}

impl Default for Resources {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Res<'a, T: Resource> {
    value: &'a T,
    borrow: &'a AtomicBorrow,
}

impl<'a, T: Resource> Deref for Res<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value
    }
}

impl<'a, T: Resource> Drop for Res<'a, T> {
    fn drop(&mut self) {
        self.borrow.release();
    }
}

pub struct ResMut<'a, T: Resource> {
    value: &'a mut T,
    borrow: &'a AtomicBorrow,
}

impl<'a, T: Resource> Deref for ResMut<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value
    }
}

impl<'a, T: Resource> DerefMut for ResMut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.value
    }
}

impl<'a, T: Resource> Drop for ResMut<'a, T> {
    fn drop(&mut self) {
        self.borrow.release_mut();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get_resource() {
        let mut resources = Resources::new();
        resources.insert(42u32);

        let res = resources.get::<u32>().unwrap();
        assert_eq!(*res, 42);
    }

    #[test]
    fn test_insert_and_get_mut_resource() {
        let mut resources = Resources::new();
        resources.insert(42u32);

        {
            let mut res_mut = resources.get_mut::<u32>().unwrap();
            *res_mut = 100;
        }

        let res = resources.get::<u32>().unwrap();
        assert_eq!(*res, 100);
    }

    #[test]
    #[should_panic(expected = "Resource of this type is already borrowed")]
    fn test_borrow_conflict() {
        let mut resources = Resources::new();
        resources.insert(42u32);

        let _res1 = resources.get::<u32>().unwrap();
        let _res2 = resources.get_mut::<u32>().unwrap(); // This should panic
    }
}
