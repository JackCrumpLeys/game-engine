use std::any::{Any, TypeId};
use std::collections::HashMap;

// An Entity is just a unique ID. Using a struct provides more type safety than a raw integer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Entity(u64);

// The World manages all entities and components.
pub struct World {
    entities: Vec<Entity>,
    component_stores: HashMap<TypeId, Box<dyn Any>>,
    next_entity_id: u64,
}

impl World {
    pub fn new() -> Self {
        World {
            entities: Vec::new(),
            component_stores: HashMap::new(),
            next_entity_id: 0,
        }
    }

    pub fn spawn_entity(&mut self) -> Entity {
        let entity = Entity(self.next_entity_id);
        self.next_entity_id += 1;
        self.entities.push(entity);
        entity
    }

    pub fn add_component<T: 'static>(&mut self, entity: Entity, component: T) {
        let type_id = TypeId::of::<T>();

        // Get the specific component store for this type, or create it if it doesn't exist.
        let store = self
            .component_stores
            .entry(type_id)
            .or_insert_with(|| Box::new(Vec::<Option<T>>::new()));

        // Downcast the `Box<dyn Any>` to our concrete `Vec<Option<T>>`.
        if let Some(component_vec) = store.downcast_mut::<Vec<Option<T>>>() {
            let entity_id = entity.0 as usize;
            debug_assert!(entity_id <= 32 * 1024 * 1_000_000); // This would use all my ram and we clearly have an issue

            if entity_id >= component_vec.len() {
                component_vec.resize_with(entity_id + 1, Default::default)
            }

            component_vec[entity_id] = Some(component)
        }
    }

    // This is a simplified query that gets all components of a single type.
    pub fn query<T: 'static>(&self) -> Vec<(Entity, &T)> {
        let mut results = Vec::new();
        let type_id = TypeId::of::<T>();

        if let Some(store) = self.component_stores.get(&type_id)
            && let Some(component_vec) = store.downcast_ref::<Vec<Option<T>>>()
        {
            for (i, component_opt) in component_vec.iter().enumerate() {
                if let Some(component) = component_opt {
                    results.push((Entity(i as u64), component))
                }
            }
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_component() {
        let mut world = World::new();
        let entity = world.spawn_entity();

        world.add_component(entity, 42);

        let components = world.query::<i32>();
        assert_eq!(components, vec![(entity, &42)]);
    }

    #[test]
    fn test_query() {
        let mut world = World::new();
        let entity1 = world.spawn_entity();
        let entity2 = world.spawn_entity();

        world.add_component(entity1, 42);
        world.add_component(entity2, 24);

        let components = world.query::<i32>();
        assert_eq!(components, vec![(entity1, &42), (entity2, &24)]);
    }
}
