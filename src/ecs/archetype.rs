use crate::ecs::component::{ComponentId, ComponentRegistry};
use crate::ecs::entity::Entity;
use crate::ecs::storage::Column;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArchetypeId(pub u32);

pub struct Archetype {
    pub id: ArchetypeId,
    // kept sorted for consistent hashing/lookups
    pub component_ids: Vec<ComponentId>,
    entities: Vec<Entity>,
    columns: HashMap<ComponentId, Column>,
}

impl Archetype {
    pub fn new(
        id: ArchetypeId,
        component_ids: Vec<ComponentId>,
        registry: &ComponentRegistry,
    ) -> Self {
        Archetype {
            id,
            columns: {
                let mut cols = HashMap::new();
                for &comp_id in &component_ids {
                    // TODO: Faster method for batch insert?
                    if let Some(meta) = registry.get_meta(comp_id) {
                        cols.insert(comp_id, Column::new(meta.layout));
                    } else {
                        panic!("ComponentId {:?} not found in registry", comp_id);
                    }
                }
                cols
            },
            component_ids: {
                let mut ids = component_ids;
                ids.sort_unstable();
                ids
            },
            entities: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entities.len()
    }

    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }

    /// Reserves a slot for an entity.
    /// Returns the row index where the data should be written.
    /// Caller is responsible for writing data to columns immediately after!
    pub fn push_entity(&mut self, entity: Entity) -> usize {
        self.entities.push(entity);
        self.entities.len() - 1
    }

    /// Removes the entity at `row`.
    /// Returns the Entity that was swapped into this spot (if any).
    /// panics if row is out of bounds.
    pub fn swap_remove(&mut self, row: usize) -> Option<Entity> {
        if row >= self.entities.len() {
            panic!("Row index out of bounds");
        }

        // 1. Swap remove from columns
        for column in self.columns.values_mut() {
            column.swap_remove(row);
        }

        // 2. Swap remove from entities
        // If we are removing the very last element, nothing moves to fill the gap.
        if row == self.entities.len() - 1 {
            self.entities.pop();
            None
        } else {
            // Remove the dead entity
            self.entities.swap_remove(row);

            // The entity that was at the end is now at 'row'.
            // We must tell the world that this entity has moved.
            Some(self.entities[row])
        }
    }

    // Helper to access a specific column safely
    pub fn column(&mut self, id: ComponentId) -> Option<&mut Column> {
        self.columns.get_mut(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::component::{ComponentId, ComponentRegistry};

    #[test]
    fn test_archetype_ops() {
        // Setup registry
        let mut registry = ComponentRegistry::new();
        let pos_id = registry.register::<u32>();
        let vel_id = registry.register::<f32>(); // different size/type

        // Create Archetype [Pos, Vel]
        let arch_id = ArchetypeId(0);
        let mut arch = Archetype::new(arch_id, vec![pos_id, vel_id], &registry);

        // Fake adding an entity
        let e1 = Entity::new(0, 0);
        let row = arch.push_entity(e1);

        // Manually push data to columns (simulating what World will do)
        unsafe {
            arch.column(pos_id).unwrap().push(100u32, 0);
            arch.column(vel_id).unwrap().push(1.0f32, 0);
        }

        assert_eq!(arch.len(), 1);

        // Remove
        arch.swap_remove(row);
        assert_eq!(arch.len(), 0);
        // Columns should be empty too
        assert_eq!(arch.columns[&pos_id].len(), 0); // Need to make len public in Column or add getter
    }
}
