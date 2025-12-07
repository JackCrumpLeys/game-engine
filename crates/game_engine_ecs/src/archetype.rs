use crate::component::{ComponentId, ComponentMask, ComponentRegistry};
use crate::entity::Entity;
use crate::storage::Column;
use std::ops::Deref;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ArchetypeId(pub usize);

impl ArchetypeId {
    pub fn new(id: usize) -> Self {
        ArchetypeId(id)
    }
}

impl Deref for ArchetypeId {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct Archetype {
    pub id: ArchetypeId,
    // kept sorted for consistent hashing/lookups
    pub component_ids: Vec<ComponentId>,
    pub component_mask: ComponentMask,
    entities: Vec<Entity>,
    pub columns: Vec<Option<Box<Column>>>,
}

impl Archetype {
    pub fn new(
        id: ArchetypeId,
        component_mask: ComponentMask,
        registry: &ComponentRegistry,
    ) -> Self {
        Archetype {
            id,
            columns: {
                // for &comp_id in &component_mask.to_ids() {
                //     // TODO: Faster method for batch insert?
                //     if let Some(meta) = registry.get_meta(comp_id) {
                //         cols.insert(comp_id, Column::new(meta.layout));
                //     } else {
                //         panic!("ComponentId {comp_id:?} not found in registry");
                //     }
                // }
                let mut cols = Vec::with_capacity(ComponentMask::CAPACITY);

                for comp_id in 0..ComponentMask::CAPACITY {
                    let cid = ComponentId(comp_id);
                    if component_mask.has_id(cid) {
                        if let Some(meta) = registry.get_meta(cid) {
                            cols.push(Some(Box::new(Column::from_meta(meta))));
                        } else {
                            panic!("ComponentId {cid:?} not found in registry");
                        }
                    } else {
                        cols.push(None);
                    }
                }

                cols
            },
            component_ids: { component_mask.to_ids() },
            component_mask,
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

    /// Reserves many slots for entities.
    /// returns the starting row index where data should be written.
    /// caller is responsible for writing data to columns immediately after!
    pub fn push_entities(&mut self, entities: &[Entity]) -> usize {
        let start_row = self.entities.len();
        self.entities.extend_from_slice(entities);
        start_row
    }

    /// Removes the entity at `row`.
    /// Returns the Entity that was swapped into this spot (if any).
    /// panics if row is out of bounds.
    pub fn swap_remove(&mut self, row: usize) -> Option<Entity> {
        if row >= self.entities.len() {
            panic!("Row index out of bounds");
        }

        // 1. Swap remove from columns
        for column_id in self.component_mask.to_ids() {
            let column = self
                .column_mut(&column_id)
                .expect("Column should exist for component ID");

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

    pub fn columns(&self) -> Vec<&Column> {
        let mut res = Vec::with_capacity(self.component_ids.len());
        for col in &self.component_ids {
            res.push(self.column(col).unwrap());
        }
        res
    }

    pub fn columns_mut(&mut self) -> impl Iterator<Item = &mut Column> {
        self.columns.iter_mut().filter_map(|c| c.as_deref_mut())
    }

    /// Helper to access a specific column safely and mutably
    pub fn column_mut(&mut self, id: &ComponentId) -> Option<&mut Column> {
        self.columns.get_mut(id.0).and_then(|c| c.as_deref_mut())
    }

    /// Helper to access a specific column safely
    pub fn column(&self, id: &ComponentId) -> Option<&Column> {
        self.columns.get(id.0).and_then(|c| c.as_deref())
    }

    /// Helper to Borrow a specific column
    /// Returns false if the column does not exist or is already borrowed
    pub fn borrow_column(&self, id: &ComponentId) -> bool {
        self.columns
            .get(id.0)
            .is_some_and(|c| c.as_deref().is_some_and(|c| c.borrow_state().borrow()))
    }

    /// Helper to borrow a specific column mutably
    /// returns false if the column does not exist or is already borrowed
    pub fn borrow_column_mut(&self, id: &ComponentId) -> bool {
        self.columns
            .get(id.0)
            .is_some_and(|c| c.as_deref().is_some_and(|c| c.borrow_state().borrow_mut()))
    }

    /// Releases a previously borrowed column
    pub fn release_column(&self, id: ComponentId) {
        if let Some(Some(c)) = self.columns.get(id.0) {
            c.borrow_state().release();
        }
    }

    /// Releases a previously mutably borrowed column
    pub fn release_column_mut(&self, id: ComponentId) {
        if let Some(Some(c)) = self.columns.get(id.0) {
            c.borrow_state().release_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::ComponentRegistry;

    #[test]
    fn test_archetype_ops() {
        // Setup registry
        let mut registry = ComponentRegistry::new();
        let pos_id = registry.register::<u32>();
        let vel_id = registry.register::<f32>(); // different size/type

        // Create Archetype [Pos, Vel]
        let arch_id = ArchetypeId(0);
        let mut arch = Archetype::new(
            arch_id,
            ComponentMask::from_ids(&[pos_id, vel_id]),
            &registry,
        );

        // Fake adding an entity
        let e1 = Entity::new(0, 0);
        let row = arch.push_entity(e1);

        // Manually push data to columns (simulating what World will do)
        unsafe {
            arch.column_mut(&pos_id).unwrap().push(100u32);
            arch.column_mut(&vel_id).unwrap().push(1.0f32);
        }

        assert_eq!(arch.len(), 1);

        // Remove
        arch.swap_remove(row);
        assert_eq!(arch.len(), 0);
        // Columns should be empty too
        assert_eq!(arch.column(&pos_id).unwrap().len(), 0);
    }
}
