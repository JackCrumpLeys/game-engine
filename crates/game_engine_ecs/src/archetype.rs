use smallvec::SmallVec;

use crate::component::{ComponentId, ComponentMask, ComponentRegistry, MAX_COMPONENTS};
use crate::entity::Entity;
use crate::storage::Column;
use std::array;
use std::ops::{Deref, Index, IndexMut};

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

#[cfg_attr(debug_assertions, derive(Debug))]
pub struct Archetype {
    pub id: ArchetypeId,
    // kept sorted for consistent hashing/lookups
    pub component_ids: SmallVec<ComponentId, MAX_COMPONENTS>,
    pub component_mask: ComponentMask,
    entities: Vec<Entity>,
    pub columns: [Option<Column>; MAX_COMPONENTS],
    /// Index: ComponentId, Value: ArchetypeId of archetype with that component added
    with: [Option<ArchetypeId>; MAX_COMPONENTS],
    /// Index: ComponentId, Value: ArchetypeId of archetype with that component removed
    without: [Option<ArchetypeId>; MAX_COMPONENTS],
}

#[cfg_attr(debug_assertions, derive(Debug))]
pub struct ArchetypeStore {
    inner: Vec<Archetype>,
    // The Lookup: Sorted by Mask
    lookup: Vec<(ComponentMask, ArchetypeId)>,
}

/// By not using a Vec directly, we can later change the storage strategy
/// We also define only 2 mutating methods: push and index_mut
impl Default for ArchetypeStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ArchetypeStore {
    pub fn new() -> Self {
        ArchetypeStore {
            inner: Vec::new(),
            lookup: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn push(&mut self, archetype: Archetype) {
        let mask = archetype.component_mask;
        let id = archetype.id;
        self.inner.push(archetype);

        // Keep lookup sorted
        let idx = self
            .lookup
            .binary_search_by_key(&mask, |(m, _)| *m)
            .unwrap_err();
        self.lookup.insert(idx, (mask, id));
    }

    #[inline(always)]
    pub fn with(
        &mut self,
        from: ArchetypeId,
        with: &ComponentId,
        registry: &ComponentRegistry,
    ) -> ArchetypeId {
        match self.inner[from.0].with(with) {
            Some(id) => id,
            None => {
                let mut new_mask = self.inner[from.0].component_mask;
                new_mask.set_id(with);
                let id = self.get_or_create_archetype_search(new_mask, registry);
                self.inner[from.0].set_with(with, id);
                self.inner[id.0].set_without(with, from);
                id
            }
        }
    }

    #[inline(always)]
    pub fn without(
        &mut self,
        from: ArchetypeId,
        without: &ComponentId,
        registry: &ComponentRegistry,
    ) -> ArchetypeId {
        match self.inner[from.0].without(without) {
            Some(id) => id,
            None => {
                let mut new_mask = self.inner[from.0].component_mask;
                new_mask.unset_id(without);
                let id = self.get_or_create_archetype_search(new_mask, registry);
                self.inner[from.0].set_without(without, id);
                self.inner[id.0].set_with(without, from);
                id
            }
        }
    }

    #[inline(always)]
    pub fn get_disjoint_archetypes(
        &mut self,
        idx1: usize,
        idx2: usize,
    ) -> (&mut Archetype, &mut Archetype) {
        assert!(idx1 != idx2, "Indices must be different");

        let (first, second) = if idx1 < idx2 {
            let (left, right) = self.inner.split_at_mut(idx2);
            (&mut left[idx1], &mut right[0])
        } else {
            let (left, right) = self.inner.split_at_mut(idx1);
            (&mut right[0], &mut left[idx2])
        };

        (first, second)
    }

    pub fn get_or_create_archetype_search(
        &mut self,
        mask: ComponentMask,
        registry: &ComponentRegistry,
    ) -> ArchetypeId {
        match self.get_id_search(&mask) {
            Some(arch) => arch,
            None => self.create_archetype(mask, registry),
        }
    }

    #[inline(always)]
    pub(crate) fn create_archetype(
        &mut self,
        component_mask: ComponentMask,
        registry: &ComponentRegistry,
    ) -> ArchetypeId {
        let id = ArchetypeId(self.len());
        let archetype = Archetype::new(id, component_mask, registry);
        self.push(archetype);
        id
    }

    pub fn get_or_create_archetype(
        &mut self,
        ids: &[ComponentId],
        registry: &ComponentRegistry,
    ) -> ArchetypeId {
        let mut curr = self.get_or_create_empty(registry);

        for id in ids.iter() {
            curr = self.with(curr, id, registry);
        }

        curr
    }

    fn get_or_create_empty(&mut self, registry: &ComponentRegistry) -> ArchetypeId {
        if self
            .lookup
            .first()
            .is_none_or(|(mask, _)| mask != &ComponentMask::new())
        {
            self.create_archetype(ComponentMask::new(), registry)
        } else {
            self.lookup[0].1
        }
    }

    #[inline(always)]
    pub fn get_id_search(&self, mask: &ComponentMask) -> Option<ArchetypeId> {
        // Binary search is extremely fast on small-medium datasets
        self.lookup
            .binary_search_by_key(mask, |(m, _)| *m)
            .ok()
            .map(|idx| self.lookup[idx].1)
    }

    /// Gets an iterator over archetypes added since the given ArchetypeId.
    #[inline(always)]
    pub fn since(&self, last_seen: ArchetypeId) -> impl Iterator<Item = &Archetype> {
        self.inner.iter().skip(last_seen.0)
    }
}

impl Deref for ArchetypeStore {
    type Target = Vec<Archetype>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl Index<ArchetypeId> for ArchetypeStore {
    type Output = Archetype;

    fn index(&self, index: ArchetypeId) -> &Self::Output {
        self.inner.get(index.0).expect("ArchetypeId out of bounds")
    }
}

impl IndexMut<ArchetypeId> for ArchetypeStore {
    fn index_mut(&mut self, index: ArchetypeId) -> &mut Self::Output {
        self.inner
            .get_mut(index.0)
            .expect("ArchetypeId out of bounds")
    }
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
                array::from_fn(|id| {
                    let cid = ComponentId(id);
                    if component_mask.has_id(&cid) {
                        if let Some(meta) = registry.get_meta(cid) {
                            Some(Column::from_meta(meta))
                        } else {
                            panic!("ComponentId {cid:?} not found in registry");
                        }
                    } else {
                        None
                    }
                })
            },
            component_ids: { component_mask.to_ids() },
            component_mask,
            entities: Vec::new(),
            with: [None; MAX_COMPONENTS],
            without: [None; MAX_COMPONENTS],
        }
    }

    pub(crate) fn with(&self, comp_id: &ComponentId) -> Option<ArchetypeId> {
        self.with[comp_id.0].or_else(|| self.component_mask.has_id(comp_id).then_some(self.id))
    }

    pub(crate) fn set_with(&mut self, comp_id: &ComponentId, arch_id: ArchetypeId) {
        self.with[comp_id.0] = Some(arch_id);
    }

    pub(crate) fn move_to(&mut self, from_row: usize, to_arch: &mut Archetype) -> Option<Entity> {
        if from_row >= self.entities.len() {
            panic!("Row index out of bounds");
        }

        // 1. Move components
        for column_id in &to_arch.component_ids {
            if !self.component_mask.has_id(column_id) {
                continue;
            }

            let from_column = self.columns[column_id.0]
                .as_mut()
                .expect("Column should exist for component ID");
            let to_column = to_arch.columns[column_id.0]
                .as_mut()
                .expect("Column should exist for component ID");

            from_column.swap_remove_into(from_row, to_column);
        }

        // 2. Move entity
        to_arch.entities.push(self.entities[from_row]);
        self.swap_remove_entities(from_row)
    }

    pub(crate) fn without(&self, comp_id: &ComponentId) -> Option<ArchetypeId> {
        self.without[comp_id.0]
    }

    pub(crate) fn set_without(&mut self, comp_id: &ComponentId, arch_id: ArchetypeId) {
        self.without[comp_id.0] = Some(arch_id);
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
        for i in 0..self.component_ids.len() {
            let column = self
                .column_mut(&self.component_ids[i].clone())
                .expect("Column should exist for component ID");

            column.swap_remove(row);
        }

        self.swap_remove_entities(row)
    }

    fn swap_remove_entities(&mut self, row: usize) -> Option<Entity> {
        // Swap remove from entities
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
        self.columns.iter_mut().filter_map(|c| c.as_mut())
    }

    /// Helper to access a specific column safely and mutably
    pub fn column_mut(&mut self, id: &ComponentId) -> Option<&mut Column> {
        self.columns.get_mut(id.0).and_then(|c| c.as_mut())
    }

    /// Helper to access a specific column safely
    pub fn column(&self, id: &ComponentId) -> Option<&Column> {
        self.columns.get(id.0).and_then(|c| c.as_ref())
    }

    /// Helper to Borrow a specific column
    /// Returns false if the column does not exist or is already borrowed
    pub fn borrow_column(&self, id: &ComponentId) -> bool {
        self.columns
            .get(id.0)
            .is_some_and(|c| c.as_ref().is_some_and(|c| c.borrow_state().borrow()))
    }

    /// Helper to borrow a specific column mutably
    /// returns false if the column does not exist or is already borrowed
    pub fn borrow_column_mut(&self, id: &ComponentId) -> bool {
        self.columns
            .get(id.0)
            .is_some_and(|c| c.as_ref().is_some_and(|c| c.borrow_state().borrow_mut()))
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
            ComponentMask::from_ids([pos_id, vel_id]),
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
