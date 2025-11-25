use std::alloc::Layout;
use std::any::TypeId;
use std::collections::HashMap;

/// A unique index assigned to a component type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ComponentId(pub usize);

/// Metadata required to store and serialise a component.
#[derive(Debug, Clone)]
pub struct ComponentMeta {
    pub name: &'static str,
    pub layout: Layout,
    // We will add more reflection data here later (field names, etc.)
}

/// The trait that all ECS data must implement.
/// We require Sized because Archetypes need to calculate strides.
pub trait Component: 'static + Send + Sync + Sized {}

// Blanket impl so users don't have to manually impl Component
impl<T: 'static + Send + Sync + Sized> Component for T {}

pub struct ComponentRegistry {
    type_to_id: HashMap<TypeId, ComponentId>,
    components: Vec<ComponentMeta>,
}

impl Default for ComponentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ComponentRegistry {
    pub fn new() -> Self {
        ComponentRegistry {
            type_to_id: HashMap::new(),
            components: Vec::new(),
        }
    }

    /// Registers a type T. If it already exists, returns the existing ID.
    /// If it's new, assigns a new ID and stores the Layout/Name.
    pub fn register<T: Component>(&mut self) -> ComponentId {
        *self.type_to_id.entry(TypeId::of::<T>()).or_insert_with(|| {
            let name = std::any::type_name::<T>();
            let layout = std::alloc::Layout::new::<T>();

            let id = ComponentId(self.components.len());

            let meta = ComponentMeta { name, layout };
            self.components.push(meta);

            id
        })
    }

    pub fn get_id<T: Component>(&self) -> Option<ComponentId> {
        self.type_to_id.get(&TypeId::of::<T>()).cloned()
    }

    pub fn get_meta(&self, id: ComponentId) -> Option<&ComponentMeta> {
        self.components.get(id.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(dead_code)]
    fn test_component_registry() {
        let mut registry = ComponentRegistry::new();

        #[derive(Debug)]
        struct Position(f32, f32);

        #[derive(Debug)]
        struct Velocity(f32, f32);

        let pos_id = registry.register::<Position>();
        let vel_id = registry.register::<Velocity>();
        let pos_id2 = registry.register::<Position>();

        assert_eq!(pos_id, pos_id2);
        assert_ne!(pos_id, vel_id);

        let pos_meta = registry.get_meta(pos_id).unwrap();
        assert_eq!(
            pos_meta.name,
            "game_engine_ecs::component::tests::test_component_registry::Position"
        );
        assert_eq!(pos_meta.layout.size(), std::mem::size_of::<Position>());

        let vel_meta = registry.get_meta(vel_id).unwrap();
        assert_eq!(
            vel_meta.name,
            "game_engine_ecs::component::tests::test_component_registry::Velocity"
        );
        assert_eq!(vel_meta.layout.size(), std::mem::size_of::<Velocity>());
    }
}
