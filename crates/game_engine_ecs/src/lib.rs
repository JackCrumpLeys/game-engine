#![feature(associated_type_defaults)]
#[macro_use]
mod macros;
mod archetype;
mod borrow;
pub mod bundle;
pub mod component;
pub mod entity;
pub mod message;
pub mod query;
pub mod resource;
pub mod schedule;
mod storage;
pub mod system;
mod thread_entity_allocator;
pub mod threading;
pub mod world;

pub mod prelude {
    pub use crate::bundle::Bundle;
    pub use crate::component::{Component, ComponentId, ComponentRegistry};
    pub use crate::entity::Entity;
    pub use crate::message::{Message, MessageReader, MessageWriter};
    pub use crate::query::Filter;
    pub use crate::resource::{Res, ResMut, Resource, Resources};
    pub use crate::system::function::{FunctionSystem, IntoSystem};
    pub use crate::system::{Query, System, SystemParam};
    pub use crate::world::World;
}
