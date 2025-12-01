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
mod storage;
pub mod system;
pub mod world;

pub mod prelude {
    pub use crate::bundle::Bundle;
    pub use crate::component::{Component, ComponentId, ComponentRegistry};
    pub use crate::entity::Entity;
    pub use crate::query::{Filter, QueryInner};
    pub use crate::resource::{Res, ResMut, Resource, Resources};
    pub use crate::world::World;
}
