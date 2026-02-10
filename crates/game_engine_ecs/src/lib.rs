#![feature(associated_type_defaults)]
#[macro_use]
mod macros;
mod archetype;
mod borrow;
pub mod bundle;
pub mod component;
pub mod entity;
pub mod message;
pub mod observers;
pub mod query;
pub mod resource;
pub mod schedule;
mod storage;
pub mod system;
mod thread_entity_allocator;
pub mod threading;
pub mod world;

pub mod prelude {
    // Core World and Entities
    pub use crate::entity::Entity;
    pub use crate::world::World;

    // Components and Bundles
    pub use crate::bundle::Bundle;
    pub use crate::component::{Component, ComponentId};

    // Resources
    pub use crate::resource::{Res, ResMut, Resource, Resources};

    // Systems and Parameters
    pub use crate::system::{Local, Query, System, function::IntoSystem};

    // Command Buffer (for spawning/despawning/inserting deferred)
    pub use crate::system::command::Command;

    // Query Filters
    pub use crate::query::{Changed, Filter, With, Without};

    // Scheduling and System Configuration
    pub use crate::schedule::{IntoSystemConfigs, IntoSystemSet, Schedule, SystemSet};

    // Messaging / Events
    pub use crate::message::{Message, MessageReader, MessageWriter};
}

extern crate self as game_engine_ecs;

#[cfg(feature = "tracy")]
#[global_allocator]
static GLOBAL: tracy_client::ProfiledAllocator<std::alloc::System> =
    tracy_client::ProfiledAllocator::new(std::alloc::System, 100);
