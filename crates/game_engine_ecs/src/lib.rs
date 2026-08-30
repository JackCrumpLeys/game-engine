#![feature(associated_type_defaults)]
#[macro_use]
mod macros;
pub mod app;
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
    // App and Plugins
    pub use crate::app::{App, Plugin};

    // Core World and Entities
    pub use crate::entity::Entity;
    pub use crate::world::World;

    // Components and Bundles
    pub use crate::bundle::Bundle;
    pub use crate::component::{Component, ComponentId};

    // Resources
    pub use crate::resource::{Res, ResMut, Resource, Resources, TLRes, TLResMut};

    // Systems and Parameters
    pub use crate::system::function::IntoSystem;
    pub use crate::system::{Local, Query, System, SystemParam};

    // Command Buffer (for spawning/despawning/inserting deferred)
    pub use crate::system::command::Commands;

    // Query & Filters
    // QueryState is added for manual World::query() outside of systems
    pub use crate::query::{Changed, Filter, QueryState, With, Without};

    // Scheduling and System Configuration
    pub use crate::schedule::{IntoSystemConfigs, IntoSystemSet, Schedule, SystemSet};

    // Messaging / Queues
    pub use crate::message::{Message, MessageReader, MessageWriter};

    // Observers / Event Triggers
    pub use crate::observers::{IntoObserverSystem, Trigger, TriggerEvent};
}

extern crate self as game_engine_ecs;

#[cfg(feature = "tracy")]
#[global_allocator]
static GLOBAL: tracy_client::ProfiledAllocator<std::alloc::System> =
    tracy_client::ProfiledAllocator::new(System, 100);
