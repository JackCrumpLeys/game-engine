use crate::schedule::Schedule;
use crate::world::World;

pub trait Plugin {
    fn build(&self, app: &mut App);
}

pub struct App {
    pub world: World,
    pub schedule: Schedule,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    pub fn new() -> Self {
        Self {
            world: World::new(),
            schedule: Schedule::new(),
        }
    }

    pub fn add_plugin<P: Plugin>(&mut self, plugin: P) -> &mut Self {
        plugin.build(self);
        self
    }

    pub fn add_systems<M>(
        &mut self,
        configs: impl crate::schedule::IntoSystemConfigs<M>,
    ) -> &mut Self {
        self.schedule.add_systems(configs, &mut self.world);
        self
    }
}
