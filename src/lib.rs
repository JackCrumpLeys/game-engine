pub mod input;
pub mod render;
pub mod text;

use std::sync::Arc;
use std::time::Instant;

use crossbeam_channel::bounded;
use game_engine_ecs::app::{App, Plugin};
use game_engine_ecs::schedule::{IntoSystemConfigs, Schedule};
use game_engine_render::slot_allocator::GpuSlotAllocator;

use crate::input::{Camera2D, InputState};
use crate::render::RenderManager;
use crate::render::storage::{RenderChannelStatus, RenderDelta, RenderPacket};
use crate::render::texture::TextureServer;
use crate::text::font;

use winit::event_loop::{ControlFlow, EventLoop};

// ── Re-exports for user convenience (prelude) ────────────────────────

pub mod prelude {
    pub use crate::input::{Camera2D, InputState, KeyCode, MouseButton};
    pub use crate::render::components::Position;
    pub use crate::render::storage::{RenderChannelStatus, RenderPacket};
    pub use crate::text::font::FontAtlas as FontAtlasData;
    pub use crate::{
        Engine, FontAtlas, FrameData, SimTime, define_mut_methods, define_render_stack, sync_fields,
    };

    pub use game_engine_derive::Component;
    pub use game_engine_ecs::prelude::*;
    pub use game_engine_ecs::system::command::Commands;
    pub use game_engine_render::slot_allocator::GpuSlotAllocator;
    pub use game_engine_shaders_types::packet::*;
    pub use game_engine_shaders_types::shapes::ColorRGBA;
    pub use game_engine_shaders_types::{Vec2, vec2};
}

// ── Shared types ─────────────────────────────────────────────────────

/// Data sent from the ECS simulation thread to the render thread each tick.
pub struct FrameData {
    pub delta: RenderDelta,
    pub dirty: Vec<Vec<u32>>,
    pub input_snapshot: Option<InputState>,
}

/// Global simulation time resource.
#[derive(Default, Clone)]
pub struct SimTime {
    pub phase: f32,
    pub frame_count: u64,
}

/// Font atlas resource wrapping the loaded font data.
pub struct FontAtlas {
    pub atlas: Arc<font::FontAtlas>,
}

// ── Engine Builder ───────────────────────────────────────────────────

/// Configuration for the window.
pub struct WindowConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            title: "Game Engine".into(),
            width: 1280,
            height: 720,
        }
    }
}

/// The main engine builder. Replaces the old monolithic `GameApp`.
///
/// # Example
/// ```ignore
/// Engine::new()
///     .insert_resource(SimTime::default())
///     .add_startup_system(spawn_my_scene)
///     .add_systems(animate_system)
///     .run()
/// ```
pub struct Engine {
    app: App,
    startup_configs: Vec<game_engine_ecs::schedule::SystemConfig>,
    window_config: WindowConfig,
    font_bytes: Option<&'static [u8]>,
    atlas: Arc<font::FontAtlas>,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    pub fn new() -> Self {
        let mut app = App::new();

        // Insert core resources eagerly so plugins/systems can reference them during init
        app.world.resources_mut().insert(RenderPacket::new());
        app.world.resources_mut().insert(GpuSlotAllocator::new());
        app.world
            .resources_mut()
            .insert(RenderChannelStatus { is_full: false });
        app.world.resources_mut().insert(InputState::default());
        app.world.resources_mut().insert(Camera2D::default());

        let texture_server = Arc::new(TextureServer::new());
        let font_bytes = include_bytes!("../assets/Inter.ttf");

        let atlas = font::load_font(font_bytes).expect("Failed to load font");
        app.world.resources_mut().insert(texture_server.clone());

        let atlas = Arc::new(atlas);
        app.world.resources_mut().insert(FontAtlas {
            atlas: atlas.clone(),
        });

        Self {
            app,
            startup_configs: Vec::new(),
            window_config: WindowConfig::default(),
            font_bytes: None,
            atlas,
        }
    }

    /// Set window configuration.
    pub fn with_window(mut self, config: WindowConfig) -> Self {
        self.window_config = config;
        self
    }

    /// Set custom font bytes. If not called, the bundled Inter font is used.
    pub fn with_font(mut self, font_bytes: &'static [u8]) -> Self {
        self.font_bytes = Some(font_bytes);
        self
    }

    /// Insert a resource into the ECS world.
    pub fn insert_resource<R: 'static + Send + Sync>(mut self, resource: R) -> Self {
        self.app.world.resources_mut().insert(resource);
        self
    }

    /// Add a plugin that configures the ECS app.
    pub fn add_plugin<P: Plugin>(mut self, plugin: P) -> Self {
        self.app.add_plugin(plugin);
        self
    }

    /// Add systems that run every frame.
    pub fn add_systems<M: 'static>(mut self, systems: impl IntoSystemConfigs<M>) -> Self {
        self.app.add_systems(systems);
        self
    }

    /// Add systems that run only once on the first frame.
    pub fn add_startup_system<M: 'static>(mut self, system: impl IntoSystemConfigs<M>) -> Self {
        let configs = system.into_configs();
        self.startup_configs.extend(configs);
        self
    }

    /// Build and run the engine. This blocks forever.
    pub fn run(mut self) -> Result<(), Box<dyn std::error::Error>> {
        let (tx, rx) = bounded::<FrameData>(2);
        let (input_tx, input_rx) = bounded::<InputState>(2);
        println!("font atlis");
        // Ensure SimTime exists (user may have already inserted it)
        self.app
            .world
            .resources_mut()
            .get_or_insert_default::<SimTime>();

        // --- 4. Build startup schedule ---
        let mut startup_schedule = Schedule::new();
        for config in self.startup_configs {
            startup_schedule.add_systems(config, &mut self.app.world);
        }

        // --- 5. ECS Simulation Thread ---
        let mut sim_world = self.app.world;
        let mut schedule = self.app.schedule;

        let window_title = self.window_config.title.clone();

        std::thread::Builder::new()
            .name("ECS Simulation Loop".into())
            .spawn(move || {
                // Run startup systems exactly once
                startup_schedule.update(&mut sim_world);

                let mut tick_time = Instant::now();
                let mut ticks = 0u64;
                log::info!("ECS Background thread started.");

                loop {
                    // 1. Drain input from render thread
                    while let Ok(input) = input_rx.try_recv() {
                        let resources = sim_world.resources_mut();
                        *resources.get_mut::<InputState>().unwrap() = input;
                    }

                    // 2. Update channel status
                    {
                        let resources = sim_world.resources_mut();
                        let mut status = resources.get_mut::<RenderChannelStatus>().unwrap();
                        status.is_full = tx.is_full();
                    }

                    // 3. Run simulation
                    schedule.update(&mut sim_world);

                    // 4. Send to render thread
                    if !tx.is_full() {
                        let resources = sim_world.resources_mut();
                        let mut packet = resources.get_mut::<RenderPacket>().unwrap();
                        let mut allocator = resources.get_mut::<GpuSlotAllocator>().unwrap();

                        let dirty = allocator.take_dirty();
                        packet.snapped_at = Instant::now();
                        let delta = packet.take_delta();

                        let _ = tx.try_send(FrameData {
                            delta,
                            dirty,
                            input_snapshot: None,
                        });
                    }

                    ticks += 1;
                    if tick_time.elapsed().as_secs_f32() >= 1.0 {
                        log::info!("SIMULATION THREAD: {ticks} TPS");
                        ticks = 0;
                        tick_time = Instant::now();
                    }
                }
            })
            .unwrap();

        // --- 6. Main Thread (Winit / Vulkan) ---
        let mut packet = RenderPacket::new();
        packet.path_curves = Arc::new(self.atlas.curves.clone());

        let mut render_manager = RenderManager::new(rx, input_tx);
        render_manager.packet = packet;
        render_manager.window_title = window_title;

        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);
        event_loop.run_app(&mut render_manager)?;

        Ok(())
    }
}

pub type GameEngineResult<T> = Result<T, Box<dyn std::error::Error>>;
