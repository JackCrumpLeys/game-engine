//! render module:
//! Makes the window, initializes Vulkan, manages swapchain, and contains the render world ECS.

pub mod components;
pub mod packet;
pub mod pass;
pub mod plugin;
mod shaders;
pub mod stack;
pub mod storage;
pub mod texture;

use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender};
use game_engine_ecs::world::World;
use crate::input::{InputAccumulator, InputState};
use glam::Vec2;
use vulkano::{
    VulkanLibrary,
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
        QueueFlags,
        physical::{PhysicalDevice, PhysicalDeviceType},
    },
    format::Format,
    image::Image,
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    swapchain::{PresentMode, Surface, Swapchain, SwapchainCreateInfo},
};

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};

use crate::{
    GameEngineResult,
    render::{
        pass::PassManager,
        storage::RenderPacket,
    },
};


use crate::FrameData;

pub struct RenderManager {
    // Dummy world just for PassManager's thread-local allocators
    pub render_world: World,
    pub render_ctx: Option<RenderContext>,

    // Channel from ECS sim thread
    pub rx: Receiver<FrameData>,
    // Channel to send input state to ECS sim thread
    pub input_tx: Sender<InputState>,
    pub packet: RenderPacket,

    pub camera_pos: Vec2,
    pub zoom: f32,
    /// Textures buffered while render_ctx wasn't ready yet.
    buffered_textures: Vec<texture::PendingTexture>,
    pub accum_render_time: std::time::Duration,
    input_accumulator: InputAccumulator,
    pub window_title: String,
}

#[allow(dead_code)] // TODO: Remove when more fields are used
pub struct RenderContext {
    pub pass_manager: PassManager,
    pub instance: Arc<Instance>,
    pub physical_device: Arc<PhysicalDevice>,
    pub surface: Arc<Surface>,
    pub window: Arc<Window>,
    pub device: Arc<Device>,
    pub render_queue: Arc<Queue>,
    pub transfer_queue: Arc<Queue>,
    pub swapchain: Arc<Swapchain>,
    pub swapchain_images: Vec<Arc<Image>>,
}

impl RenderManager {
    pub fn window(&self) -> GameEngineResult<Arc<Window>> {
        match &self.render_ctx {
            Some(rcx) => Ok(rcx.window.clone()),
            None => Err("Render context not initialized".into()),
        }
    }

    fn recreate_swapchain(&mut self) {
        let rcx = match &mut self.render_ctx {
            Some(rcx) => rcx,
            None => return,
        };

        let (new_swapchain, new_images) = rcx
            .swapchain
            .recreate(SwapchainCreateInfo {
                image_extent: rcx.window.inner_size().into(),
                ..rcx.swapchain.create_info()
            })
            .expect("Failed to recreate swapchain");

        log::debug!("Swapchain recreated with {} images", new_images.len());

        rcx.pass_manager.resize(&new_images);

        rcx.swapchain = new_swapchain;
        rcx.swapchain_images = new_images;
    }

    pub fn new(rx: Receiver<FrameData>, input_tx: Sender<InputState>) -> Self {
        let mut render_world = World::new();
        render_world.resources_mut().insert(RenderPacket::new());
        Self {
            render_world,
            render_ctx: None,
            buffered_textures: Vec::new(),
            rx,
            input_tx,
            camera_pos: glam::Vec2::ZERO,
            zoom: 1.0,
            accum_render_time: std::time::Duration::ZERO,
            input_accumulator: InputAccumulator::new(),
            packet: RenderPacket::new(),
            window_title: "Game Engine".into(),
        }
    }

    pub fn world_mut(&mut self) -> &mut World {
        &mut self.render_world
    }
}

impl ApplicationHandler for RenderManager {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        log::info!("Resumed. Initializing Vulkan and creating window...");

        // --- Instance Creation ---
        let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
        log::debug!("Vulkan api ver = {}", library.api_version());
        let mut enable_validation = false;
        if let Ok(prop) = library.layer_properties() {
            for layer in prop {
                if layer.name() == "VK_LAYER_KHRONOS_validation" {
                    enable_validation = true;
                    break;
                }
            }
        }

        let mut enabled_layers = vec![];
        if enable_validation {
            enabled_layers.push("VK_LAYER_KHRONOS_validation".to_owned());
        }

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_layers,
                enabled_extensions: Surface::required_extensions(event_loop).unwrap(),
                ..Default::default()
            },
        )
        .expect("failed to create instance");



        // --- Window Creation ---
        let mut attributes = Window::default_attributes()
            .with_title(&self.window_title);

        #[cfg(target_os = "linux")]
        {
            use winit::platform::wayland::WindowAttributesExtWayland;
            attributes = WindowAttributesExtWayland::with_name(attributes, "rust-bg-class", "rust-bg-class");
        }

        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .unwrap(),
        );

        // --- Surface Creation ---
        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

        // --- Physical Device Selection (with surface support) ---
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .expect("could not enumerate devices")
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.contains(QueueFlags::GRAPHICS)
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                _ => 4,
            })
            .expect("No suitable physical device found");

        // get secondary queue family for transfer operations

        let transfer_queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_, q)| {
                q.queue_flags.contains(QueueFlags::TRANSFER)
                    && !q.queue_flags.contains(QueueFlags::GRAPHICS)
            })
            .map(|i| i as u32)
            .unwrap_or(queue_family_index); // fallback to graphics queue if no separate transfer
        // queue
        log::info!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type
        );

        // --- Logical Device and Queue Creation ---
        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![
                    QueueCreateInfo {
                        queue_family_index,
                        ..Default::default()
                    },
                    QueueCreateInfo {
                        queue_family_index: transfer_queue_family_index,
                        ..Default::default()
                    },
                ],
                enabled_extensions: DeviceExtensions {
                    khr_swapchain: true,
                    khr_vulkan_memory_model: true,
                    ..Default::default()
                },
                enabled_features: DeviceFeatures {
                    vulkan_memory_model: true,
                    descriptor_indexing: true,
                    shader_sampled_image_array_non_uniform_indexing: true,
                    runtime_descriptor_array: true,
                    descriptor_binding_variable_descriptor_count: true,
                    descriptor_binding_partially_bound: true,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .expect("Failed to create logical device");

        let render_queue = queues.next().unwrap();
        let transfer_queue = queues.next().unwrap();

        // --- Swapchain Capabilities and Format Selection ---
        let surface_capabilities = physical_device
            .surface_capabilities(&surface, Default::default())
            .expect("Failed to get surface capabilities");

        let image_format = physical_device
            .surface_formats(&surface, Default::default())
            .expect("Failed to get surface formats")
            .into_iter()
            .find(|(format, _)| matches!(format, Format::B8G8R8A8_SRGB | Format::R8G8B8A8_SRGB))
            .unwrap_or_else(|| {
                physical_device
                    .surface_formats(&surface, Default::default())
                    .unwrap()[0]
            })
            .0;

        let present_mode = PresentMode::Fifo;

        log::info!("Selected image format: {image_format:?}, present mode: {present_mode:?}");

        // --- Swapchain Creation ---
        let (swapchain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count.max(2),
                image_format,
                image_extent: window.inner_size().into(),
                image_usage: vulkano::image::ImageUsage::COLOR_ATTACHMENT,
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .unwrap(),
                present_mode,
                ..Default::default()
            },
        )
        .expect("Failed to create swapchain");

        log::info!("Swapchain created with {} images", images.len());

        let pass_manager = PassManager::new(
            device.clone(),
            swapchain.clone(),
            &images,
            window.clone(),
            &mut self.render_world,
        )
        .unwrap();

        let rcx = RenderContext {
            pass_manager,
            instance,
            physical_device,
            window,
            device,
            render_queue,
            transfer_queue,
            surface,
            swapchain,
            swapchain_images: images,
        };

        self.render_ctx = Some(rcx);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // Feed all events through the input accumulator
        self.input_accumulator.process_event(&event);

        match event {
            // Camera controls (built-in pan/zoom via mouse)
            WindowEvent::MouseWheel { delta, .. } => {
                let amount = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => (pos.y / 100.0) as f32,
                };
                self.zoom *= 1.0 + (amount * 0.1);
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(_) | WindowEvent::Focused(_) => {
                self.recreate_swapchain();
            }
            WindowEvent::RedrawRequested => {
                // Send input snapshot to ECS thread
                let snapshot = self.input_accumulator.take_snapshot();

                // Built-in camera pan via mouse drag
                if snapshot.mouse_button_held(winit::event::MouseButton::Left) {
                    let delta = glam::vec2(snapshot.mouse_delta.x, -snapshot.mouse_delta.y);
                    self.camera_pos -= delta / self.zoom;
                }

                let _ = self.input_tx.try_send(snapshot);

                if let Some(rctx) = &mut self.render_ctx {
                    
                    // 1. Drain ALL pending frames and merge them safely!
                    while let Ok(frame) = self.rx.try_recv() {
                        rctx.pass_manager.push_packet_dirty(frame.delta, frame.dirty);
                    }

                    // 2. Draw using the completely merged state
                    let _ = rctx.pass_manager.do_pass(
                        &self.render_world,
                        rctx.swapchain.clone(),
                        rctx.render_queue.clone(),
                        rctx.transfer_queue.clone(),
                        self.camera_pos,
                        self.zoom,
                    );
                }

                // Ask Winit to instantly redraw again (Uncapped FPS)
                if let Some(rcx) = &self.render_ctx {
                    rcx.window.request_redraw();
                }
            }
            _ => (),
        }
    }
}
