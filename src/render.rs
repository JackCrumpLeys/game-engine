//! render module:
//! Makes the window, initializes Vulkan, manages swapchain, and contains the render world ECS.
pub mod components;
pub mod packet;
mod pass;
mod shaders;
pub mod storage;

use std::sync::Arc;

use game_engine_ecs::world::World;
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
    platform::x11::WindowAttributesExtX11,
    window::{Window, WindowId},
};

use crate::{
    GameEngineResult,
    render::{packet::SnapshotPair, pass::PassManager, storage::RenderPacket},
};

pub struct RenderManager {
    render_world: World,
    render_ctx: Option<RenderContext>,
    snapshot_pair: Option<SnapshotPair>,

    pub camera_pos: Vec2,
    pub zoom: f32,
    mouse_pressed: bool,
    last_mouse_pos: Vec2,
}

#[allow(dead_code)] // TODO: Remove when more fields are used
struct RenderContext {
    pass_manager: PassManager,
    instance: Arc<Instance>,
    physical_device: Arc<PhysicalDevice>,
    surface: Arc<Surface>,
    window: Arc<Window>,
    device: Arc<Device>,
    render_queue: Arc<Queue>,
    transfer_queue: Arc<Queue>,
    swapchain: Arc<Swapchain>,
    swapchain_images: Vec<Arc<Image>>,
}

impl Default for RenderManager {
    fn default() -> Self {
        Self::new()
    }
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

    pub fn new() -> Self {
        Self {
            render_world: World::new(),
            render_ctx: None,
            snapshot_pair: None,
            camera_pos: glam::Vec2::ZERO,
            zoom: 1.0,
            mouse_pressed: false,
            last_mouse_pos: glam::Vec2::ZERO,
        }
    }

    pub fn world_mut(&mut self) -> &mut World {
        &mut self.render_world
    }

    pub fn push_snapshot(&mut self, snapshot: RenderPacket) {
        match self.snapshot_pair {
            Some(ref mut pair) => {
                pair.push_new(snapshot.snapped_at);
                if let Some(rctx) = &mut self.render_ctx {
                    rctx.pass_manager.push_packet(snapshot);
                }
            }
            None => {
                self.snapshot_pair =
                    Some(SnapshotPair::new(snapshot.snapped_at, snapshot.snapped_at));
            }
        }
    }
}

impl ApplicationHandler for RenderManager {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        log::info!("Resumed. Initializing Vulkan and creating window...");

        // --- Instance Creation ---
        let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
        log::debug!("Vulkan api ver = {}", library.api_version());
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: Surface::required_extensions(event_loop).unwrap(),
                ..Default::default()
            },
        )
        .expect("failed to create instance");

        // --- Window Creation ---
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("my-rust-bg")
                        .with_name("rust-bg-class", "rust-bg-class"),
                )
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

        let present_mode = physical_device
            .surface_present_modes(&surface, Default::default())
            .expect("Failed to get present modes")
            .into_iter()
            .find(|&mode| mode == PresentMode::Fifo)
            .unwrap_or(PresentMode::Fifo);

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

        let pass_manager =
            PassManager::new(device.clone(), swapchain.clone(), &images, window.clone()).unwrap();

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
        match event {
            // TODO: Proper key recording and shi
            WindowEvent::MouseWheel { delta, .. } => {
                let amount = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => (pos.y / 100.0) as f32,
                };
                // Exponential zoom feels more natural
                self.zoom *= 1.0 + (amount * 0.1);
                // self.zoom = self.zoom.clamp(0.01, 100.0);
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if button == winit::event::MouseButton::Left {
                    self.mouse_pressed = state.is_pressed();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let pos = glam::vec2(position.x as f32, -position.y as f32);
                if self.mouse_pressed {
                    let delta = pos - self.last_mouse_pos;
                    // We divide by zoom so panning speed matches world movement
                    self.camera_pos -= delta / self.zoom;
                }
                self.last_mouse_pos = pos;
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(_) | WindowEvent::Focused(_) => {
                // Recreate swapchain when window is resized or focus changes
                self.recreate_swapchain();
            }
            WindowEvent::RedrawRequested => {
                log::trace!("--- FRAME START ---");
                match &mut self.render_ctx {
                    Some(rcx) => {
                        // Handle rendering here
                        if self.snapshot_pair.is_none() {
                            // Finally do pass
                            match rcx.pass_manager.do_pass(
                                rcx.swapchain.clone(),
                                rcx.render_queue.clone(),
                                rcx.transfer_queue.clone(),
                                self.camera_pos,
                                self.zoom,
                            ) {
                                Ok(r) => match r {
                                    pass::PassResult::Success => {
                                        log::trace!("Frame rendered successfully")
                                    }
                                    pass::PassResult::SwapchainOutOfDate => {
                                        log::warn!(
                                            "Swapchain out of date during rendering, recreating..."
                                        );
                                        self.recreate_swapchain();
                                    }
                                },
                                Err(e) => log::error!("Rendering error: {e:?}"),
                            }
                        } else {
                            log::warn!("No snapshot pair available for rendering");
                        }
                    }
                    None => {
                        log::warn!("Render context not initialized");
                    }
                }
                log::trace!("--- FRAME END ---");
            }
            _ => (),
        }
    }
}
