/// render module:
/// Makes the window, initializes Vulkan, manages swapchain, and contains the render world ECS.
pub mod components;
mod packet;
mod pass;

use std::sync::Arc;

use vulkano::{
    VulkanLibrary,
    buffer::{Buffer, BufferContents},
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
        physical::{PhysicalDevice, PhysicalDeviceType},
    },
    format::Format,
    image::Image,
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    pipeline::graphics::vertex_input::Vertex,
    swapchain::{PresentMode, Surface, Swapchain, SwapchainCreateInfo},
};
use winit::{
    application::ApplicationHandler,
    dpi::Position,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};

use crate::{
    ecs::World,
    render::{
        components::Renderable,
        packet::{Interpolate, RenderPacket},
    },
};

pub struct RenderManager {
    render_world: World,
    instance: Option<Arc<Instance>>,
    physical_device: Option<Arc<PhysicalDevice>>,
    window: Option<Arc<Window>>,
    device: Option<Arc<Device>>,
    queue: Option<Arc<Queue>>,
    surface: Option<Arc<Surface>>,
    swapchain: Option<Arc<Swapchain>>,
    swapchain_images: Vec<Arc<Image>>,
    snapshot_pair: Option<SnapshotPair>,
}

/// The renderer will interpolate between two snapshots to produce smooth animations.
pub struct SnapshotPair {
    old: RenderPacket,
    new: RenderPacket,
}

impl SnapshotPair {
    pub fn new(old: RenderPacket, new: RenderPacket) -> Self {
        Self { old, new }
    }

    /// replaces the "new" snapshot with the provided one, and moves the previous "new" to "old"
    pub fn push_new(&mut self, new: RenderPacket) {
        self.old = std::mem::replace(&mut self.new, new);
    }

    pub fn interpolate(&self) -> RenderPacket {
        let now = std::time::Instant::now();
        let duration_since_old = now.duration_since(self.old.snapped_at);
        /// WIP TODO
        self.old.interpolate(&self.new, factor)
    }
}

impl RenderManager {
    fn recreate_swapchain(&mut self) {
        let window = match &self.window {
            Some(w) => w,
            None => return,
        };
        let old_swapchain = match &self.swapchain {
            Some(s) => s,
            None => return,
        };

        let (new_swapchain, new_images) = old_swapchain
            .recreate(SwapchainCreateInfo {
                image_extent: window.inner_size().into(),
                ..old_swapchain.create_info()
            })
            .expect("Failed to recreate swapchain");

        log::info!("Swapchain recreated with {} images", new_images.len());

        self.swapchain = Some(new_swapchain);
        self.swapchain_images = new_images;
    }

    pub fn new() -> Self {
        Self {
            render_world: World::new(),
            instance: None,
            physical_device: None,
            window: None,
            device: None,
            queue: None,
            surface: None,
            swapchain: None,
            swapchain_images: Vec::new(),
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
                .create_window(Window::default_attributes())
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

        log::info!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type
        );

        // --- Logical Device and Queue Creation ---
        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: DeviceExtensions {
                    khr_swapchain: true,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .expect("Failed to create logical device");

        let queue = queues.next().unwrap();

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
            .unwrap_or_else(|| PresentMode::Fifo);

        log::info!(
            "Selected image format: {:?}, present mode: {:?}",
            image_format,
            present_mode
        );

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

        // --- Store Core Objects ---
        self.instance = Some(instance);
        self.physical_device = Some(physical_device);
        self.window = Some(window);
        self.device = Some(device);
        self.queue = Some(queue);
        self.surface = Some(surface);
        self.swapchain = Some(swapchain);
        self.swapchain_images = images;
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(_) | WindowEvent::Focused(_) => {
                // Recreate swapchain when window is resized or focus changes
                self.recreate_swapchain();
            }
            WindowEvent::RedrawRequested => {
                // Placeholder render system logic
                log::info!("--- FRAME START ---");
                let render_query = self.render_world.query::<Renderable>();
                let position_query = self.render_world.query::<Position>();

                for (entity, renderable) in render_query {
                    if let Some((_, pos)) = position_query.iter().find(|(e, _)| *e == entity) {
                        log::info!(
                            "Drawing entity {:?} with mesh {} at {:?}",
                            entity,
                            renderable.mesh_id,
                            pos
                        );
                    }
                }
            }
            _ => (),
        }
    }
}
