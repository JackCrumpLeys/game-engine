mod ecs;

use std::{
    error::Error,
    sync::Arc,
    time::{Duration, UNIX_EPOCH},
};
use vulkano::{
    VulkanLibrary,
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
        physical::{PhysicalDevice, PhysicalDeviceType},
    },
    format::Format,
    image::Image,
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions},
    swapchain::{PresentMode, Surface, Swapchain, SwapchainCreateInfo},
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

use ecs::World;

// --- Components for our ECS ---
#[derive(Debug)]
struct Position {
    x: f32,
    y: f32,
}
#[derive(Debug)]
struct Renderable {
    mesh_id: u32,
}

// Our main application struct
pub struct GameApp {
    instance: Option<Arc<Instance>>,
    physical_device: Option<Arc<PhysicalDevice>>,
    window: Option<Arc<Window>>,
    device: Option<Arc<Device>>,
    queue: Option<Arc<Queue>>,
    surface: Option<Arc<Surface>>,
    swapchain: Option<Arc<Swapchain>>,
    swapchain_images: Vec<Arc<Image>>,
    world: World,
}

impl Default for GameApp {
    fn default() -> Self {
        Self {
            instance: None,
            physical_device: None,
            window: None,
            device: None,
            queue: None,
            surface: None,
            swapchain: None,
            swapchain_images: Vec::new(),
            world: World::new(),
        }
    }
}

impl GameApp {
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
}

impl ApplicationHandler for GameApp {
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
            .unwrap_or(PresentMode::Fifo);

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

        // --- Test Scene in ECS ---
        let entity1 = self.world.spawn_entity();
        self.world
            .add_component(entity1, Position { x: 0.0, y: 0.0 });
        self.world.add_component(entity1, Renderable { mesh_id: 1 });
        let entity2 = self.world.spawn_entity();
        self.world
            .add_component(entity2, Position { x: 10.0, y: 5.0 });
        let entity3 = self.world.spawn_entity();
        self.world
            .add_component(entity3, Position { x: -5.0, y: 2.0 });
        self.world.add_component(entity3, Renderable { mesh_id: 2 });
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
                let render_query = self.world.query::<Renderable>();
                let position_query = self.world.query::<Position>();

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
