/// render module:
/// Makes the window, initializes Vulkan, manages swapchain, and contains the render world ECS.
pub mod components;
pub mod packet;
mod pass;

use std::sync::Arc;

use game_engine_ecs::world::World;
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
        packet::{RenderPacket, SnapshotPair},
        pass::PassManager,
    },
};

pub struct RenderManager {
    render_world: World,
    render_ctx: Option<RenderContext>,
    snapshot_pair: Option<SnapshotPair>,
}

#[allow(dead_code)] // TODO: Remove when more fields are used
struct RenderContext {
    pass_manager: PassManager,
    instance: Arc<Instance>,
    physical_device: Arc<PhysicalDevice>,
    surface: Arc<Surface>,
    window: Arc<Window>,
    device: Arc<Device>,
    queue: Arc<Queue>,
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
        }
    }

    pub fn world_mut(&mut self) -> &mut World {
        &mut self.render_world
    }

    pub fn push_snapshot(&mut self, snapshot: RenderPacket) {
        match self.snapshot_pair {
            Some(ref mut pair) => {
                pair.push_new(snapshot);
            }
            None => {
                self.snapshot_pair = Some(SnapshotPair::new(snapshot.clone(), snapshot));
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
            queue,
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
                        if let Some(snapshot_pair) = &mut self.snapshot_pair {
                            // Interpolate between snapshots
                            let interpolated_snapshot = snapshot_pair.interpolate();

                            // Render the interpolated snapshot
                            rcx.pass_manager.load_packet(&interpolated_snapshot);
                            // Finally do pass
                            match rcx.pass_manager.do_pass(
                                rcx.swapchain.clone(),
                                rcx.device.clone(),
                                rcx.queue.clone(),
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
