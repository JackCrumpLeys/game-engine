// main.rs or lib.rs

use std::sync::Arc;
use vulkano::VulkanLibrary;
use vulkano::device::QueueFlags;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

// Our main application struct
struct VulkanApp {
    instance: Arc<Instance>,
    physical_device: Arc<PhysicalDevice>,
    // We will add more here soon: Window, EventLoop, Logical Device, Queue, etc.
}

impl VulkanApp {
    pub fn new(event_loop: &EventLoop<()>) -> Self {
        // --- Instance Creation ---
        let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
        // In the future, we'll need extensions for windowing. For now, this is fine.
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                ..Default::default()
            },
        )
        .expect("failed to create instance");

        // --- Physical Device Selection ---
        let physical_device = instance
            .enumerate_physical_devices()
            .expect("could not enumerate devices")
            .filter(|p| {
                p.queue_family_properties()
                    .iter()
                    .any(|q| q.queue_flags.contains(QueueFlags::GRAPHICS))
            })
            .min_by_key(|p| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .expect("No suitable physical device found");

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type
        );

        Self {
            instance,
            physical_device,
        }
    }

    pub fn run(self, event_loop: EventLoop<()>) {
        // For now, let's just create a window and a basic event loop.
        // We are not drawing anything yet.
        let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());

        event_loop.run(|event, _, control_flow| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            _ => (),
        });
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let app = VulkanApp::new(&event_loop);
    app.run(event_loop);
}
