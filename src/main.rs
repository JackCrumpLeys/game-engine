// main.rs

use std::sync::Arc;
use vulkano::{
    VulkanLibrary,
    device::QueueFlags,
    device::physical::{PhysicalDevice, PhysicalDeviceType},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

// Our main application struct, implementing the ApplicationHandler trait
#[derive(Default)]
struct GameApp {
    instance: Option<Arc<Instance>>,
    physical_device: Option<Arc<PhysicalDevice>>,
    window: Option<Arc<Window>>,
}

impl ApplicationHandler for GameApp {
    // This is called when the event loop is ready to run.
    // It's the perfect place for one-time setup.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        println!("Resumed. Initializing Vulkan and creating window...");

        // --- Instance Creation ---
        let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");

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

        // --- Window Creation ---
        // A window can only be created inside an active event loop.
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );

        self.instance = Some(instance);
        self.physical_device = Some(physical_device);
        self.window = Some(window);
    }

    // This is our new "main loop" for handling events.
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                // This is where our drawing logic will go.
                // For now, we do nothing.
            }
            _ => (),
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();

    // Continuously runs the event loop, ideal for games.
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = GameApp::default();
    event_loop.run_app(&mut app).unwrap();
}
