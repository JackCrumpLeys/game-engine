use bytemuck::Pod;
use std::sync::Arc;
use std::time::Instant;
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{BufferContents, BufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo,
    SubpassEndInfo,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, Queue};
use vulkano::image::Image;
use vulkano::memory::allocator::{MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::swapchain::{Swapchain, SwapchainPresentInfo, acquire_next_image};
use vulkano::sync::{self, GpuFuture};
use vulkano::{Validated, VulkanError};
use winit::window::Window;

use crate::GameEngineResult;
use crate::render::packet::{RenderPacketContents, SnapshotPair};
use crate::render::pipeline::GpuPipeline;
use crate::render::pipeline::triangle::TrianglePipeline;
use game_engine_shaders_types::PushConstants;

pub struct PassManager {
    /// The specialized pipeline for drawing triangles
    triangle_pipeline: TrianglePipeline,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    /// Used for dynamic, per-frame uploads of SSBO data
    storage_allocator: SubbufferAllocator,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    viewport: Viewport,
    start_time: Instant,
}

impl PassManager {
    pub fn new(
        device: Arc<Device>,
        swapchain: Arc<Swapchain>,
        images: &[Arc<Image>],
        window: Arc<Window>,
    ) -> GameEngineResult<Self> {
        // 1. Create the RenderPass (Clear -> Draw -> Store)
        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )?;

        // 2. Initialize Framebuffers and Pipeline
        let framebuffers = window_size_dependent_setup(images, &render_pass);
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        // Use the trait initialization
        let triangle_pipeline = TrianglePipeline::new(device.clone(), subpass)?;

        // 3. Initialize Allocators
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let storage_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::STORAGE_BUFFER,
                // HOST_SEQUENTIAL_WRITE is essential for the CPU to stream data to the GPU every frame
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window.inner_size().into(),
            depth_range: 0.0..=1.0,
        };

        let previous_frame_end = Some(sync::now(device).boxed());

        Ok(Self {
            start_time: Instant::now(),
            triangle_pipeline,
            render_pass,
            framebuffers,
            command_buffer_allocator,
            descriptor_set_allocator,
            storage_allocator,
            previous_frame_end,
            viewport,
        })
    }

    /// Records and submits the rendering commands for the current frame
    pub fn do_pass(
        &mut self,
        snapshots: &SnapshotPair,
        swapchain: Arc<Swapchain>,
        queue: Arc<Queue>,
    ) -> GameEngineResult<PassResult> {
        let time = self.start_time.elapsed().as_secs_f32();
        let window_size = self.viewport.extent;

        // Clean up resources from finished frames
        if let Some(future) = self.previous_frame_end.as_mut() {
            future.cleanup_finished();
        }

        // Acquire the next image from the swapchain
        let (image_index, mut suboptimal, acquire_future) =
            match acquire_next_image(swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => return Ok(PassResult::SwapchainOutOfDate),
                Err(e) => return Err(e.into()),
            };

        // --- 1. UPLOAD DATA ---
        // Map simulation data into GPU-side Storage Buffers (SSBOs)
        let old_buffer = self.upload_contents(&snapshots.old.triangles)?;
        let new_buffer = self.upload_contents(&snapshots.new.triangles)?;
        let alive_buffer = self.upload_indices(&snapshots.new.triangles.active_indices)?;

        // --- 2. DESCRIPTOR SETS ---
        // Set 0: Global Snapshots (Binding 0=Old, 1=New, 2=Alive)
        let layout = &self.triangle_pipeline.pipeline().layout().set_layouts()[0];
        let set0 = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, old_buffer),
                WriteDescriptorSet::buffer(1, new_buffer),
                WriteDescriptorSet::buffer(2, alive_buffer),
            ],
            [],
        )?;

        // --- 3. RECORD COMMANDS ---
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        // Calculate the smooth interpolation factor for the GPU
        let factor = snapshots.interpolation_factor();

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(
                        self.framebuffers[image_index as usize].clone(),
                    )
                },
                SubpassBeginInfo::default(),
            )?
            .set_viewport(0, [self.viewport.clone()].into_iter().collect())?
            .bind_pipeline_graphics(self.triangle_pipeline.pipeline().clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.triangle_pipeline.pipeline().layout().clone(),
                0,
                set0,
            )?
            .push_constants(
                self.triangle_pipeline.pipeline().layout().clone(),
                0,
                PushConstants {
                    factor,
                    position_offset: [0.0, 0.0].into(),
                    time,
                    resolution: window_size.into(),
                },
            )?;

        // Execute the draw command from the concrete pipeline
        let instance_count = snapshots.new.triangles.active_indices.len() as u32;
        self.triangle_pipeline
            .record_draw(&mut builder, instance_count);

        builder.end_render_pass(SubpassEndInfo::default())?;

        // --- 4. EXECUTION ---
        let command_buffer = builder.build()?;
        let future = self
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(queue.clone(), command_buffer)?
            .then_swapchain_present(
                queue,
                SwapchainPresentInfo::swapchain_image_index(swapchain, image_index),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(f) => {
                self.previous_frame_end = Some(f.boxed());
            }
            Err(VulkanError::OutOfDate) => {
                suboptimal = true;
                self.previous_frame_end =
                    Some(sync::now(self.triangle_pipeline.pipeline().device().clone()).boxed());
            }
            Err(e) => return Err(e.into()),
        }

        Ok(if suboptimal {
            PassResult::SwapchainOutOfDate
        } else {
            PassResult::Success
        })
    }

    /// Helper to allocate and fill a Storage Buffer slice from a generic Vec
    fn upload_contents<T: Pod + BufferContents + Send>(
        &self,
        contents: &RenderPacketContents<T>,
    ) -> GameEngineResult<vulkano::buffer::Subbuffer<[T]>> {
        // We upload the entire "database" of triangles
        let subbuffer = self
            .storage_allocator
            .allocate_slice(contents.data.len() as u64)?;
        {
            let mut writer = subbuffer.write()?;
            writer.copy_from_slice(&contents.data);
        }
        Ok(subbuffer)
    }

    /// Helper to allocate and fill the active index buffer (the indirection list)
    fn upload_indices(
        &self,
        indices: &[u32],
    ) -> GameEngineResult<vulkano::buffer::Subbuffer<[u32]>> {
        let subbuffer = self
            .storage_allocator
            .allocate_slice(indices.len() as u64)?;
        {
            let mut writer = subbuffer.write()?;
            writer.copy_from_slice(indices);
        }
        Ok(subbuffer)
    }

    /// Handles window resizing by recreating framebuffers
    pub fn resize(&mut self, images: &[Arc<Image>]) {
        self.framebuffers = window_size_dependent_setup(images, &self.render_pass);
        // We also update the viewport dimensions for the next draw
        if let Some(image) = images.first() {
            let extent = image.extent();
            self.viewport.extent = [extent[0] as f32, extent[1] as f32];
        }
    }
}

pub enum PassResult {
    Success,
    SwapchainOutOfDate,
}

/// Creates the Framebuffer objects for each swapchain image
fn window_size_dependent_setup(
    images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = vulkano::image::view::ImageView::new_default(image.clone()).unwrap();

            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}
