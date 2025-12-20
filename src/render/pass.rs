use bytemuck::Pod;
use game_engine_shaders_types::packet::GpuTrianglePacket;
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
use crate::render::packet::{RenderPacket, RenderPacketContents, SnapshotPair};
use crate::render::pipeline::GpuPipeline;
use crate::render::pipeline::triangle::TrianglePipeline;
use crate::render::storage::{InterpolationCache, PingPongBuffer};
use game_engine_shaders_types::PushConstants;

pub struct PassManager {
    /// The specialized pipeline for drawing triangles
    triangle_pipeline: TrianglePipeline,
    triangle_buffer_mgr: PingPongBuffer<GpuTrianglePacket>,
    triangle_buf_cache: Option<InterpolationCache<GpuTrianglePacket>>,

    snapshots: SnapshotPair,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    memory_allocator: Arc<StandardMemoryAllocator>,
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
                buffer_usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
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

        let triangle_buffer_mgr = PingPongBuffer::new(memory_allocator.clone(), 1024)?;

        let snapshots = SnapshotPair::new_empty();

        Ok(Self {
            start_time: Instant::now(),
            triangle_pipeline,
            triangle_buffer_mgr,
            triangle_buf_cache: None,
            snapshots,
            render_pass,
            framebuffers,
            command_buffer_allocator,
            descriptor_set_allocator,
            memory_allocator,
            storage_allocator,
            previous_frame_end,
            viewport,
        })
    }

    pub fn push_packet(&mut self, packet: RenderPacket) {
        self.snapshots.push_new(packet);
        self.triangle_buf_cache = None; // Invalidate cache
    }

    /// Records and submits the rendering commands for the current frame
    pub fn do_pass(
        &mut self,
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

        if self.snapshots.new.triangles.active_indices.is_empty() {
            // Nothing to draw this frame
            self.previous_frame_end = Some(
                self.previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_swapchain_present(
                        queue,
                        SwapchainPresentInfo::swapchain_image_index(swapchain, image_index),
                    )
                    .then_signal_fence_and_flush()?
                    .boxed(),
            );
            return Ok(PassResult::Success);
        }

        // 1. PREPARE UPLOADS
        // We separate the upload commands into their own buffer.
        // If we have a cached state (no new data), this buffer remains None.
        let (old_buffer, new_buffer, alive_buffer, upload_buffer) = if let Some(cache) =
            self.triangle_buf_cache.clone()
        {
            (
                cache.old_buffer,
                cache.new_buffer,
                cache.alive_indices,
                None,
            )
        } else {
            let mut upload_builder = AutoCommandBufferBuilder::primary(
                self.command_buffer_allocator.clone(),
                queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )?;

            // Record copies (transfer commands)
            let (old_buffer, new_buffer) = self.triangle_buffer_mgr.prepare_frame(
                self.memory_allocator.clone(),
                &self.storage_allocator,
                &mut upload_builder,
                &self.snapshots.new.triangles.data,
                &self.snapshots.new.triangles.newly_spawned_indices,
            )?;

            // Upload indices (writes to mapped memory, no command buffer needed usually,
            // but we do it here to keep flow consistent)
            let alive_buffer = self.upload_indices(&self.snapshots.new.triangles.active_indices)?;

            self.triangle_buf_cache = Some(InterpolationCache {
                old_buffer: old_buffer.clone(),
                new_buffer: new_buffer.clone(),
                alive_indices: alive_buffer.clone(),
            });

            (
                old_buffer,
                new_buffer,
                alive_buffer,
                Some(upload_builder.build()?),
            )
        };

        // 2. RECORD DRAW COMMANDS
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        // --- 2. DESCRIPTOR SETS ---
        let layout = &self.triangle_pipeline.pipeline().layout().set_layouts()[0];
        let set0 = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, old_buffer), // Binding 0
                WriteDescriptorSet::buffer(1, new_buffer), // Binding 1
                WriteDescriptorSet::buffer(2, alive_buffer),
            ],
            [],
        )?;

        // Calculate the smooth interpolation factor for the GPU
        let factor = self.snapshots.interpolation_factor();
        println!("Interpolation Factor: {}", factor);

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
        let instance_count = self.snapshots.new.triangles.active_indices.len() as u32;
        self.triangle_pipeline
            .record_draw(&mut builder, instance_count);

        builder.end_render_pass(SubpassEndInfo::default())?;

        // --- 4. EXECUTION ---
        let command_buffer = builder.build()?;

        // Start with the previous frame's end + image acquisition
        let future = self.previous_frame_end.take().unwrap().join(acquire_future);

        // If we have uploads, execute them FIRST
        let future = if let Some(upload_buffer) = upload_buffer {
            future.then_execute(queue.clone(), upload_buffer)?.boxed()
        } else {
            future.boxed()
        };

        // Then execute the draw commands and present
        let future = future
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
