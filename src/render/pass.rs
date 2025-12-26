use game_engine_shaders_types::packet::{
    CIRCLE_SHAPE_INDEX, InstancePointer, TRIANGLE_SHAPE_INDEX,
};
use std::sync::Arc;
use std::time::Instant;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo,
    SubpassEndInfo,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::{Device, Queue};
use vulkano::image::Image;
use vulkano::memory::allocator::{MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, ColorBlendAttachmentState, ColorBlendState,
};
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass};
use vulkano::swapchain::{Swapchain, SwapchainPresentInfo, acquire_next_image};
use vulkano::sync::{self, GpuFuture};
use vulkano::{Validated, VulkanError, single_pass_renderpass};
use winit::window::Window;

use crate::GameEngineResult;
use crate::render::packet::SnapshotPair;
use crate::render::shaders;
use crate::render::storage::{RenderPacket, RenderSystem};
use game_engine_shaders_types::{PushConstants, create_camera_matrix};

pub struct PassManager {
    // REFACTOR: The RenderSystem now manages all GPU data channels.
    system: RenderSystem,
    // REFACTOR: The GraphicsPipeline is now a single, shared resource.
    pipeline: Arc<GraphicsPipeline>,

    new_packet: RenderPacket,

    snapshots: SnapshotPair,
    pub(crate) render_pass: Arc<RenderPass>,
    pub(crate) framebuffers: Vec<Arc<Framebuffer>>,
    pub(crate) command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub(crate) descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub(crate) memory_allocator: Arc<StandardMemoryAllocator>,
    pub(crate) storage_allocator: SubbufferAllocator,
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
        let render_pass = single_pass_renderpass!(
            device.clone(),
            attachments: { color: { format: swapchain.image_format(), samples: 1, load_op: Clear, store_op: Store } },
            pass: { color: [color], depth_stencil: {} },
        )?;

        let framebuffers = window_size_dependent_setup(images, &render_pass);

        let pipeline = {
            let vs = shaders::game_engine_shaders::load(device.clone())?
                .entry_point("main_vs")
                .unwrap();
            let fs = shaders::game_engine_shaders::load(device.clone())?
                .entry_point("main_fs")
                .unwrap();
            let stages = vec![
                vulkano::pipeline::PipelineShaderStageCreateInfo::new(vs),
                vulkano::pipeline::PipelineShaderStageCreateInfo::new(fs),
            ];
            let layout = vulkano::pipeline::PipelineLayout::new(
                device.clone(),
                vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo::from_stages(
                    &stages,
                )
                .into_pipeline_layout_create_info(device.clone())?,
            )?;

            GraphicsPipeline::new(
                device.clone(),
                None,
                vulkano::pipeline::graphics::GraphicsPipelineCreateInfo {
                    stages: stages.into(),
                    vertex_input_state: Some(Default::default()),
                    input_assembly_state: Some(Default::default()),
                    viewport_state: Some(
                        vulkano::pipeline::graphics::viewport::ViewportState::default(),
                    ),
                    rasterization_state: Some(
                        vulkano::pipeline::graphics::rasterization::RasterizationState::default(),
                    ),
                    multisample_state: Some(Default::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        1,
                        ColorBlendAttachmentState {
                            blend: Some(AttachmentBlend::alpha()),
                            ..Default::default()
                        },
                    )),
                    dynamic_state: [vulkano::pipeline::DynamicState::Viewport]
                        .into_iter()
                        .collect(),
                    subpass: Some(
                        vulkano::render_pass::Subpass::from(render_pass.clone(), 0)
                            .unwrap()
                            .into(),
                    ),
                    ..vulkano::pipeline::graphics::GraphicsPipelineCreateInfo::layout(layout)
                },
            )?
        };

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let storage_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        Ok(Self {
            system: RenderSystem::new(memory_allocator.clone()), // REFACTOR
            pipeline,
            new_packet: RenderPacket::new(),
            snapshots: SnapshotPair::new_empty(),
            render_pass,
            framebuffers,
            command_buffer_allocator: Arc::new(StandardCommandBufferAllocator::new(
                device.clone(),
                Default::default(),
            )),
            descriptor_set_allocator: Arc::new(StandardDescriptorSetAllocator::new(
                device.clone(),
                Default::default(),
            )),
            memory_allocator,
            storage_allocator,
            previous_frame_end: Some(sync::now(device).boxed()),
            viewport: Viewport {
                offset: [0.0, 0.0],
                extent: window.inner_size().into(),
                depth_range: 0.0..=1.0,
            },
            start_time: Instant::now(),
        })
    }

    pub fn push_packet(&mut self, packet: RenderPacket) {
        self.snapshots.push_new(packet.snapped_at);
        self.new_packet = packet;
        self.system.mark_all_dirty();
    }

    pub fn do_pass(
        &mut self,
        swapchain: Arc<Swapchain>,
        queue: Arc<Queue>,
        _transfer_queue: Arc<Queue>, // Removing this for now to solve the lock issue
        camera_pos: glam::Vec2,
        zoom: f32,
    ) -> GameEngineResult<PassResult> {
        // Reset the future timeline
        self.previous_frame_end = Some(sync::now(queue.device().clone()).boxed());
        // --- 0. CLEANUP ---
        if let Some(future) = self.previous_frame_end.as_mut() {
            future.cleanup_finished();
        }

        let (image_index, mut suboptimal, acquire_future) =
            match acquire_next_image(swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => return Ok(PassResult::SwapchainOutOfDate),
                Err(e) => return Err(e.into()),
            };

        // --- 1. DATA PREP ---
        let mut instance_map: Vec<InstancePointer> = self
            .new_packet
            .triangles
            .active_indices
            .iter()
            .map(|&storage_index| InstancePointer {
                shape_type: TRIANGLE_SHAPE_INDEX,
                local_index: storage_index,
            })
            .collect();

        instance_map.extend(
            self.new_packet
                .circles
                .active_indices
                .iter()
                .map(|&storage_index| InstancePointer {
                    shape_type: CIRCLE_SHAPE_INDEX,
                    local_index: storage_index,
                }),
        );

        if instance_map.is_empty() {
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

        let future = self.previous_frame_end.take().unwrap().join(acquire_future);

        // --- 2. SINGLE COMMAND BUFFER ---
        // We record BOTH uploads and draws into the same primary command buffer.
        // This solves the locking issue because Vulkano sees them as a sequence
        // in a single timeline and inserts the correct barriers automatically.
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        // A. Record Uploads
        let map_buffer = self.system.upload_all(
            self.memory_allocator.clone(),
            &self.storage_allocator,
            &mut builder, // Using the SAME builder
            &self.new_packet,
            &instance_map,
        )?;

        // B. Create Descriptors
        let descriptor_sets = self.system.create_descriptor_sets(
            &self.pipeline,
            self.descriptor_set_allocator.clone(),
            map_buffer,
        )?;

        let push_constants = PushConstants {
            view_proj: create_camera_matrix(
                self.viewport.extent[0],
                self.viewport.extent[1],
                camera_pos,
                zoom,
            ),
            time: self.start_time.elapsed().as_secs_f32(),
            factor: self.snapshots.interpolation_factor(),
            ..Default::default()
        };

        // C. Record Draw
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.1, 0.1, 0.1, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(
                        self.framebuffers[image_index as usize].clone(),
                    )
                },
                SubpassBeginInfo::default(),
            )?
            .set_viewport(0, [self.viewport.clone()].into_iter().collect())?
            .bind_pipeline_graphics(self.pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                descriptor_sets,
            )?
            .push_constants(self.pipeline.layout().clone(), 0, push_constants)?;

        unsafe {
            builder.draw(6, instance_map.len() as u32, 0, 0)?;
        }

        builder.end_render_pass(SubpassEndInfo::default())?;

        let command_buffer = builder.build()?;

        // --- 3. EXECUTION ---
        let future = future
            .then_execute(queue.clone(), command_buffer)?
            .then_swapchain_present(
                queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(swapchain, image_index),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(f) => {
                self.previous_frame_end = Some(f.boxed());
            }
            Err(VulkanError::OutOfDate) => {
                suboptimal = true;
                self.previous_frame_end = Some(sync::now(queue.device().clone()).boxed());
            }
            Err(e) => return Err(e.into()),
        }

        Ok(if suboptimal {
            PassResult::SwapchainOutOfDate
        } else {
            PassResult::Success
        })
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
