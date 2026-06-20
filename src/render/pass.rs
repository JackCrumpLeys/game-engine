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
use vulkano::shader::ShaderStages;
use vulkano::swapchain::{Swapchain, SwapchainPresentInfo, acquire_next_image};
use vulkano::sync::{self, GpuFuture};
use vulkano::{Validated, VulkanError, single_pass_renderpass};
use winit::window::Window;

#[derive(Default, Debug)]
pub struct PassProfiler {
    pub frames: u32,
    pub cleanup: std::time::Duration,
    pub acquire: std::time::Duration,
    pub cmd_alloc: std::time::Duration,
    pub upload: std::time::Duration,
    pub desc_sets: std::time::Duration,
    pub record_draw: std::time::Duration,
    pub build_cmd: std::time::Duration,
    pub submit: std::time::Duration,
    pub total: std::time::Duration,
}

use game_engine_ecs::threading::FromWorldThread;
use game_engine_ecs::world::World;

use crate::GameEngineResult;
use crate::render::packet::SnapshotPair;
use crate::render::shaders;
use crate::render::storage::{RenderDelta, RenderPacket, RenderSystem};
use game_engine_shaders_types::{PushConstants, create_camera_matrix};

struct ThreadLocalAllocators {
    staging_allocator: SubbufferAllocator,
}

impl FromWorldThread for ThreadLocalAllocators {
    fn new_thread_local(_thread_id: usize, world: &World) -> Self {
        let memory_allocator = world
            .resources()
            .get::<Arc<StandardMemoryAllocator>>()
            .unwrap()
            .clone();

        Self {
            staging_allocator: SubbufferAllocator::new(
                memory_allocator,
                SubbufferAllocatorCreateInfo {
                    buffer_usage: BufferUsage::TRANSFER_SRC,
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST,
                    ..Default::default()
                },
            ),
        }
    }
}

pub struct PassManager {
    system: RenderSystem,
    pipeline: Arc<GraphicsPipeline>,

    new_delta: RenderDelta,
    dirty_per_channel: Option<Vec<Vec<u32>>>,

    needs_upload: bool,

    snapshots: SnapshotPair,
    pub(crate) render_pass: Arc<RenderPass>,
    pub(crate) framebuffers: Vec<Arc<Framebuffer>>,
    pub(crate) command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub(crate) descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub(crate) memory_allocator: Arc<StandardMemoryAllocator>,
    last_frame_fence: Option<vulkano::sync::future::FenceSignalFuture<Box<dyn GpuFuture>>>,
    viewport: Viewport,
    start_time: Instant,
    pub profiler: PassProfiler,
}

impl PassManager {
    pub fn new(
        device: Arc<Device>,
        swapchain: Arc<Swapchain>,
        images: &[Arc<Image>],
        window: Arc<Window>,
        world: &mut World,
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
            let mut layout_create_infos =
                vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo::from_stages(
                    &stages,
                );

            if let Some(set0) = layout_create_infos.set_layouts.get_mut(0) {
                for binding in set0.bindings.values_mut() {
                    binding.stages |= ShaderStages::VERTEX;
                }
            }

            if let Some(set2) = layout_create_infos.set_layouts.get_mut(2)
                && let Some(binding1) = set2.bindings.get_mut(&1)
            {
                binding1.descriptor_count = 1024;
                binding1.binding_flags |=
                    vulkano::descriptor_set::layout::DescriptorBindingFlags::PARTIALLY_BOUND;
                binding1.binding_flags |= vulkano::descriptor_set::layout::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT;
            }

            let layout = vulkano::pipeline::PipelineLayout::new(
                device.clone(),
                layout_create_infos.into_pipeline_layout_create_info(device.clone())?,
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

        // Add shared resources for thread-local creation
        world.resources_mut().insert(device.clone());
        world.resources_mut().insert(memory_allocator.clone());
        world
            .thread_local_res_mut()
            .insert(ThreadLocalAllocators::new_thread_local(0, world));

        Ok(Self {
            system: RenderSystem::new(memory_allocator.clone()),
            pipeline,
            new_delta: RenderDelta::default(),
            dirty_per_channel: None,
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
            last_frame_fence: None,
            viewport: Viewport {
                offset: [0.0, 0.0],
                extent: window.inner_size().into(),
                depth_range: 0.0..=1.0,
            },
            start_time: Instant::now(),
            needs_upload: true,
            profiler: PassProfiler::default(),
        })
    }

    pub fn push_packet(&mut self, mut delta: RenderDelta) {
        if self.needs_upload {
            self.snapshots.new = delta.snapped_at;
            delta.merge_previous(std::mem::take(&mut self.new_delta));
            self.new_delta = delta;
            self.dirty_per_channel = None;
        } else {
            self.snapshots.push_new(delta.snapped_at);
            self.new_delta = delta;
            self.dirty_per_channel = None;
            self.needs_upload = true;
        }
    }

    /// Push a packet with per-channel dirty slot lists.
    /// Only the listed slots in each channel will be uploaded.
    pub fn push_packet_dirty(&mut self, mut delta: RenderDelta, dirty: Vec<Vec<u32>>) {
        if self.needs_upload {
            self.snapshots.new = delta.snapped_at;
            delta.merge_previous(std::mem::take(&mut self.new_delta));
            self.new_delta = delta;

            if let Some(existing_dirty) = &mut self.dirty_per_channel {
                for (i, slots) in dirty.into_iter().enumerate() {
                    existing_dirty[i].extend(slots);
                    existing_dirty[i].sort_unstable();
                    existing_dirty[i].dedup();
                }
            } else {
                self.dirty_per_channel = Some(dirty);
            }
        } else {
            self.snapshots.push_new(delta.snapped_at);
            self.new_delta = delta;
            self.dirty_per_channel = Some(dirty);
            self.needs_upload = true;
        }
    }

    pub fn do_pass(
        &mut self,
        world: &World,
        swapchain: Arc<Swapchain>,
        queue: Arc<Queue>,
        _transfer_queue: Arc<Queue>,
        camera_pos: glam::Vec2,
        zoom: f32,
    ) -> GameEngineResult<PassResult> {
        #[cfg(feature = "tracy")]
        let _pass_span = tracy_client::span!("do_pass");

        log::trace!("do_pass frame start");
        log::trace!("do_pass start");
        let start_total = Instant::now();
        let mut t = Instant::now();

        // --- 0. SYNC: one frame in flight. Block render thread until prior frame's
        // GPU work is fully complete, then drop the fence to release its resources.
        // (Blocking here is free: the render thread has no other work; the ECS sim
        // thread runs independently.)
        if let Some(f) = self.last_frame_fence.take() {
            f.wait(None)?;
        }
        self.profiler.cleanup += t.elapsed();
        t = Instant::now();

        log::trace!("do_pass acquire_next_image pending");
        #[cfg(feature = "tracy")]
        let _acquire_span = tracy_client::span!("Acquire Swapchain Image");
        let (image_index, mut suboptimal, acquire_future) =
            match acquire_next_image(swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => return Ok(PassResult::SwapchainOutOfDate),
                Err(e) => return Err(e.into()),
            };
        #[cfg(feature = "tracy")]
        drop(_acquire_span);
        self.profiler.acquire += t.elapsed();
        t = Instant::now();

        let entity_count = self.system.total_entity_count;
        log::trace!("entity_count: {entity_count}");

        // --- 1. PREPARE COMMANDS ---
        let mut upload_builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        self.profiler.cmd_alloc += t.elapsed();
        t = Instant::now();

        // A. Record Uploads (data channels + entity_heads + op_pool)
        let thread_res = world.thread_local_res_mut();
        if !thread_res.contains::<ThreadLocalAllocators>() {
            thread_res.insert(ThreadLocalAllocators::new_thread_local(0, world));
        }
        let staging_allocator = &thread_res
            .get::<ThreadLocalAllocators>()
            .unwrap()
            .staging_allocator;

        let packet = world
            .resources()
            .get::<RenderPacket>()
            .expect("RenderPacket resource missing from world");

        // Only touch the GPU upload path (and advance interpolation heads) when the sim
        // thread actually delivered new data. On no-data frames we re-bind the existing
        // old/new buffers and redraw with a fresh interpolation factor.
        let did_upload = self.needs_upload;
        if did_upload {
            if let Some(dirty) = &self.dirty_per_channel {
                self.system.upload_all_dirty(
                    self.memory_allocator.clone(),
                    staging_allocator,
                    &mut upload_builder,
                    &packet,
                    &self.new_delta,
                    dirty,
                )?
            } else {
                self.system.upload_all(
                    self.memory_allocator.clone(),
                    staging_allocator,
                    &mut upload_builder,
                    &packet,
                    &self.new_delta,
                )?
            };

            self.needs_upload = false;
            self.new_delta = RenderDelta::default();
            self.dirty_per_channel = None;
        }

        log::trace!(
            "do_pass: entity_count={} head_index={} eh.head={} op.head={} path.head={} did_upload={}",
            self.system.total_entity_count,
            self.system.head_index,
            self.system.entity_heads.head,
            self.system.op_pool.head,
            self.system.path_curves.head,
            did_upload
        );

        self.profiler.upload += t.elapsed();
        t = Instant::now();

        // Build the frame's future. Include the upload command buffer only if we recorded one.
        let future: Box<dyn GpuFuture> = if did_upload {
            let upload_cmd = upload_builder.build()?;
            sync::now(queue.device().clone())
                .boxed()
                .then_execute(queue.clone(), upload_cmd)?
                .join(acquire_future)
                .boxed()
        } else {
            // No uploads recorded; drop the empty builder and just join the acquire.
            drop(upload_builder);
            sync::now(queue.device().clone())
                .boxed()
                .join(acquire_future)
                .boxed()
        };

        let descriptor_sets = self
            .system
            .get_descriptor_sets(&self.pipeline, self.descriptor_set_allocator.clone())?;
        self.profiler.desc_sets += t.elapsed();
        t = Instant::now();

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
        log::trace!("do_pass Record Draw");
        #[cfg(feature = "tracy")]
        let _cmd_span = tracy_client::span!("Record Draw");

        let mut draw_builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        draw_builder
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
                descriptor_sets.to_vec(),
            )?
            .push_constants(self.pipeline.layout().clone(), 0, push_constants)?;

        // One draw call: 6 vertices (quad) per entity, entity_count instances
        log::trace!("do_pass builder draw");
        let entity_count = self.system.total_entity_count;
        if entity_count > 0 {
            unsafe {
                draw_builder.draw(6, entity_count as u32, 0, 0)?;
            }
        }

        draw_builder.end_render_pass(SubpassEndInfo::default())?;
        self.profiler.record_draw += t.elapsed();
        t = Instant::now();

        let draw_cmd = draw_builder.build()?;
        self.profiler.build_cmd += t.elapsed();
        t = Instant::now();
        #[cfg(feature = "tracy")]
        drop(_cmd_span);

        log::trace!("do_pass Record Draw done");

        // --- 3. EXECUTION ---
        log::trace!("do_pass submit command buffer");
        #[cfg(feature = "tracy")]
        let _submit_span = tracy_client::span!("Submit & Present");
        let future = future
            .then_execute(queue.clone(), draw_cmd)?
            .then_swapchain_present(
                queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(swapchain, image_index),
            )
            .boxed()
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(f) => {
                log::trace!("do_pass executed success");
                self.last_frame_fence = Some(f);
                log::trace!(
                    "do_pass frame submitted | head_index={}",
                    self.system.head_index
                );
            }
            Err(VulkanError::OutOfDate) => {
                log::trace!("OutOfDate: dropping fence — chain broken!");
                suboptimal = true;
                self.last_frame_fence = None;
            }
            Err(e) => {
                log::trace!("do_pass executed error: {e:?}");
                return Err(e.into());
            }
        }

        self.profiler.submit += t.elapsed();
    }
}
