use std::collections::HashSet;
use std::sync::Arc;

use game_engine_shaders_types::PushConstants;
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
/// Manegement of rendering passes
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo,
    SubpassContents, SubpassEndInfo,
};
use vulkano::device::{Device, Queue};
use vulkano::image::Image;
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::{MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::swapchain::{Swapchain, SwapchainPresentInfo, acquire_next_image};
use vulkano::sync::{self, GpuFuture};
use vulkano::{Validated, VulkanError};
use winit::window::Window;

use crate::GameEngineResult;
use crate::render::packet::{RenderPacket, VulVertex};

pub struct PassManager {
    start_time: std::time::Instant,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    pipeline: Arc<GraphicsPipeline>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    viewport: Viewport,
    subbuffers: Buffers,
    subbuffer_allocator: SubbufferAllocator,
}

// Eventully we will have much more of a multi file layout and proper render graph
pub struct Buffers {
    pub vertex_buffer: Subbuffer<[VulVertex]>,
}

impl PassManager {
    pub fn new(
        device: Arc<Device>,
        swapchain: Arc<Swapchain>,
        images: &[Arc<Image>],
        window: Arc<Window>,
    ) -> GameEngineResult<Self> {
        // The next step is to create a *render pass*, which is an object that describes where the
        // output of the graphics pipeline will go. It describes the layout of the images where the
        // colors, depth and/or stencil information will be written.
        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                // `color` is a custom name we give to the first and only attachment.
                color: {
                    // `format: <ty>` indicates the type of the format of the image. This has to be
                    // one of the types of the `vulkano::format` module (or alternatively one of
                    // your structs that implements the `FormatDesc` trait). Here we use the same
                    // format as the swapchain.
                    format: swapchain.image_format(),
                    // `samples: 1` means that we ask the GPU to use one sample to determine the
                    // value of each pixel in the color attachment. We could use a larger value
                    // (multisampling) for antialiasing. An example of this can be found in
                    // msaa-renderpass.rs.
                    samples: 1,
                    // `load_op: Clear` means that we ask the GPU to clear the content of this
                    // attachment at the start of the drawing.
                    load_op: Clear,
                    // `store_op: Store` means that we ask the GPU to store the output of the draw
                    // in the actual image. We could also ask it to discard the result.
                    store_op: Store,
                },
            },
            pass: {
                // We use the attachment named `color` as the one and only color attachment.
                color: [color],
                // No depth-stencil attachment is indicated with empty brackets.
                depth_stencil: {},
            },
        )?;

        // The render pass we created above only describes the layout of our framebuffers. Before
        // we can draw we also need to create the actual framebuffers.
        //
        // Since we need to draw to multiple images, we are going to create a different framebuffer
        // for each image.
        let framebuffers = window_size_dependent_setup(images, &render_pass);

        // print!(env!("game_engine_shaders.spv"));

        mod shaders {
            vulkano_shaders::shader! {
                bytes: "/home/jackc/Documents/code/rust/game_engine/target/spirv-builder/spirv-unknown-vulkan1.4/release/deps/game_engine_shaders.spv"
            }
        }

        // Before we draw, we have to create what is called a **pipeline**. A pipeline describes
        // how a GPU operation is to be performed. It is similar to an OpenGL program, but it also
        // contains many settings for customization, all baked into a single object. For drawing,
        // we create a **graphics** pipeline, but there are also other types of pipeline.
        let pipeline = {
            // First, we load the shaders that the pipeline will use: the vertex shader and the
            // fragment shader.
            //
            // A Vulkan shader can in theory contain multiple entry points, so we have to specify
            // which one.
            let vs = shaders::load(device.clone())?
                .entry_point("main_vs")
                .unwrap();
            let fs = shaders::load(device.clone())?
                .entry_point("main_fs")
                .unwrap();

            // Automatically generate a vertex input state from the vertex shader's input
            // interface, that takes a single vertex buffer containing `Vertex` structs.
            let vertex_input_state = VulVertex::per_vertex().definition(&vs)?;

            // Make a list of the shader stages that the pipeline will have.
            let stages = vec![
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            // We must now create a **pipeline layout** object, which describes the locations and
            // types of descriptor sets and push constants used by the shaders in the pipeline.
            //
            // Multiple pipelines can share a common layout object, which is more efficient. The
            // shaders in a pipeline must use a subset of the resources described in its pipeline
            // layout, but the pipeline layout is allowed to contain resources that are not present
            // in the shaders; they can be used by shaders in other pipelines that share the same
            // layout. Thus, it is a good idea to design shaders so that many pipelines have common
            // resource locations, which allows them to share pipeline layouts.
            //
            // Since we only have one pipeline in this example, and thus one pipeline layout, we
            // automatically generate the layout from the resources used in the shaders. In a real
            // application, you would specify this information manually so that you can re-use one
            // layout in multiple pipelines.
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(device.clone())?,
            )?;

            // We have to indicate which subpass of which render pass this pipeline is going to be
            // used in. The pipeline will only be usable from this particular subpass.
            let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

            // Finally, create the pipeline.
            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into(),
                    // How vertex data is read from the vertex buffers into the vertex shader.
                    vertex_input_state: Some(vertex_input_state),
                    // How vertices are arranged into primitive shapes. The default primitive shape
                    // is a triangle.
                    input_assembly_state: Some(InputAssemblyState::default()),
                    // How primitives are transformed and clipped to fit the framebuffer. We use a
                    // resizable viewport, set to draw over the entire window.
                    viewport_state: Some(ViewportState::default()),
                    // How polygons are culled and converted into a raster of pixels. The default
                    // value does not perform any culling.
                    rasterization_state: Some(RasterizationState::default()),
                    // How multiple fragment shader samples are converted to a single pixel value.
                    // The default value does not perform any multisampling.
                    multisample_state: Some(MultisampleState::default()),
                    // How pixel values are combined with the values already present in the
                    // framebuffer. The default value overwrites the old value with the new one,
                    // without any blending.
                    color_blend_state: Some(ColorBlendState {
                        attachments: vec![ColorBlendAttachmentState::default()],
                        ..Default::default()
                    }),
                    // Dynamic states allows us to specify parts of the pipeline settings when
                    // recording the command buffer, before we perform drawing. Here, we specify
                    // that the viewport should be dynamic.
                    dynamic_state: HashSet::from_iter(vec![DynamicState::Viewport].into_iter()),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )?
        };

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let subbuffer_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::VERTEX_BUFFER,
                // HOST_SEQUENTIAL_WRITE means: CPU writes once, GPU reads once.
                // Perfect for dynamic 2D geometry.
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        // Allocate a dummy empty buffer just so `self.vertex_buffer` has something valid to start with
        let vertex_buffer = subbuffer_allocator
            .allocate_slice::<VulVertex>(3)
            .expect("failed to allocate initial buffer");

        // Dynamic viewports allow us to recreate just the viewport when the window is resized.
        // Otherwise we would have to recreate the whole pipeline.
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window.inner_size().into(),
            depth_range: 0.0..=1.0,
        };

        let previous_frame_end = Some(sync::now(device.clone()).boxed());

        Ok(PassManager {
            viewport,
            render_pass,
            framebuffers,
            pipeline,
            command_buffer_allocator: Arc::new(StandardCommandBufferAllocator::new(
                device.clone(),
                Default::default(),
            )),
            previous_frame_end,
            start_time: std::time::Instant::now(),
            subbuffer_allocator,
            subbuffers: Buffers { vertex_buffer },
        })
    }

    pub fn resize(&mut self, images: &[Arc<Image>]) {
        self.framebuffers = window_size_dependent_setup(images, &self.render_pass);
    }

    pub fn load_packet(&mut self, packet: &RenderPacket) {
        let vertex_data = packet.vertex_buffer();

        if vertex_data.is_empty() {
            return;
        }

        // 1. Ask the allocator for a slice of memory big enough for our vertices
        let buffer = self
            .subbuffer_allocator
            .allocate_slice(vertex_data.len() as u64)
            .expect("Failed to allocate vertex buffer");

        // 2. Map the memory and write to it
        // syntax: .write() locks the CPU-side pointer to copy data
        {
            let mut buffer_map = buffer.write().unwrap();
            buffer_map.copy_from_slice(vertex_data);
        }

        // 3. Store it for the draw call
        // The allocator guarantees this new 'buffer' does NOT conflict with
        // what the GPU is currently reading from the previous frame.
        self.subbuffers.vertex_buffer = buffer;
    }

    pub fn do_pass(
        &mut self,
        swapchain: Arc<Swapchain>,
        device: Arc<Device>,
        queue: Arc<Queue>,
    ) -> GameEngineResult<PassResult> {
        let window_size = self.viewport.extent;
        let time = self.start_time.elapsed().as_secs_f32();

        if let Some(gpu_future) = self.previous_frame_end.as_mut() {
            gpu_future.cleanup_finished();
        }

        // Before we can draw on the output, we have to *acquire* an image from the
        // swapchain. If no image is available (which happens if you submit draw commands
        // too quickly), then the function will block. This operation returns the index of
        // the image that we are allowed to draw upon.
        //
        // This function can block if no image is available. The parameter is an optional
        // timeout after which the function call will return an error.
        let (image_index, mut suboptimal, acquire_future) =
            match acquire_next_image(swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    return Ok(PassResult::SwapchainOutOfDate);
                }
                Err(e) => {
                    return Err(Box::new(e));
                }
            };

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        builder
            .push_constants(
                self.pipeline.layout().clone(),
                0,
                PushConstants {
                    position_offset: [0.0, 0.0].into(),
                    resolution: window_size.into(),
                    time,
                },
            )?
            // Before we can draw, we have to *enter a render pass*.
            .begin_render_pass(
                RenderPassBeginInfo {
                    // A list of values to clear the attachments with. This list contains
                    // one item for each attachment in the render pass. In this case, there
                    // is only one attachment, and we clear it with a blue color.
                    //
                    // Only attachments that have `AttachmentLoadOp::Clear` are provided
                    // with clear values, any others should use `None` as the clear value.
                    clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],

                    ..RenderPassBeginInfo::framebuffer(
                        self.framebuffers[image_index as usize].clone(),
                    )
                },
                SubpassBeginInfo {
                    // The contents of the first (and only) subpass. This can be either
                    // `Inline` or `SecondaryCommandBuffers`. The latter is a bit more
                    // advanced and is not covered here.
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )?
            // We are now inside the first subpass of the render pass.
            .set_viewport(0, [self.viewport.clone()].into_iter().collect())?
            .bind_pipeline_graphics(self.pipeline.clone())?
            .bind_vertex_buffers(0, self.subbuffers.vertex_buffer.clone())?;

        // BEGIN RENDER PASS

        unsafe { builder.draw(self.subbuffers.vertex_buffer.len() as u32, 1, 0, 0) }?;

        // END RENDER PASS

        builder.end_render_pass(SubpassEndInfo::default())?;

        let command_buffer = builder.build()?;

        let future = self
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(queue.clone(), command_buffer)?
            // The color output is now expected to contain our triangle. But in order to
            // show it on the screen, we have to *present* the image by calling
            // `then_swapchain_present`.
            //
            // This function does not actually present the image immediately. Instead it
            // submits a present command at the end of the queue. This means that it will
            // only be presented once the GPU has finished executing the command buffer
            // that draws the triangle.
            .then_swapchain_present(
                queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                self.previous_frame_end = Some(future.boxed());
            }
            Err(VulkanError::OutOfDate) => {
                suboptimal = true;
                self.previous_frame_end = Some(sync::now(device.clone()).boxed());
            }
            Err(e) => {
                self.previous_frame_end = Some(sync::now(device.clone()).boxed());
                return Err(Box::new(e));
            }
        }

        Ok(if suboptimal {
            PassResult::SwapchainOutOfDate
        } else {
            PassResult::Success
        })
    }
}

pub enum PassResult {
    Success,
    SwapchainOutOfDate,
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(
    images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();

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
