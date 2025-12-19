pub(super) mod triangle;

use crate::GameEngineResult;
use crate::render::packet::SnapshotPair;
use std::sync::Arc;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::descriptor_set::DescriptorSet;
use vulkano::device::Device;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::Subpass;

/// A trait for any rendering pipeline that pulls data from Snapshot SSBOs.
pub trait GpuPipeline: Sized {
    /// Initializes the pipeline, loading shaders and creating the GPU state.
    fn new(device: Arc<Device>, subpass: Subpass) -> GameEngineResult<Self>;

    /// Returns the underlying Vulkano graphics pipeline.
    fn pipeline(&self) -> &Arc<GraphicsPipeline>;

    /// Records the draw call into the command buffer.
    /// The PassManager handles binding the descriptor sets;
    /// this handles the actual 'draw' command logic.
    fn record_draw(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        instance_count: u32,
    );
}
