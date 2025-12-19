use std::collections::HashSet;
use std::sync::Arc;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::device::Device;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::{GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::render_pass::Subpass;

use crate::GameEngineResult;
use crate::render::pipeline::GpuPipeline;
use crate::render::shaders;

pub struct TrianglePipeline {
    pipeline: Arc<GraphicsPipeline>,
}

impl GpuPipeline for TrianglePipeline {
    fn new(device: Arc<Device>, subpass: Subpass) -> GameEngineResult<Self> {
        // Use the macro-generated shader loader from your build artifacts
        let vs = shaders::game_engine_shaders::load(device.clone())?
            .entry_point("triangle::main_vs")
            .unwrap();
        let fs = shaders::game_engine_shaders::load(device.clone())?
            .entry_point("triangle::main_fs")
            .unwrap();

        let stages = vec![
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];

        let layout = PipelineLayout::new(
            device.clone(),
            vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())?,
        )?;

        let pipeline = GraphicsPipeline::new(
            device,
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into(),
                // Vertex Pulling: Empty input state
                vertex_input_state: Some(VertexInputState::default()),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState {
                    attachments: vec![ColorBlendAttachmentState::default()],
                    ..Default::default()
                }),
                dynamic_state: HashSet::from_iter(
                    vec![vulkano::pipeline::DynamicState::Viewport].into_iter(),
                ),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )?;

        Ok(Self { pipeline })
    }
    fn pipeline(&self) -> &Arc<GraphicsPipeline> {
        &self.pipeline
    }

    fn record_draw(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        instance_count: u32,
    ) {
        if instance_count > 0 {
            // Draw exactly 3 vertices (one triangle) per instance
            unsafe {
                builder.draw(3, instance_count, 0, 0).unwrap();
            }
        }
    }
}

// TBC: Updating PassManager to use these partitioned files
