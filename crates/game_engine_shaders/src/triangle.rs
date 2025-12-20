use core::f32::consts::PI;
use game_engine_shaders_types::PushConstants;
use game_engine_shaders_types::packet::GpuTrianglePacket;
use game_engine_shaders_types::packet::Interpolate;
use game_engine_shaders_types::vec4;
use spirv_std::glam::{Vec2, Vec3, Vec4};
use spirv_std::num_traits::Float;
use spirv_std::spirv;

#[spirv(fragment)]
pub fn main_fs(
    #[spirv(frag_coord)] frag_coord: Vec4,
    #[spirv(push_constant)] constants: &PushConstants,
    output: &mut Vec4,
) {
    *output = vec4(0.3, 0.6, 0.9, 1.0);
}

#[spirv(vertex)]
pub fn main_vs(
    #[spirv(instance_index)] tri_idx: usize, // Which triangle
    #[spirv(vertex_index)] vert_idx: usize,  // 0, 1, or 2 (the corner)

    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] old_buffer: &[GpuTrianglePacket],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] new_buffer: &[GpuTrianglePacket],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] alive_indices: &[usize],

    #[spirv(push_constant)] constants: &PushConstants,
    #[spirv(position)] out_pos: &mut Vec4,
) {
    // Look up the specific triangle in the database
    let render_idx = alive_indices[tri_idx];

    // Lerp the ENTIRE triangle from its old state to its new state
    let tri = old_buffer[render_idx].interpolate(&new_buffer[render_idx], constants.factor);

    // Pick the specific vertex for this shader thread
    let position = tri.vertices[vert_idx];

    let pos = position + constants.position_offset;
    *out_pos = pos.extend(0.0).extend(1.0);
}
