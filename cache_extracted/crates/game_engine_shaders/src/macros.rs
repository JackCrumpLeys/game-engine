#[macro_export]
macro_rules! define_renderer {
    (
        $( $id:path => $name:ident : $packet:ty [ $b_old:literal, $b_new:literal ] ),* $(,)?
    ) => {

        // --- BOUNDS RESOLVER (Vertex Shader) ---
        // Iterates the operation stack for an entity and accumulates the bounding box.
        fn resolve_bounds(
            start: u32,
            len: u32,
            op_pool: &[InstancePointer],
            factor: f32,
            $(
                paste::paste!([<old_ $name>]): &[$packet],
                paste::paste!([<new_ $name>]): &[$packet],
            )*
        ) -> crate::sdf::Bounds {
            let mut b = crate::sdf::Bounds::empty();

            let mut i: u32 = 0;
            while i < len {
                let ptr = op_pool[(start + i) as usize];

                $(
                    if ptr.shape_type == $id {
                        let item = crate::fetch(
                            ptr.local_index,
                            paste::paste!([<old_ $name>]),
                            paste::paste!([<new_ $name>]),
                            factor,
                        );
                        b = <$packet as crate::sdf::ComputeBounds>::compute_bounds(&item, b);
                    }
                )*

                i += 1;
            }
            b
        }

        // --- COLOR RESOLVER (Fragment Shader) ---
        // Iterates the operation stack and applies each SDF operation sequentially.
        fn resolve_color(
            start: u32,
            len: u32,
            op_pool: &[InstancePointer],
            uv: Vec2,
            factor: f32,
            $(
                paste::paste!([<old_ $name>]): &[$packet],
                paste::paste!([<new_ $name>]): &[$packet],
            )*
        ) -> Vec4 {
            let mut res = crate::sdf::SdfResult::empty();

            let mut i: u32 = 0;
            while i < len {
                let ptr = op_pool[(start + i) as usize];

                $(
                    if ptr.shape_type == $id {
                        let item = crate::fetch(
                            ptr.local_index,
                            paste::paste!([<old_ $name>]),
                            paste::paste!([<new_ $name>]),
                            factor,
                        );
                        res = <$packet as crate::sdf::ComputeSdf>::apply(&item, uv, res);
                    }
                )*

                i += 1;
            }

            res.color
        }

        // --- VERTEX SHADER ---
        #[spirv(vertex)]
        pub fn main_vs(
            #[spirv(instance_index)] idx: usize,
            #[spirv(vertex_index)] vert_idx: usize,

            // Descriptor Set 1: Entity headers + operation pool
            #[spirv(storage_buffer, descriptor_set = 1, binding = 0)]
            entity_headers: &[GpuEntityHead],

            #[spirv(storage_buffer, descriptor_set = 1, binding = 1)]
            op_pool: &[InstancePointer],

            // Descriptor Set 0: Per-shape-type data buffers (ping-pong)
            $(
                #[spirv(storage_buffer, descriptor_set = 0, binding = $b_old)]
                paste::paste!([<old_ $name>]): &[$packet],

                #[spirv(storage_buffer, descriptor_set = 0, binding = $b_new)]
                paste::paste!([<new_ $name>]): &[$packet],
            )*

            #[spirv(push_constant)] constants: &PushConstants,

            // Outputs
            #[spirv(position)] out_pos: &mut Vec4,
            out_uv: &mut Vec2,
            #[spirv(flat)] out_instance_idx: &mut u32,
        ) {
            let head = entity_headers[idx];
            let factor = constants.factor;

            let bounds = resolve_bounds(
                head.stack_start, head.stack_len, op_pool, factor,
                $( paste::paste!([<old_ $name>]), paste::paste!([<new_ $name>]) ),*
            );

            // Generate a Quad from 6 vertices (two triangles)
            let uv_norm = match vert_idx {
                0 => vec2(-1.0, -1.0),
                1 => vec2( 1.0, -1.0),
                2 => vec2( 1.0,  1.0),
                3 => vec2(-1.0, -1.0),
                4 => vec2( 1.0,  1.0),
                5 => vec2(-1.0,  1.0),
                _ => Vec2::ZERO,
            };

            let half_size = bounds.size * 0.5;
            let local_pos = uv_norm * half_size;
            let world_pos = bounds.center + local_pos;

            // Transform to clip space
            *out_pos = constants.view_proj * world_pos.extend(0.0).extend(1.0);

            // Pass absolute world coordinate to fragment shader
            *out_uv = world_pos;

            *out_instance_idx = idx as u32;
        }

        // --- FRAGMENT SHADER ---
        #[spirv(fragment)]
        pub fn main_fs(
            in_uv: Vec2,
            #[spirv(flat)] in_instance_idx: u32,

            // Descriptor Set 1: Entity headers + operation pool
            #[spirv(storage_buffer, descriptor_set = 1, binding = 0)]
            entity_headers: &[GpuEntityHead],

            #[spirv(storage_buffer, descriptor_set = 1, binding = 1)]
            op_pool: &[InstancePointer],

            // Descriptor Set 0: Per-shape-type data buffers (ping-pong)
            $(
                #[spirv(storage_buffer, descriptor_set = 0, binding = $b_old)]
                paste::paste!([<old_ $name>]): &[$packet],

                #[spirv(storage_buffer, descriptor_set = 0, binding = $b_new)]
                paste::paste!([<new_ $name>]): &[$packet],
            )*

            #[spirv(push_constant)] constants: &PushConstants,
            output: &mut Vec4,
        ) {
            let head = entity_headers[in_instance_idx as usize];
            let factor = constants.factor;

            *output = resolve_color(
                head.stack_start, head.stack_len, op_pool, in_uv, factor,
                $( paste::paste!([<old_ $name>]), paste::paste!([<new_ $name>]) ),*
            );
        }
    };
}
