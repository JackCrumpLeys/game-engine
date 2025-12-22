#[macro_export]
macro_rules! define_renderer {
    (
        $( $id:path => $name:ident : $packet:ty [ $b_old:literal, $b_new:literal ] ),* $(,)?
    ) => {
        // --- VERTEX SHADER ---
        #[spirv(vertex)]
        pub fn main_vs(
            #[spirv(instance_index)] idx: usize,
            #[spirv(vertex_index)] vert_idx: usize,

            #[spirv(storage_buffer, descriptor_set = 1, binding = 0)]
            instance_map: &[InstancePointer],

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

            #[spirv(flat)] out_shape_type: &mut u32,
            #[spirv(flat)] out_local_index: &mut u32,
        ) {
            let ptr = instance_map[idx];
            let factor = constants.factor;

            let (center, size) = match ptr.shape_type {
                $(
                    $id => {
                        let packet = fetch(
                            ptr.local_index,
                            paste::paste!([<old_ $name>]),
                            paste::paste!([<new_ $name>]),
                            factor,
                        );
                        packet.bounds()
                    }
                ),*
                _ => (Vec2::ZERO, Vec2::ZERO),
            };

            // Generate a Quad from 6 vertices (Standard CCW)
            // Indices: 0, 1, 2 (First Tri) | 2, 1, 3 (Second Tri) -> (Actually 0,1,2, 0,2,3 for 6 verts)
            // This maps vert_idx 0..6 to corners of a square [-1, -1] to [1, 1]
            let uv_norm = match vert_idx {
                0 => vec2(-1.0, -1.0),
                1 => vec2( 1.0, -1.0),
                2 => vec2( 1.0,  1.0),
                3 => vec2(-1.0, -1.0),
                4 => vec2( 1.0,  1.0),
                5 => vec2(-1.0,  1.0),
                _ => Vec2::ZERO,
            };

            let half_size = size * 0.5;
            let local_pos = uv_norm * half_size;
            let world_pos = center + local_pos; // Calculate absolute screen position

            // Transform to Clip Space (NDC)
            *out_pos = constants.view_proj * world_pos.extend(0.0).extend(1.0);

            // Pass the absolute coordinate to the fragment shader
            *out_uv = world_pos;

            *out_shape_type = ptr.shape_type;
            *out_local_index = ptr.local_index;
        }

        // --- FRAGMENT SHADER ---
        #[spirv(fragment)]
        pub fn main_fs(
            in_uv: Vec2, // This is local_pos (offset in pixels from shape center)
            #[spirv(flat)] in_shape_type: u32,
            #[spirv(flat)] in_local_index: u32,

            $(
                #[spirv(storage_buffer, descriptor_set = 0, binding = $b_old)]
                paste::paste!([<old_ $name>]): &[$packet],

                #[spirv(storage_buffer, descriptor_set = 0, binding = $b_new)]
                paste::paste!([<new_ $name>]): &[$packet],
            )*

            #[spirv(push_constant)] constants: &PushConstants,
            output: &mut Vec4,
        ) {
            let factor = constants.factor;

            match in_shape_type {
                $(
                    $id => {
                        let packet = fetch(
                            in_local_index,
                            paste::paste!([<old_ $name>]),
                            paste::paste!([<new_ $name>]),
                            factor,
                        );
                        // in_uv is the relative pixel coordinate.
                        // shape logic usually adds this back to 'center' to get the world pixel pos.
                        *output = packet.pixel_color(in_uv, constants);
                    }
                ),*
                _ => {
                    *output = Vec4::ZERO;
                    spirv_std::arch::kill();
                }
            };
        }
    };
}
