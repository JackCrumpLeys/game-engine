use crate::render::{
    components::{GpuAllocations, Position, RenderStack, ShapeStack},
    storage::RenderPacket,
};
use game_engine_ecs::{
    app::{App, Plugin},
    prelude::ResMut,
    query::Changed,
    system::Query,
};
use game_engine_render::slot_allocator::GpuSlotAllocator;
use game_engine_shaders_types::packet::InstancePointer;
use std::sync::Arc;

pub struct RenderPlugin;

impl Plugin for RenderPlugin {
    fn build(&self, app: &mut App) {
        app.world
            .resources_mut()
            .get_or_insert_default::<GpuSlotAllocator>();
        app.world
            .resources_mut()
            .get_or_insert_default::<RenderPacket>();

        app.schedule.add_systems(
            (
                sync_gpu_data_system,   // Update bytes in primitive channels (Changed only)
                rebuild_op_pool_system, // Rebuild op_pool and entity_heads (Runs on all)
            ),
            &mut app.world,
        );
    }
}

/// 1. DATA SYNC: Updates the actual packet bytes using the macro.
/// Only uploads changed shapes to prevent scatter-copying the whole buffer every frame.
fn sync_gpu_data_system(
    query: Query<(&ShapeStack, &GpuAllocations), Changed<ShapeStack>>,
    mut render_packet: ResMut<RenderPacket>,
    mut allocator: ResMut<GpuSlotAllocator>,
) {
    let packet = &mut *render_packet;

    // Pre-resize all channels!
    macro_rules! pre_resize {
        ($( ($idx:expr, $const:ident, $channel:ident, $builder:ident, $enum:ident, $pack:ident) ),*) => {
            $(
                let old_len = packet.$channel.data.len();
                let new_len = allocator.max_slots[$idx as usize] as usize;
                if old_len < new_len {
                    std::sync::Arc::make_mut(&mut packet.$channel.data).resize(new_len, Default::default());
                    packet.$channel.newly_spawned_indices.extend((old_len as u32)..(new_len as u32));
                }
            )*
        };
    }
    game_engine_shaders_types::for_all_shapes!(pre_resize);

    query.for_each(|(shape_stack, allocs)| {
        for (i, shape) in shape_stack.0.iter().enumerate() {
            let slot = allocs.slots[i] as usize;
            let kind = shape.shape_index();

            // Mark dirty for GPU upload
            allocator.mark_dirty(kind, allocs.slots[i]);

            // Write the shape data to the typed channels
            macro_rules! write_shape {
                ($( ($idx:expr, $const:ident, $channel:ident, $builder:ident, $enum_name:ident, $packet_type:ident) ),*) => {
                    match shape {
                        $(
                            game_engine_shaders_types::packet::Shape::$enum_name(data_packet) => {
                                let data = std::sync::Arc::make_mut(&mut packet.$channel.data);
                                data[slot] = *data_packet;
                            }
                        )*
                    }
                };
            }
            game_engine_shaders_types::for_all_shapes!(write_shape);
        }
    });
}

/// 2. POOL & TRANSFORM SYNC: Rebuilds the op_pool, entity_heads, and transforms
fn rebuild_op_pool_system(
    query: Query<(&ShapeStack, &RenderStack, &GpuAllocations, &Position)>,
    mut render_packet: ResMut<RenderPacket>,
) {
    let packet = &mut *render_packet;
    let needs_rebuild = packet.op_pool.is_empty();

    // MOST IMPORTANT: detect deep clones
    log::debug!(
        "rebuild_op_pool: needs_rebuild={} op_pool_strong={} entity_heads_strong={} (>1 => make_mut CLONES)",
        needs_rebuild,
        Arc::strong_count(&packet.op_pool),
        Arc::strong_count(&packet.entity_heads)
    );

    let entity_heads = Arc::make_mut(&mut packet.entity_heads);
    let op_pool = Arc::make_mut(&mut packet.op_pool);

    // 2. Single-pass fused update
    query.for_each(|(shape_stack, stack, allocs, pos)| {
        let idx = stack.gpu_head_idx as usize;

        // Optimistic resize: only happens when entity count actually grows
        if idx >= entity_heads.len() {
            entity_heads.resize(idx + 1024, Default::default()); // Resize in chunks
        }

        let head = &mut entity_heads[idx];
        head.position = [pos.x, pos.y];

        if needs_rebuild {
            packet.op_pool_dirty = true;
            let op_start = op_pool.len() as u32;
            for (i, shape) in shape_stack.0.iter().enumerate() {
                op_pool.push(InstancePointer::new(shape.shape_index(), allocs.slots[i]));
            }
            head.stack_start = op_start;
            head.stack_len = stack.op_pool_len as u32;
        }
    });
}
