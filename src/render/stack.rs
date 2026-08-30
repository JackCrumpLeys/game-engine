use crate::render::packet::RenderPacketContents;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Copy, Clone, Debug)]
pub enum FreedAllocation {
    StackId(u32),
    OpPoolRange { start: u32, len: u32 },
    Slot { shape_kind: u32, slot: u32 },
}

pub static FREED_ALLOCATIONS: std::sync::Mutex<Vec<FreedAllocation>> =
    std::sync::Mutex::new(Vec::new());

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct StackTypeId(pub usize);

static NEXT_STACK_TYPE_ID: AtomicUsize = AtomicUsize::new(0);

#[inline]
pub fn allocate_stack_type_id() -> StackTypeId {
    StackTypeId(NEXT_STACK_TYPE_ID.fetch_add(1, Ordering::Relaxed))
}

/// Kept in the Render Backend (Main Thread). Stores the physical GPU locations.
#[derive(Clone, Debug)]
pub struct StackAlloc {
    pub head_id: u32,
    pub op_pool_start: u32,
    pub op_pool_len: u32,
    /// The physical slot IDs in the primitive channels (in the exact order of the macro)
    pub slots: smallvec::SmallVec<(u32, u32), 8>,
}

pub trait ShapeKindProvider {
    const SHAPE_KIND: u32;
}

pub trait ChannelAccess<T> {
    fn get_channel_mut(&mut self) -> &mut RenderPacketContents<T>;
}

#[macro_export]
macro_rules! define_render_stack {
    // Single stack variant (legacy/convenience)
    (
        $struct_name:ident => [
            $( $field_name:ident : $packet_type:ident ),* $(,)?
        ]
    ) => {
        $crate::define_render_stack!(
            [< $struct_name Plugin >] => {
                $struct_name => [ $( $field_name : $packet_type ),* ]
            }
        );
    };

    // Multi-stack variant with explicit plugin name
    (
        $plugin_name:ident => {
            $(
                $struct_name:ident => [
                    $( $field_name:ident : $packet_type:ident ),* $(,)?
                ]
            ),* $(,)?
        }
    ) => {
        paste::paste! {
            $(
                // 1. THE LOGICAL DATA (Mutate this in your parallel systems!)
                #[derive(game_engine_derive::Component, Clone, Debug)]
                pub struct $struct_name {
                    position: glam::Vec2,
                    $( $field_name: $packet_type, )*
                    /// Bit 0: position, Bit 1..N: fields
                    pub(crate) dirty_mask: u32,
                }

                impl Default for $struct_name {
                    fn default() -> Self {
                        Self {
                            position: Default::default(),
                            $( $field_name: Default::default(), )*
                            dirty_mask: !0, // Start fully dirty to ensure first sync uploads everything
                        }
                    }
                }

                impl $struct_name {
                    pub fn position(&self) -> glam::Vec2 { self.position }
                    pub fn position_mut(&mut self) -> &mut glam::Vec2 {
                        self.dirty_mask |= 1;
                        &mut self.position
                    }

                    $crate::define_mut_methods!(1 << 1, $( $field_name : $packet_type ),*);
                }

                // 2. THE GPU POINTERS (Attached to the entity on spawn)
                #[derive(game_engine_derive::Component, Clone, Debug)]
                pub struct [< $struct_name Alloc >] {
                    pub head_idx: u32,
                    pub op_pool_start: u32,
                    $( pub [< $field_name _slot >]: u32, )*
                }

                impl Drop for [< $struct_name Alloc >] {
                    fn drop(&mut self) {
                        let head_idx = self.head_idx;
                        let op_pool_start = self.op_pool_start;
                        let len = 0 $(+ { let _ = stringify!($field_name); 1 })*;
                        let slots = [
                            $(
                                (
                                    <$packet_type as $crate::render::stack::ShapeKindProvider>::SHAPE_KIND,
                                    self.[< $field_name _slot >],
                                ),
                            )*
                        ];
                        if let Ok(mut freed) = $crate::render::stack::FREED_ALLOCATIONS.lock() {
                            freed.push($crate::render::stack::FreedAllocation::StackId(head_idx));
                            freed.push($crate::render::stack::FreedAllocation::OpPoolRange { start: op_pool_start, len });
                            for (kind, slot) in slots {
                                freed.push($crate::render::stack::FreedAllocation::Slot { shape_kind: kind, slot });
                            }
                        }
                    }
                }

                impl $struct_name {
                    pub fn new(position: glam::Vec2, $( $field_name : $packet_type ),* ) -> Self {
                        Self {
                            position,
                            $( $field_name, )*
                            dirty_mask: !0,
                        }
                    }

                    // Call this when spawning to reserve GPU memory!
                    pub fn spawn_alloc(
                        allocator: &mut game_engine_render::slot_allocator::GpuSlotAllocator,
                        op_pool: &mut Vec<game_engine_shaders_types::packet::InstancePointer>,
                    ) -> [< $struct_name Alloc >] {
                        let head_idx = allocator.alloc_stack_id();

                        let len = 0 $(+ { let _ = stringify!($field_name); 1 })*;
                        let op_pool_start = if let Some(start) = allocator.alloc_op_pool_range(len) {
                            start
                        } else {
                            let start = op_pool.len() as u32;
                            op_pool.resize(op_pool.len() + len as usize, Default::default());
                            start
                        };

                        let mut _idx = 0;
                        $(
                            let shape_kind = <$packet_type as $crate::render::stack::ShapeKindProvider>::SHAPE_KIND;
                            let [< $field_name _slot >] = allocator.alloc_slot(shape_kind);
                            op_pool[(op_pool_start + _idx) as usize] = game_engine_shaders_types::packet::InstancePointer::new(shape_kind, [< $field_name _slot >]);
                            _idx += 1;
                        )*

                        [< $struct_name Alloc >] {
                            head_idx,
                            op_pool_start,
                            $( [< $field_name _slot >], )*
                        }
                    }
                }

                // 3. THE SYNC SYSTEM (Runs sequentially at end of frame in O(N) flat time)
                pub fn [< sync_ $struct_name:snake _system >](
                    // ONLY runs on entities that actually changed!
                    mut query: game_engine_ecs::system::Query<
                        (&mut $struct_name, &[< $struct_name Alloc >]),
                        game_engine_ecs::query::Changed<$struct_name>
                    >,
                    mut packet_res: game_engine_ecs::resource::ResMut<$crate::render::storage::RenderPacket>,
                    mut allocator: game_engine_ecs::resource::ResMut<game_engine_render::slot_allocator::GpuSlotAllocator>,
                    channel_status: game_engine_ecs::resource::Res<$crate::render::storage::RenderChannelStatus>,
                    mut last_success_tick: game_engine_ecs::system::Local<game_engine_ecs::world::ChangeTick>,
                ) {
                    let mut freed_items = Vec::new();
                    if let Ok(mut freed) = $crate::render::stack::FREED_ALLOCATIONS.lock() {
                        if !freed.is_empty() {
                            freed_items = std::mem::take(&mut *freed);
                        }
                    }
                    if !freed_items.is_empty() {
                        let packet = &mut *packet_res;
                        let heads = std::sync::Arc::make_mut(&mut packet.entity_heads);
                        for item in freed_items {
                            match item {
                                $crate::render::stack::FreedAllocation::StackId(id) => {
                                    allocator.free_stack_id(id);
                                    if (id as usize) < heads.len() {
                                        heads[id as usize] = Default::default();
                                    }
                                }
                                $crate::render::stack::FreedAllocation::OpPoolRange { start, len } => {
                                    allocator.free_op_pool_range(start, len);
                                }
                                $crate::render::stack::FreedAllocation::Slot { shape_kind, slot } => {
                                    allocator.free_slot(shape_kind, slot);
                                }
                             }
                        }
                        packet_res.op_pool_dirty = true;
                    }

                    if channel_status.is_full {
                        return;
                    }

                    let packet = &mut *packet_res;

                    // Safety: We obtain mutable pointers to distinct fields of RenderPacket.
                    let packet_ptr = packet as *mut $crate::render::storage::RenderPacket;
                    let heads = unsafe { std::sync::Arc::make_mut(&mut (*packet_ptr).entity_heads) };

                    if heads.len() < allocator.stack_count as usize {
                        heads.resize(allocator.stack_count as usize, Default::default());
                    }

                    $(
                        let [< $field_name _ch >]: *mut _ = $crate::render::stack::ChannelAccess::<$packet_type>::get_channel_mut(unsafe { &mut *packet_ptr });
                        let [< $field_name _data >] = unsafe { std::sync::Arc::make_mut(&mut (*[< $field_name _ch >]).data) };
                        let [< $field_name _spawned >] = unsafe { &mut (*[< $field_name _ch >]).newly_spawned_indices };

                        let shape_kind = <$packet_type as $crate::render::stack::ShapeKindProvider>::SHAPE_KIND;
                        let old_len = [< $field_name _data >].len();
                        let new_len = allocator.max_slots[shape_kind as usize] as usize;
                        if old_len < new_len {
                            [< $field_name _data >].resize(new_len, Default::default());
                            [< $field_name _spawned >].extend((old_len as u32)..(new_len as u32));
                        }
                    )*

                    let mut last_tick = *last_success_tick;
                    if last_tick.0 == 0 {
                        last_tick = query.tick();
                    }

                    query.for_each_mut_since(last_tick, |(mut data, alloc)| {
                        let mask = data.dirty_mask;
                        if mask == 0 { return; }

                        // Bit 0: Position/Head update
                        if (mask & 1) != 0 {
                            heads[alloc.head_idx as usize] = game_engine_shaders_types::packet::GpuEntityHead {
                                stack_start: alloc.op_pool_start,
                                stack_len: 0 $(+ { let _ = stringify!($field_name); 1 })*,
                                position: [data.position.x, data.position.y],
                            };
                        }


                        // Bits 1..N: Field updates
                        $crate::sync_fields!(data, alloc, mask, 1 << 1, allocator, $( $field_name : [< $field_name _data >], [< $field_name _spawned >], $packet_type ),*);

                        // Reset dirty mask after successful sync
                        data.dirty_mask = 0;
                    });

                    *last_success_tick = query.tick();
                }
            )*

            pub struct $plugin_name;
            impl game_engine_ecs::app::Plugin for $plugin_name {
                fn build(&self, app: &mut game_engine_ecs::app::App) {
                    $(
                        app.add_systems([< sync_ $struct_name:snake _system >]);
                    )*
                }
            }
        }
    };
}

#[macro_export]
macro_rules! define_mut_methods {
    ($mask:expr, $field_name:ident : $packet_type:ident $(, $($rest_field:ident : $rest_type:ident),* )? ) => {
        ::paste::paste! {
            pub fn [<$field_name _mut>](&mut self) -> &mut $packet_type {
                self.dirty_mask |= $mask;
                &mut self.$field_name
            }
        }
        $(
            $crate::define_mut_methods!($mask << 1, $($rest_field : $rest_type),*);
        )?
    };
    ($mask:expr, ) => {};
}

#[macro_export]
macro_rules! sync_fields {
    ($data:ident, $alloc:ident, $mask:ident, $bit:expr, $allocator:ident, $field_name:ident : $data_vec:ident, $spawned_vec:ident, $packet_type:ident $(, $($rest_field:ident : $rest_data:ident, $rest_spawned:ident, $rest_type:ident),* )? ) => {
        if ($mask & $bit) != 0 {
            ::paste::paste! {
                let slot: usize = $alloc.[< $field_name _slot >] as usize;
                $data_vec[slot] = $data.$field_name;
                let shape_kind = <$packet_type as $crate::render::stack::ShapeKindProvider>::SHAPE_KIND;
                $allocator.mark_dirty(shape_kind, $alloc.[< $field_name _slot >]);
            }
        }
        $(
            $crate::sync_fields!($data, $alloc, $mask, $bit << 1, $allocator, $($rest_field : $rest_data, $rest_spawned, $rest_type),*);
        )?
    };
    ($data:ident, $alloc:ident, $mask:ident, $bit:expr, $allocator:ident, ) => {};
}
