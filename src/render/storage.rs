use crate::GameEngineResult;
use crate::render::packet::RenderPacketContents;
use bytemuck::Pod;
use game_engine_shaders_types::packet::*;
use std::fmt::Debug;
use std::sync::Arc;
use std::time::Instant;
use vulkano::buffer::allocator::SubbufferAllocator;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::CopyBufferToImageInfo;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, BufferCopy, CopyBufferInfo, PrimaryAutoCommandBuffer,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::DeviceOwned;
use vulkano::format::Format;
use vulkano::image::sampler::{
    Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode,
};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::Pipeline;


pub struct RenderChannelStatus {
    pub is_full: bool,
}

pub struct TripleBuffer<T: Pod + Send + Sync> {
    buffers: [Subbuffer<[T]>; 3],
    pub head: usize,
    capacity: u64,
}

impl<T: Pod + Send + Sync + Debug> TripleBuffer<T> {
    pub fn new(allocator: Arc<StandardMemoryAllocator>, capacity: u64) -> GameEngineResult<Self> {
        let usage =
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC;
        let alloc_info = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        };

        // Pre-allocate all 3 buffers once
        Ok(Self {
            buffers: [
                Buffer::new_slice(
                    allocator.clone(),
                    BufferCreateInfo {
                        usage,
                        ..Default::default()
                    },
                    alloc_info.clone(),
                    capacity,
                )?,
                Buffer::new_slice(
                    allocator.clone(),
                    BufferCreateInfo {
                        usage,
                        ..Default::default()
                    },
                    alloc_info.clone(),
                    capacity,
                )?,
                Buffer::new_slice(
                    allocator,
                    BufferCreateInfo {
                        usage,
                        ..Default::default()
                    },
                    alloc_info,
                    capacity,
                )?,
            ],
            head: 0,
            capacity,
        })
    }

    fn resize(
        &mut self,
        allocator: Arc<StandardMemoryAllocator>,
        needed: u64,
    ) -> GameEngineResult<bool> {
        let new_capacity = needed.next_power_of_two().max(self.capacity.max(64));
        if new_capacity <= self.capacity {
            return Ok(false);
        }

        log::debug!("RESIZE<{}> cap {} -> {} head={}",
            std::any::type_name::<T>(), self.capacity, new_capacity, self.head);

        let usage =
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC;
        let alloc_info = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        };

        let new_buffers = [
            Buffer::new_slice(allocator.clone(), BufferCreateInfo { usage, ..Default::default() }, alloc_info.clone(), new_capacity)?,
            Buffer::new_slice(allocator.clone(), BufferCreateInfo { usage, ..Default::default() }, alloc_info.clone(), new_capacity)?,
            Buffer::new_slice(allocator,        BufferCreateInfo { usage, ..Default::default() }, alloc_info,        new_capacity)?,
        ];
        // Safe to drop old buffers immediately: the single per-frame fence guarantees
        // the previous frame (the only one that could reference them) is fully complete.
        self.buffers = new_buffers;
        self.capacity = new_capacity;
        Ok(true)
    }

    pub fn prepare_frame_dirty(
        &mut self,
        allocator: Arc<StandardMemoryAllocator>,
        staging_allocator: &SubbufferAllocator,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        data: &[T],
        dirty_slots: &[u32],
        newly_spawned_indices: &[u32],
    ) -> GameEngineResult<bool> {
        let needed = data.len() as u64;
        let resized = if needed > self.capacity {
            self.resize(allocator, needed)?
        } else {
            false
        };

        log::trace!("TripleBuffer::advance head {} -> {}", self.head, (self.head + 1) % 3);
        self.head = (self.head + 1) % 3;
        log::trace!("advance<{}> head={} needed={} cap={}",
            std::any::type_name::<T>(), self.head, needed, self.capacity);

        if needed == 0 {
            return Ok(resized);
        }

        let new_idx = self.head;
        let old_idx = (self.head + 2) % 3;

        let old_buf = self.buffers[old_idx].clone();
        let new_buf = self.buffers[new_idx].clone();

        // 1. Establish baseline in New by copying Old
        builder.copy_buffer(CopyBufferInfo::buffers(
            old_buf.clone().slice(0..needed),
            new_buf.clone().slice(0..needed),
        ))?;

        // 2. Apply dirty updates to New
        if !dirty_slots.is_empty() {
            // BULK COPY IS SAFE HERE: `data` contains the full valid state for all slots.
            // Overwriting non-dirty slots in the gap with `data[slot]` writes the exact same value
            // that is already in `new_buf`, bypassing thousands of slow Vulkan copy regions.
            let min_slot = *dirty_slots.first().unwrap();
            let max_slot = *dirty_slots.last().unwrap();
            let range_count = (max_slot - min_slot + 1) as usize;

            let staging = staging_allocator.allocate_slice::<T>(range_count as u64)?;
            staging
                .write()?
                .copy_from_slice(&data[min_slot as usize..=max_slot as usize]);

            let elem_size = std::mem::size_of::<T>() as u64;
            let mut copy_info = CopyBufferInfo::buffers(staging, new_buf.clone());
            copy_info.regions.clear();
            copy_info.regions.push(BufferCopy {
                src_offset: 0,
                dst_offset: (min_slot as u64) * elem_size,
                size: (range_count as u64) * elem_size,
                ..Default::default()
            });
            builder.copy_buffer(copy_info)?;
        }

        // 3. Patch newly spawned entities in Old
        if !newly_spawned_indices.is_empty() {
            let mut spawned = newly_spawned_indices.to_vec();
            spawned.sort_unstable();
            spawned.dedup();

            let count = spawned.len() as u64;
            let patch_staging = staging_allocator.allocate_slice::<T>(count)?;
            let mut writer = patch_staging.write()?;
            let elem_size = std::mem::size_of::<T>() as u64;

            let mut copy_info = CopyBufferInfo::buffers(patch_staging.clone(), old_buf);
            copy_info.regions.clear();

            let mut current_start_i = 0;
            for i in 1..=spawned.len() {
                if i == spawned.len() || spawned[i] != spawned[i - 1] + 1 {
                    let chunk_len = i - current_start_i;
                    let start_slot = spawned[current_start_i];

                    for j in 0..chunk_len {
                        writer[current_start_i + j] = data[(start_slot as usize) + j];
                    }

                    copy_info.regions.push(BufferCopy {
                        src_offset: (current_start_i as u64) * elem_size,
                        dst_offset: (start_slot as u64) * elem_size,
                        size: (chunk_len as u64) * elem_size,
                        ..Default::default()
                    });

                    current_start_i = i;
                }
            }
            drop(writer);
            builder.copy_buffer(copy_info)?;
        }

        Ok(resized)
    }

    pub fn prepare_frame(
        &mut self,
        allocator: Arc<StandardMemoryAllocator>,
        staging_allocator: &SubbufferAllocator,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        new_data: &[T],
        newly_spawned_indices: &[u32],
    ) -> GameEngineResult<bool> {
        let needed = new_data.len() as u64;
        let resized = if needed > self.capacity {
            self.resize(allocator, needed)?
        } else {
            false
        };

        log::trace!("TripleBuffer::advance head {} -> {}", self.head, (self.head + 1) % 3);
        self.head = (self.head + 1) % 3;
        log::trace!("advance<{}> head={} needed={} cap={}",
            std::any::type_name::<T>(), self.head, needed, self.capacity);

        if needed == 0 {
            return Ok(resized);
        }

        let new_idx = self.head;
        let old_idx = (self.head + 2) % 3;

        let old_buf = self.buffers[old_idx].clone();
        let new_buf = self.buffers[new_idx].clone();

        // Upload all data to New
        let staging = staging_allocator.allocate_slice::<T>(needed)?;
        staging.write()?.copy_from_slice(new_data);
        builder.copy_buffer(CopyBufferInfo::buffers(
            staging,
            new_buf.clone().slice(0..needed),
        ))?;

        // Patch newly spawned entities into Old
        if !newly_spawned_indices.is_empty() {
            let mut spawned = newly_spawned_indices.to_vec();
            spawned.sort_unstable();
            spawned.dedup();

            let count = spawned.len() as u64;
            let patch_staging = staging_allocator.allocate_slice::<T>(count)?;
            let mut writer = patch_staging.write()?;
            let elem_size = std::mem::size_of::<T>() as u64;

            let mut copy_info = CopyBufferInfo::buffers(patch_staging.clone(), old_buf);
            copy_info.regions.clear();

            let mut current_start_i = 0;
            for i in 1..=spawned.len() {
                if i == spawned.len() || spawned[i] != spawned[i - 1] + 1 {
                    let chunk_len = i - current_start_i;
                    let start_slot = spawned[current_start_i];

                    for j in 0..chunk_len {
                        writer[current_start_i + j] = new_data[(start_slot as usize) + j];
                    }

                    copy_info.regions.push(BufferCopy {
                        src_offset: (current_start_i as u64) * elem_size,
                        dst_offset: (start_slot as u64) * elem_size,
                        size: (chunk_len as u64) * elem_size,
                        ..Default::default()
                    });

                    current_start_i = i;
                }
            }
            drop(writer);
            builder.copy_buffer(copy_info)?;
        }

        Ok(resized)
    }

    pub fn current_buffers(&self) -> (Subbuffer<[T]>, Subbuffer<[T]>) {
        let old_idx = (self.head + 2) % 3;
        (
            self.buffers[old_idx].clone(),
            self.buffers[self.head].clone(),
        )
    }

    pub fn write_entire_frame(
        &mut self,
        allocator: Arc<StandardMemoryAllocator>,
        staging_allocator: &SubbufferAllocator,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        new_data: &[T],
    ) -> GameEngineResult<bool> {
        let needed = new_data.len() as u64;
        let resized = if needed > self.capacity {
            self.resize(allocator, needed)?
        } else {
            false
        };

        log::trace!("TripleBuffer::advance head {} -> {}", self.head, (self.head + 1) % 3);
        self.head = (self.head + 1) % 3;
        log::trace!("advance<{}> head={} needed={} cap={}",
            std::any::type_name::<T>(), self.head, needed, self.capacity);

        if needed == 0 {
            return Ok(resized);
        }

        let new_idx = self.head;
        let new_buf = self.buffers[new_idx].clone();

        let staging = staging_allocator.allocate_slice::<T>(needed)?;
        staging.write()?.copy_from_slice(new_data);
        builder.copy_buffer(CopyBufferInfo::buffers(staging, new_buf.slice(0..needed)))?;
        Ok(resized)
    }
}

pub struct GpuChannel<P: Pod + Send + Sync + GpuBindingProvider> {
    buffer: TripleBuffer<P>,

    // CACHE: We store the writes, not the full Set.
    // If this is Some, we are clean. If None, we are dirty.
    cached_writes: Option<[WriteDescriptorSet; 2]>,
}

impl<P: Pod + Send + Sync + GpuBindingProvider + Debug> GpuChannel<P> {
    pub fn new(alloc: Arc<StandardMemoryAllocator>, cap: u64) -> Self {
        Self {
            buffer: TripleBuffer::new(alloc, cap).unwrap(),
            cached_writes: None,
        }
    }

    /// mark dirty. We update even without changes because otherwise the interpolation
    /// will get out of sync.
    pub fn mark_dirty(&mut self) {
        self.cached_writes = None;
    }

    pub fn ensure_gpu_sync(
        &mut self,
        alloc: Arc<StandardMemoryAllocator>,
        staging: &SubbufferAllocator,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        newly_spawned_indices: &[u32],
        data: &[P],
    ) -> GameEngineResult<bool> {
        // Fix: Removed early return to enforce buffer advancement
        let resized =
            self.buffer
                .prepare_frame(alloc, staging, builder, data, newly_spawned_indices)?;

        let (old_buf, new_buf) = self.buffer.current_buffers();
        self.cached_writes = Some([
            WriteDescriptorSet::buffer(P::OLD_BINDING, old_buf),
            WriteDescriptorSet::buffer(P::NEW_BINDING, new_buf),
        ]);

        Ok(resized)
    }

    pub fn ensure_gpu_sync_dirty(
        &mut self,
        alloc: Arc<StandardMemoryAllocator>,
        staging: &SubbufferAllocator,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        dirty_slots: &[u32],
        newly_spawned_indices: &[u32],
        data: &[P],
    ) -> GameEngineResult<bool> {
        // Fix: Removed early return to enforce buffer advancement
        let resized = self.buffer.prepare_frame_dirty(
            alloc,
            staging,
            builder,
            data,
            dirty_slots,
            newly_spawned_indices,
        )?;

        let (old_buf, new_buf) = self.buffer.current_buffers();
        self.cached_writes = Some([
            WriteDescriptorSet::buffer(P::OLD_BINDING, old_buf),
            WriteDescriptorSet::buffer(P::NEW_BINDING, new_buf),
        ]);

        Ok(resized)
    }

    /// Binding Phase.
    /// Returns the cached instructions for the RenderSystem to aggregate.
    pub fn get_write_descriptors(&self) -> [WriteDescriptorSet; 2] {
        // Expect ensures we crash if we forgot to call ensure_gpu_sync first
        self.cached_writes
            .clone()
            .expect("GpuChannel not synced before binding!")
    }
}

macro_rules! define_render_system {
    ($( ($idx:expr, $const_name:ident, $channel_name:ident, $builder_name:ident, $enum_name:ident, $packet_type:ident) ),*) => {

        /// Data and metadata for a single GPU channel transfer.
        #[derive(Clone, Debug)]
        pub struct ChannelDelta<T> {
            pub newly_spawned_indices: Vec<u32>,
            pub data: Arc<Vec<T>>,
        }

        /// The transient per-frame data transferred to the GPU in O(1) time.
        #[derive(Clone, Debug)]
        pub struct RenderDelta {
            pub snapped_at: Instant,
            pub op_pool: Arc<Vec<InstancePointer>>,
            pub entity_heads: Arc<Vec<GpuEntityHead>>,
            pub path_curves: Arc<Vec<GpuCurve>>,
            pub pending_textures: Vec<crate::render::texture::PendingTexture>,
            pub op_pool_dirty: bool,
            $(
                pub $channel_name: ChannelDelta<$packet_type>,
            )*
        }

        impl RenderDelta {
            pub fn merge_previous(&mut self, previous: RenderDelta) {
                self.pending_textures.splice(0..0, previous.pending_textures);
                self.op_pool_dirty |= previous.op_pool_dirty;
                $(
                    self.$channel_name.newly_spawned_indices.extend(previous.$channel_name.newly_spawned_indices);
                    self.$channel_name.newly_spawned_indices.sort_unstable();
                    self.$channel_name.newly_spawned_indices.dedup();
                )*
            }
        }

        impl Default for RenderDelta {
            fn default() -> Self {
                Self {
                    snapped_at: Instant::now(),
                    op_pool: Arc::new(Vec::new()),
                    entity_heads: Arc::new(Vec::new()),
                    path_curves: Arc::new(Vec::new()),
                    pending_textures: Vec::new(),
                    op_pool_dirty: false,
                    $(
                        $channel_name: ChannelDelta {
                            newly_spawned_indices: Vec::new(),
                            data: Arc::new(Vec::new()),
                        },
                    )*
                }
            }
        }

        /// A unified snapshot of the simulation world at a specific moment in time.
        #[derive(Clone, Debug)]
        pub struct RenderPacket {
            /// When this snapshot was captured.
            pub snapped_at: Instant,
            /// Data channels for each primitive type.
            $(
                pub $channel_name: RenderPacketContents<$packet_type>,
            )*
            /// The operation pool: a flat list of InstancePointers that entity stacks reference.
            pub op_pool: Arc<Vec<InstancePointer>>,
            /// The entity headers: one per drawable entity, pointing into op_pool.
            pub entity_heads: Arc<Vec<GpuEntityHead>>,
            /// Flattened Bezier segments for analytic glyph rendering.
            pub path_curves: Arc<Vec<GpuCurve>>,
            /// Textures that have been loaded by the game since the last snapshot.
            pub pending_textures: Vec<crate::render::texture::PendingTexture>,
            /// Set when op_pool contents change without a length change
            /// (e.g. slot recycling during churn). Forces a re-upload.
            pub op_pool_dirty: bool,
        }

        impl Default for RenderPacket {
            fn default() -> Self {
                Self::new()
            }
        }

        impl RenderPacket {
            pub fn new() -> Self {
                Self {
                    snapped_at: Instant::now(),
                    $(
                        $channel_name: RenderPacketContents::new(),
                    )*
                    op_pool: Arc::new(Vec::new()),
                    entity_heads: Arc::new(Vec::new()),
                    path_curves: Arc::new(Vec::new()),
                    pending_textures: Vec::new(),
                    op_pool_dirty: false,
                }
            }

            /// Extracts the transient frame data in O(1) without cloning the heavy data arrays.
            pub fn take_delta(&mut self) -> RenderDelta {
                RenderDelta {
                    snapped_at: self.snapped_at,
                    op_pool: self.op_pool.clone(),
                    entity_heads: self.entity_heads.clone(),
                    path_curves: self.path_curves.clone(),
                    pending_textures: std::mem::take(&mut self.pending_textures),
                    op_pool_dirty: std::mem::replace(&mut self.op_pool_dirty, false),
                    $(
                        $channel_name: ChannelDelta {
                            newly_spawned_indices: std::mem::take(&mut self.$channel_name.newly_spawned_indices),
                            data: self.$channel_name.data.clone(), // O(1) Arc clone!
                        },
                    )*
                }
            }

            /// Start building a new drawable entity.
            pub fn entity(&mut self) -> EntityBuilder<'_> {
                EntityBuilder {
                    packet: self,
                    stack_start: 0, // set on first push
                    count: 0,
                    started: false,
                }
            }
        }

        /// Fluent builder for assembling an entity's render stack.
        pub struct EntityBuilder<'a> {
            packet: &'a mut RenderPacket,
            stack_start: u32,
            count: u32,
            started: bool,
        }

        impl<'a> EntityBuilder<'a> {
            fn push_op(&mut self, ptr: InstancePointer) {
                if !self.started {
                    self.stack_start = self.packet.op_pool.len() as u32;
                    self.started = true;
                }
                Arc::make_mut(&mut self.packet.op_pool).push(ptr);
                self.count += 1;
            }

            $(
                pub fn $builder_name(mut self, packet: $packet_type) -> Self {
                    let data = Arc::make_mut(&mut self.packet.$channel_name.data);
                    let idx = data.len() as u32;
                    data.push(packet);
                    self.packet.$channel_name.active_indices.push(idx);
                    self.push_op(InstancePointer::new($const_name, idx));
                    self
                }
            )*

            /// Finalize this entity and add it to the packet.
            pub fn spawn(self) {
                if self.count > 0 {
                    Arc::make_mut(&mut self.packet.entity_heads).push(GpuEntityHead {
                        stack_start: self.stack_start,
                        stack_len: self.count,
                        position: [0.0; 2],
                    });
                }
            }
        }


        $(
            impl $crate::render::stack::ShapeKindProvider for $packet_type {
                const SHAPE_KIND: u32 = $const_name;
            }

            // 2. Tell RenderPacket how to yield the specific channel for this type
            impl $crate::render::stack::ChannelAccess<$packet_type> for RenderPacket {
                #[inline(always)]
                fn get_channel_mut(&mut self) -> &mut RenderPacketContents<$packet_type> {
                    &mut self.$channel_name
                }
            }
        )*

        /// Manages all GPU data channels for the stack-based renderer.
        pub struct RenderSystem {
            pub stat_last_log: std::time::Instant,
            pub stat_accum_bytes: usize,
            pub stat_accum_counts: Vec<usize>,
            pub stat_frame_count: usize,

            // -- Data Channels (ping-pong interpolated) --
            $( pub $channel_name: GpuChannel<$packet_type>, )*

            // -- Per-frame allocators for non-interpolated buffers --
            pub entity_heads: TripleBuffer<GpuEntityHead>,
            pub op_pool: TripleBuffer<InstancePointer>,
            pub path_curves: TripleBuffer<GpuCurve>,
            pub last_op_pool_len: usize,
            pub last_path_curves_len: usize,
            pub total_entity_count: usize,

            cached_set_0: [Option<Arc<DescriptorSet>>; 3],
            cached_set_1: [Option<Arc<DescriptorSet>>; 3],
            cached_set_2: Option<Arc<DescriptorSet>>,
            pub head_index: usize,

            pub textures: Vec<Arc<ImageView>>,
            pub sampler: Arc<Sampler>,
        }

        impl RenderSystem {
            /// Creates a new RenderSystem, initializing a GpuChannel for each defined shape.
            pub fn new(alloc: Arc<StandardMemoryAllocator>) -> Self {
                let sampler = Sampler::new(
                    alloc.device().clone(),
                    SamplerCreateInfo {
                        mag_filter: Filter::Linear,
                        min_filter: Filter::Linear,
                        mipmap_mode: SamplerMipmapMode::Linear,
                        address_mode: [SamplerAddressMode::ClampToEdge; 3],
                        ..Default::default()
                    },
                ).unwrap();

                Self {
                    stat_last_log: std::time::Instant::now(),
                    stat_accum_bytes: 0,
                    stat_accum_counts: vec![0; game_engine_shaders_types::packet::NUM_SHAPE_KINDS],
                    stat_frame_count: 0,
                    $(
                        $channel_name: GpuChannel::new(alloc.clone(), 1024),
                    )*
                    entity_heads: TripleBuffer::new(alloc.clone(), 1024).unwrap(),
                    op_pool: TripleBuffer::new(alloc.clone(), 1024).unwrap(),
                    path_curves: TripleBuffer::new(alloc, 1024).unwrap(),
                    last_op_pool_len: 0,
                    last_path_curves_len: 0,
                    total_entity_count: 0,
                    cached_set_0: [None, None, None],
                    cached_set_1: [None, None, None],
                    cached_set_2: None,
                    head_index: 0,
                    textures: Vec::new(),
                    sampler,
                }
            }

            /// Marks all managed GpuChannels as dirty.
            pub fn mark_all_dirty(&mut self) {
                $( self.$channel_name.mark_dirty(); )*
            }

            /// Records all necessary buffer uploads for the current frame.
            pub fn upload_all(
                &mut self,
                alloc: Arc<StandardMemoryAllocator>,
                staging: &SubbufferAllocator,
                builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
                _packet: &RenderPacket,
                delta: &RenderDelta,
            ) -> GameEngineResult<()> {
                #[cfg(feature = "tracy")]
                let _span = tracy_client::span!("RenderSystem::upload_gpu_data");

                let mut _ch_idx = 0;
                let mut frame_bytes = 0;
                $(
                    let mut resized = false;
                    resized |= self.$channel_name.ensure_gpu_sync(
                        alloc.clone(), staging, builder, &delta.$channel_name.newly_spawned_indices, &delta.$channel_name.data,
                    )?;

                    if resized {
                        self.cached_set_0 = [None, None, None];
                    }

                    _ch_idx += 1;
                )*

                let mut resized = false;
                resized |= self.entity_heads.write_entire_frame(
                    alloc.clone(),
                    staging,
                    builder,
                    &delta.entity_heads,
                )?;
                if !delta.entity_heads.is_empty() {
                    self.total_entity_count = delta.entity_heads.len();
                }

                if delta.op_pool.len() != self.last_op_pool_len || delta.op_pool_dirty {
                    resized |= self.op_pool.write_entire_frame(
                         alloc.clone(), staging, builder, &delta.op_pool,
                    )?;
                    self.last_op_pool_len = delta.op_pool.len();
                }

                if delta.path_curves.len() != self.last_path_curves_len {
                    resized |= self.path_curves.write_entire_frame(
                        alloc.clone(), staging, builder, &delta.path_curves,
                    )?;
                    self.last_path_curves_len = delta.path_curves.len();
                }

                if resized {
                    self.cached_set_1 = [None, None, None];
                }

                // Update head index to match one of our triple buffers
                self.head_index = self.entity_heads.head;

                // 4. Upload textures
                if self.textures.is_empty() {
                    // Create dummy fallback texture for index 0
                    let image = Image::new(
                        alloc.clone(),
                        ImageCreateInfo {
                            image_type: ImageType::Dim2d,
                            format: Format::R8G8B8A8_UNORM,
                            extent: [1, 1, 1],
                            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                            ..Default::default()
                        },
                        vulkano::memory::allocator::AllocationCreateInfo {
                            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                            ..Default::default()
                        },
                    )?;
                    let staging_buf = staging.allocate_slice::<u8>(4)?;
                    staging_buf.write()?.copy_from_slice(&[255, 255, 255, 255]);
                    builder.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                        staging_buf,
                        image.clone(),
                    ))?;
                    self.textures.push(ImageView::new_default(image)?);
                    self.cached_set_2 = None;
                }

                if !delta.pending_textures.is_empty() {
                    self.cached_set_2 = None;
                }

                for pending in &delta.pending_textures {
                    let handle_id = pending.handle.0 as usize;

                    // Resize to fit this handle, filling with the fallback texture
                    if handle_id >= self.textures.len() {
                        self.textures.resize(handle_id + 1, self.textures[0].clone());
                    }

                    // Read bytes into staging buffer
                    let staging_buf = staging.allocate_slice::<u8>(pending.rgba8_data.len() as u64)?;
                    staging_buf.write()?.copy_from_slice(&pending.rgba8_data);

                    // Create the device-local image
                    let image = Image::new(
                        alloc.clone(),
                        ImageCreateInfo {
                            image_type: ImageType::Dim2d,
                            format: Format::R8G8B8A8_UNORM,
                            extent: [pending.width, pending.height, 1],
                            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                            ..Default::default()
                        },
                        vulkano::memory::allocator::AllocationCreateInfo {
                            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                            ..Default::default()
                        },
                    )?;

                    // Command to copy
                    builder.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                        staging_buf,
                        image.clone(),
                    ))?;

                    self.textures[handle_id] = ImageView::new_default(image)?;
                }

                frame_bytes += delta.entity_heads.len() * std::mem::size_of::<GpuEntityHead>();
                frame_bytes += delta.op_pool.len() * std::mem::size_of::<InstancePointer>();
                for pending in &delta.pending_textures {
                    frame_bytes += pending.rgba8_data.len();
                }
                self.stat_accum_bytes += frame_bytes;
                self.stat_frame_count += 1;

                if self.stat_last_log.elapsed().as_secs_f32() >= 1.0 {
                    log::info!(
                        "GPU Upload Stats (1s | {} frames): Total: {} bytes",
                        self.stat_frame_count,
                        self.stat_accum_bytes
                    );

                    // Log per-channel amounts
                    let mut _log_idx = 0;
                    $(
                        let count = self.stat_accum_counts[_log_idx];
                        if count > 0 {
                            log::info!("  - {}: {} elements", stringify!($channel_name), count);
                        }
                        _log_idx += 1;
                    )*

                    // Reset stats
                    self.stat_last_log = std::time::Instant::now();
                    self.stat_accum_bytes = 0;
                    self.stat_accum_counts.fill(0);
                    self.stat_frame_count = 0;
                }

                Ok(())
            }

            /// Like upload_all, but only uploads dirty slots per channel.
            /// `dirty_per_channel` must have exactly one entry per channel, in the
            /// same order as the define_render_system!() declaration.
            pub fn upload_all_dirty(
                &mut self,
                alloc: Arc<StandardMemoryAllocator>,
                staging: &SubbufferAllocator,
                builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
                _packet: &RenderPacket,
                delta: &RenderDelta,
                dirty_per_channel: &[Vec<u32>],
            ) -> GameEngineResult<()> {
                #[cfg(feature = "tracy")]
                let _span = tracy_client::span!("RenderSystem::upload_gpu_data");

                let mut _ch_idx = 0usize;
                let mut frame_bytes = 0;
                $(
                        {
                            let dirty = dirty_per_channel.get(_ch_idx).map(|v| v.as_slice()).unwrap_or(&[]);
                            let spawned = &delta.$channel_name.newly_spawned_indices;
                            let mut resized = false;
                            resized |= self.$channel_name.ensure_gpu_sync_dirty(
                                alloc.clone(), staging, builder, dirty, spawned, &delta.$channel_name.data,
                            )?;

                            if resized {
                                self.cached_set_0 = [None, None, None];
                            }
                        }
                        _ch_idx += 1;
                )*

                let mut resized = false;
                resized |= self.entity_heads.write_entire_frame(
                    alloc.clone(),
                    staging,
                    builder,
                    &delta.entity_heads,
                )?;
                if !delta.entity_heads.is_empty() {
                    self.total_entity_count = delta.entity_heads.len();
                }

                if delta.op_pool.len() != self.last_op_pool_len || delta.op_pool_dirty {
                    resized |= self.op_pool.write_entire_frame(
                         alloc.clone(), staging, builder, &delta.op_pool,
                    )?;
                    self.last_op_pool_len = delta.op_pool.len();
                }

                if delta.path_curves.len() != self.last_path_curves_len {
                    resized |= self.path_curves.write_entire_frame(
                        alloc.clone(), staging, builder, &delta.path_curves,
                    )?;
                    self.last_path_curves_len = delta.path_curves.len();
                }

                if resized {
                    self.cached_set_1 = [None, None, None];
                }

                // Update head index to match one of our triple buffers
                self.head_index = self.entity_heads.head;

                // Textures
                if self.textures.is_empty() {
                    let image = Image::new(
                        alloc.clone(),
                        ImageCreateInfo {
                            image_type: ImageType::Dim2d,
                            format: Format::R8G8B8A8_UNORM,
                            extent: [1, 1, 1],
                            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                            ..Default::default()
                        },
                        vulkano::memory::allocator::AllocationCreateInfo {
                            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                            ..Default::default()
                        },
                    )?;
                    let staging_buf = staging.allocate_slice::<u8>(4)?;
                    staging_buf.write()?.copy_from_slice(&[255, 255, 255, 255]);
                    builder.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                        staging_buf,
                        image.clone(),
                    ))?;
                    self.textures.push(ImageView::new_default(image)?);
                    self.cached_set_2 = None;
                }

                if !delta.pending_textures.is_empty() {
                    self.cached_set_2 = None;
                }
                for pending in &delta.pending_textures {
                    let handle_id = pending.handle.0 as usize;
                    if handle_id >= self.textures.len() {
                        self.textures.resize(handle_id + 1, self.textures[0].clone());
                    }
                    let staging_buf = staging.allocate_slice::<u8>(pending.rgba8_data.len() as u64)?;
                    staging_buf.write()?.copy_from_slice(&pending.rgba8_data);
                    let image = Image::new(
                        alloc.clone(),
                        ImageCreateInfo {
                            image_type: ImageType::Dim2d,
                            format: Format::R8G8B8A8_UNORM,
                            extent: [pending.width, pending.height, 1],
                            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                            ..Default::default()
                        },
                        vulkano::memory::allocator::AllocationCreateInfo {
                            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                            ..Default::default()
                        },
                    )?;
                    builder.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                        staging_buf,
                        image.clone(),
                    ))?;
                    self.textures[handle_id] = ImageView::new_default(image)?;
                }

                frame_bytes += delta.entity_heads.len() * std::mem::size_of::<GpuEntityHead>();
                frame_bytes += delta.op_pool.len() * std::mem::size_of::<InstancePointer>();
                for pending in &delta.pending_textures {
                    frame_bytes += pending.rgba8_data.len();
                }
                self.stat_accum_bytes += frame_bytes;
                self.stat_frame_count += 1;

                if self.stat_last_log.elapsed().as_secs_f32() >= 1.0 {
                    log::info!(
                        "GPU Upload Stats (1s | {} frames): Total: {} bytes",
                        self.stat_frame_count,
                        self.stat_accum_bytes
                    );

                    // Log per-channel amounts
                    let mut _log_idx = 0;
                    $(
                        let count = self.stat_accum_counts[_log_idx];
                        if count > 0 {
                            log::info!("  - {}: {} elements", stringify!($channel_name), count);
                        }
                        _log_idx += 1;
                    )*

                    // Reset stats
                    self.stat_last_log = std::time::Instant::now();
                    self.stat_accum_bytes = 0;
                    self.stat_accum_counts.fill(0);
                    self.stat_frame_count = 0;
                }

                Ok(())
            }

            pub fn get_descriptor_sets(
                &mut self,
                pipeline: &Arc<GraphicsPipeline>,
                set_allocator: Arc<StandardDescriptorSetAllocator>,
            ) -> GameEngineResult<[Arc<DescriptorSet>; 3]> {
                let layouts = pipeline.layout().set_layouts();
                let head = self.head_index;
                log::trace!("get_descriptor_sets using head={} | set1_cached={}", head, self.cached_set_1[head].is_some());

                // 1. Set 0
                if self.cached_set_0[head].is_none() {
                    let mut all_writes = Vec::new();
                    $( all_writes.extend(self.$channel_name.get_write_descriptors()); )*

                    self.cached_set_0[head] = Some(DescriptorSet::new(
                        set_allocator.clone(),
                        layouts.get(0).unwrap().clone(),
                        all_writes,
                        [],
                    )?);
                }

                // 2. Set 1
                if self.cached_set_1[head].is_none() {
                    let (old_head_buffer, new_head_buffer) = self.entity_heads.current_buffers();
                    let (_, pool_buffer) = self.op_pool.current_buffers();
                    let (_, path_curves_buffer) = self.path_curves.current_buffers();

                    self.cached_set_1[head] = Some(DescriptorSet::new(
                        set_allocator.clone(),
                        layouts.get(1).unwrap().clone(),
                        [
                            WriteDescriptorSet::buffer(0, old_head_buffer),
                            WriteDescriptorSet::buffer(1, new_head_buffer),
                            WriteDescriptorSet::buffer(2, pool_buffer),
                            WriteDescriptorSet::buffer(3, path_curves_buffer),
                        ],
                        [],
                    )?);
                }

                // 3. Set 2
                if self.cached_set_2.is_none() {
                    let mut set_2_writes = vec![WriteDescriptorSet::sampler(0, self.sampler.clone())];
                    if !self.textures.is_empty() {
                        set_2_writes.push(WriteDescriptorSet::image_view_array(
                            1,
                            0, // starting index
                            self.textures.clone(),
                        ));
                    }

                    self.cached_set_2 = Some(DescriptorSet::new_variable(
                        set_allocator,
                        layouts.get(2).unwrap().clone(),
                        self.textures.len().max(1) as u32,
                        set_2_writes,
                        [],
                    )?);
                }

                Ok([
                    self.cached_set_0[head].clone().unwrap(),
                    self.cached_set_1[head].clone().unwrap(),
                    self.cached_set_2.clone().unwrap(),
                ])
            }
        }
    };
}

game_engine_shaders_types::for_all_shapes!(define_render_system);
