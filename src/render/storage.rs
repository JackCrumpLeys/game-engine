use crate::GameEngineResult;
use crate::RenderPacketContents;
use bytemuck::Pod;
use game_engine_shaders_types::packet::GpuCirclePacket;
use game_engine_shaders_types::packet::InstancePointer;
use game_engine_shaders_types::packet::{GpuBindingProvider, GpuTrianglePacket};
use std::fmt::Debug;
use std::mem;
use std::sync::Arc;
use std::time::Instant;
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, BufferCopy, CopyBufferInfo, PrimaryAutoCommandBuffer,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::Pipeline;

pub struct PingPongBuffer<T: Pod + Send + Sync> {
    buffer_a: Subbuffer<[T]>,
    buffer_b: Subbuffer<[T]>,
    /// If true, 'A' is currently the destination for new data (making 'B' the old data)
    capacity: u64,
}

impl<T: Pod + Send + Sync + Debug> PingPongBuffer<T> {
    pub fn new(allocator: Arc<StandardMemoryAllocator>, capacity: u64) -> GameEngineResult<Self> {
        let usage =
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC;
        let layout = BufferCreateInfo {
            usage,
            ..Default::default()
        };
        let alloc_info = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        };

        // Start with two empty buffers
        let buffer_a = Buffer::new_slice::<T>(
            allocator.clone(),
            layout.clone(),
            alloc_info.clone(),
            capacity,
        )?;
        let buffer_b = Buffer::new_slice::<T>(allocator, layout, alloc_info, capacity)?;

        Ok(Self {
            buffer_a, // Represents "Old"
            buffer_b, // Represents "New"
            capacity,
        })
    }

    pub fn prepare_frame(
        &mut self,
        allocator: Arc<StandardMemoryAllocator>,
        staging_allocator: &SubbufferAllocator,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        new_data: &[T],
        newly_spawned_indices: &[u32],
    ) -> GameEngineResult<()> {
        if (new_data.len() as u64) > self.capacity {
            self.resize(allocator.clone(), new_data.len() as u64, builder)?;
        }

        if new_data.is_empty() {
            // No data to upload, skip.
            return Ok(());
        }

        // dbg!(&new_data);

        // --- 1. CREATE NEXT NEW BUFFER ---
        // Every frame, we allocate a fresh "New" buffer. This is immutable and safe.
        let usage =
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC;
        let next_new = Buffer::new_slice::<T>(
            allocator.clone(),
            BufferCreateInfo {
                usage,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            self.capacity,
        )?;

        // Upload everything to the next new buffer
        let new_data_len = new_data.len() as u64;
        let staging = staging_allocator.allocate_slice::<T>(new_data_len)?;
        staging.write()?.copy_from_slice(new_data);
        builder.copy_buffer(CopyBufferInfo::buffers(
            staging,
            next_new.clone().slice(0..new_data_len),
        ))?;

        // --- 2. HANDLE THE "OLD" BUFFER ---
        let next_old = if !newly_spawned_indices.is_empty() {
            // SPAWN CASE: We need to patch the data from the previous "New" buffer
            // so the entities don't fly in from (0,0).
            let patched_old = Buffer::new_slice::<T>(
                allocator,
                BufferCreateInfo {
                    usage,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                self.capacity,
            )?;

            // A. Copy the entire previous "New" buffer to the fresh "Old" buffer
            builder.copy_buffer(CopyBufferInfo::buffers(
                self.buffer_b.clone(),
                patched_old.clone(),
            ))?;

            // B. Patch the new entities into this clone
            let count = newly_spawned_indices.len();
            let patch_staging = staging_allocator.allocate_slice::<T>(count as u64)?;
            {
                let mut writer = patch_staging.write()?;
                for (i, &game_idx) in newly_spawned_indices.iter().enumerate() {
                    writer[i] = new_data[game_idx as usize];
                }
            }

            let size = mem::size_of::<T>() as u64;
            let regions: Vec<BufferCopy> = newly_spawned_indices
                .iter()
                .enumerate()
                .map(|(i, &game_idx)| BufferCopy {
                    src_offset: (i as u64) * size,
                    dst_offset: (game_idx as u64) * size,
                    size,
                    ..Default::default()
                })
                .collect();

            let mut copy_info = CopyBufferInfo::buffers(patch_staging, patched_old.clone());
            copy_info.regions.clear();
            copy_info.regions.extend(regions);
            builder.copy_buffer(copy_info)?;

            patched_old
        } else {
            // NORMAL CASE: The previous frame's "New" is perfectly fine to be the "Old".
            // Since we never mutated it, the GPU can keep reading it.
            self.buffer_b.clone()
        };

        // --- 3. ROTATE ---
        // The Arcs are updated. The previous self.buffer_a is dropped on CPU,
        // but stays alive on GPU until the previous frame's Future is cleared.
        self.buffer_a = next_old;
        self.buffer_b = next_new;

        Ok(())
    }

    fn current_buffers(&self) -> (Subbuffer<[T]>, Subbuffer<[T]>) {
        (self.buffer_a.clone(), self.buffer_b.clone())
    }

    fn resize(
        &mut self,
        _allocator: Arc<StandardMemoryAllocator>,
        needed: u64,
        _builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> GameEngineResult<()> {
        // Just update capacity. The next prepare_frame will allocate at this size.
        self.capacity = needed.next_power_of_two();
        Ok(())
    }
}

pub struct GpuChannel<P: Pod + Send + Sync + GpuBindingProvider> {
    buffer: PingPongBuffer<P>,

    // CACHE: We store the writes, not the full Set.
    // If this is Some, we are clean. If None, we are dirty.
    cached_writes: Option<[WriteDescriptorSet; 2]>,
}

impl<P: Pod + Send + Sync + GpuBindingProvider + Debug> GpuChannel<P> {
    pub fn new(alloc: Arc<StandardMemoryAllocator>, cap: u64) -> Self {
        Self {
            buffer: PingPongBuffer::new(alloc, cap).unwrap(),
            cached_writes: None,
        }
    }

    /// mark dirty. We update even without changes becuase otherwise the interpolation
    /// will get out of sync.
    pub fn mark_dirty(&mut self) {
        self.cached_writes = None;
    }

    /// Step 2: Upload Phase (Conditional)
    pub fn ensure_gpu_sync(
        &mut self,
        alloc: Arc<StandardMemoryAllocator>,
        staging: &SubbufferAllocator,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        newly_spawned_indices: &[u32],
        data: &[P],
    ) -> GameEngineResult<()> {
        // If we have cached writes, it means we successfully uploaded/flipped already.
        if self.cached_writes.is_some() {
            return Ok(());
        }

        // 1. Perform Upload / Flip
        self.buffer
            .prepare_frame(alloc, staging, builder, data, newly_spawned_indices)?;

        // 2. Regenerate Cache immediately
        // We do this now so we don't have to look up buffer pointers later
        let (old_buf, new_buf) = self.buffer.current_buffers();

        self.cached_writes = Some([
            WriteDescriptorSet::buffer(P::OLD_BINDING, old_buf),
            WriteDescriptorSet::buffer(P::NEW_BINDING, new_buf),
        ]);

        Ok(())
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
    (
        // We only need the Name and the Type.
        // Bindings are pulled from the GpuBindingProvider trait.
        $( $channel_name:ident : $packet_type:ty ),* $(,)?
    ) => {

        /// A unified snapshot of the simulation world at a specific moment in time.
        #[derive(Clone, Debug)]
        pub struct RenderPacket {
            /// When this snapshot was captured.
            pub snapped_at: Instant,
            /// Collection of all triangle primitives in the world.
            $(
                pub $channel_name: RenderPacketContents<$packet_type>,
            )*
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
                }
            }
        }

        /// Manages all GPU data channels for the Uber-Shader renderer.
        /// This struct is the sole owner of the GPU-side PingPongBuffers.
        pub struct RenderSystem {
            // -- Generated Fields --
            // pub triangles: GpuChannel<GpuTrianglePacket>,
            // pub circles: GpuChannel<GpuCirclePacket>,
            // ...
            $( pub $channel_name: GpuChannel<$packet_type>, )*

            // -- Fixed Fields --
            instance_map_allocator: SubbufferAllocator,
        }

        impl RenderSystem {
            /// Creates a new RenderSystem, initializing a GpuChannel for each defined shape.
            pub fn new(alloc: Arc<StandardMemoryAllocator>) -> Self {
                Self {
                    $(
                        $channel_name: GpuChannel::new(alloc.clone(), 1024),
                    )*
                    instance_map_allocator: SubbufferAllocator::new(
                        alloc.clone(),
                        SubbufferAllocatorCreateInfo {
                            buffer_usage: BufferUsage::STORAGE_BUFFER,
                            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                            ..Default::default()
                        }
                    ),
                }
            }

            /// Marks all managed GpuChannels as dirty.
            /// This should be called by the PassManager whenever a new RenderPacket is pushed,
            /// signaling that the GPU state is now out of sync with the latest simulation tick.
            pub fn mark_all_dirty(&mut self) {
                $( self.$channel_name.mark_dirty(); )*
            }

            /// Records all necessary buffer uploads for the current frame.
            /// Iterates through each channel, checks its dirty state, and records copy
            /// commands if necessary. Also uploads the master instance map.
            ///
            /// Returns the Subbuffer for the instance_map, which is needed for binding.
            pub fn upload_all(
                &mut self,
                alloc: Arc<StandardMemoryAllocator>,
                staging: &SubbufferAllocator,
                builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
                packet: &RenderPacket, // The latest simulation state
                sorted_map: &[InstancePointer],
            ) -> GameEngineResult<Subbuffer<[InstancePointer]>> {
                // 1. Sync all shape data channels (these have internal dirty checks)
                $(
                    self.$channel_name.ensure_gpu_sync(
                        alloc.clone(),
                        staging,
                        builder,
                        &packet.$channel_name.newly_spawned_indices, // Assumes RenderPacket has a field with this name
                        &packet.$channel_name.data,
                    )?;
                )*

                // 2. Upload the Master Instance Map (always needed)
                let map_subbuffer = self.instance_map_allocator.allocate_slice(sorted_map.len() as u64)?;
                map_subbuffer.write()?.copy_from_slice(sorted_map);

                Ok(map_subbuffer)
            }

            /// Creates the two DescriptorSets required by the pipeline for a given frame.
            ///
            /// - **Set 0:** Contains all the `old`/`new` data buffers for every shape type.
            /// - **Set 1:** Contains the `instance_map` for indirection.
            pub fn create_descriptor_sets(
                &self,
                pipeline: &Arc<GraphicsPipeline>,
                set_allocator: Arc<StandardDescriptorSetAllocator>,
                instance_map_buffer: Subbuffer<[InstancePointer]>,
            ) -> GameEngineResult<Vec<Arc<DescriptorSet>>> {
                let layouts = pipeline.layout().set_layouts();
                let mut all_writes = Vec::new();

                // 1. Gather writes for Set 0 from all channels
                $(
                    all_writes.extend(self.$channel_name.get_write_descriptors());
                )*

                let set_0 = DescriptorSet::new(
                    set_allocator.clone(),
                    layouts.get(0).unwrap().clone(),
                    all_writes,
                    []
                )?;

                // 2. Create Set 1
                let set_1 = DescriptorSet::new(
                    set_allocator,
                    layouts.get(1).unwrap().clone(),
                    [WriteDescriptorSet::buffer(0, instance_map_buffer)], // Binding 0 for the map
                    []
                )?;

                Ok(vec![set_0, set_1])
            }
        }
    };
}

define_render_system!(
    triangles: GpuTrianglePacket,
    circles: GpuCirclePacket
);
