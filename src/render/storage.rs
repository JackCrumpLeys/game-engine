use crate::GameEngineResult;
use bytemuck::Pod;
use std::mem;
use std::sync::Arc;
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, BufferCopy, CopyBufferInfo, PrimaryAutoCommandBuffer,
};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::sync::{AccessFlags, DependencyInfo, MemoryBarrier, PipelineStage, PipelineStages};

pub struct PingPongBuffer<T: Pod + Send + Sync> {
    buffer_a: Subbuffer<[T]>,
    buffer_b: Subbuffer<[T]>,
    /// If true, 'A' is currently the destination for new data (making 'B' the old data)
    a_is_dest: bool,
    capacity: u64,
}

impl<T: Pod + Send + Sync> PingPongBuffer<T> {
    pub fn new(allocator: Arc<StandardMemoryAllocator>, capacity: u64) -> GameEngineResult<Self> {
        let usage =
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC;
        let layout = BufferCreateInfo {
            usage,
            ..Default::default()
        };
        // DEVICE_LOCAL is crucial for shader performance
        let alloc_info = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        };

        // Create two buffers residing in VRAM
        let buffer_a = Buffer::new_slice::<T>(
            allocator.clone(),
            layout.clone(),
            alloc_info.clone(),
            capacity,
        )?;
        let buffer_b = Buffer::new_slice::<T>(allocator, layout, alloc_info, capacity)?;

        Ok(Self {
            buffer_a,
            buffer_b,
            a_is_dest: true,
            capacity,
        })
    }

    // Returns (old_buffer, new_buffer)
    pub fn prepare_frame(
        &mut self,
        allocator: Arc<StandardMemoryAllocator>,
        staging_allocator: &SubbufferAllocator,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        new_data: &[T],
        newly_spawned_indices: &[u32],
    ) -> GameEngineResult<(Subbuffer<[T]>, Subbuffer<[T]>)> {
        if (new_data.len() as u64) > self.capacity {
            self.resize(allocator, new_data.len() as u64, builder)?;
        }

        let (new_gpu, old_gpu) = if self.a_is_dest {
            (&self.buffer_a, &self.buffer_b)
        } else {
            (&self.buffer_b, &self.buffer_a)
        };

        if new_data.is_empty() {
            return Ok((old_gpu.clone(), new_gpu.clone()));
        }

        // We need to patch in newly spawned entities into the OLD buffer
        if !newly_spawned_indices.is_empty() {
            let count = newly_spawned_indices.len();

            // Allocate ONE single staging buffer for all spawns
            // We cast to u8 (bytes) to make offset math easier in the copy command later
            let patch_staging = staging_allocator.allocate_slice(count as u64)?;

            // Write data tightly packed into staging
            {
                let mut writer = patch_staging.write()?;

                for (staging_index, &game_index) in newly_spawned_indices.iter().enumerate() {
                    if let Some(val) = new_data.get(game_index as usize) {
                        writer[staging_index] = *val;
                    }
                }
            }

            let size = mem::size_of::<T>() as u64;
            // Construct Copy Regions
            // We want to tell the GPU: "Take 1st item from staging -> put at index X in GPU"
            // "Take 2nd item from staging -> put at index Y in GPU"
            let regions: Vec<BufferCopy> = newly_spawned_indices
                .iter()
                .enumerate()
                .map(|(staging_index, &game_index)| BufferCopy {
                    src_offset: (staging_index as u64) * size,
                    dst_offset: (game_index as u64) * size,
                    size,
                    ..Default::default()
                })
                .collect();

            // Issue ONE single command for all 10k copies
            let mut copy_buffer_info = CopyBufferInfo::buffers(patch_staging, old_gpu.clone());
            copy_buffer_info.regions.clear();
            copy_buffer_info.regions.extend(regions);
            builder.copy_buffer(copy_buffer_info)?;
        }

        // UPLOAD NEW DATA
        let new_data_len = new_data.len() as u64;
        let main_staging = staging_allocator.allocate_slice::<T>(new_data_len)?;
        main_staging.write()?.copy_from_slice(new_data);

        builder.copy_buffer(CopyBufferInfo::buffers(
            main_staging,
            new_gpu.clone().slice(0..new_data_len),
        ))?;

        // 5. Swap
        let result = (old_gpu.clone(), new_gpu.clone());
        self.a_is_dest = !self.a_is_dest;
        Ok(result)
    }

    fn resize(
        &mut self,
        allocator: Arc<StandardMemoryAllocator>,
        needed: u64,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> GameEngineResult<()> {
        let new_cap = needed.next_power_of_two();
        // Create new larger buffers
        let new_mgr = Self::new(allocator, new_cap)?;

        let (new_gpu, old_gpu) = if self.a_is_dest {
            (&mut self.buffer_a, &mut self.buffer_b)
        } else {
            (&mut self.buffer_b, &mut self.buffer_a)
        };

        let new_old_gpu = &new_mgr.buffer_b;

        builder.copy_buffer(CopyBufferInfo::buffers(
            old_gpu.clone(),
            new_old_gpu.clone().slice(0..self.capacity),
        ))?;

        *new_gpu = new_mgr.buffer_a;
        *old_gpu = new_mgr.buffer_b;
        self.capacity = new_cap;

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct InterpolationCache<T: Pod + Send + Sync> {
    pub(crate) old_buffer: Subbuffer<[T]>,
    pub(crate) new_buffer: Subbuffer<[T]>,
    pub(crate) alive_indices: Subbuffer<[u32]>,
}
