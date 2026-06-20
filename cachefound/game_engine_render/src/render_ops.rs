use crate::stack_store::RenderStackStore;
use crate::thread_local_queue::{RenderOp, ThreadLocalRenderQueue};
use game_engine_ecs::prelude::*;
use game_engine_ecs::system::{SystemAccess, SystemParam, UnsafeWorldCell};
use game_engine_shaders_types::packet::*;
use smallvec::SmallVec;

/// System parameter for queueing render operations.
/// Uses a thread-local queue — zero contention, fully parallel.
pub struct RenderOps<'w> {
    queue: TLResMut<'w, ThreadLocalRenderQueue>,
}

impl<'w> SystemParam for RenderOps<'w> {
    type State = <TLResMut<'static, ThreadLocalRenderQueue> as SystemParam>::State;
    type Item<'a> = RenderOps<'a>;

    const DYNAMIC: bool = false;

    fn init_state(world: &mut World, access: &mut SystemAccess) -> Self::State {
        world.thread_local_res_mut().get_or_insert_default::<ThreadLocalRenderQueue>();
        world.resources_mut().get_or_insert_default::<RenderStackStore>();
        TLResMut::<ThreadLocalRenderQueue>::init_state(world, access)
    }

    unsafe fn get_param<'a>(
        state: &'a mut Self::State,
        world: &'a UnsafeWorldCell<'a>,
    ) -> Self::Item<'a> {
        RenderOps {
            queue: unsafe { TLResMut::get_param(state, world) },
        }
    }
}

impl<'w> RenderOps<'w> {
    /// Start building a new entity's render stack (full rebuild).
    pub fn entity(&mut self, entity: Entity) -> EntityStackBuilder<'_> {
        EntityStackBuilder {
            queue: &mut self.queue,
            entity,
            ops: SmallVec::new(),
        }
    }

    /// Patch specific shapes in an existing entity's stack.
    /// Only the patched channels get marked dirty — untouched channels stay clean.
    pub fn patch_entity(&mut self, entity: Entity) -> PatchBuilder<'_> {
        PatchBuilder {
            queue: &mut self.queue,
            entity,
            stack_idx: 0,
            patches: SmallVec::new(),
        }
    }

    /// Free an entity's render stack (despawn).
    pub fn free(&mut self, entity: Entity) {
        self.queue.push(RenderOp::Free { entity, stack_idx: 0 });
    }
}

// ── Full stack builder ───────────────────────────────────────────────

pub struct EntityStackBuilder<'a> {
    queue: &'a mut ThreadLocalRenderQueue,
    entity: Entity,
    ops: SmallVec<Shape, 8>,
}

impl<'a> EntityStackBuilder<'a> {
    pub fn push(mut self, shape: Shape) -> Self {
        self.ops.push(shape);
        self
    }

    pub fn circle(self, p: GpuCirclePacket) -> Self            { self.push(Shape::Circle(p)) }
    pub fn rect(self, p: GpuRectPacket) -> Self                { self.push(Shape::Rect(p)) }
    pub fn triangle(self, p: GpuTrianglePacket) -> Self        { self.push(Shape::Triangle(p)) }
    pub fn rounded_rect(self, p: GpuRoundedRectPacket) -> Self { self.push(Shape::RoundedRect(p)) }
    pub fn line_segment(self, p: GpuLineSegmentPacket) -> Self { self.push(Shape::LineSegment(p)) }
    pub fn sprite(self, p: GpuSpritePacket) -> Self            { self.push(Shape::Sprite(p)) }
    pub fn stroke(self, p: GpuStrokePacket) -> Self            { self.push(Shape::Stroke(p)) }
    pub fn clip_rect(self, p: GpuClipRectPacket) -> Self       { self.push(Shape::ClipRect(p)) }
    pub fn linear_gradient(self, p: GpuLinearGradientPacket) -> Self  { self.push(Shape::LinearGradient(p)) }
    pub fn radial_gradient(self, p: GpuRadialGradientPacket) -> Self  { self.push(Shape::RadialGradient(p)) }
    pub fn glyph(self, p: GpuGlyphPacket) -> Self              { self.push(Shape::Glyph(p)) }

    /// Finalise and push to the thread-local queue.
    pub fn spawn(self) {
        self.queue.push(RenderOp::Stacked {
            entity: self.entity,
            ops: self.ops,
        });
    }
}

// ── Patch builder ────────────────────────────────────────────────────

pub struct PatchBuilder<'a> {
    queue: &'a mut ThreadLocalRenderQueue,
    entity: Entity,
    stack_idx: u8,
    patches: SmallVec<(u8, Shape), 4>,
}

impl<'a> PatchBuilder<'a> {
    /// Set which stack index to patch (default 0).
    pub fn stack(mut self, idx: u8) -> Self {
        self.stack_idx = idx;
        self
    }

    /// Update the shape at `op_offset` in the stack.
    /// Only this shape's channel gets marked dirty.
    pub fn update(mut self, op_offset: u8, shape: Shape) -> Self {
        self.patches.push((op_offset, shape));
        self
    }

    /// Submit the patches.
    pub fn apply(self) {
        if !self.patches.is_empty() {
            self.queue.push(RenderOp::Patch {
                entity: self.entity,
                stack_idx: self.stack_idx,
                patches: self.patches,
            });
        }
    }
}
