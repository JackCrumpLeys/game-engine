use std::{marker::PhantomData, ops::Deref};

use crate::prelude::{Res, ResMut, Resource};
use crate::system::{SystemAccess, SystemParam, UnsafeWorldCell};
use crate::world::World;

pub trait Message: 'static + Send + Sync {}
impl<T: Resource> Message for T {}

/// Double buffered message queue for ECS systems.
pub(crate) struct MessageQueue<T: Message> {
    lowest_id: MessageId<T>,
    front_buffer: Vec<T>,
    back_buffer: Vec<T>,
}

impl<T: Message> Default for MessageQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MessageId<T: Message>(u64, PhantomData<T>);

impl<T: Message> MessageId<T> {
    pub fn new(id: u64) -> Self {
        Self(id, PhantomData)
    }
}

impl<T: Message> Deref for MessageId<T> {
    type Target = u64;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Message> MessageQueue<T> {
    pub fn new() -> Self {
        Self {
            front_buffer: Vec::new(),
            back_buffer: Vec::new(),
            lowest_id: MessageId::new(0),
        }
    }

    pub fn push(&mut self, message: T) {
        self.back_buffer.push(message);
    }

    #[allow(dead_code)] // TODO
    pub fn swap_buffers(&mut self) {
        self.lowest_id = MessageId::new(*self.lowest_id + self.front_buffer.len() as u64);
        std::mem::swap(&mut self.front_buffer, &mut self.back_buffer);
        self.back_buffer.clear();
    }

    pub fn index_of(&self, id: &MessageId<T>) -> Option<usize> {
        let index = **id as isize - *self.lowest_id as isize;
        if index as usize > self.front_buffer.len() + self.back_buffer.len() {
            None
        } else {
            // We include index == len case to allow readers to have an empty state
            Some(index as usize)
        }
    }

    pub fn clamp_id(&self, id: &MessageId<T>) -> MessageId<T> {
        // Check if behind (Lagging)
        if **id < *self.lowest_id {
            return MessageId::new(*self.lowest_id);
        }

        // Check if ahead (Future)
        let offset = **id - *self.lowest_id;
        let total_len = (self.front_buffer.len() + self.back_buffer.len()) as u64;

        if offset > total_len {
            return MessageId::new(*self.lowest_id + total_len);
        }

        // Valid
        MessageId::new(**id)
    }

    #[inline(always)]
    pub fn get_internal(&self, index: usize) -> Option<&T> {
        if index < self.front_buffer.len() {
            Some(&self.front_buffer[index])
        } else if self.back_buffer.len() + self.front_buffer.len() > index {
            Some(&self.back_buffer[index - self.front_buffer.len()])
        } else {
            None
        }
    }

    #[allow(dead_code)] // TODO
    pub fn len(&self) -> usize {
        self.front_buffer.len() + self.back_buffer.len()
    }

    #[inline(always)]
    fn all_after(&self, index: usize) -> Vec<&T> {
        if index < self.front_buffer.len() {
            self.front_buffer[index..]
                .iter()
                .chain(self.back_buffer.iter())
                .collect()
        } else if self.back_buffer.len() + self.front_buffer.len() > index {
            self.back_buffer[index - self.front_buffer.len()..]
                .iter()
                .collect()
        } else {
            Vec::new()
        }
    }
}

pub struct MessageReader<'a, T: Message> {
    queue: Res<'a, MessageQueue<T>>,
    from_id: &'a mut MessageId<T>,
}

impl<T: Message> MessageReader<'_, T> {
    pub fn read_next(&mut self) -> Option<&T> {
        let index = self
            .queue
            .index_of(self.from_id)
            .expect("MessageReader in invalid state");

        let message = self.queue.get_internal(index);
        if message.is_some() {
            self.from_id.0 += 1;
        }
        message
    }

    pub fn read_next_with_id(&mut self) -> Option<(MessageId<T>, &T)> {
        let index = self
            .queue
            .index_of(self.from_id)
            .expect("MessageReader in invalid state");

        let message = self.queue.get_internal(index);
        if let Some(msg) = message {
            let id = MessageId::new(**self.from_id);
            self.from_id.0 += 1;
            Some((id, msg))
        } else {
            None
        }
    }

    /// An iterator over all unread messages.
    /// It will only advance the internal index when the iterator is advaced
    pub fn iter(&mut self) -> impl Iterator<Item = &T> {
        let index = self
            .queue
            .index_of(self.from_id)
            .expect("MessageReader in invalid state");

        let messages = self.queue.all_after(index);
        ReadIterator {
            messages,
            index: self.from_id,
            offset: *self.queue.lowest_id,
            vec_index: 0,
        }
        .map(|(_, msg)| msg)
    }

    pub fn iter_with_id(&mut self) -> impl Iterator<Item = (MessageId<T>, &T)> {
        let index = self
            .queue
            .index_of(self.from_id)
            .expect("MessageReader in invalid state");

        let messages = self.queue.all_after(index);
        ReadIterator {
            messages,
            index: self.from_id,
            offset: *self.queue.lowest_id,
            vec_index: 0,
        }
    }
}

pub struct ReadIterator<'a, T: Message> {
    messages: Vec<&'a T>,
    index: &'a mut MessageId<T>,
    offset: u64,
    vec_index: usize,
}

impl<'a, T: Message> Iterator for ReadIterator<'a, T> {
    type Item = (MessageId<T>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.messages.is_empty() || self.vec_index >= self.messages.len() {
            None
        } else {
            let msg = self.messages[self.vec_index];
            let id = MessageId::new(**self.index + self.offset);
            self.index.0 += 1;
            self.vec_index += 1;
            Some((id, msg))
        }
    }
}

pub struct MessageWriter<'a, T: Message> {
    queue: ResMut<'a, MessageQueue<T>>,
}

impl<'a, T: Message> MessageWriter<'a, T> {
    pub fn write(&mut self, message: T) {
        self.queue.push(message);
    }
}

impl<T: Message> SystemParam for MessageWriter<'_, T> {
    type State = ();
    type Item<'w> = MessageWriter<'w, T>;

    fn init_state(world: &mut World, access: &mut SystemAccess) -> Self::State {
        access.write_resource::<T>();
        world.resources_mut().register::<MessageQueue<T>>();
    }

    unsafe fn get_param<'w>(
        _state: &'w mut Self::State,
        u_world: &UnsafeWorldCell<'w>,
    ) -> Self::Item<'w> {
        let world = unsafe { u_world.world_mut() };

        // We inser in init_state, so it must exist L
        let queue = world.resources_mut().get_mut::<MessageQueue<T>>().unwrap();

        MessageWriter { queue }
    }
}

impl<T: Message> SystemParam for MessageReader<'_, T> {
    type State = MessageId<T>;
    type Item<'w> = MessageReader<'w, T>;

    fn init_state(world: &mut World, access: &mut SystemAccess) -> Self::State {
        access.read_resource::<T>();

        world.resources_mut().register::<MessageQueue<T>>();

        MessageId::new(0)
    }

    unsafe fn get_param<'w>(
        read_from: &'w mut Self::State,
        u_world: &UnsafeWorldCell<'w>,
    ) -> Self::Item<'w> {
        let world = unsafe { u_world.world() };

        // We inser in init_state, so it must exist L
        let queue = world.resources().get::<MessageQueue<T>>().unwrap();
        // We should always be at most len + offset, as the reader wont increase beyond
        // that

        debug_assert!(
            queue.index_of(read_from).is_some(),
            "SystemParam in invalid state"
        );

        read_from.0 = queue.clamp_id(read_from).0;

        MessageReader {
            from_id: read_from,
            queue,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test payload
    #[derive(Debug, PartialEq, Clone)]
    struct Event(u32);

    #[test]
    fn test_queue_lifecycle() {
        let mut queue = MessageQueue::<Event>::new();

        // 1. Initial State
        assert_eq!(queue.len(), 0);
        assert_eq!(*queue.lowest_id, 0);

        // 2. Push to Back Buffer
        queue.push(Event(1));
        queue.push(Event(2));
        assert_eq!(queue.len(), 2);
        // Direct internal access check
        assert_eq!(queue.get_internal(0), Some(&Event(1)));
        assert_eq!(queue.get_internal(1), Some(&Event(2)));

        // 3. Swap Buffers (Frame Boundary)
        // Back buffer moves to Front. Back clears.
        queue.swap_buffers();

        assert_eq!(queue.len(), 2); // Still have the messages
        assert_eq!(*queue.lowest_id, 0); // IDs haven't shifted yet because front was empty before

        // 4. Push new messages (Frame 2)
        queue.push(Event(3));

        // Front: [1, 2], Back: [3]
        assert_eq!(queue.len(), 3);
        assert_eq!(queue.get_internal(2), Some(&Event(3)));

        // 5. Swap Buffers (Frame Boundary 2)
        // Old Front [1, 2] is discarded.
        // Old Back [3] becomes Front.
        // Back is cleared.
        queue.swap_buffers();

        assert_eq!(queue.len(), 1); // Only Event(3) remains
        assert_eq!(*queue.lowest_id, 2); // IDs 0 and 1 are gone
        assert_eq!(queue.get_internal(0), Some(&Event(3))); // Index 0 is now ID 2
    }

    #[test]
    fn test_clamp_lagging_reader() {
        let mut queue = MessageQueue::<Event>::new();
        let id = MessageId::new(0);

        // Push 2 events, Swap, Push 1 event, Swap
        // Events 0 and 1 are lost. Event 2 remains.
        queue.push(Event(0));
        queue.push(Event(1));
        queue.swap_buffers();
        queue.push(Event(2));
        queue.swap_buffers();

        assert_eq!(*queue.lowest_id, 2);

        // Reader is at 0. Queue starts at 2.
        // Should clamp up to 2.
        let clamped = queue.clamp_id(&id);
        assert_eq!(*clamped, 2);
    }

    #[test]
    fn test_clamp_future_reader() {
        let mut queue = MessageQueue::<Event>::new();
        queue.push(Event(0)); // ID 0

        // Reader thinks it is at ID 100
        let id = MessageId::new(100);

        // Max valid ID is 1 (next message to be written)
        let clamped = queue.clamp_id(&id);
        assert_eq!(*clamped, 1);
    }

    #[test]
    fn test_clamp_valid_reader() {
        let mut queue = MessageQueue::<Event>::new();
        queue.push(Event(0));
        queue.push(Event(1));

        // Reader is at 1. Valid.
        let id = MessageId::new(1);
        let clamped = queue.clamp_id(&id);
        assert_eq!(*clamped, 1);
    }

    #[test]
    fn test_manual_iteration_logic() {
        let mut queue = MessageQueue::<Event>::new();

        // Setup: Front=[10], Back=[20]
        queue.push(Event(10));
        queue.swap_buffers();
        queue.push(Event(20));

        // Reader starts at beginning (ID 0)
        let mut reader_id = MessageId::new(0);
        let clamped_id = queue.clamp_id(&reader_id);

        // Calculate internal start index
        let start_idx = (*clamped_id - *queue.lowest_id) as usize;
        let mut results = Vec::new();

        for i in start_idx..queue.len() {
            if let Some(msg) = queue.get_internal(i) {
                results.push(msg.clone());
                reader_id.0 += 1;
            }
        }

        assert_eq!(results, vec![Event(10), Event(20)]);
        assert_eq!(*reader_id, 2);
    }
}
