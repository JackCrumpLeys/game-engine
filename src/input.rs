//! Input system: bridges Winit events to the ECS simulation thread.

use glam::Vec2;
use std::collections::HashSet;

/// Keyboard key codes (subset of winit's KeyCode for common use).
pub use winit::keyboard::KeyCode;

/// Mouse button identifiers.
pub use winit::event::MouseButton;

/// Snapshot of input state for a single frame.
/// Sent from the render thread to the ECS simulation thread via channel.
#[derive(Clone, Debug, Default)]
pub struct InputState {
    pub keys_held: HashSet<KeyCode>,
    pub keys_pressed: HashSet<KeyCode>,
    pub keys_released: HashSet<KeyCode>,
    pub mouse_position: Vec2,
    pub mouse_delta: Vec2,
    pub mouse_buttons_held: HashSet<MouseButton>,
    pub mouse_buttons_pressed: HashSet<MouseButton>,
    pub mouse_buttons_released: HashSet<MouseButton>,
    pub scroll_delta: f32,
}

impl InputState {
    pub fn key_held(&self, key: KeyCode) -> bool {
        self.keys_held.contains(&key)
    }

    pub fn key_just_pressed(&self, key: KeyCode) -> bool {
        self.keys_pressed.contains(&key)
    }

    pub fn key_just_released(&self, key: KeyCode) -> bool {
        self.keys_released.contains(&key)
    }

    pub fn mouse_position(&self) -> Vec2 {
        self.mouse_position
    }

    pub fn mouse_delta(&self) -> Vec2 {
        self.mouse_delta
    }

    pub fn mouse_button_held(&self, button: MouseButton) -> bool {
        self.mouse_buttons_held.contains(&button)
    }

    pub fn mouse_button_just_pressed(&self, button: MouseButton) -> bool {
        self.mouse_buttons_pressed.contains(&button)
    }

    pub fn scroll_delta(&self) -> f32 {
        self.scroll_delta
    }
}

/// Accumulates raw Winit events on the render thread.
/// Call `take_snapshot()` to produce an `InputState` for the ECS thread.
pub struct InputAccumulator {
    keys_held: HashSet<KeyCode>,
    keys_pressed: HashSet<KeyCode>,
    keys_released: HashSet<KeyCode>,
    mouse_position: Vec2,
    prev_mouse_position: Vec2,
    mouse_buttons_held: HashSet<MouseButton>,
    mouse_buttons_pressed: HashSet<MouseButton>,
    mouse_buttons_released: HashSet<MouseButton>,
    scroll_delta: f32,
}

impl Default for InputAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl InputAccumulator {
    pub fn new() -> Self {
        Self {
            keys_held: HashSet::new(),
            keys_pressed: HashSet::new(),
            keys_released: HashSet::new(),
            mouse_position: Vec2::ZERO,
            prev_mouse_position: Vec2::ZERO,
            mouse_buttons_held: HashSet::new(),
            mouse_buttons_pressed: HashSet::new(),
            mouse_buttons_released: HashSet::new(),
            scroll_delta: 0.0,
        }
    }

    /// Process a single Winit WindowEvent.
    pub fn process_event(&mut self, event: &winit::event::WindowEvent) {
        use winit::event::{ElementState, WindowEvent};

        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                if let winit::keyboard::PhysicalKey::Code(key) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            if !event.repeat {
                                self.keys_pressed.insert(key);
                            }
                            self.keys_held.insert(key);
                        }
                        ElementState::Released => {
                            self.keys_released.insert(key);
                            self.keys_held.remove(&key);
                        }
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let amount = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => (pos.y / 100.0) as f32,
                };
                self.scroll_delta += amount;
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_position = glam::vec2(position.x as f32, position.y as f32);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                match state {
                    ElementState::Pressed => {
                        self.mouse_buttons_pressed.insert(*button);
                        self.mouse_buttons_held.insert(*button);
                    }
                    ElementState::Released => {
                        self.mouse_buttons_released.insert(*button);
                        self.mouse_buttons_held.remove(button);
                    }
                }
            }
            _ => {}
        }
    }

    /// Produce a snapshot of the current input state and reset per-frame accumulators.
    pub fn take_snapshot(&mut self) -> InputState {
        let delta = self.mouse_position - self.prev_mouse_position;
        self.prev_mouse_position = self.mouse_position;

        

        InputState {
            keys_held: self.keys_held.clone(),
            keys_pressed: std::mem::take(&mut self.keys_pressed),
            keys_released: std::mem::take(&mut self.keys_released),
            mouse_position: self.mouse_position,
            mouse_delta: delta,
            mouse_buttons_held: self.mouse_buttons_held.clone(),
            mouse_buttons_pressed: std::mem::take(&mut self.mouse_buttons_pressed),
            mouse_buttons_released: std::mem::take(&mut self.mouse_buttons_released),
            scroll_delta: std::mem::replace(&mut self.scroll_delta, 0.0),
        }
    }
}

/// Camera state for 2D rendering. Shared between ECS and render thread.
#[derive(Clone, Debug)]
pub struct Camera2D {
    pub position: Vec2,
    pub zoom: f32,
}

impl Default for Camera2D {
    fn default() -> Self {
        Self {
            position: Vec2::ZERO,
            zoom: 1.0,
        }
    }
}
