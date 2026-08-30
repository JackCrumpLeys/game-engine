use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TextureHandle(pub u32);

#[derive(Clone, Debug)]
pub struct PendingTexture {
    pub handle: TextureHandle,
    pub width: u32,
    pub height: u32,
    pub rgba8_data: Vec<u8>,
}

#[derive(Default)]
pub struct TextureServer {
    next_id: AtomicU32,
    pub pending: Mutex<Vec<PendingTexture>>,
}

impl TextureServer {
    pub fn new() -> Self {
        Self {
            // Reserve 0 for the default/fallback blank texture
            next_id: AtomicU32::new(1),
            pending: Mutex::new(Vec::new()),
        }
    }

    pub fn load_rgba8(&self, width: u32, height: u32, data: Vec<u8>) -> TextureHandle {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let handle = TextureHandle(id);

        self.pending.lock().unwrap().push(PendingTexture {
            handle,
            width,
            height,
            rgba8_data: data,
        });

        handle
    }

    pub fn load_from_bytes(&self, bytes: &[u8]) -> Result<TextureHandle, String> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| e.to_string())?
            .to_rgba8();
        let (width, height) = img.dimensions();
        Ok(self.load_rgba8(width, height, img.into_raw()))
    }
}
