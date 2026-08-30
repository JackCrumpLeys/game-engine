//! Analytical TrueType font rasterizer via GPU signed distance fields.

use game_engine_shaders_types::packet::GpuCurve;
use glam::{Vec2, vec2};
use std::collections::HashMap;
use ttf_parser::OutlineBuilder;

#[derive(Clone)]
pub struct FontAtlas {
    /// A massive buffer of all analytical Bezier curves across all glyphs
    pub curves: Vec<GpuCurve>,
    pub glyphs: HashMap<char, GlyphMetrics>,
    pub ascender: f32,
    pub descender: f32,
    pub line_gap: f32,
    pub units_per_em: f32,
}

#[derive(Clone, Debug)]
pub struct GlyphMetrics {
    pub line_start: u32,
    pub line_count: u32,
    pub advance: f32,
    pub bearing_x: f32,
    pub bearing_y: f32,
    pub bbox_min: Vec2,
    pub bbox_max: Vec2,
}

pub fn load_font(font_bytes: &[u8]) -> Result<FontAtlas, String> {
    let face = ttf_parser::Face::parse(font_bytes, 0).map_err(|e| format!("TTF parse: {e}"))?;

    let units_per_em = face.units_per_em() as f32;
    let ascender = face.ascender() as f32;
    let descender = face.descender() as f32;
    let line_gap = face.line_gap() as f32;

    let mut curves = Vec::new();
    let mut glyphs = HashMap::new();
    let charset: Vec<char> = (0x20u8..=0x7E).map(|b| b as char).collect();

    for &ch in &charset {
        if let Some(glyph_id) = face.glyph_index(ch) {
            let advance = face.glyph_hor_advance(glyph_id).unwrap_or(0) as f32;
            let line_start = curves.len() as u32;

            let mut builder = PathCurveBuilder {
                curves: &mut curves,
                current_pos: Vec2::ZERO,
                contour_start: Vec2::ZERO,
                bbox_min: Vec2::splat(f32::MAX),
                bbox_max: Vec2::splat(f32::MIN),
            };

            if face.outline_glyph(glyph_id, &mut builder).is_some() {
                builder.close(); // Ensure contour finishes off
            }

            let line_count = (builder.curves.len() as u32) - line_start;
            let bbox_min = if line_count > 0 {
                builder.bbox_min
            } else {
                Vec2::ZERO
            };
            let bbox_max = if line_count > 0 {
                builder.bbox_max
            } else {
                Vec2::ZERO
            };

            glyphs.insert(
                ch,
                GlyphMetrics {
                    line_start,
                    line_count,
                    advance,
                    bearing_x: bbox_min.x,
                    bearing_y: bbox_max.y,
                    bbox_min,
                    bbox_max,
                },
            );
        }
    }

    Ok(FontAtlas {
        curves,
        glyphs,
        ascender,
        descender,
        line_gap,
        units_per_em,
    })
}

struct PathCurveBuilder<'a> {
    curves: &'a mut Vec<GpuCurve>,
    current_pos: Vec2,
    contour_start: Vec2,
    bbox_min: Vec2,
    bbox_max: Vec2,
}

impl<'a> PathCurveBuilder<'a> {
    fn add_point(&mut self, p: Vec2) {
        self.bbox_min = self.bbox_min.min(p);
        self.bbox_max = self.bbox_max.max(p);
    }

    fn add_curve(&mut self, p1: Vec2, p2: Vec2) {
        self.curves.push(GpuCurve {
            p0: self.current_pos,
            p1,
            p2,
            _pad: Vec2::ZERO,
        });
        self.add_point(p1);
        self.add_point(p2);
        self.current_pos = p2;
    }

    // Recursively split cubics into quadratics (or lines) because our GPU only does quadratics
    fn flatten_cubic(&mut self, p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2) {
        let dev1 = p0 - 2.0 * p1 + p2;
        let dev2 = p1 - 2.0 * p2 + p3;
        if dev1.length_squared().max(dev2.length_squared()) < 64.0 {
            // Treat as a line (degenerate quadratic where p1 is midpoint)
            self.add_curve((self.current_pos + p3) * 0.5, p3);
        } else {
            let p01 = (p0 + p1) * 0.5;
            let p12 = (p1 + p2) * 0.5;
            let p23 = (p2 + p3) * 0.5;
            let p012 = (p01 + p12) * 0.5;
            let p123 = (p12 + p23) * 0.5;
            let p0123 = (p012 + p123) * 0.5;
            self.flatten_cubic(p0, p01, p012, p0123);
            self.flatten_cubic(p0123, p123, p23, p3);
        }
    }
}

impl<'a> ttf_parser::OutlineBuilder for PathCurveBuilder<'a> {
    fn move_to(&mut self, x: f32, y: f32) {
        let p = vec2(x, y);
        self.current_pos = p;
        self.contour_start = p;
        self.add_point(p);
    }
    fn line_to(&mut self, x: f32, y: f32) {
        let p2 = vec2(x, y);
        let p1 = (self.current_pos + p2) * 0.5; // Degenerate quadratic
        self.add_curve(p1, p2);
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.add_curve(vec2(x1, y1), vec2(x, y));
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.flatten_cubic(self.current_pos, vec2(x1, y1), vec2(x2, y2), vec2(x, y));
    }
    fn close(&mut self) {
        if self.current_pos.distance_squared(self.contour_start) > 1e-4 {
            let p1 = (self.current_pos + self.contour_start) * 0.5;
            self.add_curve(p1, self.contour_start);
        }
    }
}
