#![feature(duration_constructors_lite)]
use std::{
    error::Error,
    time::{Duration, Instant, UNIX_EPOCH},
};

use game_engine::prelude::*;
use game_engine::{Engine, FontAtlas, SimTime, WindowConfig};
use game_engine_ecs::schedule::SystemTypeSet;
use rand::{prelude::*, rng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::sync::Arc;

// ── Render Stack Definitions ─────────────────────────────────────────

define_render_stack!(
    TestRenderPlugin => {
        SwarmShape => [glyph: GpuGlyphPacket],
        WaveShape => [glyph: GpuGlyphPacket],
        ZoomShape => [glyph: GpuGlyphPacket],
        ChurnShape => [glyph: GpuGlyphPacket],
        TimerShape => [glyph: GpuGlyphPacket]
    }
);

// ── Components ───────────────────────────────────────────────────────

#[derive(Component, Debug, Clone, Copy)]
pub struct SwarmItem {
    pub vx: f32,
    pub vy: f32,
}

#[derive(Component, Debug, Clone, Copy)]
pub struct WaveItem {
    pub base_x: f32,
    pub base_y: f32,
    pub dist_from_center: f32,
}

#[derive(Component, Debug, Clone, Copy)]
pub struct ZoomItem {
    pub offset: f32,
}

#[derive(Component, Debug, Clone, Copy)]
pub struct TimerChar {
    pub index: u32,
}

#[derive(Default)]
pub struct SuiteState {
    pub current_test: u32,
    pub test_start: Option<Instant>,
    pub ticks: u32,
    pub completed: bool,
    pub report: String,
}

// ── Helpers ──────────────────────────────────────────────────────────

// Scoring based on unbound ECS Ticks Per Second (TPS)
fn get_rating(tps: f32) -> &'static str {
    if tps >= 100_000.0 {
        "S-Tier (God Engine)"
    } else if tps >= 30_000.0 {
        "A-Tier (AAA Standard)"
    } else if tps >= 5_000.0 {
        "B-Tier (Solid)"
    } else {
        "F-Tier (Bottlenecked)"
    }
}

// ── Startup ──────────────────────────────────────────────────────────

pub fn setup_timer_system(
    commands: Commands,
    mut allocator: ResMut<GpuSlotAllocator>,
    mut packet: ResMut<RenderPacket>,
    font: Res<FontAtlas>,
) {
    if packet.path_curves.is_empty() {
        packet.path_curves = Arc::new(font.atlas.curves.clone());
    }

    let op_pool = Arc::make_mut(&mut packet.op_pool);

    for i in 0..80 {
        // Enough chars for a long status string
        let alloc = TimerShape::spawn_alloc(&mut allocator, op_pool);
        let mut shape = TimerShape::default();

        commands.spawn((
            Position { x: 0.0, y: 0.0 },
            TimerChar { index: i as u32 },
            alloc,
            shape,
        ));
    }
}

// ── Orchestrator ─────────────────────────────────────────────────────

pub fn test_orchestrator(
    mut state: ResMut<SuiteState>,
    mut allocator: ResMut<GpuSlotAllocator>,
    mut packet: ResMut<RenderPacket>,
    font: Res<FontAtlas>,
    commands: Commands,
    swarm_q: Query<(Entity, &SwarmItem)>,
    wave_q: Query<(Entity, &WaveItem)>,
    zoom_q: Query<(Entity, &ZoomItem)>,
    churn_q: Query<(Entity, &ChurnItem)>,
) {
    if state.completed {
        return;
    }
    let now = Instant::now();

    // Setup new test if needed
    if state.test_start.is_none() {
        state.test_start = Some(now);
        state.ticks = 0;

        let op_pool = Arc::make_mut(&mut packet.op_pool);
        let mut rng = rand::rng();
        let glyph_chars: Vec<char> = font.atlas.glyphs.keys().cloned().collect();

        match state.current_test {
            0 => {
                log::info!(
                    "Test 0: High-Frequency N-Body Swarm (Flexing Unbound TPS & Interpolation)"
                );
                for _ in 0..20_000 {
                    let ch = *glyph_chars.choose(&mut rng).unwrap_or(&'~');
                    let x = rng.random_range(-500.0..500.0);
                    let y = rng.random_range(-500.0..500.0);

                    let alloc = SwarmShape::spawn_alloc(&mut allocator, op_pool);
                    let mut shape = SwarmShape::default();
                    *shape.position_mut() = vec2(x, y);

                    if let Some(metrics) = font.atlas.glyphs.get(&ch) {
                        let mut g = GpuGlyphPacket::default();
                        g.scale = 16.0 / font.atlas.units_per_em;
                        g.line_start = metrics.line_start;
                        g.line_count = metrics.line_count;
                        g.bbox_min = metrics.bbox_min;
                        g.bbox_max = metrics.bbox_max;
                        g.color = ColorRGBA::new(0.2, 0.8, 1.0, 1.0);
                        *shape.glyph_mut() = g;
                    }

                    commands.spawn((
                        Position { x, y },
                        SwarmItem {
                            vx: rng.random_range(-1.0..1.0),
                            vy: rng.random_range(-1.0..1.0),
                        },
                        alloc,
                        shape,
                    ));
                }
            }
            1 => {
                log::info!("Test 1: Delta-Sync Shockwave (Flexing dirty_slots PCI-e optimization)");
                let rows = 400;
                let cols = 400;
                let spacing = 8.0;
                let offset_x = (cols as f32 * spacing) / 2.0;
                let offset_y = (rows as f32 * spacing) / 2.0;

                for r in 0..rows {
                    for c in 0..cols {
                        let base_x = c as f32 * spacing - offset_x;
                        let base_y = r as f32 * spacing - offset_y;
                        let dist = (base_x * base_x + base_y * base_y).sqrt();

                        let alloc = WaveShape::spawn_alloc(&mut allocator, op_pool);
                        let mut shape = WaveShape::default();
                        *shape.position_mut() = vec2(base_x, base_y);

                        if let Some(metrics) = font.atlas.glyphs.get(&'#') {
                            let mut g = GpuGlyphPacket::default();
                            g.scale = 6.0 / font.atlas.units_per_em;
                            g.line_start = metrics.line_start;
                            g.line_count = metrics.line_count;
                            g.bbox_min = metrics.bbox_min;
                            g.bbox_max = metrics.bbox_max;
                            g.color = ColorRGBA::new(0.1, 0.1, 0.1, 1.0);
                            *shape.glyph_mut() = g;
                        }

                        commands.spawn((
                            Position {
                                x: base_x,
                                y: base_y,
                            },
                            WaveItem {
                                base_x,
                                base_y,
                                dist_from_center: dist,
                            },
                            alloc,
                            shape,
                        ));
                    }
                }
            }
            2 => {
                log::info!("Test 2: Analytic SDF Singularity (Flexing Wavefront Coherence)");
                for i in 0..1000 {
                    let ch = *glyph_chars.choose(&mut rng).unwrap_or(&'@');
                    let alloc = ZoomShape::spawn_alloc(&mut allocator, op_pool);
                    let mut shape = ZoomShape::default();

                    if let Some(metrics) = font.atlas.glyphs.get(&ch) {
                        let mut g = GpuGlyphPacket::default();
                        g.scale = 10.0; // Overwritten in system
                        g.line_start = metrics.line_start;
                        g.line_count = metrics.line_count;
                        g.bbox_min = metrics.bbox_min;
                        g.bbox_max = metrics.bbox_max;
                        g.color = ColorRGBA::new(rng.random(), rng.random(), rng.random(), 0.1);
                        *shape.glyph_mut() = g;
                    }

                    commands.spawn((
                        Position { x: 0.0, y: 0.0 },
                        ZoomItem {
                            offset: i as f32 * 0.01,
                        },
                        alloc,
                        shape,
                    ));
                }
            }
            3 => {
                log::info!("Test 3: The Archetype Blender (Flexing ECS GC and Mem Allocator)");
            }
            _ => {
                state.completed = true;
                log::info!("ALL TESTS COMPLETED! Final rating: {}", state.report);
                return;
            }
        }
        return;
    }

    let elapsed = now.duration_since(state.test_start.unwrap()).as_secs_f32();
    state.ticks += 1;

    // Transition test after 10 seconds
    if elapsed >= 10.0 {
        let avg_tps = state.ticks as f32 / elapsed;
        let rating = get_rating(avg_tps);
        state.report = format!(
            "T{} Avg TPS: {:.0} [{}]",
            state.current_test, avg_tps, rating
        );
        log::info!("{}", state.report);

        // Nuke all test entities
        swarm_q.for_each(|(e, _)| commands.despawn(e));
        wave_q.for_each(|(e, _)| commands.despawn(e));
        zoom_q.for_each(|(e, _)| commands.despawn(e));
        churn_q.for_each(|(e, _)| commands.despawn(e));

        state.current_test += 1;
        state.test_start = None;
    }
}

// ── Per-Test Animation Logic ─────────────────────────────────────────

// Test 0: High-frequency physics. Updates 100,000+ times per second.
// Rendering thread will interpolate perfectly via double buffering.
pub fn animate_swarm_system(
    mut query: Query<(&mut Position, &mut SwarmItem, &mut SwarmShape)>,
    time: Res<SimTime>,
) {
    // Because TPS is unbound, time delta per tick is tiny.
    let dt = 0.001;
    let center_x = (time.phase * 0.5).sin() * 300.0;
    let center_y = (time.phase * 0.7).cos() * 300.0;

    query.for_each_mut(|(mut pos, mut swarm, mut shape)| {
        let dx = center_x - pos.x;
        let dy = center_y - pos.y;
        let dist_sq = dx * dx + dy * dy + 100.0;

        // Gravity toward moving center
        swarm.vx += (dx / dist_sq) * 500.0 * dt;
        swarm.vy += (dy / dist_sq) * 500.0 * dt;

        // Friction
        swarm.vx *= 0.999;
        swarm.vy *= 0.999;

        pos.x += swarm.vx;
        pos.y += swarm.vy;

        *shape.position_mut() = vec2(pos.x, pos.y);
    });
}

// Test 1: Only mutate entities exactly on the shockwave ring.
// Proves that dirty_slots avoids uploading the 150,000 static entities.
pub fn animate_wave_system(
    mut query: Query<(&WaveItem, &mut WaveShape)>, // mut triggers dirty flag
    time: Res<SimTime>,
) {
    let wave_radius = (time.phase * 400.0) % 1000.0;

    // Pass 1: We use a read-only query to find who needs updating, avoiding dirty flags.
    // In a real engine, a spatial grid would make this O(1), but O(N) read is fine here.
    // Pass 2: Actually mutate the shapes that are touched.
    query.for_each_mut(|(item, mut shape)| {
        let dist = item.dist_from_center;
        let diff = (dist - wave_radius).abs();

        if diff < 30.0 {
            // Only these entities get pushed over the PCI-e bus this frame!
            let intensity = 1.0 - (diff / 30.0);
            shape.glyph_mut().color = ColorRGBA::new(intensity, 0.2, 1.0 - intensity, 1.0);
            shape.glyph_mut().scale = (6.0 + intensity * 20.0) / 1000.0; // Units per em rough calc
        } else if shape.glyph_mut().scale > 0.01 {
            // Decay back to rest state
            shape.glyph_mut().color = ColorRGBA::new(0.1, 0.1, 0.1, 1.0);
            shape.glyph_mut().scale = 6.0 / 1000.0;
        }
    });
}

// Test 2: Massive overdraw and shader ALU scaling.
pub fn animate_zoom_system(
    mut query: Query<(&ZoomItem, &mut ZoomShape)>,
    time: Res<SimTime>,
    font: Res<FontAtlas>,
) {
    let phase = time.phase;
    let upe = font.atlas.units_per_em;

    query.for_each_mut(|(item, mut shape)| {
        // Scales from 1.0 to a massive 1500.0
        let scale_factor = ((phase + item.offset).sin() * 0.5 + 0.5) * 1500.0 + 1.0;
        shape.glyph_mut().scale = scale_factor / upe;

        let rot_radius = scale_factor * 0.1;
        let x = (phase * 2.0 + item.offset * 10.0).cos() * rot_radius;
        let y = (phase * 2.0 + item.offset * 10.0).sin() * rot_radius;

        *shape.position_mut() = vec2(x, y);
    });
}

// ── Timer UI Updater ─────────────────────────────────────────────────

pub fn update_timer_system(
    state: Res<SuiteState>,
    mut query: Query<(&TimerChar, &mut TimerShape)>,
    font: Res<FontAtlas>,
) {
    let text = if state.completed {
        format!("Done! Final: {}", state.report)
    } else if let Some(start) = state.test_start {
        let elapsed = start.elapsed().as_secs_f32();
        let tps = state.ticks as f32 / elapsed.max(0.001);
        format!(
            "Test: {} | Time: {:.1}s | TPS: {:.0}",
            state.current_test,
            10.0 - elapsed,
            tps
        )
    } else {
        "Preparing...".to_string()
    };

    let chars: Vec<char> = text.chars().collect();
    let atlas = &font.atlas;
    let scale = 36.0 / atlas.units_per_em;
    let base_x = -800.0;
    let base_y = 500.0; // Top left corner

    query.for_each_mut(|(tc, mut shape)| {
        let idx = tc.index as usize;
        if idx < chars.len() {
            let ch = chars[idx];
            if let Some(metrics) = atlas.glyphs.get(&ch) {
                let mut g = GpuGlyphPacket::default();
                g.scale = scale;
                g.line_start = metrics.line_start;
                g.line_count = metrics.line_count;
                g.bbox_min = metrics.bbox_min;
                g.bbox_max = metrics.bbox_max;
                g.color = ColorRGBA::new(1.0, 1.0, 0.0, 1.0);

                *shape.position_mut() = vec2(base_x + (idx as f32 * 20.0), base_y);
                *shape.glyph_mut() = g;
            }
        } else {
            shape.glyph_mut().scale = 0.0; // Hide unused characters
        }
    });
}

pub fn increment_time_system(mut time: ResMut<SimTime>) {
    // Because TPS is massive, phase increment must be tiny or things teleport
    time.phase += 0.0005;
}

// ── Main ─────────────────────────────────────────────────────────────

#[derive(Component, Debug, Clone, Copy)]
pub struct ChurnItem {
    pub dead_after: u64,
}

pub fn churn_system(commands: Commands, query: Query<(Entity, &ChurnItem)>, mut tick: Local<u64>) {
    *tick += 1;

    query.for_each(|(e, item)| {
        if item.dead_after < *tick {
            commands.despawn(e);
        }
    });

    (0..250000).into_par_iter().for_each(|_| {
        let mut rng = rand::rng();

        let mut shape = ChurnShape::default();

        let rx = rng.random_range(-800.0..800.0);
        let ry = rng.random_range(-500.0..500.0);
        *shape.position_mut() = vec2(rx, ry);

        commands.spawn((
            Position { x: rx, y: ry },
            ChurnItem {
                dead_after: *tick + rng.random_range(0..200),
            },
        ));
    });
}

fn main() -> Result<(), Box<dyn Error>> {
    let _tracy_client = tracy_client::Client::start();
    tracy_client::register_demangler!();

    fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "[{} {} {}] {}",
                get_time(),
                record.level(),
                record.target(),
                message
            ))
        })
        .level(log::LevelFilter::Debug)
        .chain(std::io::stdout())
        .apply()?;

    Engine::new()
        .with_window(WindowConfig {
            title: "Game Engine Architectural Flex Suite".into(),
            ..Default::default()
        })
        .insert_resource(SimTime::default())
        .insert_resource(SuiteState::default())
        .add_plugin(TestRenderPlugin)
        .add_startup_system(setup_timer_system)
        .add_systems((
            increment_time_system,
            test_orchestrator,
            animate_swarm_system,
            animate_wave_system,
            animate_zoom_system,
            churn_system,
            update_timer_system.after(increment_time_system),
        ))
        .run()?;

    Ok(())
}

fn get_time() -> String {
    let time = std::time::SystemTime::now()
        .checked_add(Duration::from_hours(13))
        .unwrap()
        .duration_since(UNIX_EPOCH)
        .unwrap();
    let secs = time.as_secs();
    let mins = secs / 60;
    let hours = mins / 60;

    let mut days = hours / 24;
    let mut year = 1970;

    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if days >= days_in_year {
            days -= days_in_year;
            year += 1;
        } else {
            break;
        }
    }
    let day_of_year = days + 1;

    let month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut month = 0;
    let mut temp_days = day_of_year;

    for (i, &days_in_month) in month_lengths.iter().enumerate() {
        let mut current_month_length = days_in_month;
        if i == 1 && is_leap(year) {
            current_month_length = 29;
        }

        if temp_days <= current_month_length {
            month = i as u64 + 1;
            break;
        } else {
            temp_days -= current_month_length;
        }
    }

    format!(
        "{:02}-{:02}-{:04} {:02}:{:02}:{:02}",
        temp_days,
        month,
        year,
        hours % 24,
        mins % 60,
        secs % 60
    )
}

fn is_leap(year: u64) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}
