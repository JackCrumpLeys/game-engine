
import os

# Define the precise file mappings (1-based line ranges, inclusive)
FILE_MAPPINGS = [
    ("crates/game_engine_derive/src/lib.rs", 2, 155),
    ("crates/game_engine_ecs/benches/bevy_v_me.rs", 156, 367),
    ("crates/game_engine_ecs/benches/flush.rs", 368, 484),
    ("crates/game_engine_ecs/benches/iter_bench.rs", 485, 795),
    ("crates/game_engine_ecs/benches/real_tough.rs", 796, 993),
    ("crates/game_engine_ecs/benches/spawn_bench.rs", 994, 1255),
    ("crates/game_engine_ecs/benches/system_ecs_vs_vec.rs", 1256, 1659),
    ("crates/game_engine_ecs/examples/complex_demo.rs", 1660, 1820),
    ("crates/game_engine_ecs/examples/get.rs", 1821, 2033),
    ("crates/game_engine_ecs/examples/minimal_all.rs", 2034, 2265),
    ("crates/game_engine_ecs/examples/perf_spawn.rs", 2266, 2338),
    ("crates/game_engine_ecs/src/app.rs", 2339, 2378),
    ("crates/game_engine_ecs/src/archetype.rs", 2379, 2826),
    ("crates/game_engine_ecs/src/borrow.rs", 2827, 3050),
    ("crates/game_engine_ecs/src/bundle.rs", 3051, 3224),
    ("crates/game_engine_ecs/src/component.rs", 3225, 3624),
    ("crates/game_engine_ecs/src/entity.rs", 3625, 3962),
    ("crates/game_engine_ecs/src/lib.rs", 3963, 4030),
    ("crates/game_engine_ecs/src/macros.rs", 4031, 4118),
    ("crates/game_engine_ecs/src/message.rs", 4119, 4528),
    ("crates/game_engine_ecs/src/observers.rs", 4529, 5621),
    ("crates/game_engine_ecs/src/query.rs", 5622, 7464),
    ("crates/game_engine_ecs/src/resource.rs", 7465, 7827),
    ("crates/game_engine_ecs/src/schedule.rs", 7828, 8787),
    ("crates/game_engine_ecs/src/storage.rs", 8788, 9842),
    ("crates/game_engine_ecs/src/system.rs", 9843, 10439),
    ("crates/game_engine_ecs/src/system/command.rs", 10440, 11178),
    ("crates/game_engine_ecs/src/system/function.rs", 11179, 11865),
    ("crates/game_engine_ecs/src/thread_entity_allocator.rs", 11865, 12056),
    ("crates/game_engine_ecs/src/threading.rs", 12056, 12437),
    ("crates/game_engine_ecs/src/world.rs", 12437, 13801),
    ("crates/game_engine_render/src/lib.rs", 13802, 13803),
    ("crates/game_engine_render/src/render_ops.rs", 13804, 13933),
    ("crates/game_engine_render/src/shape_markers.rs", 13934, 14059),
    ("crates/game_engine_render/src/slot_allocator.rs", 14060, 14197),
    ("crates/game_engine_shaders/src/lib.rs", 14198, 14255),
    ("crates/game_engine_shaders/src/macros.rs", 14256, 14457),
    ("crates/game_engine_shaders_types/src/lib.rs", 14458, 14524),
    ("crates/game_engine_shaders_types/src/packet.rs", 14525, 14694),
    ("crates/game_engine_shaders_types/src/sdf.rs", 14695, 14740),
    ("crates/game_engine_shaders_types/src/shapes/bezier_path.rs", 14741, 14932),
    ("crates/game_engine_shaders_types/src/shapes/circle.rs", 14933, 14972),
    ("crates/game_engine_shaders_types/src/shapes/clip.rs", 14973, 15018),
    ("crates/game_engine_shaders_types/src/shapes/data.rs", 15019, 15102),
    ("crates/game_engine_shaders_types/src/shapes/glyph.rs", 15103, 15217),
    ("crates/game_engine_shaders_types/src/shapes/gradient.rs", 15218, 15293),
    ("crates/game_engine_shaders_types/src/shapes/line_segment.rs", 15294, 15340),
    ("crates/game_engine_shaders_types/src/shapes/mod.rs", 15341, 15365),
    ("crates/game_engine_shaders_types/src/shapes/polygon.rs", 15366, 15444),
    ("crates/game_engine_shaders_types/src/shapes/rect.rs", 15445, 15485),
    ("crates/game_engine_shaders_types/src/shapes/rounded_rect.rs", 15486, 15529),
    ("crates/game_engine_shaders_types/src/shapes/sprite.rs", 15530, 15607),
    ("crates/game_engine_shaders_types/src/shapes/stroked.rs", 15608, 15675),
    ("crates/game_engine_utils/src/graph.rs", 15676, 16290),
    ("crates/game_engine_utils/src/lib.rs", 16291, 16388),
    ("src/input.rs", 16389, 16566),
    ("src/lib.rs", 16567, 16833),
    ("src/main.rs", 16834, 17397),
    ("src/render.rs", 17398, 17794),
    ("src/render/components.rs", 17795, 17840),
    ("src/render/packet.rs", 17841, 17955),
    ("src/render/pass.rs", 17956, 18438),
    ("src/render/plugin.rs", 18439, 18489),
    ("src/render/stack.rs", 18490, 18659),
    ("src/render/shaders.rs", 18660, 18664),
    ("src/render/storage.rs", 18665, 20037),
    ("src/render/texture.rs", 20038, 20088),
    ("src/text/font.rs", 20089, 20254),
    ("src/text/mod.rs", 20255, 20257),
]

def main():
    cat_path = 'cachefound/cat'
    if not os.path.exists(cat_path):
        print(f"Error: {cat_path} not found.")
        return

    print("Reading concatenated dump...")
    with open(cat_path, 'r', encoding='utf-8', errors='replace') as f:
        # readlines keeps the trailing newlines intact
        lines = f.readlines()

    print(f"Loaded {len(lines)} lines.")
    
    for filepath, start_line, end_line in FILE_MAPPINGS:
        # Translate 1-based line numbers to 0-based Python indices
        # start_line is index start_line - 1
        # end_line is inclusive, so we slice up to end_line
        file_lines = lines[start_line - 1 : end_line]
        
        # Strip any leading/trailing blank elements cleanly if needed, 
        # but otherwise preserve formatting
        content = "".join(file_lines)
        
        # Create directories if they do not exist
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
            
        with open(filepath, 'w', encoding='utf-8') as out_f:
            out_f.write(content)
            
        print(f"Extracted: {filepath} ({start_line}-{end_line})")

    print("\nExtraction process completed.")

if __name__ == '__main__':
    main()
