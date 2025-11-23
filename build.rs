use spirv_builder::{MetadataPrintout, SpirvBuilder, SpirvMetadata};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=crates/game_engine_shaders/src");
    SpirvBuilder::new("crates/game_engine_shaders", "spirv-unknown-vulkan1.4")
        .print_metadata(MetadataPrintout::Full)
        .spirv_metadata(SpirvMetadata::NameVariables)
        .build()?;
    Ok(())
}
