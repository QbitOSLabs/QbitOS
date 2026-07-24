// QbitOS services build script.
//
// Generates Rust types from the Protocol Buffers schema when the `grpc`
// feature is enabled. When `grpc` is disabled, this is a no-op.

#[cfg(feature = "grpc")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .out_dir(std::env::var("OUT_DIR").unwrap())
        .compile_protos(&["proto/qbitos.proto"], &["proto"])?;
    println!("cargo:rerun-if-changed=proto/qbitos.proto");
    Ok(())
}

#[cfg(not(feature = "grpc"))]
fn main() {
    // No-op when grpc feature is disabled.
}
