//! `qbitos` CLI binary — top-level entry point.

use std::sync::Arc;

use clap::Parser;
use qbitos_cli::{commands::Command, init_tracing, App};

#[derive(Parser)]
#[command(name = "qbitos")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "QbitOS — a Rust 2024 native multimodal cognitive AI operating system")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();
    let cli = Cli::parse();
    let app = Arc::new(App::new());

    match cli.command {
        Some(command) => qbitos_cli::commands::dispatch(&app, command).await,
        None => qbitos_cli::commands::dispatch(&app, Command::Repl).await,
    }
}
