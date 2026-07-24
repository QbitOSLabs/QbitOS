//! qbitos-data CLI: list loaders, inspect files, convert formats.

use std::path::PathBuf;
use clap::{Parser, Subcommand};

use qbitos_data::{
    DatasetLoader, DatasetRecordStream, DatasetRegistry, LoadOptions, SyntheticGenerator, SyntheticSpec,
};

#[derive(Parser)]
#[command(name = "qbitos-data")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "QbitOS dataset loader CLI — 40+ formats supported")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List all registered loaders.
    ListLoaders,
    /// Show information about a dataset (format, schema, record count).
    Inspect {
        /// Path or URL to the dataset.
        path: PathBuf,
    },
    /// Load a dataset and print records as JSONL to stdout.
    Load {
        /// Path or URL to the dataset.
        path: PathBuf,
        /// Maximum number of records to print.
        #[arg(long, default_value = "10")]
        max: usize,
        /// Offset to start from.
        #[arg(long, default_value = "0")]
        offset: usize,
    },
    /// Generate a synthetic dataset and print records as JSONL.
    Synthetic {
        /// Kind of synthetic data.
        #[arg(long, value_enum, default_value = "random-json")]
        kind: SyntheticKindArg,
        /// Number of records.
        #[arg(long, default_value = "100")]
        count: usize,
        /// Random seed.
        #[arg(long, default_value = "42")]
        seed: u64,
    },
}

#[derive(clap::ValueEnum, Clone, Copy)]
enum SyntheticKindArg {
    #[value(name = "random-json")]
    RandomJson,
    #[value(name = "conversation")]
    Conversation,
    #[value(name = "trajectory")]
    Trajectory,
    #[value(name = "preference")]
    Preference,
    #[value(name = "numbers")]
    Numbers,
}

impl From<SyntheticKindArg> for qbitos_data::SyntheticKind {
    fn from(arg: SyntheticKindArg) -> Self {
        match arg {
            SyntheticKindArg::RandomJson => Self::RandomJson,
            SyntheticKindArg::Conversation => Self::Conversation,
            SyntheticKindArg::Trajectory => Self::Trajectory,
            SyntheticKindArg::Preference => Self::Preference,
            SyntheticKindArg::Numbers => Self::Numbers,
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let registry = DatasetRegistry::with_default_loaders();

    match cli.command {
        Command::ListLoaders => {
            println!("{:<15} {:<10} {}", "ID", "Extensions", "Name");
            println!("{}", "-".repeat(80));
            for loader in registry.list() {
                println!("{:<15} {:<10} {}", loader.id(), loader.extensions().join(","), loader.name());
            }
        }
        Command::Inspect { path } => {
            let loader = registry.select(&path)?;
            println!("Format:  {}", loader.id());
            println!("Name:    {}", loader.name());
            let schema = loader.detect_schema(&path).await?;
            if !schema.is_empty() {
                println!("Schema:");
                for field in &schema.fields {
                    println!("  - {}: {:?}{}", field.name, field.field_type, if field.nullable { " (nullable)" } else { "" });
                }
            }
            let mut stream = loader.load_with_options(&path, &LoadOptions::new().with_max_records(1)).await?;
            let mut count = 0;
            while stream.next_record().await?.is_some() {
                count += 1;
            }
            println!("Sample records: {count}");
        }
        Command::Load { path, max, offset } => {
            let loader = registry.select(&path)?;
            let opts = LoadOptions::new().with_max_records(max).with_offset(offset);
            let mut stream = loader.load_with_options(&path, &opts).await?;
            while let Some(record) = stream.next_record().await? {
                println!("{}", serde_json::to_string(&record)?);
            }
        }
        Command::Synthetic { kind, count, seed } => {
            let kind = match kind {
                SyntheticKindArg::RandomJson => qbitos_data::SyntheticKind::RandomJson,
                SyntheticKindArg::Conversation => qbitos_data::SyntheticKind::Conversation,
                SyntheticKindArg::Trajectory => qbitos_data::SyntheticKind::Trajectory,
                SyntheticKindArg::Preference => qbitos_data::SyntheticKind::Preference,
                SyntheticKindArg::Numbers => qbitos_data::SyntheticKind::Numbers,
            };
            let spec = SyntheticSpec { kind, count, seed };
            let generator = SyntheticGenerator::new(spec);
            for record in generator.generate() {
                println!("{}", serde_json::to_string(&record)?);
            }
        }
    }
    Ok(())
}
