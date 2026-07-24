//! Persistence layer.
//!
//! Saves and loads the QbitOS state (memory episodes, skills, world model,
//! dataset catalog) to a JSON file on disk so the CLI survives across
//! invocations.
//!
//! The state is stored at `<data_dir>/qbitos-state.json` by default. The
//! data dir is `$XDG_DATA_HOME/qbitos` or `~/.local/share/qbitos` on Linux,
//! `~/Library/Application Support/qbitos` on macOS, and
//! `%APPDATA%\qbitos` on Windows.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use qbitos_memory::episodic::Episode;
use qbitos_skills::registry::Skill;

use crate::app::App;

/// The on-disk state snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    /// Schema version.
    pub version: u32,
    /// When the snapshot was taken.
    pub saved_at: chrono::DateTime<chrono::Utc>,
    /// All episodic memories.
    pub episodes: Vec<Episode>,
    /// All registered skills.
    pub skills: Vec<Skill>,
    /// Knowledge graph triples.
    pub triples: Vec<TripleSnapshot>,
    /// Entities.
    pub entities: Vec<EntitySnapshot>,
    /// Dataset catalog entries.
    pub datasets: Vec<DatasetEntrySnapshot>,
}

/// A serializable knowledge-graph triple.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleSnapshot {
    /// Subject label.
    pub subject: String,
    /// Predicate.
    pub predicate: String,
    /// Object label.
    pub object: String,
}

/// A serializable entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntitySnapshot {
    /// Entity name.
    pub name: String,
    /// Entity kind (as string).
    pub kind: String,
    /// Attributes.
    pub attributes: std::collections::BTreeMap<String, serde_json::Value>,
}

/// A serializable dataset catalog entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetEntrySnapshot {
    /// Entry name.
    pub name: String,
    /// Path or URL.
    pub path: String,
    /// Format id.
    pub format: String,
    /// Tags.
    pub tags: Vec<String>,
}

/// The persistence manager.
pub struct Persistence {
    state_path: PathBuf,
}

impl Persistence {
    /// Create a new persistence manager with the default data directory.
    #[must_use]
    pub fn new() -> Self {
        let data_dir = Self::default_data_dir();
        std::fs::create_dir_all(&data_dir).ok();
        Self {
            state_path: data_dir.join("qbitos-state.json"),
        }
    }

    /// Create a persistence manager with an explicit state path.
    #[must_use]
    pub fn with_path(path: PathBuf) -> Self {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        Self { state_path: path }
    }

    /// Returns the default data directory for the current platform.
    fn default_data_dir() -> PathBuf {
        if let Ok(dir) = std::env::var("QBITOS_DATA_DIR") {
            return PathBuf::from(dir);
        }
        if let Ok(dir) = std::env::var("XDG_DATA_HOME") {
            return PathBuf::from(dir).join("qbitos");
        }
        if let Some(home) = dirs_or_home() {
            #[cfg(target_os = "macos")]
            {
                return home.join("Library").join("Application Support").join("qbitos");
            }
            #[cfg(target_os = "windows")]
            {
                return home.join("AppData").join("Roaming").join("qbitos");
            }
            #[cfg(not(any(target_os = "macos", target_os = "windows")))]
            {
                return home.join(".local").join("share").join("qbitos");
            }
        }
        PathBuf::from("/tmp/qbitos")
    }

    /// Returns the state file path.
    #[must_use]
    pub fn path(&self) -> &Path { &self.state_path }

    /// Save the current state of the app to disk.
    ///
    /// # Errors
    /// Returns an error if serialization or writing fails.
    pub fn save(&self, app: &App) -> anyhow::Result<()> {
        let snapshot = self.snapshot(app);
        let json = serde_json::to_string_pretty(&snapshot)?;
        std::fs::write(&self.state_path, json)?;
        tracing::info!(target: "qbitos.persistence", path = %self.state_path.display(), "state saved");
        Ok(())
    }

    /// Load the state from disk into the app.
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or parsed.
    pub fn load(&self, app: &App) -> anyhow::Result<()> {
        if !self.state_path.exists() {
            tracing::debug!(target: "qbitos.persistence", "no state file, starting fresh");
            return Ok(());
        }
        let json = std::fs::read_to_string(&self.state_path)?;
        let snapshot: StateSnapshot = serde_json::from_str(&json)?;
        self.restore(&snapshot, app);
        tracing::info!(target: "qbitos.persistence",
            episodes = snapshot.episodes.len(),
            skills = snapshot.skills.len(),
            triples = snapshot.triples.len(),
            "state loaded");
        Ok(())
    }

    /// Build a snapshot from the app's current state.
    fn snapshot(&self, app: &App) -> StateSnapshot {
        // Episodes — use recall_similar with a wildcard query to fetch all
        let episodes = app.memory.episodic.recall_similar(
            &serde_json::json!({"query": "*"}),
            usize::MAX,
        );

        // Skills
        let skills = app.skills.registry.list();

        // Knowledge graph triples — now using the public all_triples() method
        let raw_triples = app.world_model.knowledge_graph.all_triples();
        let triples: Vec<TripleSnapshot> = raw_triples.into_iter()
            .map(|(subject, predicate, object)| TripleSnapshot {
                subject, predicate, object,
            })
            .collect();

        // Datasets — we don't track them yet in the app
        let datasets = Vec::new();

        StateSnapshot {
            version: 1,
            saved_at: chrono::Utc::now(),
            episodes,
            skills,
            triples,
            entities: Vec::new(),
            datasets,
        }
    }

    /// Restore a snapshot into the app.
    fn restore(&self, snapshot: &StateSnapshot, app: &App) {
        // Episodes
        for ep in &snapshot.episodes {
            let content = ep.object.content.clone();
            let outcome = ep.outcome;
            // Re-record the episode
            app.memory.record_episode(content, outcome);
        }
        // Skills
        for skill in &snapshot.skills {
            app.skills.registry.register(skill.clone());
        }
        // Triples
        for triple in &snapshot.triples {
            app.world_model.knowledge_graph.add(
                &triple.subject,
                &triple.predicate,
                &triple.object,
            );
        }
    }

    /// Returns `true` if a state file exists.
    #[must_use]
    pub fn exists(&self) -> bool {
        self.state_path.exists()
    }
}

impl Default for Persistence {
    fn default() -> Self {
        Self::new()
    }
}

/// Get the user's home directory (without depending on the `dirs` crate).
fn dirs_or_home() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistence_round_trip() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let persistence = Persistence::with_path(tmp.path().to_path_buf());
        let app = Arc::new(App::new());

        // Add some state
        app.memory.record_episode(
            serde_json::json!({"event": "test"}),
            Outcome::Success,
        );
        app.skills.registry.register(Skill::new("test_skill", "a test"));
        app.world_model.knowledge_graph.add("dog", "is_a", "mammal");

        // Save
        persistence.save(&app).unwrap();
        assert!(persistence.exists());

        // Load into a fresh app
        let app2 = Arc::new(App::new());
        persistence.load(&app2).unwrap();

        // Verify
        assert_eq!(app2.memory.episodic.len(), 1);
        assert!(app2.skills.registry.get_by_name("test_skill").is_some());
    }
}
