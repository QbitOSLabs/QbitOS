//! Local config database for the QbitOS desktop app.
//!
//! Uses SQLite (embedded, via `rusqlite` with the `bundled` feature) to persist
//! user-specific data on the local machine:
//!
//! - **Connection settings**: server URLs, auth tokens, API keys
//! - **Query history**: prompts the user has sent, with timestamps and responses
//! - **Engine configs**: HNSW parameters, embedding dimensions, backend preferences
//! - **UI preferences**: theme, language, active view, sidebar state
//!
//! This is NOT a memory backend for the cognitive OS — it is purely the
//! desktop app's local persistence layer. The database file lives at
//! `~/.qbitos/desktop.db` (or the platform equivalent).
//!
//! # Schema
//!
//! ```sql
//! CREATE TABLE connection_settings (
//!     id          INTEGER PRIMARY KEY AUTOINCREMENT,
//!     name        TEXT NOT NULL UNIQUE,
//!     url         TEXT NOT NULL,
//!     auth_token  TEXT,
//!     is_default  BOOLEAN NOT NULL DEFAULT 0,
//!     created_at  TEXT NOT NULL,
//!     updated_at  TEXT NOT NULL
//! );
//!
//! CREATE TABLE query_history (
//!     id          INTEGER PRIMARY KEY AUTOINCREMENT,
//!     prompt      TEXT NOT NULL,
//!     response    TEXT,
//!     endpoint    TEXT,
//!     duration_ms INTEGER,
//!     success     BOOLEAN NOT NULL DEFAULT 1,
//!     created_at  TEXT NOT NULL
//! );
//!
//! CREATE TABLE engine_config (
//!     key         TEXT PRIMARY KEY,
//!     value       TEXT NOT NULL,
//!     category    TEXT NOT NULL,
//!     updated_at  TEXT NOT NULL
//! );
//!
//! CREATE TABLE ui_preferences (
//!     key         TEXT PRIMARY KEY,
//!     value       TEXT NOT NULL,
//!     updated_at  TEXT NOT NULL
//! );
//! ```

use anyhow::{Context, Result};
use chrono::Utc;
use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

/// A saved connection profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionProfile {
    /// Unique database ID.
    pub id: Option<i64>,
    /// Human-readable name (e.g. "Local Dev", "Production").
    pub name: String,
    /// Server URL (e.g. `http://127.0.0.1:8080`).
    pub url: String,
    /// Optional Bearer token.
    pub auth_token: Option<String>,
    /// Whether this is the default connection.
    pub is_default: bool,
    /// Creation timestamp (RFC 3339).
    pub created_at: String,
    /// Last update timestamp (RFC 3339).
    pub updated_at: String,
}

/// A query history entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryHistoryEntry {
    /// Unique database ID.
    pub id: Option<i64>,
    /// The prompt/input the user sent.
    pub prompt: String,
    /// The response received (if any).
    pub response: Option<String>,
    /// Which API endpoint was called.
    pub endpoint: Option<String>,
    /// Request duration in milliseconds.
    pub duration_ms: Option<i64>,
    /// Whether the request succeeded.
    pub success: bool,
    /// Timestamp (RFC 3339).
    pub created_at: String,
}

/// An engine configuration key-value pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfigEntry {
    /// Config key (e.g. "hnsw_ef_search", "embedding_dim").
    pub key: String,
    /// Config value (JSON string).
    pub value: String,
    /// Category (e.g. "vector", "memory", "inference").
    pub category: String,
    /// Last update timestamp.
    pub updated_at: String,
}

/// A UI preference key-value pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiPreference {
    /// Preference key (e.g. "theme", "active_view", "sidebar_width").
    pub key: String,
    /// Preference value (JSON string).
    pub value: String,
    /// Last update timestamp.
    pub updated_at: String,
}

/// The local config database handle. Thread-safe via `Mutex<Connection>`.
pub struct LocalDb {
    conn: Arc<Mutex<Connection>>,
}

impl LocalDb {
    /// Open (or create) the local database at the default path.
    ///
    /// The database file is placed at:
    /// - Linux: `~/.local/share/qbitos/desktop.db`
    /// - macOS: `~/Library/Application Support/qbitos/desktop.db`
    /// - Windows: `%APPDATA%\qbitos\desktop.db`
    ///
    /// # Errors
    /// Returns an error if the directory cannot be created or the database
    /// cannot be opened.
    pub fn open_default() -> Result<Self> {
        let path = Self::default_db_path()?;
        Self::open(&path)
    }

    /// Open (or create) the database at the given path.
    ///
    /// # Errors
    /// Returns an error if the database cannot be opened or migrated.
    pub fn open(path: &PathBuf) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create db directory: {}", parent.display()))?;
        }
        let conn = Connection::open(path)
            .with_context(|| format!("failed to open db: {}", path.display()))?;

        // Performance pragmas.
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;\
             PRAGMA synchronous = NORMAL;\
             PRAGMA cache_size = -8192;\
             PRAGMA temp_store = MEMORY;\
             PRAGMA mmap_size = 268435456;",
        )?;

        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        db.migrate()?;
        Ok(db)
    }

    /// Open an in-memory database (for testing).
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        db.migrate()?;
        Ok(db)
    }

    /// Get the default database path for the current platform.
    fn default_db_path() -> Result<PathBuf> {
        let dir = if cfg!(target_os = "macos") {
            // directories v5: use BaseDirs::data_dir() for the user's data directory.
            directories::BaseDirs::new().and_then(|bd| Some(bd.data_dir().to_path_buf()))
        } else if cfg!(target_os = "windows") {
            std::env::var("APPDATA").ok().map(PathBuf::from)
        } else {
            std::env::var("XDG_DATA_HOME")
                .ok()
                .map(PathBuf::from)
                .or_else(|| {
                    std::env::var("HOME")
                        .ok()
                        .map(|h| PathBuf::from(h).join(".local").join("share"))
                })
        }
        .ok_or_else(|| anyhow::anyhow!("cannot determine data directory"))?;

        Ok(dir.join("qbitos").join("desktop.db"))
    }

    /// Run schema migrations.
    fn migrate(&self) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute_batch(
            r#"CREATE TABLE IF NOT EXISTS connection_settings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL UNIQUE,
                url         TEXT NOT NULL,
                auth_token  TEXT,
                is_default  INTEGER NOT NULL DEFAULT 0,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS conn_settings_default_idx
                ON connection_settings(is_default) WHERE is_default = 1;

            CREATE TABLE IF NOT EXISTS query_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt      TEXT NOT NULL,
                response    TEXT,
                endpoint    TEXT,
                duration_ms INTEGER,
                success     INTEGER NOT NULL DEFAULT 1,
                created_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS query_history_created_idx
                ON query_history(created_at DESC);

            CREATE TABLE IF NOT EXISTS engine_config (
                key         TEXT PRIMARY KEY,
                value       TEXT NOT NULL,
                category    TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS engine_config_category_idx
                ON engine_config(category);

            CREATE TABLE IF NOT EXISTS ui_preferences (
                key         TEXT PRIMARY KEY,
                value       TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );"#,
        )?;
        Ok(())
    }

    // ─── Connection Settings ─────────────────────────────────────────────

    /// Save a connection profile. If `name` already exists, it is updated.
    pub fn save_connection(&self, profile: &ConnectionProfile) -> Result<i64> {
        let conn = self.conn.lock();
        let now = Utc::now().to_rfc3339();
        // If this is the default, unset any previous default.
        if profile.is_default {
            conn.execute(
                "UPDATE connection_settings SET is_default = 0 WHERE is_default = 1",
                [],
            )?;
        }
        conn.execute(
            "INSERT INTO connection_settings (name, url, auth_token, is_default, created_at, updated_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
             ON CONFLICT(name) DO UPDATE SET \
                url = excluded.url, auth_token = excluded.auth_token, \
                is_default = excluded.is_default, updated_at = excluded.updated_at",
            params![
                &profile.name,
                &profile.url,
                &profile.auth_token,
                profile.is_default as i64,
                &now,
                &now,
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// List all saved connection profiles.
    pub fn list_connections(&self) -> Result<Vec<ConnectionProfile>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT id, name, url, auth_token, is_default, created_at, updated_at \
             FROM connection_settings ORDER BY is_default DESC, name ASC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(ConnectionProfile {
                id: Some(row.get(0)?),
                name: row.get(1)?,
                url: row.get(2)?,
                auth_token: row.get(3)?,
                is_default: row.get::<_, i64>(4)? != 0,
                created_at: row.get(5)?,
                updated_at: row.get(6)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Get the default connection profile.
    pub fn default_connection(&self) -> Result<Option<ConnectionProfile>> {
        let conn = self.conn.lock();
        let row = conn
            .query_row(
                "SELECT id, name, url, auth_token, is_default, created_at, updated_at \
                 FROM connection_settings WHERE is_default = 1 LIMIT 1",
                [],
                |row| {
                    Ok(ConnectionProfile {
                        id: Some(row.get(0)?),
                        name: row.get(1)?,
                        url: row.get(2)?,
                        auth_token: row.get(3)?,
                        is_default: true,
                        created_at: row.get(5)?,
                        updated_at: row.get(6)?,
                    })
                },
            )
            .optional()?;
        Ok(row)
    }

    /// Delete a connection profile by ID.
    pub fn delete_connection(&self, id: i64) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute("DELETE FROM connection_settings WHERE id = ?1", params![id])?;
        Ok(())
    }

    // ─── Query History ───────────────────────────────────────────────────

    /// Add a query to the history.
    pub fn add_query(&self, entry: &QueryHistoryEntry) -> Result<i64> {
        let conn = self.conn.lock();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO query_history (prompt, response, endpoint, duration_ms, success, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                &entry.prompt,
                &entry.response,
                &entry.endpoint,
                entry.duration_ms,
                entry.success as i64,
                &now,
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// List recent query history (most recent first).
    pub fn list_queries(&self, limit: usize) -> Result<Vec<QueryHistoryEntry>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT id, prompt, response, endpoint, duration_ms, success, created_at \
             FROM query_history ORDER BY created_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(QueryHistoryEntry {
                id: Some(row.get(0)?),
                prompt: row.get(1)?,
                response: row.get(2)?,
                endpoint: row.get(3)?,
                duration_ms: row.get(4)?,
                success: row.get::<_, i64>(5)? != 0,
                created_at: row.get(6)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Clear all query history.
    pub fn clear_queries(&self) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute("DELETE FROM query_history", [])?;
        Ok(())
    }

    /// Search query history by prompt text.
    pub fn search_queries(&self, search: &str, limit: usize) -> Result<Vec<QueryHistoryEntry>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT id, prompt, response, endpoint, duration_ms, success, created_at \
             FROM query_history WHERE prompt LIKE ?1 ORDER BY created_at DESC LIMIT ?2",
        )?;
        let pattern = format!("%{search}%");
        let rows = stmt.query_map(params![pattern, limit as i64], |row| {
            Ok(QueryHistoryEntry {
                id: Some(row.get(0)?),
                prompt: row.get(1)?,
                response: row.get(2)?,
                endpoint: row.get(3)?,
                duration_ms: row.get(4)?,
                success: row.get::<_, i64>(5)? != 0,
                created_at: row.get(6)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    // ─── Engine Config ───────────────────────────────────────────────────

    /// Set an engine config value.
    pub fn set_engine_config(&self, key: &str, value: &str, category: &str) -> Result<()> {
        let conn = self.conn.lock();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO engine_config (key, value, category, updated_at) \
             VALUES (?1, ?2, ?3, ?4) \
             ON CONFLICT(key) DO UPDATE SET value = excluded.value, category = excluded.category, updated_at = excluded.updated_at",
            params![key, value, category, now],
        )?;
        Ok(())
    }

    /// Get an engine config value.
    pub fn get_engine_config(&self, key: &str) -> Result<Option<EngineConfigEntry>> {
        let conn = self.conn.lock();
        let row = conn
            .query_row(
                "SELECT key, value, category, updated_at FROM engine_config WHERE key = ?1",
                params![key],
                |row| {
                    Ok(EngineConfigEntry {
                        key: row.get(0)?,
                        value: row.get(1)?,
                        category: row.get(2)?,
                        updated_at: row.get(3)?,
                    })
                },
            )
            .optional()?;
        Ok(row)
    }

    /// List all engine configs in a category.
    pub fn list_engine_configs(&self, category: &str) -> Result<Vec<EngineConfigEntry>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT key, value, category, updated_at FROM engine_config WHERE category = ?1 ORDER BY key",
        )?;
        let rows = stmt.query_map(params![category], |row| {
            Ok(EngineConfigEntry {
                key: row.get(0)?,
                value: row.get(1)?,
                category: row.get(2)?,
                updated_at: row.get(3)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// List all engine configs.
    pub fn all_engine_configs(&self) -> Result<Vec<EngineConfigEntry>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT key, value, category, updated_at FROM engine_config ORDER BY category, key",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(EngineConfigEntry {
                key: row.get(0)?,
                value: row.get(1)?,
                category: row.get(2)?,
                updated_at: row.get(3)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Delete an engine config.
    pub fn delete_engine_config(&self, key: &str) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute("DELETE FROM engine_config WHERE key = ?1", params![key])?;
        Ok(())
    }

    // ─── UI Preferences ──────────────────────────────────────────────────

    /// Set a UI preference.
    pub fn set_ui_preference(&self, key: &str, value: &str) -> Result<()> {
        let conn = self.conn.lock();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO ui_preferences (key, value, updated_at) \
             VALUES (?1, ?2, ?3) \
             ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            params![key, value, now],
        )?;
        Ok(())
    }

    /// Get a UI preference.
    pub fn get_ui_preference(&self, key: &str) -> Result<Option<String>> {
        let conn = self.conn.lock();
        let row = conn
            .query_row(
                "SELECT value FROM ui_preferences WHERE key = ?1",
                params![key],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        Ok(row)
    }

    /// List all UI preferences.
    pub fn all_ui_preferences(&self) -> Result<Vec<UiPreference>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT key, value, updated_at FROM ui_preferences ORDER BY key",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(UiPreference {
                key: row.get(0)?,
                value: row.get(1)?,
                updated_at: row.get(2)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Delete a UI preference.
    pub fn delete_ui_preference(&self, key: &str) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute("DELETE FROM ui_preferences WHERE key = ?1", params![key])?;
        Ok(())
    }

    // ─── Maintenance ─────────────────────────────────────────────────────

    /// Vacuum the database to reclaim space.
    pub fn vacuum(&self) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute_batch("VACUUM")?;
        Ok(())
    }

    /// Checkpoint the WAL log.
    pub fn checkpoint(&self) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")?;
        Ok(())
    }

    /// Get the database file size in bytes.
    pub fn db_size_bytes(&self) -> Result<u64> {
        // Read the page count and page size.
        let conn = self.conn.lock();
        let page_count: i64 = conn
            .query_row("PRAGMA page_count", [], |row| row.get(0))?;
        let page_size: i64 = conn
            .query_row("PRAGMA page_size", [], |row| row.get(0))?;
        Ok((page_count * page_size) as u64)
    }
}

impl Default for LocalDb {
    fn default() -> Self {
        Self::open_default().expect("failed to open default local db")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_db() -> LocalDb {
        LocalDb::open_in_memory().unwrap()
    }

    #[test]
    fn connection_settings_crud() {
        let db = make_db();
        let profile = ConnectionProfile {
            id: None,
            name: "Local Dev".into(),
            url: "http://127.0.0.1:8080".into(),
            auth_token: Some("secret".into()),
            is_default: true,
            created_at: String::new(),
            updated_at: String::new(),
        };
        let id = db.save_connection(&profile).unwrap();
        assert!(id > 0);

        // List.
        let conns = db.list_connections().unwrap();
        assert_eq!(conns.len(), 1);
        assert_eq!(conns[0].name, "Local Dev");
        assert!(conns[0].is_default);

        // Default.
        let default = db.default_connection().unwrap().unwrap();
        assert_eq!(default.url, "http://127.0.0.1:8080");

        // Update.
        let mut updated = profile.clone();
        updated.url = "http://localhost:9090".into();
        db.save_connection(&updated).unwrap();
        let conns = db.list_connections().unwrap();
        assert_eq!(conns[0].url, "http://localhost:9090");

        // Delete.
        db.delete_connection(conns[0].id.unwrap()).unwrap();
        assert_eq!(db.list_connections().unwrap().len(), 0);
    }

    #[test]
    fn default_connection_unsets_previous() {
        let db = make_db();
        let p1 = ConnectionProfile {
            id: None, name: "A".into(), url: "http://a".into(),
            auth_token: None, is_default: true, created_at: String::new(), updated_at: String::new(),
        };
        db.save_connection(&p1).unwrap();
        let p2 = ConnectionProfile {
            id: None, name: "B".into(), url: "http://b".into(),
            auth_token: None, is_default: true, created_at: String::new(), updated_at: String::new(),
        };
        db.save_connection(&p2).unwrap();
        let default = db.default_connection().unwrap().unwrap();
        assert_eq!(default.name, "B");
    }

    #[test]
    fn query_history_crud() {
        let db = make_db();
        let entry = QueryHistoryEntry {
            id: None,
            prompt: "What is 2+2?".into(),
            response: Some("4".into()),
            endpoint: Some("/api/v1/cognitive/cycle".into()),
            duration_ms: Some(42),
            success: true,
            created_at: String::new(),
        };
        db.add_query(&entry).unwrap();
        db.add_query(&QueryHistoryEntry {
            id: None,
            prompt: "Hello".into(),
            response: None,
            endpoint: None,
            duration_ms: None,
            success: false,
            created_at: String::new(),
        }).unwrap();

        let history = db.list_queries(10).unwrap();
        assert_eq!(history.len(), 2);

        let searched = db.search_queries("2+2", 10).unwrap();
        assert_eq!(searched.len(), 1);
        assert_eq!(searched[0].prompt, "What is 2+2?");

        db.clear_queries().unwrap();
        assert_eq!(db.list_queries(10).unwrap().len(), 0);
    }

    #[test]
    fn engine_config_crud() {
        let db = make_db();
        db.set_engine_config("hnsw_ef_search", "64", "vector").unwrap();
        db.set_engine_config("embedding_dim", "384", "vector").unwrap();
        db.set_engine_config("max_tokens", "256", "inference").unwrap();

        let val = db.get_engine_config("hnsw_ef_search").unwrap().unwrap();
        assert_eq!(val.value, "64");
        assert_eq!(val.category, "vector");

        let vector_configs = db.list_engine_configs("vector").unwrap();
        assert_eq!(vector_configs.len(), 2);

        let all = db.all_engine_configs().unwrap();
        assert_eq!(all.len(), 3);

        db.delete_engine_config("max_tokens").unwrap();
        assert!(db.get_engine_config("max_tokens").unwrap().is_none());
    }

    #[test]
    fn ui_preferences_crud() {
        let db = make_db();
        db.set_ui_preference("theme", "dark").unwrap();
        db.set_ui_preference("active_view", "chat").unwrap();
        db.set_ui_preference("sidebar_width", "240").unwrap();

        assert_eq!(db.get_ui_preference("theme").unwrap().unwrap(), "dark");
        assert_eq!(db.get_ui_preference("active_view").unwrap().unwrap(), "chat");

        // Update existing.
        db.set_ui_preference("theme", "light").unwrap();
        assert_eq!(db.get_ui_preference("theme").unwrap().unwrap(), "light");

        let all = db.all_ui_preferences().unwrap();
        assert_eq!(all.len(), 3);

        db.delete_ui_preference("sidebar_width").unwrap();
        assert!(db.get_ui_preference("sidebar_width").unwrap().is_none());
    }

    #[test]
    fn db_size_is_positive() {
        let db = make_db();
        db.set_ui_preference("test", "value").unwrap();
        let size = db.db_size_bytes().unwrap();
        assert!(size > 0);
    }
}
