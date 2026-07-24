//! Permissions.
//!
//! Every tool call requires explicit permission. Permissions are scoped:
//! `filesystem.read:/tmp/*` allows reading files under `/tmp/`, while
//! `terminal.exec:bash` allows executing bash. Permissions are stored in
//! a [`PermissionStore`] keyed by agent id.

use ahash::{AHashMap, AHashSet};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// A single permission: a (action, scope) pair.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Permission {
    /// Action: `read`, `write`, `exec`, `network`, etc.
    pub action: String,
    /// Scope: a glob pattern like `/tmp/*` or `https://api.example.com/*`.
    pub scope: String,
}

impl Permission {
    /// Create a new permission.
    #[must_use]
    pub fn new(action: impl Into<String>, scope: impl Into<String>) -> Self {
        Self { action: action.into(), scope: scope.into() }
    }

    /// Check whether this permission matches a (action, target) pair.
    #[must_use]
    pub fn matches(&self, action: &str, target: &str) -> bool {
        if self.action != action && self.action != "*" {
            return false;
        }
        glob_match(&self.scope, target)
    }
}

/// Glob match: supports `*` (any sequence) and `?` (any single char).
fn glob_match(pattern: &str, target: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = target.chars().collect();
    glob_match_inner(&p, &t)
}

fn glob_match_inner(p: &[char], t: &[char]) -> bool {
    if p.is_empty() {
        return t.is_empty();
    }
    match p[0] {
        '*' => {
            // Try matching the rest of the pattern at every position in t
            for i in 0..=t.len() {
                if glob_match_inner(&p[1..], &t[i..]) {
                    return true;
                }
            }
            false
        }
        '?' => {
            if t.is_empty() {
                return false;
            }
            glob_match_inner(&p[1..], &t[1..])
        }
        c => {
            if t.is_empty() || t[0] != c {
                return false;
            }
            glob_match_inner(&p[1..], &t[1..])
        }
    }
}

/// A set of permissions.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PermissionSet {
    /// The permissions.
    pub permissions: AHashSet<Permission>,
}

impl PermissionSet {
    /// Create a new empty set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a permission.
    pub fn add(&mut self, perm: Permission) {
        self.permissions.insert(perm);
    }

    /// Check whether the set grants `action` on `target`.
    #[must_use]
    pub fn allows(&self, action: &str, target: &str) -> bool {
        self.permissions.iter().any(|p| p.matches(action, target))
    }

    /// Returns `true` if the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.permissions.is_empty()
    }
}

/// A store of permission sets, keyed by agent id.
#[derive(Clone)]
pub struct PermissionStore {
    sets: Arc<RwLock<AHashMap<Uuid, PermissionSet>>>,
}

impl PermissionStore {
    /// Create a new empty store.
    #[must_use]
    pub fn new() -> Self {
        Self { sets: Arc::new(RwLock::new(AHashMap::new())) }
    }

    /// Grant a permission to an agent.
    pub fn grant(&self, agent: Uuid, perm: Permission) {
        self.sets.write().entry(agent).or_default().add(perm);
    }

    /// Grant multiple permissions to an agent.
    pub fn grant_all(&self, agent: Uuid, perms: PermissionSet) {
        let mut sets = self.sets.write();
        let entry = sets.entry(agent).or_default();
        for p in perms.permissions {
            entry.add(p);
        }
    }

    /// Check whether an agent has `action` permission on `target`.
    #[must_use]
    pub fn check(&self, agent: Uuid, action: &str, target: &str) -> bool {
        self.sets.read().get(&agent).map_or(false, |s| s.allows(action, target))
    }

    /// Get a copy of an agent's permission set.
    #[must_use]
    pub fn get(&self, agent: Uuid) -> PermissionSet {
        self.sets.read().get(&agent).cloned().unwrap_or_default()
    }

    /// Revoke all permissions for an agent.
    pub fn revoke_all(&self, agent: Uuid) {
        self.sets.write().remove(&agent);
    }
}

impl Default for PermissionStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_matches() {
        assert!(glob_match("/tmp/*", "/tmp/foo.txt"));
        assert!(glob_match("/tmp/*", "/tmp/bar/baz.txt"));
        assert!(!glob_match("/tmp/*", "/etc/passwd"));
        assert!(glob_match("*://api.example.com/*", "https://api.example.com/v1/users"));
        assert!(glob_match("a?c", "abc"));
        assert!(!glob_match("a?c", "ac"));
    }

    #[test]
    fn permission_check() {
        let store = PermissionStore::new();
        let agent = Uuid::now_v7();
        store.grant(agent, Permission::new("read", "/tmp/*"));
        assert!(store.check(agent, "read", "/tmp/foo.txt"));
        assert!(!store.check(agent, "read", "/etc/passwd"));
        assert!(!store.check(agent, "write", "/tmp/foo.txt"));
    }
}
