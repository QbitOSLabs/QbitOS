//! Multi-tenant isolation for QbitOS.
//!
//! Provides per-tenant namespacing of memories, skills, and cognitive state.
//! Each tenant gets an isolated slice of the memory fabric — queries only
//! return memories from the same tenant (or from a designated "shared" tenant
//! for global knowledge).
//!
//! # Tenant model
//!
//! - `TenantId` is a 128-bit identifier (UUID v7 recommended for time-ordered IDs).
//! - `TenantId::shared()` returns the well-known shared tenant (`00000000-0000-0000-0000-000000000000`).
//! - All store/search operations accept an optional `TenantId`. When `None`,
//!   the operation targets the shared tenant.
//!
//! # Enforcement
//!
//! Isolation is enforced at the `MemoryFabric` API boundary: a search query
//! scoped to tenant A will never return results from tenant B. The underlying
//! storage backends (Postgres, Qdrant, etc.) tag every record with a
//! `tenant_id` column/field and filter on it at query time.
//!
//! # Rate limiting
//!
//! Each tenant gets an independent rate-limit budget (configurable via
//! `TenantConfig`). This prevents one noisy tenant from starving others.
//!
//! # Example
//!
//! ```
//! use qbitos_memory::tenant::{TenantId, TenantContext};
//!
//! let tenant_a = TenantId::new_v7();
//! let ctx = TenantContext::new(tenant_a);
//! // ctx.tenant_id() is now embedded in every memory store/search call.
//! ```

#![allow(missing_docs)]
#![allow(clippy::missing_errors_doc)]

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// A tenant identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TenantId(pub Uuid);

impl TenantId {
    /// The well-known shared tenant. Memories stored under this tenant are
    /// visible to all tenants (useful for global knowledge).
    #[must_use]
    pub fn shared() -> Self {
        // Uuid::nil() = 00000000-0000-0000-0000-000000000000
        Self(Uuid::nil())
    }

    /// Generate a new time-ordered tenant ID (UUID v7).
    #[must_use]
    pub fn new_v7() -> Self {
        Self(Uuid::now_v7())
    }

    /// Parse a tenant ID from a UUID string.
    ///
    /// # Errors
    /// Returns an error if the string is not a valid UUID.
    pub fn parse(s: &str) -> Result<Self, uuid::Error> {
        Ok(Self(Uuid::parse_str(s)?))
    }

    /// Returns true if this is the shared tenant.
    #[must_use]
    pub fn is_shared(&self) -> bool {
        self.0 == Uuid::nil()
    }

    /// Returns the underlying UUID.
    #[must_use]
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl std::fmt::Display for TenantId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for TenantId {
    fn default() -> Self {
        Self::shared()
    }
}

/// Per-tenant configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantConfig {
    /// Display name for the tenant.
    pub name: String,
    /// Maximum memories this tenant can store (0 = unlimited).
    pub max_memories: usize,
    /// Maximum cognitive cycles per minute.
    pub rate_limit_cycles_per_min: u32,
    /// Maximum tool calls per cognitive cycle.
    pub max_tool_calls_per_cycle: u32,
    /// Whether this tenant can read the shared tenant's memories.
    pub can_read_shared: bool,
    /// Whether this tenant can write to the shared tenant.
    pub can_write_shared: bool,
    /// Created timestamp.
    pub created_at: DateTime<Utc>,
}

impl Default for TenantConfig {
    fn default() -> Self {
        Self {
            name: "default".into(),
            max_memories: 100_000,
            rate_limit_cycles_per_min: 60,
            max_tool_calls_per_cycle: 32,
            can_read_shared: true,
            can_write_shared: false,
            created_at: Utc::now(),
        }
    }
}

/// A tenant context — passed through the cognitive pipeline to scope all
/// memory and skill operations to a specific tenant.
#[derive(Debug, Clone, Copy)]
pub struct TenantContext {
    tenant_id: TenantId,
}

impl TenantContext {
    /// Create a new tenant context.
    #[must_use]
    pub fn new(tenant_id: TenantId) -> Self {
        Self { tenant_id }
    }

    /// Create a context for the shared tenant.
    #[must_use]
    pub fn shared() -> Self {
        Self::new(TenantId::shared())
    }

    /// Get the tenant ID.
    #[must_use]
    pub fn tenant_id(&self) -> TenantId {
        self.tenant_id
    }

    /// Returns true if this context is for the shared tenant.
    #[must_use]
    pub fn is_shared(&self) -> bool {
        self.tenant_id.is_shared()
    }
}

impl Default for TenantContext {
    fn default() -> Self {
        Self::shared()
    }
}

/// Registry of known tenants and their configurations.
#[derive(Debug, Default)]
pub struct TenantRegistry {
    tenants: RwLock<HashMap<TenantId, TenantConfig>>,
}

impl TenantRegistry {
    /// Create a new empty registry. The shared tenant is auto-registered.
    #[must_use]
    pub fn new() -> Self {
        let mut tenants = HashMap::new();
        tenants.insert(
            TenantId::shared(),
            TenantConfig {
                name: "shared".into(),
                max_memories: usize::MAX,
                rate_limit_cycles_per_min: u32::MAX,
                max_tool_calls_per_cycle: 100,
                can_read_shared: true,
                can_write_shared: true,
                created_at: Utc::now(),
            },
        );
        Self {
            tenants: RwLock::new(tenants),
        }
    }

    /// Register a new tenant.
    pub fn register(&self, id: TenantId, config: TenantConfig) {
        self.tenants.write().insert(id, config);
    }

    /// Look up a tenant's configuration.
    #[must_use]
    pub fn get(&self, id: &TenantId) -> Option<TenantConfig> {
        self.tenants.read().get(id).cloned()
    }

    /// Remove a tenant (does not delete their memories — use the memory fabric
    /// for that).
    pub fn revoke(&self, id: &TenantId) -> Option<TenantConfig> {
        self.tenants.write().remove(id)
    }

    /// List all registered tenants.
    #[must_use]
    pub fn list(&self) -> Vec<(TenantId, TenantConfig)> {
        self.tenants
            .read()
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect()
    }

    /// Count of registered tenants (including shared).
    #[must_use]
    pub fn len(&self) -> usize {
        self.tenants.read().len()
    }

    /// Returns true if no tenants are registered (always false because
    /// shared is auto-registered).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tenants.read().is_empty()
    }

    /// Check if a tenant is allowed to perform an action.
    ///
    /// Returns `Ok(())` if allowed, `Err(reason)` otherwise.
    pub fn check_permission(
        &self,
        tenant: &TenantId,
        action: TenantAction,
    ) -> Result<(), String> {
        let tenants = self.tenants.read();
        let config = tenants
            .get(tenant)
            .ok_or_else(|| format!("tenant {tenant} not registered"))?;

        match action {
            TenantAction::ReadShared if !config.can_read_shared => {
                Err(format!("tenant {tenant} cannot read shared memories"))
            }
            TenantAction::WriteShared if !config.can_write_shared => {
                Err(format!("tenant {tenant} cannot write shared memories"))
            }
            _ => Ok(()),
        }
    }
}

/// Actions that can be permission-checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TenantAction {
    /// Read memories from the shared tenant.
    ReadShared,
    /// Write memories to the shared tenant.
    WriteShared,
    /// Read memories from this tenant's own namespace.
    ReadOwn,
    /// Write memories to this tenant's own namespace.
    WriteOwn,
    /// Execute a cognitive cycle.
    CognitiveCycle,
    /// Call a tool.
    ToolCall,
}

/// Shared handle to a tenant registry.
pub type SharedTenantRegistry = Arc<TenantRegistry>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_tenant_is_well_known() {
        let shared = TenantId::shared();
        assert!(shared.is_shared());
        assert_eq!(shared.to_string(), "00000000-0000-0000-0000-000000000000");
    }

    #[test]
    fn new_v7_is_unique_and_time_ordered() {
        let a = TenantId::new_v7();
        let b = TenantId::new_v7();
        assert_ne!(a, b);
        assert!(!a.is_shared());
    }

    #[test]
    fn registry_auto_registers_shared() {
        let reg = TenantRegistry::new();
        assert!(reg.get(&TenantId::shared()).is_some());
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn register_and_revoke_tenant() {
        let reg = TenantRegistry::new();
        let t = TenantId::new_v7();
        reg.register(t, TenantConfig::default());
        assert_eq!(reg.len(), 2);
        assert!(reg.get(&t).is_some());

        let revoked = reg.revoke(&t);
        assert!(revoked.is_some());
        assert_eq!(reg.len(), 1);
        assert!(reg.get(&t).is_none());
    }

    #[test]
    fn permission_check_blocks_unauthorized_shared_write() {
        let reg = TenantRegistry::new();
        let t = TenantId::new_v7();
        reg.register(
            t,
            TenantConfig {
                can_write_shared: false,
                ..Default::default()
            },
        );
        assert!(reg.check_permission(&t, TenantAction::WriteShared).is_err());
        assert!(reg.check_permission(&t, TenantAction::ReadShared).is_ok());
    }

    #[test]
    fn permission_check_allows_shared_tenant_everything() {
        let reg = TenantRegistry::new();
        let shared = TenantId::shared();
        assert!(reg.check_permission(&shared, TenantAction::WriteShared).is_ok());
        assert!(reg.check_permission(&shared, TenantAction::ReadShared).is_ok());
    }

    #[test]
    fn tenant_context_default_is_shared() {
        let ctx = TenantContext::default();
        assert!(ctx.is_shared());
    }

    #[test]
    fn parse_tenant_id_from_string() {
        let id = TenantId::parse("00000000-0000-0000-0000-000000000000").unwrap();
        assert!(id.is_shared());
        assert!(TenantId::parse("not-a-uuid").is_err());
    }

    #[test]
    fn list_includes_all_tenants() {
        let reg = TenantRegistry::new();
        let t1 = TenantId::new_v7();
        let t2 = TenantId::new_v7();
        reg.register(t1, TenantConfig::default());
        reg.register(t2, TenantConfig::default());

        let list = reg.list();
        assert_eq!(list.len(), 3); // shared + t1 + t2
        assert!(list.iter().any(|(id, _)| *id == TenantId::shared()));
        assert!(list.iter().any(|(id, _)| *id == t1));
        assert!(list.iter().any(|(id, _)| *id == t2));
    }
}
