//! Entity engine.
//!
//! Tracks named entities (people, places, organizations, etc.) extracted
//! from observations. Each entity has a type, a set of attributes, and
//! relations to other entities.

use ahash::AHashMap;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// Unique identifier for an entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntityId(pub Uuid);

impl EntityId {
    /// Mint a fresh id.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }
}

impl Default for EntityId {
    fn default() -> Self {
        Self::new()
    }
}

/// The kind of entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityKind {
    /// A person.
    Person,
    /// An organization.
    Organization,
    /// A location.
    Location,
    /// A date or time.
    DateTime,
    /// A document or file.
    Document,
    /// A concept.
    Concept,
    /// An event.
    Event,
    /// A custom entity.
    Custom,
}

/// A relation between two entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRelation {
    /// Relation type (e.g. "works_for", "located_in", "authored").
    pub kind: String,
    /// Target entity.
    pub target: EntityId,
    /// Weight in `[0.0, 1.0]`.
    pub weight: f32,
}

/// An entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Unique id.
    pub id: EntityId,
    /// Display name.
    pub name: String,
    /// Entity kind.
    pub kind: EntityKind,
    /// Attributes (key-value pairs).
    pub attributes: AHashMap<String, serde_json::Value>,
    /// Relations to other entities.
    pub relations: Vec<EntityRelation>,
    /// When the entity was first seen.
    pub first_seen_at: DateTime<Utc>,
    /// When the entity was last mentioned.
    pub last_mentioned_at: DateTime<Utc>,
    /// Mention count.
    pub mention_count: u32,
}

impl Entity {
    /// Create a new entity.
    #[must_use]
    pub fn new(name: impl Into<String>, kind: EntityKind) -> Self {
        Self {
            id: EntityId::new(),
            name: name.into(),
            kind,
            attributes: AHashMap::new(),
            relations: Vec::new(),
            first_seen_at: Utc::now(),
            last_mentioned_at: Utc::now(),
            mention_count: 0,
        }
    }

    /// Record a mention.
    pub fn mention(&mut self) {
        self.mention_count += 1;
        self.last_mentioned_at = Utc::now();
    }
}

/// The entity engine.
pub struct EntityEngine {
    entities: RwLock<AHashMap<EntityId, Entity>>,
    by_name: RwLock<AHashMap<String, EntityId>>,
}

impl EntityEngine {
    /// Create a new empty engine.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entities: RwLock::new(AHashMap::new()),
            by_name: RwLock::new(AHashMap::new()),
        }
    }

    /// Register an entity. If an entity with the same name already exists,
    /// it is returned (with mention count bumped).
    pub fn register(&self, name: impl Into<String>, kind: EntityKind) -> EntityId {
        let name = name.into();
        // Check by name
        if let Some(id) = self.by_name.read().get(&name).copied() {
            let mut entities = self.entities.write();
            if let Some(e) = entities.get_mut(&id) {
                e.mention();
            }
            return id;
        }
        let entity = Entity::new(name.clone(), kind);
        let id = entity.id;
        self.entities.write().insert(id, entity);
        self.by_name.write().insert(name, id);
        id
    }

    /// Look up an entity by id.
    #[must_use]
    pub fn get(&self, id: EntityId) -> Option<Entity> {
        self.entities.read().get(&id).cloned()
    }

    /// Look up an entity by name.
    #[must_use]
    pub fn get_by_name(&self, name: &str) -> Option<Entity> {
        let id = *self.by_name.read().get(name)?;
        self.get(id)
    }

    /// Add a relation between two entities.
    pub fn relate(&self, from: EntityId, relation: impl Into<String>, to: EntityId, weight: f32) -> bool {
        let mut entities = self.entities.write();
        let Some(e) = entities.get_mut(&from) else { return false };
        e.relations.push(EntityRelation {
            kind: relation.into(),
            target: to,
            weight,
        });
        true
    }

    /// Set an attribute on an entity.
    pub fn set_attribute(&self, id: EntityId, key: impl Into<String>, value: serde_json::Value) -> bool {
        let mut entities = self.entities.write();
        let Some(e) = entities.get_mut(&id) else { return false };
        e.attributes.insert(key.into(), value);
        true
    }

    /// List all entities of a given kind.
    #[must_use]
    pub fn by_kind(&self, kind: EntityKind) -> Vec<Entity> {
        self.entities.read().values().filter(|e| e.kind == kind).cloned().collect()
    }

    /// Number of entities.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entities.read().len()
    }
}

impl Default for EntityEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// A shared entity engine.
pub type SharedEntityEngine = Arc<EntityEngine>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_relate() {
        let engine = EntityEngine::new();
        let alice = engine.register("Alice", EntityKind::Person);
        let acme = engine.register("Acme Corp", EntityKind::Organization);
        engine.relate(alice, "works_for", acme, 1.0);
        let alice_entity = engine.get(alice).unwrap();
        assert_eq!(alice_entity.relations.len(), 1);
        assert_eq!(alice_entity.relations[0].kind, "works_for");
    }
}
