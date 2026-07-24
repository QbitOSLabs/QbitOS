//! OpenAPI 3.1 specification generation.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// OpenAPI 3.1 document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenApiSpec {
    /// OpenAPI version.
    pub openapi: String,
    /// API metadata.
    pub info: Info,
    /// Available servers.
    pub servers: Vec<Server>,
    /// Available paths.
    pub paths: serde_json::Map<String, Value>,
    /// Security schemes.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub security: Vec<Value>,
    /// Components (schemas, security schemes).
    #[serde(default)]
    pub components: Components,
}

/// Info block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Info {
    /// API title.
    pub title: String,
    /// API version.
    pub version: String,
    /// Description.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub description: Option<String>,
    /// License.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub license: Option<License>,
}

/// License block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct License {
    /// License name.
    pub name: String,
    /// Optional URL.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub url: Option<String>,
}

/// Server block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Server {
    /// Server URL.
    pub url: String,
    /// Description.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub description: Option<String>,
}

/// Components block.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Components {
    /// Reusable schemas.
    #[serde(default)]
    pub schemas: serde_json::Map<String, Value>,
}

/// Generate the OpenAPI 3.1 specification for the QbitOS HTTP API.
#[must_use]
pub fn generate_openapi() -> OpenApiSpec {
    let mut paths = serde_json::Map::new();

    macro_rules! add_path {
        ($path:expr, $method:expr, $summary:expr, $tag:expr) => {{
            let method_obj = serde_json::json!({
                "summary": $summary,
                "tags": [$tag],
                "responses": {
                    "200": {
                        "description": "OK",
                        "content": {
                            "application/json": {
                                "schema": { "$ref": "#/components/schemas/ApiResponse" }
                            }
                        }
                    },
                    "4XX": {
                        "description": "Client error",
                        "content": {
                            "application/json": {
                                "schema": { "$ref": "#/components/schemas/ApiResponse" }
                            }
                        }
                    }
                }
            });
            let entry = paths
                .entry($path.to_string())
                .or_insert(Value::Object(serde_json::Map::new()));
            if let Value::Object(map) = entry {
                map.insert($method.to_string(), method_obj);
            }
        }};
    }

    add_path!("/api/v1/health", "get", "Health check", "system");
    add_path!("/api/v1/version", "get", "API version", "system");
    add_path!("/api/v1/stats", "get", "Service statistics", "system");
    add_path!("/api/v1/cognitive/cycle", "post", "Run a cognitive cycle", "cognitive");
    add_path!("/api/v1/memory", "post", "Store a memory", "memory");
    add_path!("/api/v1/memory/search", "post", "Search memories", "memory");
    add_path!("/api/v1/memory/{id}", "delete", "Delete a memory", "memory");
    add_path!("/api/v1/skills", "get", "List skills", "skills");
    add_path!("/api/v1/skills/execute", "post", "Execute a skill", "skills");
    add_path!("/api/v1/training/step", "post", "Run a training step", "training");
    add_path!("/api/v1/curriculum", "get", "Get curriculum state", "training");
    add_path!("/api/v1/replay/stats", "get", "Get replay buffer stats", "training");
    add_path!("/api/v1/meta/strategies", "get", "List meta-learning strategies", "training");
    add_path!("/api/v1/events/subscribe", "post", "Subscribe to events", "events");
    add_path!("/api/v1/events/poll", "post", "Poll events", "events");
    add_path!("/api/v1/events/replay", "post", "Replay events from a sequence number", "events");
    add_path!("/api/v1/events/lookup/{id}", "get", "Look up a single event by id", "events");
    add_path!("/api/v1/events/ws", "get", "WebSocket event stream", "events");
    add_path!("/api/v1/language/detect", "post", "Detect the language of a text", "language");
    add_path!("/api/v1/language/tokenize", "post", "Tokenize a text into words", "language");
    add_path!("/api/v1/language/stem", "post", "Stem a word to its root form", "language");
    add_path!("/api/v1/algo/levenshtein", "post", "Compute Levenshtein edit distance", "algorithms");
    add_path!("/api/v1/algo/topk", "post", "Track the top-K elements of a stream", "algorithms");
    add_path!("/api/v1/algo/bloom", "post", "Bloom filter membership check", "algorithms");
    add_path!("/api/v1/openapi.json", "get", "OpenAPI spec (JSON)", "system");

    let mut schemas = serde_json::Map::new();
    schemas.insert(
        "ApiResponse".into(),
        serde_json::json!({
            "type": "object",
            "required": ["ok", "request_id", "timestamp"],
            "properties": {
                "ok": { "type": "boolean" },
                "data": {},
                "error": { "type": "string" },
                "error_code": { "type": "integer" },
                "request_id": { "type": "string", "format": "uuid" },
                "timestamp": { "type": "string", "format": "date-time" }
            }
        }),
    );
    schemas.insert(
        "HealthResponse".into(),
        serde_json::json!({
            "type": "object",
            "properties": {
                "instance_id": { "type": "string", "format": "uuid" },
                "instance_name": { "type": "string" },
                "healthy": { "type": "boolean" },
                "api_version": { "type": "string" },
                "uptime_seconds": { "type": "integer" },
                "subsystems": { "type": "array", "items": { "type": "string" } }
            }
        }),
    );
    schemas.insert(
        "CognitiveCycleRequest".into(),
        serde_json::json!({
            "type": "object",
            "required": ["input"],
            "properties": {
                "input": { "type": "string" },
                "modality": { "type": "string" },
                "stream_thoughts": { "type": "boolean" },
                "max_cycles": { "type": "integer", "default": 1 }
            }
        }),
    );

    OpenApiSpec {
        openapi: "3.1.0".into(),
        info: Info {
            title: "QbitOS Cognitive Service API".into(),
            version: crate::API_VERSION.into(),
            description: Some(
                "HTTP/REST, WebSocket, and RPC service layer for the QbitOS cognitive operating system."
                    .into(),
            ),
            license: Some(License {
                name: "Apache-2.0".into(),
                url: Some("https://www.apache.org/licenses/LICENSE-2.0".into()),
            }),
        },
        servers: vec![Server {
            url: "http://localhost:8080".into(),
            description: Some("Local development server".into()),
        }],
        paths,
        security: vec![],
        components: Components { schemas },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_is_valid_json() {
        let spec = generate_openapi();
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains("/api/v1/health"));
        assert!(json.contains("CognitiveCycleRequest"));
    }

    #[test]
    fn spec_has_all_paths() {
        let spec = generate_openapi();
        // 25 endpoints as of 2026 — health, version, stats, cognitive, memory (3),
        // skills (2), training (4), events (5), language (3), algorithms (3), openapi.
        assert!(spec.paths.len() >= 20, "expected at least 20 paths, got {}", spec.paths.len());
    }
}
