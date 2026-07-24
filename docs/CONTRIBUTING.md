# Contributing to QbitOS

Thank you for your interest in contributing to QbitOS! This document
covers the development workflow, code style, and submission process.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Adding a New Subsystem](#adding-a-new-subsystem)
- [Adding a New HTTP Endpoint](#adding-a-new-http-endpoint)
- [Adding a New Skill](#adding-a-new-skill)
- [Adding a New Tool](#adding-a-new-tool)
- [Adding a New Dataset Loader](#adding-a-new-dataset-loader)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)

---

## Code of Conduct

Be excellent to each other. Disagreements happen — focus on the
technical merits, not the person. Personal attacks, harassment, and
discriminatory language are not tolerated.

---

## Getting Started

### Prerequisites

- Rust 1.85+ (nightly recommended)
- A Linux/macOS development environment (Windows is best-effort)
- ~10 GB free disk space for build artifacts
- `pkg-config`, `libssl-dev` (for HTTPS support)

### Fork and clone

```bash
git clone https://github.com/qbitos/qbitos.git
cd qbitos
cargo build --workspace --exclude qbitos-desktop
cargo test --workspace --exclude qbitos-desktop
```

### Build configurations

```bash
# Fast dev build (no real backends, ~30s)
cargo build -p qbitos-cli

# Full build with real backends (~5 min first time, cached after)
cargo build -p qbitos-cli --features real-backends

# All DB backends
cargo build -p qbitos-cli --features all-backends

# Everything
cargo build --workspace --features qbitos-cli/real-backends --exclude qbitos-desktop
```

---

## Development Workflow

### 1. Pick or file an issue

Check the issue tracker for `good-first-issue` and `help-wanted` tags.
If you're working on something new, file an issue first to coordinate.

### 2. Create a branch

```bash
git checkout -b feat/my-feature
```

Branch naming: `feat/...`, `fix/...`, `docs/...`, `test/...`,
`refactor/...`, `chore/...`.

### 3. Make changes

Follow the [code style](#code-style). Run `cargo fmt` and `cargo clippy`
frequently.

### 4. Test

```bash
# Fast tests (no real-backends)
cargo test -p qbitos-skills -p qbitos-training -p qbitos-services

# Full tests
cargo test --workspace --features qbitos-cli/real-backends --exclude qbitos-desktop
```

### 5. Commit

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(skills): add late-binding NativeAgentBackend
fix(http): handle axum 0.8 Utf8Bytes conversion
docs(api): add WebSocket event type reference
test(services): cover persistent event store replay
```

### 6. Push and open a PR

```bash
git push -u origin feat/my-feature
```

Open a PR against `main`. Fill in the PR template. CI runs automatically.

---

## Code Style

### Rust

- Edition 2024 (set in `Cargo.toml`).
- `#![cfg_attr(not(test), warn(clippy::pedantic))]` at the top of every crate.
- `#![warn(missing_docs)]` — every public item must have a doc comment.
- Use `#[must_use]` on constructors that return `Self`.
- Use `thiserror` for error types, `anyhow` for application-level errors.
- Prefer `Arc<T>` over `Rc<T>` (everything must be `Send + Sync`).
- Use `parking_lot::RwLock` over `std::sync::RwLock` (faster, no poisoning).
- Use `ahash::AHashMap` over `std::collections::HashMap` (faster, non-cryptographic).
- Use `tokio::sync::Mutex` only when async lock-holding is required.

### Documentation

- Every public function, struct, enum, and trait must have a `///` doc comment.
- Module-level docs (`//!`) at the top of every file.
- Include `# Example` sections for non-trivial APIs.
- Cross-reference other items with `[`Item`]` syntax.

### Formatting

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
```

---

## Testing

### Test layout

- Unit tests: `#[cfg(test)] mod tests { ... }` at the bottom of each file.
- Integration tests: `tests/` directory in each crate.
- Doctests: `///` examples are compiled and run with `cargo test --doc`.

### Test naming

Use `snake_case` describing the behavior under test:

```rust
#[tokio::test]
async fn cognitive_cycle_dispatches_through_real_kernel() { ... }
```

### Feature-gated tests

For tests that require optional features:

```rust
#[cfg(feature = "real-backends")]
#[tokio::test]
async fn real_backends_wired_into_skill_module() { ... }
```

### Running tests

```bash
# All tests in a crate
cargo test -p qbitos-services

# Single test
cargo test -p qbitos-services -- test_name

# With a feature
cargo test -p qbitos-skills --features real-backends

# Doc tests
cargo test --doc -p qbitos-skills
```

### Test counts (current)

| Crate | Default | With `real-backends` |
|-------|---------|----------------------|
| qbitos-skills | 24 | 33 |
| qbitos-training | 29 | 29 |
| qbitos-services | 39 | 39 |
| qbitos-cli | 10 | 13 |
| (others) | 84 | 84 |
| **Total** | **186** | **198** |

---

## Adding a New Subsystem

1. Create a new directory: `my_subsystem/`
2. Add a `Cargo.toml`:
   ```toml
   [package]
   name = "qbitos-my-subsystem"
   version.workspace = true
   edition.workspace = true
   # ...

   [dependencies]
   serde.workspace = true
   # qbitos-memory.workspace = true (if needed)
   ```
3. Add to workspace `Cargo.toml`:
   ```toml
   members = [..., "my_subsystem"]
   qbitos-my-subsystem = { path = "my_subsystem" }
   ```
4. Add `src/lib.rs` with module-level docs and `#![warn(missing_docs)]`.
5. Wire into `cli/src/app.rs` `App` struct and `App::new()`.
6. Add tests.
7. Add a `README.md` (see [per-crate README template](#per-crate-readme-template)).
8. Update top-level `README.md` and `docs/ARCHITECTURE.md`.

---

## Adding a New HTTP Endpoint

1. Add the DTO to `services/src/http/dto.rs`.
2. Add the handler to `services/src/http/handlers.rs`:
   ```rust
   pub async fn my_endpoint(
       State(state): State<ServiceState>,
       AxumJson(req): AxumJson<MyRequest>,
   ) -> Result<Json<ApiResponse<MyResponse>>, ServiceError> {
       state.svc.record_request();
       // ... business logic ...
       Ok(Json(ApiResponse::ok(response)))
   }
   ```
3. Add the route to `services/src/http/router.rs`:
   ```rust
   .route("/api/v1/my-endpoint", post(handlers::my_endpoint))
   ```
4. Add to OpenAPI spec in `services/src/openapi.rs`.
5. Add to the docs in `docs/API.md`.
6. Add a test in `services/src/http/handlers.rs` (unit) or
   `services/tests/` (integration).

---

## Adding a New Skill

Skills are registered at runtime via the `SkillRegistry`:

```rust
use qbitos_skills::{Skill, SkillStep, StepKind};

let mut skill = Skill::new("my-skill", "Does the thing");
skill.add_step(SkillStep::new(
    "infer",
    StepKind::Infer,
    serde_json::json!({"prompt": "Do the thing"}),
));
skill.add_step(SkillStep::new(
    "return",
    StepKind::Return,
    serde_json::json!({}),
));
skill.success_criteria.push("output_is_valid".into());
let id = app.skills.registry.register(skill);
```

To make the skill available across restarts, persist it via
`qbitos state save`.

To mine skills from datasets, use the `DatasetToSkills` pipeline:

```rust
app.ds_to_skills.mine_from_path(&app.data, path, Some(1000)).await?;
```

---

## Adding a New Tool

1. Implement the `Tool` trait in `tools/src/tools/my_tool.rs`:
   ```rust
   pub struct MyTool;

   #[async_trait]
   impl Tool for MyTool {
       fn name(&self) -> &str { "my_tool" }
       fn description(&self) -> &str { "Does my thing" }
       fn action(&self) -> Action { Action::Read }
       fn scope(&self, _params: &Value) -> Scope { Scope::Local }
       async fn execute(&self, params: &Value) -> Result<Value, ToolError> {
           // ...
       }
   }
   ```
2. Register in `tools/src/tool_module.rs` `with_builtin_tools`:
   ```rust
   self.registry.register(Arc::new(crate::tools::my_tool::MyTool));
   ```
3. Add tests in `tools/src/tools/my_tool.rs`.
4. Grant permissions to agents that should use it:
   ```rust
   app.tools.grant(agent_id, Permission::Read);
   ```

---

## Adding a New Dataset Loader

1. Implement the `DatasetLoader` trait in `data/src/loaders/my_loader.rs`.
2. Register in `data/src/registry.rs` `with_default_loaders`.
3. Add the file extension to the format detection logic.
4. Add tests with sample files in `data/tests/data/`.

---

## Submitting Changes

### PR checklist

- [ ] Branch is up-to-date with `main`
- [ ] `cargo fmt --all` passes
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passes
- [ ] All tests pass: `cargo test --workspace --exclude qbitos-desktop`
- [ ] New code has doc comments
- [ ] New code has tests
- [ ] `README.md` updated if user-facing
- [ ] `docs/` updated if architecture/API changed
- [ ] `CHANGELOG.md` updated
- [ ] Conventional Commit messages

### Review process

1. Automated CI must pass (build + test + clippy + fmt).
2. At least one maintainer review required.
3. Changes requested by reviewers must be addressed.
4. Squash-merge on approval.

---

## Release Process

1. Update `CHANGELOG.md` with the new version and changes.
2. Bump `version.workspace = true` in the root `Cargo.toml`.
3. Tag: `git tag v0.X.Y && git push --tags`.
4. CI builds release binaries for Linux x86_64.
5. GitHub Release is created automatically with binaries attached.
6. Docker image published to `ghcr.io/qbitos/qbitos:v0.X.Y`.

---

## Per-crate README template

```markdown
# qbitos-<name>

<one-paragraph description>

## Overview

<what it does, why it exists>

## Architecture

<module structure, key types>

## Usage

```rust
// code example
```

## Testing

```bash
cargo test -p qbitos-<name>
```

## See Also

- [Architecture](../docs/ARCHITECTURE.md)
- [API Reference](../docs/API.md)
- [Top-level README](../README.md)
```

---

## See Also

- [`ARCHITECTURE.md`](./ARCHITECTURE.md)
- [`API.md`](./API.md)
- [`DEPLOYMENT.md`](./DEPLOYMENT.md)
- [`CHANGELOG.md`](./CHANGELOG.md)
