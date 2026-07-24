//! QbitOS Examples — end-to-end demos.
//!
//! This crate hosts the runnable examples for QbitOS. Each example lives in
//! `examples/<name>.rs` and can be run with `cargo run -p qbitos-examples
//! --example <name>`.
//!
//! # Available Examples
//!
//! - `cognitive_cycle` — exercises the HTTP REST API: health, memory store/search,
//!   cognitive cycle, training step, curriculum, stats.
//! - `ws_events` — connects to the WebSocket event stream and prints events.
//! - `rpc_client` — exercises the JSON-RPC 2.0 interface (in-process and HTTP).
//! - `self_improvement_loop` — runs a full training loop in-process.

#![cfg_attr(not(test), warn(clippy::pedantic))]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions, clippy::must_use_candidate, clippy::missing_errors_doc)]
