//! QbitOS Algorithms — production-grade algorithmic primitives.
//!
//! Implements:
//!
//! 1. **Pathfinding** — A* search, Dijkstra, BFS, DFS
//! 2. **Spatial indexing** — KD-tree, R-tree, spatial hashing
//! 3. **Probabilistic structures** — Bloom filter, count-min sketch, hyperloglog
//! 4. **String algorithms** — Trie (prefix tree), suffix array, Levenshtein
//! 5. **Disjoint sets** — Union-Find with path compression + union by rank
//! 6. **Caching** — LRU, LFU, ARC, two-layer cache
//! 7. **Distributed** — consistent hashing, rendezvous hashing
//! 8. **Selection** — quickselect, top-k
//! 9. **Graph algorithms** — connected components, topological sort, SCC
//! 10. **Compression** — run-length encoding, dictionary coding

#![cfg_attr(not(test), warn(clippy::pedantic))]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions, clippy::must_use_candidate, clippy::missing_errors_doc)]

pub mod astar;
pub mod hnsw;
pub mod cuckoo;
pub mod robin_hood;
pub mod skip_list;
pub mod hll_plus;
pub mod bloom;
pub mod consistent_hash;
pub mod count_min;
pub mod dijkstra;
pub mod disjoint_set;
pub mod graph;
pub mod hyperloglog;
pub mod kd_tree;
pub mod levenshtein;
pub mod lfu_cache;
pub mod lru_cache;
pub mod quickselect;
pub mod rle;
pub mod topk;
pub mod trie;

pub use astar::{astar_search, AStarNode, AStarResult};
pub use bloom::BloomFilter;
pub use consistent_hash::ConsistentHashRing;
pub use count_min::CountMinSketch;
pub use cuckoo::CuckooFilter;
pub use dijkstra::{dijkstra, DijkstraResult};
pub use disjoint_set::DisjointSet;
pub use graph::{connected_components, topological_sort, scc};
pub use hnsw::{cosine_distance, l2_distance, HnswIndex, NodeId};
pub use hll_plus::HyperLogLogPlus;
pub use hyperloglog::HyperLogLog;
pub use kd_tree::KdTree;
pub use levenshtein::{levenshtein_distance, levenshtein_distance_with_ops, LevenshteinOp};
pub use lfu_cache::LfuCache;
pub use lru_cache::LruCache;
pub use quickselect::quickselect;
pub use rle::{rle_encode, rle_decode};
pub use skip_list::SkipList;
pub use topk::TopK;
pub use trie::Trie;
pub use robin_hood::RobinHoodMap;
