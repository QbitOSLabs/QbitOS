//! Trie (prefix tree) for efficient prefix queries.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A trie storing arbitrary string keys with associated JSON payloads.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Trie {
    root: TrieNode,
    size: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct TrieNode {
    // BTreeMap gives sorted iteration for free, which makes prefix_search deterministic.
    children: BTreeMap<char, TrieNode>,
    payload: Option<serde_json::Value>,
    is_end: bool,
}

impl Trie {
    /// Construct an empty trie.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of stored keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Whether the trie is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Insert a key with the given payload. If the key exists, the payload is replaced.
    pub fn insert(&mut self, key: &str, payload: serde_json::Value) {
        let mut node = &mut self.root;
        for ch in key.chars() {
            node = node.children.entry(ch).or_default();
        }
        if !node.is_end {
            self.size += 1;
            node.is_end = true;
        }
        node.payload = Some(payload);
    }

    /// Look up the payload for an exact key match.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        let mut node = &self.root;
        for ch in key.chars() {
            node = node.children.get(&ch)?;
        }
        if node.is_end {
            node.payload.as_ref()
        } else {
            None
        }
    }

    /// Remove a key from the trie. Returns the removed payload if the key existed.
    pub fn remove(&mut self, key: &str) -> Option<serde_json::Value> {
        // Walk down, recording the path of (parent_node, child_char) so we can
        // prune empty branches on the way back up.
        let chars: Vec<char> = key.chars().collect();
        // Build a path of pointers to each node along the key.
        let mut path: Vec<*mut TrieNode> = Vec::with_capacity(chars.len() + 1);
        // SAFETY: we hold a mutable reference via &mut self for the whole operation.
        let root_ptr: *mut TrieNode = &mut self.root;
        path.push(root_ptr);
        unsafe {
            let mut node: *mut TrieNode = root_ptr;
            for &ch in &chars {
                let next = match (*node).children.get_mut(&ch) {
                    Some(n) => n as *mut TrieNode,
                    None => return None,
                };
                path.push(next);
                node = next;
            }
            if !(*node).is_end {
                return None;
            }
            (*node).is_end = false;
            let payload = (*node).payload.take();
            self.size -= 1;

            // Walk back up the path, pruning empty nodes.
            for i in (0..chars.len()).rev() {
                let parent = path[i];
                let child_key = chars[i];
                let child = (*parent).children.get(&child_key).map(|c| c as *const TrieNode);
                if let Some(child_ptr) = child {
                    if (*child_ptr).children.is_empty() && !(*child_ptr).is_end {
                        (*parent).children.remove(&child_key);
                    } else {
                        break;
                    }
                }
            }
            payload
        }
    }

    /// Whether the trie contains an exact key.
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    /// Return all keys with the given prefix.
    #[must_use]
    pub fn prefix_search(&self, prefix: &str) -> Vec<(String, serde_json::Value)> {
        let mut node = &self.root;
        for ch in prefix.chars() {
            match node.children.get(&ch) {
                Some(n) => node = n,
                None => return Vec::new(),
            }
        }
        let mut out = Vec::new();
        let mut current = prefix.to_string();
        self.collect(node, &mut current, &mut out);
        out
    }

    fn collect(
        &self,
        node: &TrieNode,
        current: &mut String,
        out: &mut Vec<(String, serde_json::Value)>,
    ) {
        if node.is_end {
            if let Some(p) = &node.payload {
                out.push((current.clone(), p.clone()));
            }
        }
        // BTreeMap iterates in sorted key order, so results are deterministic.
        for (&ch, child) in &node.children {
            current.push(ch);
            self.collect(child, current, out);
            current.pop();
        }
    }

    /// Return all keys that start with `prefix`, limited to `max` results.
    #[must_use]
    pub fn autocomplete(&self, prefix: &str, max: usize) -> Vec<String> {
        let mut results = self.prefix_search(prefix);
        results.truncate(max);
        results.into_iter().map(|(k, _)| k).collect()
    }

    /// Longest common prefix of all keys in the trie.
    #[must_use]
    pub fn longest_common_prefix(&self) -> String {
        let mut prefix = String::new();
        let mut node = &self.root;
        while node.children.len() == 1 && !node.is_end {
            let (&ch, child) = node.children.iter().next().unwrap();
            prefix.push(ch);
            node = child;
        }
        prefix
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.root = TrieNode::default();
        self.size = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get() {
        let mut t = Trie::new();
        t.insert("hello", serde_json::json!(1));
        t.insert("help", serde_json::json!(2));
        t.insert("he", serde_json::json!(3));
        assert_eq!(t.len(), 3);
        assert_eq!(t.get("hello"), Some(&serde_json::json!(1)));
        assert_eq!(t.get("help"), Some(&serde_json::json!(2)));
        assert_eq!(t.get("he"), Some(&serde_json::json!(3)));
        assert_eq!(t.get("hel"), None);
    }

    #[test]
    fn prefix_search_returns_all_matches() {
        let mut t = Trie::new();
        t.insert("apple", serde_json::json!(1));
        t.insert("app", serde_json::json!(2));
        t.insert("application", serde_json::json!(3));
        t.insert("banana", serde_json::json!(4));

        let results = t.prefix_search("app");
        assert_eq!(results.len(), 3);
        let keys: Vec<String> = results.into_iter().map(|(k, _)| k).collect();
        assert!(keys.contains(&"app".to_string()));
        assert!(keys.contains(&"apple".to_string()));
        assert!(keys.contains(&"application".to_string()));
    }

    #[test]
    fn autocomplete_limits_results() {
        let mut t = Trie::new();
        for i in 0..10 {
            t.insert(&format!("key{i}"), serde_json::json!(i));
        }
        let r = t.autocomplete("key", 3);
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn remove_returns_payload() {
        let mut t = Trie::new();
        t.insert("hello", serde_json::json!(1));
        assert_eq!(t.remove("hello"), Some(serde_json::json!(1)));
        assert!(!t.contains("hello"));
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn remove_nonexistent_returns_none() {
        let mut t = Trie::new();
        t.insert("hello", serde_json::json!(1));
        assert_eq!(t.remove("world"), None);
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn longest_common_prefix_works() {
        let mut t = Trie::new();
        t.insert("interspecies", serde_json::json!(1));
        t.insert("interstellar", serde_json::json!(2));
        t.insert("interstate", serde_json::json!(3));
        assert_eq!(t.longest_common_prefix(), "inters");
    }
}
