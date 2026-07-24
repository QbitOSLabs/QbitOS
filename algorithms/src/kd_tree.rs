//! KD-Tree — k-dimensional spatial indexing for nearest-neighbor search.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// A k-dimensional point.
pub type Point = Vec<f64>;

/// A KD-tree for efficient k-nearest-neighbor and range queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KdTree {
    root: Option<Box<Node>>,
    k: usize,
    size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Node {
    point: Point,
    payload: serde_json::Value,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
}

impl KdTree {
    /// Construct an empty KD-tree with dimensionality `k`.
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self {
            root: None,
            k: k.max(1),
            size: 0,
        }
    }

    /// Dimensionality of the tree.
    #[must_use]
    pub fn dimensions(&self) -> usize {
        self.k
    }

    /// Number of points in the tree.
    #[must_use]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Whether the tree is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Insert a point with an associated JSON payload.
    ///
    /// # Panics
    /// Panics if the point's dimensionality doesn't match the tree's.
    pub fn insert(&mut self, point: Point, payload: serde_json::Value) {
        assert_eq!(point.len(), self.k, "point dimension mismatch");
        let new_node = Box::new(Node {
            point,
            payload,
            left: None,
            right: None,
        });
        let root = self.root.take();
        self.root = Some(self.insert_recursive(root, new_node, 0));
        self.size += 1;
    }

    fn insert_recursive(
        &self,
        node: Option<Box<Node>>,
        new_node: Box<Node>,
        depth: u32,
    ) -> Box<Node> {
        match node {
            None => new_node,
            Some(mut n) => {
                let axis = (depth as usize) % self.k;
                if new_node.point[axis] < n.point[axis] {
                    n.left = Some(self.insert_recursive(n.left.take(), new_node, depth + 1));
                } else {
                    n.right = Some(self.insert_recursive(n.right.take(), new_node, depth + 1));
                }
                n
            }
        }
    }

    /// Find the `k` nearest neighbors of `target`.
    ///
    /// Returns a vector of (point, payload, distance²) sorted by distance ascending.
    #[must_use]
    pub fn nearest(&self, target: &Point, k: usize) -> Vec<(Point, serde_json::Value, f64)> {
        if self.root.is_none() || k == 0 {
            return Vec::new();
        }
        let mut heap: MaxHeap = Vec::with_capacity(k + 1);
        self.nearest_recursive(self.root.as_deref(), target, k, 0, &mut heap);
        // Sort ascending by distance.
        heap.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));
        heap
    }

    fn nearest_recursive(
        &self,
        node: Option<&Node>,
        target: &Point,
        k: usize,
        depth: u32,
        heap: &mut MaxHeap,
    ) {
        let Some(n) = node else {
            return;
        };
        let dist_sq = squared_distance(&n.point, target);
        if heap.len() < k {
            heap.push((n.point.clone(), n.payload.clone(), dist_sq));
        } else if let Some(&(_, _, max_d)) = heap.first() {
            if dist_sq < max_d {
                heap[0] = (n.point.clone(), n.payload.clone(), dist_sq);
                heap.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
            }
        }

        let axis = (depth as usize) % self.k;
        let diff = target[axis] - n.point[axis];
        let (near, far) = if diff < 0.0 {
            (n.left.as_deref(), n.right.as_deref())
        } else {
            (n.right.as_deref(), n.left.as_deref())
        };
        self.nearest_recursive(near, target, k, depth + 1, heap);

        // Check the far side if it could contain a closer point.
        let dist_to_plane = diff * diff;
        let should_visit_far = if heap.len() < k {
            true
        } else if let Some(&(_, _, max_d)) = heap.first() {
            dist_to_plane < max_d
        } else {
            true
        };
        if should_visit_far {
            self.nearest_recursive(far, target, k, depth + 1, heap);
        }
    }

    /// Range query: return all points within Euclidean distance `radius` of `target`.
    #[must_use]
    pub fn range(&self, target: &Point, radius: f64) -> Vec<(Point, serde_json::Value, f64)> {
        if self.root.is_none() {
            return Vec::new();
        }
        let mut out = Vec::new();
        let radius_sq = radius * radius;
        self.range_recursive(self.root.as_deref(), target, radius_sq, 0, &mut out);
        out.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));
        out
    }

    fn range_recursive(
        &self,
        node: Option<&Node>,
        target: &Point,
        radius_sq: f64,
        depth: u32,
        out: &mut Vec<(Point, serde_json::Value, f64)>,
    ) {
        let Some(n) = node else {
            return;
        };
        let dist_sq = squared_distance(&n.point, target);
        if dist_sq <= radius_sq {
            out.push((n.point.clone(), n.payload.clone(), dist_sq));
        }
        let axis = (depth as usize) % self.k;
        let diff = target[axis] - n.point[axis];
        let (near, far) = if diff < 0.0 {
            (n.left.as_deref(), n.right.as_deref())
        } else {
            (n.right.as_deref(), n.left.as_deref())
        };
        self.range_recursive(near, target, radius_sq, depth + 1, out);
        // Visit the far side if the split plane is within the radius.
        if diff * diff <= radius_sq {
            self.range_recursive(far, target, radius_sq, depth + 1, out);
        }
    }
}

type MaxHeap = Vec<(Point, serde_json::Value, f64)>;

fn squared_distance(a: &Point, b: &Point) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_finds_closest_point() {
        let mut tree = KdTree::new(2);
        tree.insert(vec![1.0, 1.0], serde_json::json!("a"));
        tree.insert(vec![2.0, 2.0], serde_json::json!("b"));
        tree.insert(vec![3.0, 3.0], serde_json::json!("c"));
        tree.insert(vec![10.0, 10.0], serde_json::json!("far"));

        let nn = tree.nearest(&vec![2.1, 2.1], 1);
        assert_eq!(nn.len(), 1);
        assert_eq!(nn[0].1, serde_json::json!("b"));
    }

    #[test]
    fn nearest_k_returns_k_results() {
        let mut tree = KdTree::new(2);
        for i in 0..10 {
            let x = i as f64;
            tree.insert(vec![x, x], serde_json::json!(i));
        }
        let nn = tree.nearest(&vec![5.0, 5.0], 3);
        assert_eq!(nn.len(), 3);
        // Closest points to (5,5) are 4,5,6.
        let ids: Vec<i64> = nn
            .iter()
            .map(|(_, v, _)| v.as_i64().unwrap())
            .collect();
        assert!(ids.contains(&4));
        assert!(ids.contains(&5));
        assert!(ids.contains(&6));
    }

    #[test]
    fn range_returns_points_within_radius() {
        let mut tree = KdTree::new(2);
        tree.insert(vec![0.0, 0.0], serde_json::json!("o"));
        tree.insert(vec![1.0, 0.0], serde_json::json!("a"));
        tree.insert(vec![2.0, 0.0], serde_json::json!("b"));
        tree.insert(vec![10.0, 10.0], serde_json::json!("far"));

        let r = tree.range(&vec![0.0, 0.0], 1.5);
        assert_eq!(r.len(), 2); // o and a
        let labels: Vec<&str> = r.iter().map(|(_, v, _)| v.as_str().unwrap()).collect();
        assert!(labels.contains(&"o"));
        assert!(labels.contains(&"a"));
    }

    #[test]
    fn empty_tree_returns_empty() {
        let tree = KdTree::new(3);
        assert!(tree.is_empty());
        assert!(tree.nearest(&vec![0.0, 0.0, 0.0], 5).is_empty());
        assert!(tree.range(&vec![0.0, 0.0, 0.0], 1.0).is_empty());
    }

    #[test]
    fn insert_increments_size() {
        let mut tree = KdTree::new(2);
        assert_eq!(tree.len(), 0);
        tree.insert(vec![1.0, 1.0], serde_json::json!(1));
        assert_eq!(tree.len(), 1);
        tree.insert(vec![2.0, 2.0], serde_json::json!(2));
        assert_eq!(tree.len(), 2);
    }
}
