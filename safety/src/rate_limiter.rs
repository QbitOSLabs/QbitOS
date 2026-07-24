//! Rate limiter — prevents resource exhaustion.

use ahash::AHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// A rate limit configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    /// Maximum number of requests in the window.
    pub max_requests: u32,
    /// Time window.
    pub window: Duration,
}

/// Rate limiter using a sliding window per key.
pub struct RateLimiter {
    limits: RwLock<AHashMap<String, Vec<Instant>>>,
    default_limit: RateLimit,
    per_key_limits: RwLock<AHashMap<String, RateLimit>>,
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self {
            limits: RwLock::new(AHashMap::new()),
            default_limit: RateLimit { max_requests: 100, window: Duration::from_secs(60) },
            per_key_limits: RwLock::new(AHashMap::new()),
        }
    }
}

impl RateLimiter {
    /// Create a new rate limiter.
    #[must_use]
    pub fn new(max_requests: u32, window: Duration) -> Self {
        Self {
            limits: RwLock::new(AHashMap::new()),
            default_limit: RateLimit { max_requests, window },
            per_key_limits: RwLock::new(AHashMap::new()),
        }
    }

    /// Set a custom limit for a specific key.
    pub fn set_limit(&self, key: &str, max_requests: u32, window: Duration) {
        self.per_key_limits.write().insert(key.to_string(), RateLimit { max_requests, window });
    }

    /// Check if a request is allowed. Returns `true` if allowed, `false` if rate-limited.
    pub fn check(&self, key: &str) -> bool {
        let now = Instant::now();
        let limit = self.per_key_limits.read().get(key).cloned().unwrap_or_else(|| self.default_limit.clone());

        let mut limits = self.limits.write();
        let timestamps = limits.entry(key.to_string()).or_default();

        // Remove expired timestamps
        timestamps.retain(|&t| now.duration_since(t) < limit.window);

        if timestamps.len() >= limit.max_requests as usize {
            return false;
        }

        timestamps.push(now);
        true
    }

    /// Get the current request count for a key in the current window.
    #[must_use]
    pub fn current_count(&self, key: &str) -> usize {
        let now = Instant::now();
        let limit = self.per_key_limits.read().get(key).cloned().unwrap_or_else(|| self.default_limit.clone());
        self.limits.read().get(key).map_or(0, |ts| ts.iter().filter(|&&t| now.duration_since(t) < limit.window).count())
    }

    /// Reset the rate limiter for a key.
    pub fn reset(&self, key: &str) {
        self.limits.write().remove(key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_under_limit() {
        let limiter = RateLimiter::new(5, Duration::from_secs(60));
        for _ in 0..5 {
            assert!(limiter.check("user1"));
        }
        assert!(!limiter.check("user1")); // 6th request should be denied
    }

    #[test]
    fn separate_keys_independent() {
        let limiter = RateLimiter::new(2, Duration::from_secs(60));
        assert!(limiter.check("a"));
        assert!(limiter.check("a"));
        assert!(!limiter.check("a"));
        assert!(limiter.check("b")); // different key
        assert!(limiter.check("b"));
        assert!(!limiter.check("b"));
    }

    #[test]
    fn custom_per_key_limit() {
        let limiter = RateLimiter::new(10, Duration::from_secs(60));
        limiter.set_limit("premium", 100, Duration::from_secs(60));
        limiter.set_limit("free", 5, Duration::from_secs(60));
        for _ in 0..5 { assert!(limiter.check("free")); }
        assert!(!limiter.check("free"));
        for _ in 0..10 { assert!(limiter.check("premium")); }
        assert!(limiter.check("premium")); // still allowed
    }
}
