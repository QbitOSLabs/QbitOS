#![allow(missing_docs)]

//! Agent reliability: circuit breaker + retry with exponential backoff + timeout.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use thiserror::Error;

/// Errors emitted by the reliability layer.
#[derive(Debug, Error)]
pub enum ReliabilityError<E> {
    /// Circuit is open; calls are short-circuited.
    #[error("circuit open (will reset after {remaining_ms}ms)")]
    CircuitOpen { remaining_ms: u64 },
    /// Timeout exceeded.
    #[error("timeout after {0:?}")]
    Timeout(Duration),
    /// All retries exhausted.
    #[error("retries exhausted: {0}")]
    RetriesExhausted(String),
    /// Inner error from the wrapped call.
    #[error(transparent)]
    Inner(#[from] E),
}

/// Circuit breaker state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CircuitState { Closed, Open, HalfOpen }

struct Inner {
    state: CircuitState,
    failure_count: u32,
    last_failure: Option<Instant>,
    probe_in_flight: bool,
}

/// A circuit breaker that opens after `threshold` consecutive failures and
/// attempts to close after `reset_timeout` has elapsed.
pub struct CircuitBreaker { threshold: u32, reset_timeout: Duration, inner: Mutex<Inner> }

impl CircuitBreaker {
    /// Create a new breaker.
    #[must_use]
    pub fn new(threshold: u32, reset_timeout: Duration) -> Self {
        Self {
            threshold, reset_timeout,
            inner: Mutex::new(Inner { state: CircuitState::Closed, failure_count: 0, last_failure: None, probe_in_flight: false }),
        }
    }
    /// Current state.
    pub fn state(&self) -> CircuitState { self.inner.lock().state }

    /// Invoke `f` through the breaker.
    pub fn call<F, T, E>(&self, f: F) -> Result<T, ReliabilityError<E>>
    where F: FnOnce() -> Result<T, E>, E: std::fmt::Display,
    {
        {
            let mut g = self.inner.lock();
            match g.state {
                CircuitState::Closed => {}
                CircuitState::Open => {
                    let elapsed = g.last_failure.map(|t| t.elapsed()).unwrap_or(self.reset_timeout);
                    if elapsed >= self.reset_timeout {
                        g.state = CircuitState::HalfOpen; g.probe_in_flight = true;
                    } else {
                        return Err(ReliabilityError::CircuitOpen { remaining_ms: (self.reset_timeout - elapsed).as_millis() as u64 });
                    }
                }
                CircuitState::HalfOpen => {
                    if g.probe_in_flight { return Err(ReliabilityError::CircuitOpen { remaining_ms: 0 }); }
                    g.probe_in_flight = true;
                }
            }
        }
        match f() {
            Ok(v) => {
                let mut g = self.inner.lock();
                g.state = CircuitState::Closed; g.failure_count = 0;
                g.last_failure = None; g.probe_in_flight = false;
                Ok(v)
            }
            Err(e) => {
                let mut g = self.inner.lock();
                g.failure_count = g.failure_count.saturating_add(1);
                g.last_failure = Some(Instant::now());
                g.probe_in_flight = false;
                if g.state == CircuitState::HalfOpen || g.failure_count >= self.threshold { g.state = CircuitState::Open; }
                Err(ReliabilityError::Inner(e))
            }
        }
    }
}

/// Retry policy with exponential backoff.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum number of retries (0 = one attempt only).
    pub max_retries: u32,
    /// Initial delay between retries.
    pub initial_delay: Duration,
    /// Backoff multiplier.
    pub backoff_factor: f32,
    /// Maximum delay between retries.
    pub max_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self { Self { max_retries: 3, initial_delay: Duration::from_millis(50), backoff_factor: 2.0, max_delay: Duration::from_secs(5) } }
}

impl RetryPolicy {
    /// Create a new policy with defaults.
    #[must_use]
    pub fn new() -> Self { Self::default() }

    /// Execute `f`, retrying on error up to `max_retries` times.
    pub fn execute<F, T, E>(&self, mut f: F) -> Result<T, ReliabilityError<E>>
    where F: FnMut() -> Result<T, E>, E: std::fmt::Display,
    {
        let mut delay = self.initial_delay;
        let mut last_err: Option<String> = None;
        for attempt in 0..=self.max_retries {
            match f() {
                Ok(v) => return Ok(v),
                Err(e) => {
                    last_err = Some(e.to_string());
                    if attempt == self.max_retries { break; }
                    std::thread::sleep(delay);
                    delay = Duration::from_millis((delay.as_millis() as f32 * self.backoff_factor) as u64).min(self.max_delay);
                }
            }
        }
        Err(ReliabilityError::RetriesExhausted(last_err.unwrap_or_default()))
    }
}

/// Execute a closure with a hard timeout (sync; uses a worker thread).
pub fn with_timeout<F, T>(timeout: Duration, f: F) -> Result<T, ReliabilityError<std::io::Error>>
where F: FnOnce() -> T + Send + 'static, T: Send + 'static,
{
    use std::sync::mpsc;
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || { let _ = tx.send(f()); });
    rx.recv_timeout(timeout).map_err(|_| ReliabilityError::Timeout(timeout))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    #[test]
    fn breaker_opens_after_threshold() {
        let b = CircuitBreaker::new(2, Duration::from_millis(50));
        let counter = Rc::new(Cell::new(0u32));
        for _ in 0..3 {
            let c = counter.clone();
            let _ = b.call(|| -> Result<i32, &str> { c.set(c.get() + 1); Err("boom") });
        }
        assert_eq!(counter.get(), 2);
        assert_eq!(b.state(), CircuitState::Open);
    }

    #[test]
    fn breaker_resets_after_timeout() {
        let b = CircuitBreaker::new(1, Duration::from_millis(20));
        let _ = b.call(|| -> Result<i32, &str> { Err("x") });
        assert_eq!(b.state(), CircuitState::Open);
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(b.call(|| -> Result<i32, &str> { Ok(42) }).unwrap(), 42);
        assert_eq!(b.state(), CircuitState::Closed);
    }

    #[test]
    fn retry_exhausts_and_returns_error() {
        let p = RetryPolicy { max_retries: 2, initial_delay: Duration::from_millis(1), backoff_factor: 2.0, max_delay: Duration::from_millis(5) };
        assert!(matches!(p.execute(|| -> Result<i32, &str> { Err("fail") }), Err(ReliabilityError::RetriesExhausted(_))));
    }

    #[test]
    fn retry_succeeds_on_second_attempt() {
        let p = RetryPolicy { max_retries: 3, initial_delay: Duration::from_millis(1), backoff_factor: 2.0, max_delay: Duration::from_millis(5) };
        let counter = Rc::new(Cell::new(0u32));
        let c = counter.clone();
        let r = p.execute(move || -> Result<i32, &str> {
            let n = c.get(); c.set(n + 1);
            if n < 1 { Err("retry") } else { Ok(99) }
        });
        assert_eq!(r.unwrap(), 99);
    }
}
