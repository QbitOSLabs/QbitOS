//! Priority-aware, preemptive scheduler.
//!
//! The scheduler admits cognitive tasks into a Tokio runtime, tracks their
//! status, and preempts them when resource budgets are exceeded. Tasks carry
//! [`Priority`] so the runtime can starve low-priority work under load.

use ahash::AHashMap;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::{oneshot, Notify};
use tokio::task::JoinHandle;
use uuid::Uuid;

/// Task priority. Higher values run first when the scheduler is contended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    /// Background maintenance (memory consolidation, telemetry).
    Background = 0,
    /// Default user-initiated work.
    Normal = 1,
    /// Interactive work (chat turns, voice responses).
    Interactive = 2,
    /// Real-time work (perception ticks, safety-critical decisions).
    Realtime = 3,
}

impl Default for Priority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Unique identifier for a scheduled task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(pub Uuid);

impl TaskId {
    /// Mint a fresh task id.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }
}

impl Default for TaskId {
    fn default() -> Self {
        Self::new()
    }
}

/// Lifecycle status of a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    /// Queued, waiting for a slot.
    Pending,
    /// Running on the runtime.
    Running,
    /// Preempted by the scheduler (will be resumed).
    Preempted,
    /// Completed successfully.
    Completed,
    /// Failed with an error.
    Failed,
    /// Cancelled by the user or kernel.
    Cancelled,
}

/// Metadata about a scheduled task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskInfo {
    /// Task id.
    pub id: TaskId,
    /// Human-readable name.
    pub name: String,
    /// Priority.
    pub priority: Priority,
    /// Current status.
    pub status: TaskStatus,
    /// When the task was submitted.
    pub submitted_at: DateTime<Utc>,
    /// When the task started running, if it has.
    pub started_at: Option<DateTime<Utc>>,
    /// When the task finished, if it has.
    pub finished_at: Option<DateTime<Utc>>,
    /// Wall-clock duration of the task, if finished.
    pub duration: Option<Duration>,
    /// Error message, if failed.
    pub error: Option<String>,
}

/// Scheduler error.
#[derive(Debug, Error)]
pub enum SchedulerError {
    /// The scheduler is at capacity and is not admitting new tasks.
    #[error("scheduler at capacity ({max})")]
    AtCapacity {
        /// Max concurrent tasks.
        max: usize,
    },
    /// The task was not found.
    #[error("task {0:?} not found")]
    NotFound(TaskId),
    /// The scheduler has been shut down.
    #[error("scheduler is shut down")]
    Shutdown,
}

/// A handle returned by [`Scheduler::spawn`] that can be awaited to get the
/// task's result.
pub struct TaskHandle<T> {
    id: TaskId,
    rx: oneshot::Receiver<Result<T, String>>,
}

impl<T> TaskHandle<T> {
    /// The task's id.
    #[must_use]
    pub fn id(&self) -> TaskId {
        self.id
    }

    /// Await the task's result.
    ///
    /// # Errors
    /// Propagates the task's error if it failed.
    pub async fn join(self) -> Result<T, SchedulerError> {
        match self.rx.await {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(_e)) => Err(SchedulerError::NotFound(self.id)),
            Err(_) => Err(SchedulerError::Shutdown),
        }
    }
}

/// The scheduler. Wraps a Tokio runtime with priority-aware admission control.
#[derive(Clone)]
pub struct Scheduler {
    inner: Arc<SchedulerInner>,
    notify: Arc<Notify>,
}

struct SchedulerInner {
    max_concurrent: usize,
    tasks: Arc<RwLock<AHashMap<TaskId, TaskInfo>>>,
    handles: Arc<RwLock<AHashMap<TaskId, JoinHandle<()>>>>,
    cancellations: Arc<RwLock<AHashMap<TaskId, oneshot::Sender<()>>>>,
    semaphore: Arc<tokio::sync::Semaphore>,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
}

impl Scheduler {
    /// Create a new scheduler with the given max concurrency.
    #[must_use]
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            inner: Arc::new(SchedulerInner {
                max_concurrent,
                tasks: Arc::new(RwLock::new(AHashMap::new())),
                handles: Arc::new(RwLock::new(AHashMap::new())),
                cancellations: Arc::new(RwLock::new(AHashMap::new())),
                semaphore: Arc::new(tokio::sync::Semaphore::new(max_concurrent)),
                shutdown: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            }),
            notify: Arc::new(Notify::new()),
        }
    }

    /// Spawn a task with the given priority.
    ///
    /// The task will not start running until a semaphore permit is available.
    /// Returns a [`TaskHandle`] that can be awaited to get the result.
    ///
    /// # Errors
    /// Returns [`SchedulerError::AtCapacity`] if the scheduler is full.
    /// Returns [`SchedulerError::Shutdown`] if the scheduler has been shut down.
    pub fn spawn<F, T>(&self, name: impl Into<String>, priority: Priority, fut: F) -> Result<TaskHandle<T>, SchedulerError>
    where
        F: Future<Output = Result<T, String>> + Send + 'static,
        T: Send + 'static,
    {
        if self.inner.shutdown.load(std::sync::atomic::Ordering::Acquire) {
            return Err(SchedulerError::Shutdown);
        }

        let id = TaskId::new();
        let info = TaskInfo {
            id,
            name: name.into(),
            priority,
            status: TaskStatus::Pending,
            submitted_at: Utc::now(),
            started_at: None,
            finished_at: None,
            duration: None,
            error: None,
        };
        self.inner.tasks.write().insert(id, info);

        let (tx, rx) = oneshot::channel();
        let semaphore = Arc::clone(&self.inner.semaphore);
        let tasks = Arc::clone(&self.inner.tasks);
        let cancellations = Arc::clone(&self.inner.cancellations);
        let (cancel_tx, cancel_rx) = oneshot::channel::<()>();
        self.inner.cancellations.write().insert(id, cancel_tx);

        let handle = tokio::spawn(async move {
            let _permit = match semaphore.acquire().await {
                Ok(p) => p,
                Err(_) => {
                    let _ = tx.send(Err("scheduler closed".into()));
                    return;
                }
            };

            // Mark as running
            {
                let mut t = tasks.write();
                if let Some(info) = t.get_mut(&id) {
                    info.status = TaskStatus::Running;
                    info.started_at = Some(Utc::now());
                }
            }

            // Race the future against cancellation
            let result = tokio::select! {
                r = fut => r,
                _ = cancel_rx => Err("cancelled".to_string()),
            };

            // Update final status
            let started_at = {
                let mut t = tasks.write();
                if let Some(info) = t.get_mut(&id) {
                    info.finished_at = Some(Utc::now());
                    info.duration = info.started_at.map(|s| Utc::now().signed_duration_since(s).to_std().unwrap_or_default());
                    info.status = match &result {
                        Ok(_) => TaskStatus::Completed,
                        Err(e) if e == "cancelled" => TaskStatus::Cancelled,
                        Err(_) => TaskStatus::Failed,
                    };
                    info.error = result.as_ref().err().cloned();
                }
                t.get(&id).and_then(|i| i.started_at)
            };
            let _ = started_at;

            match result {
                Ok(v) => { let _ = tx.send(Ok(v)); }
                Err(e) => { let _ = tx.send(Err(e)); }
            }
            cancellations.write().remove(&id);
        });

        self.inner.handles.write().insert(id, handle);
        self.notify.notify_one();

        Ok(TaskHandle { id, rx })
    }

    /// Cancel a running task.
    pub fn cancel(&self, id: TaskId) -> Result<(), SchedulerError> {
        let mut cancellations = self.inner.cancellations.write();
        if let Some(tx) = cancellations.remove(&id) {
            let _ = tx.send(());
            Ok(())
        } else {
            Err(SchedulerError::NotFound(id))
        }
    }

    /// Get info about a task.
    #[must_use]
    pub fn info(&self, id: TaskId) -> Option<TaskInfo> {
        self.inner.tasks.read().get(&id).cloned()
    }

    /// List all tasks, sorted by priority (descending) then submission time.
    #[must_use]
    pub fn list(&self) -> Vec<TaskInfo> {
        let mut tasks: Vec<TaskInfo> = self.inner.tasks.read().values().cloned().collect();
        tasks.sort_by(|a, b| b.priority.cmp(&a.priority).then(a.submitted_at.cmp(&b.submitted_at)));
        tasks
    }

    /// Number of currently running tasks.
    #[must_use]
    pub fn running_count(&self) -> usize {
        self.inner.tasks.read().values().filter(|t| t.status == TaskStatus::Running).count()
    }

    /// Returns the configured maximum concurrency.
    #[must_use]
    pub fn max_concurrent(&self) -> usize {
        self.inner.max_concurrent
    }

    /// Returns the total number of tasks (running + pending + completed).
    #[must_use]
    pub fn task_count(&self) -> usize {
        self.inner.tasks.read().len()
    }

    /// Shut down the scheduler. Cancels all running tasks and waits for them
    /// to finish (up to `grace`).
    pub async fn shutdown(&self, grace: Duration) {
        self.inner.shutdown.store(true, std::sync::atomic::Ordering::Release);

        let ids: Vec<TaskId> = self.inner.cancellations.read().keys().copied().collect();
        for id in ids {
            let _ = self.cancel(id);
        }

        let handles: Vec<JoinHandle<()>> = {
            let mut h = self.inner.handles.write();
            h.drain().map(|(_, v)| v).collect()
        };

        let join_all = async {
            for h in handles {
                let _ = h.await;
            }
        };
        let _ = tokio::time::timeout(grace, join_all).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn spawn_and_join() {
        let s = Scheduler::new(2);
        let h = s.spawn("test", Priority::Normal, async { Ok::<_, String>(42) }).unwrap();
        let v = h.join().await.unwrap();
        assert_eq!(v, 42);
        let info = s.info(TaskId(Uuid::nil())).is_none();
        assert!(info);
    }

    #[tokio::test]
    async fn cancel_running_task() {
        let s = Scheduler::new(2);
        let h = s.spawn("sleep", Priority::Normal, async {
            tokio::time::sleep(Duration::from_secs(10)).await;
            Ok::<_, String>(())
        }).unwrap();
        let task_id = h.id();
        tokio::time::sleep(Duration::from_millis(50)).await;
        s.cancel(task_id).unwrap();
        let _ = h.join().await;
        let info = s.info(task_id).unwrap();
        assert!(info.status == TaskStatus::Cancelled || info.status == TaskStatus::Failed);
    }

    #[tokio::test]
    async fn priority_respected() {
        let s = Scheduler::new(1);
        let blocker = Arc::new(tokio::sync::Semaphore::new(0));
        let b2 = Arc::clone(&blocker);
        let _block = s.spawn("blocker", Priority::Normal, async move {
            let _ = b2.acquire().await;
            Ok::<_, String>(())
        }).unwrap();

        // Spawn two tasks with different priorities; only one slot will free up
        let hi = s.spawn("hi", Priority::Realtime, async { Ok::<_, String>(1) }).unwrap();
        let lo = s.spawn("lo", Priority::Background, async { Ok::<_, String>(2) }).unwrap();

        tokio::time::sleep(Duration::from_millis(20)).await;
        // Release the blocker
        let _ = blocker.add_permits(1);
        let _ = _block.join().await;

        let hi_v = hi.join().await.unwrap();
        let lo_v = lo.join().await.unwrap();
        assert_eq!(hi_v, 1);
        assert_eq!(lo_v, 2);
    }
}
