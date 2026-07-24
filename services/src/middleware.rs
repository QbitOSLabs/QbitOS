//! Middleware utilities.

use axum::extract::Request;
use axum::http::{HeaderName, HeaderValue};
use axum::middleware::Next;
use axum::response::Response;
use std::sync::atomic::{AtomicU64, Ordering};
use uuid::Uuid;

static REQ_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Header name for the request ID propagated through the call chain.
pub const REQUEST_ID_HEADER: HeaderName = HeaderName::from_static("x-request-id");

/// Generate a request ID and attach it as a response header.
pub async fn request_id_middleware(mut req: Request, next: Next) -> Response {
    let id = req
        .headers()
        .get(&REQUEST_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(String::from)
        .unwrap_or_else(|| {
            format!(
                "req-{}-{}",
                REQ_COUNTER.fetch_add(1, Ordering::Relaxed),
                Uuid::new_v4().to_string()
            )
        });
    let _span = tracing::info_span!("http_request", request_id = %id);
    if let Ok(hv) = HeaderValue::from_str(&id) {
        req.headers_mut().insert(REQUEST_ID_HEADER, hv.clone());
    }
    let mut resp = next.run(req).await;
    if let Ok(hv) = HeaderValue::from_str(&id) {
        resp.headers_mut().insert(REQUEST_ID_HEADER, hv);
    }
    resp
}
