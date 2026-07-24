# qbitos-safety

Content filter, prompt-injection detector, output validator, rate limiter, audit log.

> Part of the [QbitOS](../README.md) project · [Architecture](../docs/ARCHITECTURE.md) · [API](../docs/API.md) · [Deployment](../docs/DEPLOYMENT.md)

## Overview

Safety infrastructure for the cognitive OS.

## Modules

- **Content filter**: Detects harmful, abusive, or inappropriate content
- **Injection detector**: Detects prompt-injection attacks
- **Output validator**: Validates agent output against schemas and policies
- **Rate limiter**: Token bucket per principal
- **Audit logger**: Records all authenticated requests for compliance

## Usage

```rust
use qbitos_safety::{ContentFilter, InjectionDetector, AuditLogger};
let filter = ContentFilter::default();
if filter.is_harmful(input) {
    return Err(ServiceError::Forbidden("harmful content".into()));
}
```

## Testing

~10 tests.
