# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Config-driven custom regex patterns (`custom_patterns`).** Regex patterns can
  now be added via YAML config instead of being hard-coded in `patterns.py`. The
  field maps a PII type name to a list of regex patterns: patterns for an existing
  type (e.g. `api_key`) are merged with the built-in ones, and new keys create brand
  new PII types (which then need a policy). Threaded through text/image/PDF detection
  and all redaction paths, with the analyzer cache keyed on the pattern set so
  per-config analyzers don't collide.

### Fixed

- **Per-message masking/whistledown silently failed with multiple messages.**
  The base `async_pre_call_hook` joins all message text, runs detection/redaction
  on the joined blob, then does `content.replace(joined, redacted)` per message —
  the joined text never matches any single message, so multi-message requests
  reached the provider unmodified. The callback wrapper now re-runs detection +
  `redact_text`/`whistledown_transform_text` per message whenever the inner handler
  returns unmodified messages.
- **`reasoning_content` was not reversed after whistledown.** Non-streaming
  responses had `message.content` restored but `message.reasoning_content`
  (e.g. DeepSeek chain-of-thought) still contained pseudonyms. The post-call hook
  now reverses it before delegating to the inner handler.
- **Streaming responses showed whistledown pseudonyms instead of original values.**
  LiteLLM v1.63.x discards callback return values in
  `async_post_call_streaming_hook` (monitoring only), so the wrapper replaces
  `proxy_server.async_data_generator` with a copy that reverses pseudonyms in each
  chunk's `delta.content`/`delta.reasoning_content` using a per-field sliding
  buffer, then flushes residual buffered text before `[DONE]`. Applied lazily on
  the first request, after full proxy startup.
- **Streaming rejection crashed when the request was blocked.** A string rejection
  with `stream: true` made LiteLLM build a `CustomStreamWrapper` with a `None`
  logging object (`AttributeError`). The wrapper now converts string rejections to
  `HTTPException(400)` before returning, so blocked streaming requests flow through
  the normal proxy exception path with a proper OpenAI-style error body.
- **Spend logs leaked original secrets and stored empty request/response columns.**
  LiteLLM snapshots the raw request into `proxy_server_request.body` *before* the
  pre-call hook transforms messages, so spend-log metadata persisted the *original*
  secrets the model never saw. The wrapper now keeps that snapshot in sync with the
  transformed messages whenever they differ (covering both the per-message path and
  single-message requests the inner handler transforms itself).
- **Blocked requests persisted the raw rejected secret to the failure spend log.**
  The wrapper now redacts all detected PII in the request before raising, so the
  failure row stores `[REDACTED_*]` instead of the original secret.
- **Empty request logs in the UI.** `_get_messages_for_spend_logs_payload` is a
  stub returning `"{}"` and the response column reads an empty payload. The wrapper
  lazily patches `get_logging_payload` (in both `spend_tracking_utils` and
  `proxy_server`) to populate `messages` (already-pseudonymized request) and
  `response` (re-pseudonymized via the inverse of the reverse-cache mapping, so the
  DB never stores restored original secrets).

See [docs/litellm_integration_fixes.md](docs/litellm_integration_fixes.md) for the
full write-up, verification matrix, and deployment notes.