# LiteLLM Integration Fixes

`ceil-dlp` ships a LiteLLM callback module (`ceil_dlp.ceil_dlp_callback`) that
wraps the base handler and fixes several integration bugs found while running
`ceil-dlp` as a LiteLLM proxy callback. These fixes live in the fork so that a
LiteLLM container built from source behaves correctly without hot-patching the
installed package.

Environment this was validated against: LiteLLM `1.63.11`
(`ghcr.io/berriai/litellm:main-latest`), `ceil-dlp` `1.3.2`, Python 3.13.

## Callback registration

LiteLLM's `get_instance_fn` resolves callback names as **file paths relative to
the config file's directory** when `config_file_path` is set. So the config must
reference a local wrapper file, not a dotted package path:

```yaml
litellm_settings:
  callbacks:
    - ceil_dlp_callback.proxy_handler_instance
```

Create a `ceil_dlp_callback.py` in the same directory as your LiteLLM config
(the `ceil-dlp install` CLI already generates one):

```python
from ceil_dlp.ceil_dlp_callback import proxy_handler_instance  # noqa: F401
```

The module reads its policy config from the `CEIL_DLP_CONFIG_PATH` environment
variable (falling back to default config):

```bash
export CEIL_DLP_CONFIG_PATH=/path/to/ceil-dlp.yaml
```

## Fixes

### 1. Streaming rejection crash

The base handler returns rejection **strings**. With `stream: true`, LiteLLM
catches the rejection and builds a `CustomStreamWrapper` with a `None` logging
object (`AttributeError: 'NoneType' object has no attribute 'model_call_details'`),
or stuffs the error into `message.content`.

The wrapper converts string rejections to `HTTPException(400)` in
`async_pre_call_hook`, flowing through the normal proxy exception path and
producing a proper OpenAI-style error JSON:

```json
{"error":{"message":"[ceil-dlp] Request blocked: ...","type":"invalid_request_error","code":"400"}}
```

### 2. `reasoning_content` not reversed (whistledown, non-streaming)

Non-streaming whistledown reversal restored `message.content` but left pseudonyms
(e.g. `DATABASE_URL_1`) in `message.reasoning_content` (DeepSeek chain-of-thought).
The wrapper reverses `reasoning_content` in `async_post_call_success_hook` before
delegating to the inner handler.

### 3. Per-message masking/whistledown fails with multiple messages

The base `async_pre_call_hook` joins all message text (e.g. system + user prompt),
runs detection/redaction on the joined blob, then does
`content.replace(joined_text, redacted_text)` per message. The joined text never
matches any single message, so masking/whistledown silently no-oped for
multi-message requests (e.g. agents with system prompts).

The wrapper's `_apply_per_message_transforms` re-runs detection +
`redact_text`/`whistledown_transform_text` per message whenever the inner handler
returns unmodified messages (detected via `_extract_text_from_messages`
comparison), and records `_whistledown_request_id` for post-call reversal.

### 4. Streaming whistledown reversal

LiteLLM v1.63.x discards callback return values in
`async_post_call_streaming_hook` (monitoring only), so chunks cannot be modified
through the hook. The wrapper replaces `proxy_server.async_data_generator` with a
copy that:

- reads `request_data["_whistledown_request_id"]` (available in `self.data`),
- reverses `delta.content` and `delta.reasoning_content` in each chunk via a
  **per-field sliding buffer** (`f"{rid}:content"` vs `f"{rid}:reasoning"`),
- flushes residual buffered text as a final chunk before `[DONE]`,
- preserves the original serialization and error handling.

**This must be applied lazily (first request), not at import time.** The callback
module is imported during config loading, which happens *before*
`def async_data_generator` executes — an eager patch would be overwritten. A
module flag (`_GEN_PATCHED`) plus an attribute on the generator itself prevent
re-wrapping.

Sliding buffer logic (`reverse_streaming_text`): pseudonyms split across chunks
(`DAT`+`ABASE`+`_URL`+`_`+`1`) are accumulated; only suffixes that are a prefix of
some pseudonym are held back, everything else emits immediately. Falls back to the
union of all active reverse-cache entries if the request-id lookup misses.

### 5. Spend-log hygiene

LiteLLM snapshots the raw request into `data["proxy_server_request"] =
{"body": copy.copy(data)}` (`litellm_pre_call_utils.py:495`) **before** the
pre-call hook transforms messages, and persists it to spend-log metadata when
`store_prompts_in_spend_logs` is on. This stored the *original* secrets the model
never saw.

The wrapper rewrites `proxy_server_request.body["messages"]` to the transformed
messages **whenever they differ from the snapshot** — unconditionally, because the
inner handler may have transformed single-message requests itself. Re-read the
final messages *after* the per-message transform block (a reference captured
before `result["messages"] = fixed` is stale and silently skips the rewrite).

### 6. Blocked requests persisted the raw rejected secret

When a request is blocked, the base handler returns a rejection string *before*
any transform, and `_ProxyDBLogger.async_post_call_failure_hook` writes a failure
spend log holding the **raw** rejected request. The wrapper redacts all detected
PII in the request (text content) before raising, so the failure row stores
`[REDACTED_*]` instead of the rejected secret.

### 7. Empty request logs + response column

`_get_messages_for_spend_logs_payload` is a stub returning `"{}"`, and the
`response` column reads an empty payload. The wrapper lazily patches
`get_logging_payload` in **both** `spend_tracking_utils` and `proxy_server`
(proxy_server imports it by name, so patching only the source module is
invisible) to fill:

- `messages` ← `kwargs["messages"]` (already pseudonymized by the pre-call
  transform),
- `response` ← `response_obj` **re-pseudonymized** via `_re_pseudonymize_response`
  (the inverse of the reverse-cache mapping), so the DB stores pseudonyms, not the
  restored originals the end user received.

Both are gated on `_should_store_prompts_and_responses_in_spend_logs()`.

## Verification

Test matrix (policy actions from `ceil-dlp.yaml`; `mode: enforce`,
`ner_strength: 2`):

| Mode (policy action) | Streaming | Non-streaming | opencode multi-msg stream |
|---|---|---|---|
| `block` (jwt_token, pem_key, ssn, credit_card) | HTTP 400 reject | HTTP 400 reject | HTTP 400 reject |
| `whistledown` (api_key, database_url, cloud_credential) | original restored | original restored | original restored |
| `mask` (email, phone) | `[REDACTED_*]` | `[REDACTED_*]` | `[REDACTED_*]` |

Spend-log verification (must use a **virtual key** — with the `master_key`,
deepseek requests never write spend logs because `standard_logging_object is
None` and the cost callback raises before `update_database`):

```bash
VK=$(curl -s http://localhost:4000/key/generate -H "Authorization: Bearer sk-master" \
  -H "Content-Type: application/json" -d '{"models":["deepseek/deepseek-v4-flash"],"max_budget":1}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin).get('key'))")

# multi-message (per-message transform path)
opencode run -m litellm/deepseek/deepseek-v4-flash --format json \
  "Repeat back: DB_CONN=postgres://root:SuperSecretP@ssw0rd!@10.0.0.15:5432/production"

# single-message (inner handler transforms; body must still be synced)
curl -s http://localhost:4000/v1/chat/completions -H "Authorization: Bearer $VK" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek/deepseek-v4-flash","messages":[{"role":"user","content":"db=postgres://root:LeakCheck@db.host:5432/prod"}],"stream":true}' >/dev/null

# blocked (failure spend log must store [REDACTED_*], not the raw JWT)
curl -s http://localhost:4000/v1/chat/completions -H "Authorization: Bearer $VK" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek/deepseek-v4-flash","messages":[{"role":"user","content":"jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"}],"stream":true}' >/dev/null

docker exec <db-container> psql -U llmproxy -d litellm -c \
  "SELECT \"startTime\", COALESCE(metadata->>'status','ok') AS st, metadata::text LIKE '%SuperSecret%' AS orig, metadata::text LIKE '%DATABASE_URL_%' AS pseudo, messages::text LIKE '%REDACTED%' AS redacted FROM \"LiteLLM_SpendLogs\" ORDER BY \"startTime\" DESC LIMIT 3;"
```

Expected: all rows `orig = f` (success rows `pseudo = t`, blocked rows
`redacted = t`).

## Deployment notes

- **Build from source, not PyPI.** Install the fork directly so the fixes are
  baked in. With Docker Compose, use a named build context and install from it:

  ```dockerfile
  # syntax=docker/dockerfile:1
  COPY --from=ceildlp . /opt/ceil-dlp
  RUN python -m pip install --no-cache-dir /opt/ceil-dlp
  ```

  ```yaml
  services:
    litellm:
      build:
        context: .
        additional_contexts:
          ceildlp: /home/you/github/ceil-dlp
  ```

  For iterative development, mount the source repo into the container and prepend
  it to `PYTHONPATH` so it shadows the installed package:

  ```yaml
  volumes:
    - /home/you/github/ceil-dlp:/app/ceil-dlp-src
  environment:
    - PYTHONPATH=/app/ceil-dlp-src
  ```

- **LiteLLM version drift.** The `async_data_generator` reimplementation mirrors
  v1.63.11's body (serialization flags, `[DONE]`, error handling). Re-diff against
  `proxy_server.py` after upgrading LiteLLM.
- **`get_logging_payload` must be patched in both modules.** `proxy_server.py`
  imports it by name, so patching only `spend_tracking_utils` is invisible.
- **Blocked-request redaction is best-effort** (text content only); images/PDFs in
  a rejected request are stored as-is.