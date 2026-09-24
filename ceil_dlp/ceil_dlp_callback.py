"""Callback module for LiteLLM integration.

This module exports a handler instance that LiteLLM can import.

Fixes applied on top of ceil-dlp's base handler:

1. Streaming-safe rejection: convert string rejections to HTTPException(400).
   The base handler returns rejection strings, and LiteLLM's streaming path
   cannot serialize them (it builds a CustomStreamWrapper with a None logging
   object and crashes, or stuffs the error into message.content).
2. Reasoning content reversal for whistledown (non-streaming responses).
3. Per-message masking/whistledown fix for multi-message requests.
4. Streaming whistledown reversal: lazily patches proxy_server's
   async_data_generator (after full startup) to reverse pseudonyms in
   streaming chunks using a per-field sliding buffer.
5. Spend-log hygiene: rewrite proxy_server_request.body to pseudonymized
   messages whenever they differ from the snapshot; wrap get_logging_payload
   to populate messages/response columns with pseudonyms. Redact all detected
   PII in the request before raising on a blocked request, so the failure
   spend log never stores the original secrets.

The handler instance reads its policy config from the CEIL_DLP_CONFIG_PATH
environment variable when set, otherwise it falls back to defaults.
"""

import copy
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Literal

from fastapi import HTTPException
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.proxy_server import UserAPIKeyAuth

from ceil_dlp.middleware import CeilDLPHandler, create_handler
from ceil_dlp.redaction import redact_text
from ceil_dlp.whistledown import whistledown_transform_text

# Create handler instance for LiteLLM
# Users can override by setting CEIL_DLP_CONFIG_PATH environment variable
_config_path = os.getenv("CEIL_DLP_CONFIG_PATH")
if _config_path and Path(_config_path).is_file():
    _inner = create_handler(config_path=_config_path)
else:
    _inner = CeilDLPHandler()


class _PatchedCeilDLPHandler(CustomLogger):

    def __init__(self, inner: CeilDLPHandler) -> None:
        super().__init__()
        self._inner = inner
        self._lock = threading.Lock()
        self._stream_buffers: dict[str, str] = {}

    def _get_replacements(self, request_id: str | None) -> dict[str, str]:
        """Replacements for a request; falls back to union of all cached
        pseudonym mappings to tolerate request-id key mismatches."""
        cache = self._inner.whistledown_cache
        if request_id:
            reps = cache.reverse_cache.get(request_id)
            if reps:
                return dict(reps)
        reps: dict[str, str] = {}
        for v in cache.reverse_cache.values():
            reps.update(v)
        return reps

    def _re_pseudonymize_response(self, request_id: str | None, response_obj: Any) -> str | None:
        """Serialize a response object back to pseudonymized JSON (inverse of
        the reversal) so the spend-log response column stores pseudonyms, not
        the original secret that the end user received."""
        try:
            reps = self._get_replacements(request_id)
            if not reps:
                return None
            inverse = {orig: pseudo for pseudo, orig in reps.items()}
            tokens = sorted(inverse.keys(), key=len, reverse=True)
            pattern = re.compile("|".join(re.escape(t) for t in tokens))

            def _sub(value: Any) -> Any:
                if isinstance(value, str):
                    return pattern.sub(lambda m: inverse[m.group()], value)
                return value

            if isinstance(response_obj, dict):
                dumped = copy.deepcopy(response_obj)
            elif hasattr(response_obj, "model_dump"):
                dumped = response_obj.model_dump()
            else:
                return None

            for choice in dumped.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                msg = choice.get("message") or choice.get("delta")
                if not isinstance(msg, dict):
                    continue
                for field in ("content", "reasoning_content"):
                    if msg.get(field):
                        msg[field] = _sub(msg[field])
            return json.dumps(dumped, default=str)
        except Exception:
            return None

    def reverse_streaming_text(self, request_id: str | None, field: str, chunk_text: str) -> str:
        """Reverse whistledown pseudonyms in a streaming chunk. Buffers any
        suffix that could be a prefix of a pseudonym (split across chunks)."""
        replacements = self._get_replacements(request_id)
        if not replacements:
            return chunk_text

        key = f"{request_id or '_fallback'}:{field}"
        buf = self._stream_buffers.get(key, "") + chunk_text
        tokens = sorted(replacements.keys(), key=len, reverse=True)
        pattern = re.compile("|".join(re.escape(t) for t in tokens))
        result = pattern.sub(lambda m: replacements[m.group()], buf)

        keep_len = 0
        upper = min(max((len(t) for t in tokens), default=1), len(result))
        for ln in range(upper, 0, -1):
            if any(t.startswith(result[-ln:]) for t in tokens):
                keep_len = ln
                break
        self._stream_buffers[key] = result[-keep_len:] if keep_len else ""
        return result[:-keep_len] if keep_len else result

    def flush_streaming_buffers(self, request_id: str | None) -> str:
        """Flush and reverse any residual buffered text at stream end."""
        parts: list[str] = []
        rid = request_id or "_fallback"
        for field in (":content", ":reasoning"):
            key = rid + field
            buf = self._stream_buffers.pop(key, "")
            if not buf:
                continue
            replacements = self._get_replacements(request_id)
            if replacements:
                tokens = sorted(replacements.keys(), key=len, reverse=True)
                pattern = re.compile("|".join(re.escape(t) for t in tokens))
                buf = pattern.sub(lambda m: replacements[m.group()], buf)
            parts.append(buf)
        return "".join(parts)

    def _reverse_chunk_inplace(self, chunk: Any, request_id: str) -> None:
        try:
            for choice in chunk.choices:
                d = getattr(choice, "delta", None)
                if d is None:
                    continue
                if getattr(d, "content", None):
                    d.content = self.reverse_streaming_text(request_id, "content", d.content)
                rc = getattr(d, "reasoning_content", None)
                if rc:
                    d.reasoning_content = self.reverse_streaming_text(
                        request_id, "reasoning", rc
                    )
        except Exception:
            pass

    def _apply_per_message_transforms(
        self, messages: list[Any], model: str, data: dict[str, Any]
    ) -> tuple[list[Any], str | None]:
        config = self._inner.config
        if config.mode != "enforce":
            return messages, None

        mask_types: dict[str, Any] = {}
        whistledown_types: dict[str, Any] = {}
        for pii_type, policy in config.policies.items():
            if not policy.enabled:
                continue
            if not self._inner._should_apply_policy(policy, model):
                continue
            if policy.action == "mask":
                mask_types[pii_type] = policy
            elif policy.action == "whistledown":
                whistledown_types[pii_type] = policy

        if not mask_types and not whistledown_types:
            return messages, None

        modified = False
        result = copy.deepcopy(messages)
        whistledown_request_id = None
        any_whistledown_applied = False

        for msg in result:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, str) or not content:
                continue

            detections = self._inner._process_pii_detection(
                [{"role": msg.get("role", "user"), "content": content}], model
            )
            masked_dets = detections[2]
            whistledown_dets = detections[3]

            subset_mask = {k: v for k, v in masked_dets.items() if k in mask_types}
            subset_whistle = {k: v for k, v in whistledown_dets.items() if k in whistledown_types}

            if subset_mask:
                redacted, _ = redact_text(
                    content, detections=subset_mask, ner_strength=config.ner_strength
                )
                if redacted != content:
                    msg["content"] = redacted
                    modified = True

            if subset_whistle:
                if whistledown_request_id is None:
                    whistledown_request_id = data.get("litellm_call_id", "unknown")
                transformed, _ = whistledown_transform_text(
                    msg["content"],
                    detections=subset_whistle,
                    cache=self._inner.whistledown_cache,
                    request_id=whistledown_request_id,
                )
                if transformed != msg["content"]:
                    msg["content"] = transformed
                    modified = True
                    any_whistledown_applied = True

        if not modified:
            return messages, None
        return result, (whistledown_request_id if any_whistledown_applied else None)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth | None,
        cache: DualCache | None,
        data: dict[str, Any],
        call_type: Literal[
            "completion", "text_completion", "embeddings",
            "image_generation", "moderation", "audio_transcription",
        ],
    ) -> dict[str, Any] | str | Exception | None:
        _ensure_data_generator_patched()
        _ensure_spend_log_patched()
        result = await self._inner.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict, cache=cache,
            data=data, call_type=call_type,
        )
        if isinstance(result, str):
            # Blocked request: the failure spend log stores the original request,
            # so redact all detected PII before raising to keep the DB clean.
            try:
                model = data.get("model", "")
                messages = data.get("messages", [])
                if messages:
                    dets, _, _, _, text, _, _ = self._inner._process_pii_detection(
                        messages, model
                    )
                    if dets and text:
                        redacted, _ = redact_text(
                            text,
                            detections=dets,
                            ner_strength=self._inner.config.ner_strength,
                        )
                        safe_msgs = self._inner._replace_text_in_messages(
                            messages, text, redacted
                        )
                        data["messages"] = safe_msgs
                        _psr = data.get("proxy_server_request") or {}
                        _psr_body = _psr.get("body")
                        if isinstance(_psr_body, dict) and "messages" in _psr_body:
                            _psr_body["messages"] = safe_msgs
                            data["proxy_server_request"] = _psr
            except Exception:
                pass
            exc = HTTPException(status_code=400, detail=result)
            exc.type = "invalid_request_error"
            exc.code = "content_policy_violation"
            raise exc

        if isinstance(result, dict):
            model = result.get("model", data.get("model", ""))
            original_messages = data.get("messages", [])
            result_messages = result.get("messages", [])
            if original_messages and result_messages:
                orig_text = self._inner._extract_text_from_messages(original_messages)
                res_text = self._inner._extract_text_from_messages(result_messages)
                if orig_text == res_text:
                    fixed, whistledown_rid = self._apply_per_message_transforms(
                        result_messages, model, data
                    )
                    if fixed is not result_messages:
                        result["messages"] = fixed
                        if whistledown_rid:
                            result["_whistledown_request_id"] = whistledown_rid
                # Always keep the pre-transform snapshot (proxy_server_request.body)
                # in sync with the messages actually sent, so spend-log metadata
                # stores pseudonyms — not originals — even when the inner ceil-dlp
                # handler did the transform itself (single-message requests).
                final_messages = result.get("messages", [])
                _psr = data.get("proxy_server_request") or {}
                _psr_body = _psr.get("body")
                if isinstance(_psr_body, dict) and "messages" in _psr_body:
                    body_text = self._inner._extract_text_from_messages(_psr_body["messages"])
                    final_text = self._inner._extract_text_from_messages(final_messages)
                    if body_text != final_text:
                        _psr_body["messages"] = copy.deepcopy(final_messages)
                        result["proxy_server_request"] = _psr
        return result

    async def async_post_call_success_hook(
        self,
        data: dict[str, Any],
        user_api_key_dict: UserAPIKeyAuth | None,
        response: Any,
    ) -> Any | None:
        rid = data.get("_whistledown_request_id")
        if rid and hasattr(response, "choices"):
            for choice in response.choices:
                if not hasattr(choice, "message"):
                    continue
                msg = choice.message
                if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                    msg.reasoning_content = self._inner.whistledown_cache.reverse_transform(
                        rid, msg.reasoning_content
                    )
        return await self._inner.async_post_call_success_hook(
            data=data, user_api_key_dict=user_api_key_dict, response=response,
        )


_instance = _PatchedCeilDLPHandler(_inner)
proxy_handler_instance = _instance  # noqa: F401  # type: ignore[unused-import]


# --- Streaming whistledown reversal ---------------------------------------
# LiteLLM v1.63.x discards callback return values in async_post_call_streaming_hook
# (proxy/utils.py always returns the original chunk). We instead replace
# proxy_server.async_data_generator with a copy that reverses pseudonyms in each
# chunk's delta before serialization.
#
# IMPORTANT: this MUST be applied lazily (first request), NOT at import time.
# ceil_dlp_callback.py is imported during config loading (proxy_server.py ~line 1796),
# which happens BEFORE `def async_data_generator` (~line 3036) executes — an eager
# patch would be overwritten by the later def. select_data_generator resolves
# async_data_generator via module-global lookup at call time, so a post-startup
# patch takes effect.

_GEN_PATCHED = False


def _ensure_data_generator_patched() -> None:
    global _GEN_PATCHED
    if _GEN_PATCHED:
        return
    try:
        import json
        import traceback

        import litellm
        from fastapi import HTTPException as _FAHTTPException
        from litellm.proxy import proxy_server as _ps
        from litellm.proxy._types import ProxyException as _PE
        from pydantic import BaseModel as _BaseModel

        if getattr(_ps.async_data_generator, "_ceil_dlp_reversing", False):
            _GEN_PATCHED = True
            return

        # Tolerate LiteLLM signature drift across versions: newer releases
        # pass extra params (request=, responses_stream_errors=, ...).
        async def _reversing_data_generator(
            response, user_api_key_dict, request_data, *args, **kwargs
        ):
            rid = (
                request_data.get("_whistledown_request_id")
                if isinstance(request_data, dict)
                else None
            )
            if rid is None and isinstance(request_data, dict):
                rid = request_data.get("litellm_call_id")
            try:
                async for chunk in response:
                    if rid and hasattr(chunk, "choices"):
                        _instance._reverse_chunk_inplace(chunk, rid)

                    chunk = await _ps.proxy_logging_obj.async_post_call_streaming_hook(
                        user_api_key_dict=user_api_key_dict, response=chunk
                    )

                    if isinstance(chunk, _BaseModel):
                        chunk = chunk.model_dump_json(exclude_none=True, exclude_unset=True)

                    try:
                        yield f"data: {chunk}\n\n"
                    except Exception as e:
                        yield f"data: {str(e)}\n\n"

                # Flush any residual buffered pseudonym text before [DONE]
                if rid:
                    leftover = _instance.flush_streaming_buffers(rid)
                    if leftover:
                        from litellm.types.utils import Delta, ModelResponse, StreamingChoices

                        final = ModelResponse(
                            id="ceil-dlp-flush",
                            choices=[StreamingChoices(delta=Delta(content=leftover))],
                        )
                        yield f"data: {final.model_dump_json(exclude_none=True, exclude_unset=True)}\n\n"

                yield "data: [DONE]\n\n"
            except Exception as e:
                _ps.verbose_proxy_logger.exception(
                    "litellm.proxy.proxy_server.async_data_generator(): Exception occured - {}".format(
                        str(e)
                    )
                )
                await _ps.proxy_logging_obj.post_call_failure_hook(
                    user_api_key_dict=user_api_key_dict,
                    original_exception=e,
                    request_data=request_data,
                )
                if isinstance(e, _FAHTTPException):
                    raise e
                error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
                proxy_exception = _PE(
                    message=getattr(e, "message", error_msg),
                    type=getattr(e, "type", "None"),
                    param=getattr(e, "param", "None"),
                    code=getattr(e, "status_code", 500),
                )
                yield f"data: {json.dumps({'error': proxy_exception.to_dict()})}\n\n"

        _reversing_data_generator._ceil_dlp_reversing = True  # type: ignore[attr-defined]
        _ps.async_data_generator = _reversing_data_generator
        _GEN_PATCHED = True
    except Exception:
        pass


# --- Spend-log prompt/response population -----------------------------------
# LiteLLM 1.63.x's _get_messages_for_spend_logs_payload is a stub returning "{}"
# and the response column reads standard_logging_payload["response"], which is
# empty for these deepseek calls. We wrap get_logging_payload to fill both
# columns (messages = already-pseudonymized request, response = re-pseudonymized
# back to pseudonyms so the DB never stores the restored original secrets).

_SPEND_PATCHED = False


def _ensure_spend_log_patched() -> None:
    global _SPEND_PATCHED
    if _SPEND_PATCHED:
        return
    try:
        import json as _json

        from litellm.proxy import proxy_server as _ps
        from litellm.proxy.spend_tracking import spend_tracking_utils as _stu
        from litellm.proxy.spend_tracking.spend_tracking_utils import (
            _should_store_prompts_and_responses_in_spend_logs,
        )

        if getattr(_stu.get_logging_payload, "_ceil_dlp_spend", False):
            _SPEND_PATCHED = True
            return

        _orig = _stu.get_logging_payload

        def _wrapped_get_logging_payload(kwargs, response_obj, start_time, end_time):
            payload = _orig(kwargs, response_obj, start_time, end_time)
            try:
                if not _should_store_prompts_and_responses_in_spend_logs():
                    return payload
                rid = (kwargs or {}).get("_whistledown_request_id")
                msgs = (kwargs or {}).get("messages")
                if msgs is not None:
                    payload["messages"] = _json.dumps(msgs, default=str)
                resp = _instance._re_pseudonymize_response(rid, response_obj)
                if resp:
                    payload["response"] = resp
            except Exception:
                pass
            return payload

        _wrapped_get_logging_payload._ceil_dlp_spend = True  # type: ignore[attr-defined]
        _stu.get_logging_payload = _wrapped_get_logging_payload
        _ps.get_logging_payload = _wrapped_get_logging_payload
        _SPEND_PATCHED = True
    except Exception:
        pass