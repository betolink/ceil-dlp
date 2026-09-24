"""Custom regex patterns for API keys and other non-standard PII.

Patterns are based on gitleaks default configuration with improvements for accuracy.
Reference: https://github.com/gitleaks/gitleaks

These patterns are used to create Presidio PatternRecognizers for detection.
Note: Validators are not supported by Presidio, so patterns with validators
have been simplified to regex-only patterns.
"""

from __future__ import annotations

import math
import re
from typing import Literal

# PatternMatch is a tuple of (matched_text, start_pos, end_pos)
PatternMatch = tuple[str, int, int]

PatternType = Literal[
    "api_key",
    "aws_credential",
    "pem_key",
    "jwt_token",
    "database_url",
    "cloud_credential",
]


# Patterns based on gitleaks default configuration
# Reference: https://github.com/gitleaks/gitleaks/blob/master/config/gitleaks.toml
PATTERNS: dict[PatternType, list[str]] = {
    "api_key": [
        # Prefix-agnostic modern keys: sk-/pk-/rk- followed by a char and
        # 19+ of [A-Za-z0-9_-]. Covers sk-proj-, sk-svcacct-, sk-FAKE-…
        r"\b(?:sk|pk|rk)-[A-Za-z0-9][A-Za-z0-9_-]{19,}\b",
        # Anthropic API keys (sk-ant-api03- prefix, 95+ chars)
        r"\bsk-ant-api03-[a-zA-Z0-9\-_]{95,}\b",
        # GitHub tokens (ghp_, gho_, ghu_, ghs_, ghr_ prefixes, 36+ chars)
        r"\bgh[opurs]_[A-Za-z0-9]{36,}\b",
        # GitLab Personal Access Token (glpat- prefix)
        r"\bglpat-[a-zA-Z0-9\-_]{20,}\b",
        # Stripe keys (sk_live_, sk_test_, rk_live_, rk_test_ prefixes)
        r"\b(?:sk|rk)_(?:live|test)_[a-zA-Z0-9]{24,}\b",
        # Slack tokens (xoxb-, xoxa-, xoxp-, xoxr-, xoxs- prefixes)
        r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
        r"\bxox[bapes]-\d+-[a-zA-Z0-9-]{27,}\b",
        # Authorization: Bearer <token> header
        r"(?i)authorization\s*[:=]\s*[\"']?bearer\s+[A-Za-z0-9._+/=-]{20,}",
        # Google API keys (AIza prefix, exactly 39 chars)
        r"\bAIza[0-9A-Za-z_-]{35}\b",
        # OCR-tolerant Google API keys: allow character misreads and count variations
        r"\bAIza[0-9A-Za-z_\-/|]{33,37}\b",
        # Azure Storage Account keys (base64, 88 chars)
        r"\b[A-Za-z0-9+/]{86}==\b",
        # OCR-tolerant Azure Storage Account keys: allow character misreads and count variations
        r"\b[A-Za-z0-9+/|]{84,90}==\b",
        # Heroku API keys (UUID format)
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        # OCR-tolerant Heroku API keys: allow hex misreads (0→O, 1→I) and slight count variations
        # Note: UUID structure is preserved, but allow character misreads within each segment
        r"\b[0-9a-f/|]{7,9}-[0-9a-f/|]{3,5}-[0-9a-f/|]{3,5}-[0-9a-f/|]{3,5}-[0-9a-f/|]{11,13}\b",
        # Mailgun API keys (key- prefix, 32 hex chars)
        r"\bkey-[0-9a-f]{32}\b",
        # OCR-tolerant Mailgun API keys: allow hex misreads (0→O, 1→I, etc.) and count variations
        r"\bkey-[0-9a-f/|]{30,34}\b",
        # SendGrid API keys (SG. prefix, base64-like, 22+ chars)
        r"\bSG\.[A-Za-z0-9_-]{22,}\b",
        # Twilio API keys (SK prefix, 32 hex chars)
        r"\bSK[0-9a-f]{32}\b",
        # OCR-tolerant Twilio API keys: allow hex misreads and count variations
        r"\bSK[0-9a-f/|]{30,34}\b",
        # Square API keys (sq0atp- or sq0csp- prefix)
        r"\bsq0[ac]sp-[0-9A-Za-z\-_]{32,}\b",
        # Square OAuth secrets (sq0csp- prefix)
        r"\bsq0csp-[0-9A-Za-z\-_]{43,}\b",
        # PayPal client ID/secret (base64-like)
        r"\b(?:access_token|client_id|client_secret)\s*[:=]\s*[A-Za-z0-9_-]{20,}\b",
        # Shopify API keys (shpat_ or shpca_ prefix)
        r"\bsh(?:pat|pca)_[a-zA-Z0-9]{32,}\b",
        # Shopify shared secret (shpss_ prefix)
        r"\bshpss_[a-zA-Z0-9]{32,}\b",
        # Shopify access token (shpat_ prefix, 32+ chars)
        r"\bshpat_[a-zA-Z0-9]{32,}\b",
        # Twitter API keys (bearer tokens, 50+ chars)
        r"\b(?:twitter|twilio)\s+[A-Za-z0-9_-]{50,}\b",
        # Facebook access tokens (EAAB prefix or base64-like)
        r"\bEAAB[a-zA-Z0-9]{100,}\b",
        # LinkedIn API keys
        r"\b(?:linkedin|li_at)\s*[:=]\s*[A-Za-z0-9_-]{20,}\b",
        # Discord bot tokens (base64-like, 59+ chars)
        r"\b(?:discord|bot)[\s:=]+[A-Za-z0-9_-]{59,}\b",
        # Generic Bearer tokens (improved pattern, 20+ chars)
        r"\bBearer\s+[A-Za-z0-9\-._~+/]{20,}=*\b",
        # Generic API key pattern (api[_-]?key, apikey, etc., 20+ chars)
        r"(?i)(?:api[_-]?key|apikey|secret[_-]?key|access[_-]?token)\s*[:=]\s*[A-Za-z0-9\-_+/=]{20,}\b",
        # Generic secret pattern (secret, password, token keywords, 16+ chars)
        r"(?i)(?:secret|password|token|key)\s*[:=]\s*[A-Za-z0-9\-_+/=]{16,}\b",
    ],
    "pem_key": [
        # RSA private keys
        r"-----BEGIN\s+RSA\s+PRIVATE\s+KEY-----[\s\S]*?-----END\s+RSA\s+PRIVATE\s+KEY-----",
        # EC private keys
        r"-----BEGIN\s+EC\s+PRIVATE\s+KEY-----[\s\S]*?-----END\s+EC\s+PRIVATE\s+KEY-----",
        # DSA private keys
        r"-----BEGIN\s+DSA\s+PRIVATE\s+KEY-----[\s\S]*?-----END\s+DSA\s+PRIVATE\s+KEY-----",
        # OPENSSH private keys
        r"-----BEGIN\s+OPENSSH\s+PRIVATE\s+KEY-----[\s\S]*?-----END\s+OPENSSH\s+PRIVATE\s+KEY-----",
        # Generic private keys
        r"-----BEGIN\s+PRIVATE\s+KEY-----[\s\S]*?-----END\s+PRIVATE\s+KEY-----",
        # PGP private keys
        r"-----BEGIN\s+PGP\s+PRIVATE\s+KEY\s+BLOCK-----[\s\S]*?-----END\s+PGP\s+PRIVATE\s+KEY\s+BLOCK-----",
    ],
    "jwt_token": [
        # JWT tokens (eyJ... format, 3 parts separated by dots, improved pattern)
        r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\b",
    ],
    "database_url": [
        # PostgreSQL connection strings
        r"postgres(?:ql)?://[^\s]+",
        # MySQL connection strings
        r"mysql://[^\s]+",
        # MongoDB connection strings
        r"mongodb(?:\+srv)?://[^\s]+",
        # Redis connection strings
        r"redis://[^\s]+",
        # Generic database URL pattern
        r"(?:database|db|connection)[\s:=]+(?:url|uri|string)[\s:=]+([a-z]+://[^\s]+)",
    ],
    "aws_credential": [
        # Access key IDs: AKIA (long-term), ASIA (STS temp), ABIA, ACCA, AROA
        r"\b(?:AKIA|ASIA|ABIA|ACCA|AROA)[0-9A-Z]{16}\b",
        # Secret access key, context-anchored (AWS secret … <40-char value>)
        r"(?i)aws[\s_-]{0,15}secret[\s_-]{0,15}(?:\w+[\s_-]{0,3}){0,2}?([A-Za-z0-9/+=]{40})\b",
        # env / Terraform forms
        r"(?i)aws_secret_access_key\s*[=:]\s*[\"']?[A-Za-z0-9/+=]{40}",
        # STS session token (long base64ish, context-anchored)
        r"(?i)(?:aws_session_token|x-amz-security-token)[\"'\s:=]{1,4}[\"']?[A-Za-z0-9/+=]{100,}",
    ],
    "cloud_credential": [
        # Google Cloud service account keys (JSON-like)
        r'"type"\s*:\s*"service_account"[\s\S]{0,2000}?"private_key"\s*:\s*"-----BEGIN',
        # AWS credentials file format
        r"\[default\]\s+aws_access_key_id\s*=\s*([A-Z0-9]{20})\s+aws_secret_access_key\s*=\s*([A-Za-z0-9/+=]{40})",
        # Azure connection strings
        r"DefaultEndpointsProtocol=https;AccountName=[^;]+;AccountKey=[A-Za-z0-9+/=]{86}==;?",
    ],
}


# Secret types that get the false-positive filter below.
SECRET_TYPES: frozenset[str] = frozenset({"api_key", "aws_credential", "cloud_credential"})

_HEX40 = re.compile(r"[0-9a-fA-F]{40}")


def _shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts: dict[str, int] = {}
    for ch in value:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(value)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def looks_like_secret(matched: str) -> bool:
    """Heuristic false-positive filter for regex-detected secrets.

    Requires a mixed alphanumeric body with at least ~3 bits/char of Shannon
    entropy, and never treats a bare 40-char hex string (e.g. a git commit
    hash) as a secret. Used for context-anchored or generic patterns.
    """
    body = matched.strip().strip("\"'").split()
    if not body:
        return False
    candidate = body[-1].strip("\"',;.:")
    if len(candidate) < 8:
        return False
    if _HEX40.fullmatch(candidate):
        return False
    has_digit = any(c.isdigit() for c in candidate)
    has_alpha = any(c.isalpha() for c in candidate)
    if not (has_digit and has_alpha):
        return False
    return _shannon_entropy(candidate) >= 3.0
