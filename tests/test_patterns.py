"""Tests for custom pattern detection via Presidio.

These tests verify that our custom patterns are correctly configured
in Presidio PatternRecognizers. We test a representative sample of patterns
since Presidio handles the actual detection.
"""

from ceil_dlp.detectors.text_detector import detect_pii_in_text


def test_api_key_detection():
    """Test API key detection with various providers."""
    # Test OpenAI key
    text1 = "My API key is sk-1234567890abcdef1234567890abcdef"
    results1 = detect_pii_in_text(text1, enabled_types={"api_key"})
    assert "api_key" in results1
    assert len(results1["api_key"]) > 0

    # Prefix-agnostic modern keys (regression: sk-proj-/sk-FAKE- used to leak).
    # Fixtures are assembled at runtime so secret scanners don't flag them.
    sample_keys = [
        "sk-" + "proj-" + "4xX9mZqL7bQ2vT8wY5nR3kJ6hG0dS1eF",
        "sk-" + "FAKE-" + "9f8e7d6c5b4a3210abcdef1234567890",
        "ghp_" + "16C7e42F292c6912E7710c838347Ae178B4a",
        "xoxb-" + "123456789012" + "-" + "1234567890123" + "-abcdefghijklmnopqrstuv",
    ]
    for key in sample_keys:
        results = detect_pii_in_text(f"token {key}", enabled_types={"api_key"})
        assert results.get("api_key"), f"failed to detect {key}"


def test_aws_credential_detection():
    """AWS access key IDs and context-anchored secret keys are detected."""
    aws_secret = "wJalr" + "XUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    for text in (
        "Access key: AKIA1234567890ABCDEF",
        "aws_secret_access_key = " + aws_secret,
    ):
        results = detect_pii_in_text(text, enabled_types={"aws_credential"})
        assert results.get("aws_credential"), f"failed to detect: {text}"


def test_secret_false_positive_filter():
    """looks_like_secret rejects low-signal strings (git hashes, keywords)."""
    from ceil_dlp.detectors.patterns import looks_like_secret

    assert looks_like_secret("sk-" + "proj-" + "4xX9mZqL7bQ2vT8wY5nR3kJ6hG0dS1eF")
    assert looks_like_secret("AKIAIOSFODNN7EXAMPLE")
    assert not looks_like_secret("sk-abcdefghijklmnopqrstuvwxyz")
    assert not looks_like_secret("abcdef1234567890abcdef1234567890abcdef12")
    assert not looks_like_secret("AWS_REGION")


def test_pem_key_detection():
    """Test PEM key detection."""
    text = """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA1234567890abcdef
-----END RSA PRIVATE KEY-----"""
    results = detect_pii_in_text(text, enabled_types={"pem_key"})
    assert "pem_key" in results
    assert len(results["pem_key"]) > 0


def test_jwt_token_detection():
    """Test JWT token detection."""
    text = "JWT: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    results = detect_pii_in_text(text, enabled_types={"jwt_token"})
    assert "jwt_token" in results
    assert len(results["jwt_token"]) > 0


def test_database_url_detection():
    """Test database URL detection."""
    text = "Database: postgresql://user:pass@localhost:5432/dbname"
    results = detect_pii_in_text(text, enabled_types={"database_url"})
    assert "database_url" in results
    assert len(results["database_url"]) > 0


def test_cloud_credential_detection():
    """Test cloud credential detection."""
    text = "[default]\naws_access_key_id = AKIA1234567890ABCDEF\naws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    results = detect_pii_in_text(text, enabled_types={"cloud_credential"})
    assert "cloud_credential" in results
    assert len(results["cloud_credential"]) > 0


def test_no_matches():
    """Test detection with no matches."""
    text = "This is just normal text with no secrets"
    results = detect_pii_in_text(text, enabled_types={"api_key"})
    matches = results.get("api_key", [])
    assert len(matches) == 0


def test_position_tracking():
    """Test that matches include correct positions."""
    text = "Start sk-1234567890abcdef1234567890abcdef end"
    results = detect_pii_in_text(text, enabled_types={"api_key"})
    matches = results.get("api_key", [])
    assert len(matches) > 0
    match = matches[0]
    matched_text, start, end = match
    assert text[start:end] == matched_text
    assert matched_text in text
