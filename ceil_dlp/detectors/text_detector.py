"""Main PII detection engine using Presidio PatternRecognizers."""

from ceil_dlp.detectors.patterns import PatternMatch
from ceil_dlp.detectors.presidio_adapter import (
    PRESIDIO_TO_PII_TYPE,
    detect_with_presidio_ensemble,
)

# All Presidio entity types supported by ceil-dlp
PRESIDIO_TYPES = frozenset(set(PRESIDIO_TO_PII_TYPE.values()))
CUSTOM_TYPES = frozenset(
    {
        "api_key",
        "pem_key",
        "jwt_token",
        "database_url",
        "cloud_credential",
    }
)
ENABLED_TYPES_DEFAULT = PRESIDIO_TYPES.union(CUSTOM_TYPES)


def detect_pii_in_text(
    text: str,
    enabled_types: set[str] | None = None,
    ner_strength: int = 3,
    custom_patterns: dict[str, list[str]] | None = None,
) -> dict[str, list[PatternMatch]]:
    """
    Detect PII in text using Presidio for standard PII and custom patterns for API keys.

    Args:
        text: Input text to scan
        enabled_types: Optional set of PII types to detect. If None, detects all types.
                      Includes all Presidio entity types (credit_card, ssn, email, phone,
                      person, location, ip_address, url, medical_license, crypto, date_time,
                      iban_code, nrp, and country-specific types like us_driver_license,
                      uk_nhs, es_nif, it_fiscal_code, etc.) plus custom types (api_key,
                      pem_key, jwt_token, database_url, cloud_credential).
        ner_strength: NER model strength:
                     - 1: en_core_web_lg only (fastest)
                     - 2: spaCy + transformer ensemble (balanced)
                     - 3: spaCy + transformer + GLiNER ensemble (best coverage, slower)
                     Defaults to 1 for backward compatibility.
        custom_patterns: Optional dict mapping PII type name to list of regex strings.
                         Patterns for existing types extend the built-in detection;
                         new keys create new PII types.

    Returns:
        Dictionary mapping PII type to list of matches.
    """
    # Custom pattern types are always valid detection types
    custom_type_names = set(custom_patterns.keys()) if custom_patterns else set()
    all_valid_types = PRESIDIO_TYPES.union(CUSTOM_TYPES).union(custom_type_names)

    # Determine which types to detect
    if enabled_types is None:
        # Default detection includes custom pattern types
        types_to_detect = ENABLED_TYPES_DEFAULT.union(custom_type_names)
    else:
        types_to_detect = frozenset(enabled_types)

    all_types = types_to_detect.intersection(all_valid_types)

    if not all_types:
        return {}

    # Use ensemble detection (handles merging when ner_strength=2)
    # detect_with_presidio_ensemble already filters by enabled_types, including custom types
    return detect_with_presidio_ensemble(
        text,
        ner_strength=ner_strength,
        enabled_types=set(all_types),
        custom_patterns=custom_patterns,
    )
