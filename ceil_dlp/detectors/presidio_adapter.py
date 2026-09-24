"""Adapter to integrate Presidio for standard PII detection."""

import logging
import os
import re
from functools import lru_cache

# Set transformers verbosity BEFORE importing anything that might use transformers
# This suppresses the "Some weights were not used" warning which is expected when loading
# BERT checkpoints for token classification (the warning itself says "This IS expected")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# Disable advisory warnings (like the "Some weights were not used" message)
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry

from ceil_dlp.detectors import patterns
from ceil_dlp.detectors.patterns import PatternMatch

logger = logging.getLogger(__name__)

# Suppress expected Presidio warnings:
# - Language mismatch warnings: Presidio loads recognizers for multiple languages (es, it, pl, etc.)
#   but we only use English. These warnings are harmless but noisy.
# - Configuration warnings: Missing optional config parameters that use defaults.
presidio_logger = logging.getLogger("presidio-analyzer")
presidio_logger.setLevel(logging.ERROR)  # Only show ERROR and above, suppress WARNING


PRESIDIO_TO_PII_TYPE: dict[str, str] = {
    # Global entities
    "CREDIT_CARD": "credit_card",
    "CRYPTO": "crypto",
    "DATE_TIME": "date_time",
    "EMAIL_ADDRESS": "email",
    "IBAN_CODE": "iban_code",
    "IP_ADDRESS": "ip_address",
    "LOCATION": "location",
    "PERSON": "person",
    "PHONE_NUMBER": "phone",
    "MEDICAL_LICENSE": "medical_license",
    "URL": "url",
    "NRP": "nrp",
    # United States
    "US_BANK_NUMBER": "us_bank_number",
    "US_DRIVER_LICENSE": "us_driver_license",
    "US_ITIN": "us_itin",
    "US_PASSPORT": "us_passport",
    "US_SSN": "ssn",
    # United Kingdom
    "UK_NHS": "uk_nhs",
    "UK_NINO": "uk_nino",
    # Spain
    "ES_NIF": "es_nif",
    "ES_NIE": "es_nie",
    # Italy
    "IT_FISCAL_CODE": "it_fiscal_code",
    "IT_DRIVER_LICENSE": "it_driver_license",
    "IT_VAT_CODE": "it_vat_code",
    "IT_PASSPORT": "it_passport",
    "IT_IDENTITY_CARD": "it_identity_card",
    # Poland
    "PL_PESEL": "pl_pesel",
    # Singapore
    "SG_NRIC_FIN": "sg_nric_fin",
    "SG_UEN": "sg_uen",
    # Australia
    "AU_ABN": "au_abn",
    "AU_ACN": "au_acn",
    "AU_TFN": "au_tfn",
    "AU_MEDICARE": "au_medicare",
    # India
    "IN_PAN": "in_pan",
    "IN_AADHAAR": "in_aadhaar",
    "IN_VEHICLE_REGISTRATION": "in_vehicle_registration",
    "IN_VOTER": "in_voter",
    "IN_PASSPORT": "in_passport",
    "IN_GSTIN": "in_gstin",
    # Finland
    "FI_PERSONAL_IDENTITY_CODE": "fi_personal_identity_code",
    # Korea
    "KR_RRN": "kr_rrn",
    # Thailand
    "TH_TNIN": "th_tnin",
    # Custom secret types (mapped from PatternRecognizer entity names)
    "API_KEY": "api_key",
    "PEM_KEY": "pem_key",
    "JWT_TOKEN": "jwt_token",
    "DATABASE_URL": "database_url",
    "CLOUD_CREDENTIAL": "cloud_credential",
    "AWS_CREDENTIAL": "aws_credential",
}


def get_pii_type_to_entities(custom_patterns: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    """Get mapping of PII type to Presidio entity names.

    Includes any custom pattern types so image/PDF redaction can map them.
    """
    mapping = {v: [k] for k, v in PRESIDIO_TO_PII_TYPE.items()}
    if custom_patterns:
        for pii_type in custom_patterns:
            mapping[pii_type] = [pii_type.upper()]
    return mapping


def _custom_patterns_key(custom_patterns: dict[str, list[str]] | None) -> frozenset:
    """Return a hashable signature for custom patterns (for analyzer caching)."""
    if not custom_patterns:
        return frozenset()
    return frozenset((pii_type, tuple(patterns)) for pii_type, patterns in custom_patterns.items())


def _create_secret_recognizers(
    custom_patterns: dict[str, list[str]] | None = None,
) -> list[PatternRecognizer]:
    """
    Create Presidio PatternRecognizer objects for custom secrets (API keys, etc.).

    Merges the built-in regex patterns with any config-provided custom patterns.
    Custom patterns for a known type extend that type's patterns; custom patterns
    for an unknown type create a brand new PII type.

    Args:
        custom_patterns: Optional dict mapping PII type name to list of regex strings.

    Returns:
        List of PatternRecognizer objects
    """
    from ceil_dlp.detectors.patterns import PATTERNS

    recognizers = []

    # Merge built-in patterns with custom patterns
    merged_patterns: dict[str, list[str]] = {
        str(pattern_type): list(patterns_list)
        for pattern_type, patterns_list in PATTERNS.items()
    }
    if custom_patterns:
        for pii_type, patterns in custom_patterns.items():
            merged_patterns.setdefault(pii_type, []).extend(patterns)

    for pattern_type, patterns_list in merged_patterns.items():
        if not patterns_list:
            continue

        presidio_patterns: list[Pattern] = []

        for regex_pattern in patterns_list:
            # Convert our regex pattern to Presidio Pattern
            presidio_pattern = Pattern(
                name=f"{pattern_type}_{len(presidio_patterns)}",
                regex=regex_pattern,
                score=0.8,  # Confidence score
            )
            presidio_patterns.append(presidio_pattern)

        if presidio_patterns:
            # Create PatternRecognizer for this secret type
            recognizer = PatternRecognizer(
                supported_entity=pattern_type.upper(),  # e.g., "API_KEY"
                patterns=presidio_patterns,
                supported_language="en",
            )
            recognizers.append(recognizer)

    return recognizers


def _get_entity_to_pii_type(
    custom_patterns: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    """Build the Presidio entity -> PII type mapping, including custom pattern types."""
    mapping = dict(PRESIDIO_TO_PII_TYPE)
    if custom_patterns:
        for pii_type in custom_patterns:
            mapping[pii_type.upper()] = pii_type
    return mapping


@lru_cache(maxsize=12)  # Cache analyzers per (strength, custom-patterns signature)
def _get_analyzer_cached(ner_strength: int, custom_patterns_key: frozenset) -> AnalyzerEngine:
    """Internal cached function - ner_strength must be 1, 2, or 3."""
    # Rebuild custom patterns dict from the hashable key
    custom_patterns: dict[str, list[str]] | None = (
        {pii_type: list(patterns) for pii_type, patterns in custom_patterns_key}
        if custom_patterns_key
        else None
    )

    # Create registry with built-in recognizers
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()

    # Add custom secret recognizers (built-in + config-provided patterns)
    secret_recognizers = _create_secret_recognizers(custom_patterns)
    for recognizer in secret_recognizers:
        registry.add_recognizer(recognizer)

    # Configure NLP engine based on strength
    if ner_strength == 1:
        # Use default (en_core_web_lg) - no special configuration needed
        return AnalyzerEngine(registry=registry)
    elif ner_strength == 3:
        try:
            from huggingface_hub.utils.tqdm import disable_progress_bars

            disable_progress_bars()
        except ImportError:
            logger.warning(
                "Failed to load HuggingFace Hub progress bars: Missing dependency. Install with: pip install huggingface_hub"
            )
        # Use GLiNER zero-shot NER model
        try:
            from presidio_analyzer.predefined_recognizers import GLiNERRecognizer

            # Map GLiNER entity types to Presidio entity types
            # GLiNER's PII model (urchade/gliner_multi_pii-v1) outputs fine-grained types
            # that we map to our standard Presidio types
            entity_mapping = {
                "person": "PERSON",
                "name": "PERSON",
                "organization": "ORGANIZATION",
                "org": "ORGANIZATION",
                "location": "LOCATION",
                "loc": "LOCATION",
                "email": "EMAIL_ADDRESS",
                "phone": "PHONE_NUMBER",
                "credit_card": "CREDIT_CARD",
                "ssn": "US_SSN",
                "ip_address": "IP_ADDRESS",
                "ip": "IP_ADDRESS",
                "url": "URL",
                "date": "DATE_TIME",
                "date_time": "DATE_TIME",
            }

            # Create GLiNER recognizer with PII-specific model
            gliner_recognizer = GLiNERRecognizer(
                model_name="urchade/gliner_multi_pii-v1",
                entity_mapping=entity_mapping,
                flat_ner=False,  # Keep nested entities
                multi_label=True,  # Allow multiple labels per span
                threshold=0.5,  # Confidence threshold
                map_location="cpu",  # Use CPU by default (can be "cuda" if GPU available)
            )

            # Register GLiNER recognizer
            registry.add_recognizer(gliner_recognizer)

            # Use default spaCy NLP engine (for tokenization, etc.)
            return AnalyzerEngine(registry=registry)
        except ImportError as e:
            logger.warning(
                f"Failed to load GLiNER (ner_strength=3): Missing dependency. "
                f"Install with: pip install gliner. Error: {e}"
            )
            raise
        except Exception as e:
            logger.warning(f"Failed to load GLiNER (ner_strength=3): {e}")
            raise
    else:  # ner_strength == 2
        # Use transformer-based NER model (best accuracy)
        try:
            # Set transformers verbosity to ERROR before importing/using transformers
            # This suppresses expected warnings like "Some weights were not used"
            try:
                from transformers.utils import logging as transformers_logging

                transformers_logging.set_verbosity_error()
            except ImportError:
                # If transformers logging utils aren't available, environment variables should handle it
                pass

            from presidio_analyzer.nlp_engine import NerModelConfiguration, TransformersNlpEngine

            model_config = [
                {
                    "lang_code": "en",
                    "model_name": {
                        "spacy": "en_core_web_sm",  # Small spaCy for tokenization, lemmatization
                        "transformers": "dslim/bert-base-NER",  # Transformer NER model
                    },
                }
            ]

            # Map transformer entity labels to Presidio entity names
            # Note: Model outputs labels with B-/I- prefixes (B-PER, I-PER), but mapping
            # should use labels without prefixes (PER, LOC, ORG, MISC)
            # The dslim/bert-base-NER model outputs: PER, LOC, ORG, MISC
            mapping = {
                "PER": "PERSON",
                "LOC": "LOCATION",
                "ORG": "ORGANIZATION",
                "MISC": "MISC",
                # Also include common variations
                "PERSON": "PERSON",
                "GPE": "LOCATION",  # Geopolitical entity
            }

            ner_config = NerModelConfiguration(
                model_to_presidio_entity_mapping=mapping,
                alignment_mode="expand",  # "strict", "contract", "expand"
                aggregation_strategy="max",  # "simple", "first", "average", "max"
                labels_to_ignore=["O"],  # Ignore "no entity" label
                stride=128,  # Increased window overlap for better coverage of long texts
                # Larger stride helps ensure entities near chunk boundaries aren't missed
            )

            tf_engine = TransformersNlpEngine(
                models=model_config, ner_model_configuration=ner_config
            )
            return AnalyzerEngine(
                registry=registry, nlp_engine=tf_engine, supported_languages=["en"]
            )
        except ImportError as e:
            logger.warning(
                f"Failed to load transformer NER model (ner_strength=2): Missing dependency. "
                f"Install with: pip install spacy-huggingface-pipelines transformers. "
                f"Falling back to default NER model. Error: {e}"
            )
            return AnalyzerEngine(registry=registry)
        except Exception as e:
            logger.warning(
                f"Failed to load transformer NER model (ner_strength=2), falling back to default: {e}"
            )
            return AnalyzerEngine(registry=registry)


def get_analyzer(
    ner_strength: int = 1,
    custom_patterns: dict[str, list[str]] | None = None,
) -> AnalyzerEngine:
    """Get cached AnalyzerEngine instance with custom secret recognizers.

    Args:
        ner_strength: NER model strength:
                     - 1: en_core_web_lg (spaCy)
                     - 2: transformer-based NER (dslim/bert-base-NER)
                     - 3: GLiNER zero-shot NER (best for long texts and hyphenated names)
                     Defaults to 1 for backward compatibility.
        custom_patterns: Optional dict mapping PII type name to list of regex strings.
                         These are merged with the built-in secret patterns.

    Returns:
        AnalyzerEngine configured with the specified NER model strength.

    Raises:
        ValueError: If ner_strength is not 1, 2, or 3.
    """
    # Validate strength BEFORE caching to ensure cache key consistency
    # This prevents multiple cache entries for the same effective strength
    if ner_strength not in (1, 2, 3):
        raise ValueError(
            f"ner_strength must be 1, 2, or 3, got {ner_strength}. "
            "Use 1 for en_core_web_lg, 2 for transformer-based NER, or 3 for GLiNER."
        )
    return _get_analyzer_cached(ner_strength, _custom_patterns_key(custom_patterns))


# Numeric-body recognizers are noisy: the digit groups inside credential
# tokens (Slack xoxb-, GitHub ghp_, OpenAI sk-, AWS AKIA…) match these
# Presidio entities. Skip such matches when they are part of a known
# credential token rather than a real license/bank number.
_TOKEN_BODY_FP_TYPES: frozenset[str] = frozenset(
    {"us_driver_license", "it_driver_license", "us_bank_number"}
)

_TOKEN_PREFIX = (
    r"xox[abprs]-|gh[opurs]_|github_pat_|glpat-|sk-|pk-|rk-|"
    r"shpat_|shpca_|shpss_|sq0[a-z]{2}-|key-|SG\.|"
    r"AKIA|ASIA|ABIA|ACCA|AROA|AIza|EAAB"
)
_TOKEN_AT_START_RE = re.compile(r"(?:" + _TOKEN_PREFIX + r")")


def _is_credential_token_fragment(text: str, start: int, matched: str) -> bool:
    """True when a (false-positive) match is really part of a credential token."""
    if _TOKEN_AT_START_RE.match(matched):
        return True
    # Widen to the whitespace-delimited word containing the match: a token's
    # numeric segments (xoxb-123456789012-1234567890123-abc) are not space
    # separated, so the second segment has no token prefix immediately before it.
    ws_start = start
    while ws_start > 0 and not text[ws_start - 1].isspace():
        ws_start -= 1
    ws_end = start + len(matched)
    while ws_end < len(text) and not text[ws_end].isspace():
        ws_end += 1
    word = text[ws_start:ws_end].lstrip("\"'`([{<").rstrip(".,;:!?)]}>\"'`")
    return bool(_TOKEN_AT_START_RE.match(word))


def _detect_with_presidio(
    text: str,
    ner_strength: int = 1,
    custom_patterns: dict[str, list[str]] | None = None,
) -> dict[str, list[PatternMatch]]:
    analyzer = get_analyzer(ner_strength=ner_strength, custom_patterns=custom_patterns)
    results = analyzer.analyze(text=text, language="en")
    entity_to_pii_type = _get_entity_to_pii_type(custom_patterns)
    detections: dict[str, list[PatternMatch]] = {}
    for result in results:
        entity_type = result.entity_type
        pii_type = entity_to_pii_type.get(entity_type)
        if pii_type:
            matched_text = text[result.start : result.end]
            # Numeric-body FP: the match is the body of a credential token
            if pii_type in _TOKEN_BODY_FP_TYPES and _is_credential_token_fragment(
                text, result.start, matched_text
            ):
                continue
            # False-positive filter for secret-shaped types
            if pii_type in patterns.SECRET_TYPES and not patterns.looks_like_secret(
                matched_text
            ):
                continue
            match = (matched_text, result.start, result.end)
            if pii_type not in detections:
                detections[pii_type] = []
            detections[pii_type].append(match)
    return detections


def detect_with_presidio_ensemble(
    text: str,
    ner_strength: int = 1,
    enabled_types: set[str] | frozenset[str] | None = None,
    custom_patterns: dict[str, list[str]] | None = None,
) -> dict[str, list[PatternMatch]]:
    """
    Detect PII using Presidio with optional ensemble approach (merging multiple NER models).

    - ner_strength=1: spaCy NER (en_core_web_lg) only
    - ner_strength=2: Ensemble of spaCy + transformer NER
    - ner_strength=3: Ensemble of spaCy + transformer + GLiNER NER (best coverage)

    Args:
        text: Input text to scan
        ner_strength: NER model strength:
                     - 1: en_core_web_lg only (fastest)
                     - 2: spaCy + transformer ensemble (balanced)
                     - 3: spaCy + transformer + GLiNER ensemble (best coverage, slower)
                     Defaults to 1 for backward compatibility.
        enabled_types: Optional set of PII types to filter results. If None, returns all detected types.
        custom_patterns: Optional dict mapping PII type name to list of regex strings.
                         These are merged with the built-in secret patterns.

    Returns:
        Dictionary mapping PII type to list of matches.
        Each match is a tuple: (matched_text, start_pos, end_pos)
    """
    # Validate ner_strength
    if ner_strength not in (1, 2, 3):
        raise ValueError(
            f"ner_strength must be 1, 2, or 3, got {ner_strength}. "
            "Use 1 for en_core_web_lg, 2 for spaCy+transformer ensemble, "
            "or 3 for spaCy+transformer+GLiNER ensemble."
        )
    ner_strength_val = ner_strength

    # NER Ensemble: If strength 2 or 3, detect with multiple models and merge
    # NOTE: We detect with all models on the ORIGINAL text, then merge.
    # This is better than sequential (detect -> redact -> detect -> redact) because:
    # 1. All models see the original text (no information loss)
    # 2. Maximum coverage from all models
    # 3. Single redaction pass (more efficient)
    # 4. No risk of missing PII that one model would catch but another already redacted
    # Sequential approach works for images because OCR can still read surrounding text
    # after redaction, but for text, redaction replaces content making it undetectable.
    if ner_strength_val == 2:
        # Two-model ensemble: spaCy + transformer
        detections_spacy = _detect_with_presidio(
            text, ner_strength=1, custom_patterns=custom_patterns
        )
        detections_transformer = _detect_with_presidio(
            text, ner_strength=2, custom_patterns=custom_patterns
        )
        detections_list = [detections_spacy, detections_transformer]
    elif ner_strength_val == 3:
        # Three-model ensemble: spaCy + transformer + GLiNER
        detections_spacy = _detect_with_presidio(
            text, ner_strength=1, custom_patterns=custom_patterns
        )
        detections_transformer = _detect_with_presidio(
            text, ner_strength=2, custom_patterns=custom_patterns
        )
        detections_gliner = _detect_with_presidio(
            text, ner_strength=3, custom_patterns=custom_patterns
        )
        detections_list = [detections_spacy, detections_transformer, detections_gliner]
    else:
        # Single model detection
        detections = _detect_with_presidio(
            text, ner_strength=ner_strength_val, custom_patterns=custom_patterns
        )
        if enabled_types:
            detections = {k: v for k, v in detections.items() if k in enabled_types}
        return detections

    # Merge detections from all models
    # For overlapping matches, prefer the one with better coverage or keep both
    merged_detections: dict[str, list[PatternMatch]] = {}

    # Collect all matches
    all_matches: dict[
        tuple[int, int], tuple[str, PatternMatch]
    ] = {}  # (start, end) -> (pii_type, match)

    # Add detections from all models
    for detections in detections_list:
        for pii_type, matches in detections.items():
            if enabled_types and pii_type not in enabled_types:
                continue
            for match in matches:
                _text, start, end = match
                key = (start, end)
                # Add match (will overwrite if same span, which is fine - later models take precedence)
                all_matches[key] = (pii_type, match)

    # Convert back to detections format
    for (_start, _end), (pii_type, match) in all_matches.items():
        if pii_type not in merged_detections:
            merged_detections[pii_type] = []
        merged_detections[pii_type].append(match)

    return merged_detections


def detect_with_presidio(
    text: str,
    ner_strength: int = 1,
    custom_patterns: dict[str, list[str]] | None = None,
) -> dict[str, list[PatternMatch]]:
    """
    Detect standard PII using Presidio.

    Args:
        text: Input text to scan
        ner_strength: NER model strength:
                     - 1: en_core_web_lg only (fastest)
                     - 2: spaCy + transformer ensemble (balanced)
                     - 3: spaCy + transformer + GLiNER ensemble (best coverage, slower)
                     Defaults to 1 for backward compatibility.
        custom_patterns: Optional dict mapping PII type name to list of regex strings.

    Returns:
        Dictionary mapping PII type to list of matches.
        Each match is a tuple: (matched_text, start_pos, end_pos)
    """
    try:
        return _detect_with_presidio(
            text, ner_strength=ner_strength, custom_patterns=custom_patterns
        )
    except Exception as e:
        raise RuntimeError("Failed to detect PII with Presidio") from e
