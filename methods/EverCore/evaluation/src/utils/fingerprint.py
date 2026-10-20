"""Deterministic identities for source data and experimental settings."""

import hashlib
import json
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from pathlib import Path


def _json_default(value):
    if is_dataclass(value):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot fingerprint {type(value).__name__}")


def fingerprint(value) -> str:
    encoder = json.JSONEncoder(
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        default=_json_default, allow_nan=False,
    )
    digest = hashlib.sha256()
    for part in encoder.iterencode(value):
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()


def public_config(value):
    """Credential rotation is not a new experiment; never persist secrets."""
    if isinstance(value, dict):
        return {
            key: public_config(item) for key, item in value.items()
            if not any(secret in key.lower() for secret in (
                "api_key", "password", "secret", "access_token", "database_url",
            ))
        }
    if isinstance(value, (list, tuple)):
        return [public_config(item) for item in value]
    return value
