"""Configuration loading and hashing."""
from pathlib import Path
import hashlib
import json
import yaml


def read_config(path):
    with open(path, encoding="utf-8") as f:
        result = yaml.safe_load(f)
    if not isinstance(result, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return result


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(Path(path), "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()
