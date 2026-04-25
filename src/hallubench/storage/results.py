"""Atomic JSON/JSONL exports, without accepting silent malformed records."""
import json
import os
import tempfile
from pathlib import Path


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temp:
        json.dump(obj, temp, indent=2, ensure_ascii=False, allow_nan=False, default=str)
        temp.write("\n")
        name = temp.name
    os.replace(name, path)


def read_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temp:
        for row in rows:
            temp.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=str) + "\n")
        name = temp.name
    os.replace(name, path)


def read_jsonl(path):
    with open(path, encoding="utf-8") as file:
        for n, line in enumerate(file, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {n}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Line {n} must be a JSON object")
                yield row
