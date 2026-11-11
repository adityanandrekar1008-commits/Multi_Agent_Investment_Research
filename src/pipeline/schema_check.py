"""Minimal JSON Schema validator for the project's own output schema.

Supports the keywords used in schemas/company_analysis.schema.json: type,
const, enum, required, properties, additionalProperties, items, oneOf,
anyOf, and local $ref ("#/$defs/..."). Avoids adding a dependency; any
standard validator (e.g. the `jsonschema` package) can be used instead.
"""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_FILE = PROJECT_ROOT / "schemas" / "company_analysis.schema.json"


def load_schema(path=SCHEMA_FILE):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _type_ok(value, expected):
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    raise ValueError(f"Unsupported type in schema: {expected}")


def validate(instance, schema, root=None, path="$"):
    """Return a list of error strings (empty when valid)."""
    root = root if root is not None else schema

    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/$defs/"):
            raise ValueError(f"Unsupported $ref: {ref}")
        return validate(instance, root["$defs"][ref[len("#/$defs/"):]], root, path)

    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_type_ok(instance, t) for t in types):
            return [f"{path}: expected {'/'.join(types)}, got "
                    f"{type(instance).__name__}"]

    errors = []
    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: expected {schema['const']!r}, got {instance!r}")
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} is not one of {schema['enum']}")

    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: missing required property '{key}'")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in properties:
                errors += validate(value, properties[key], root, f"{path}.{key}")
            elif additional is False:
                errors.append(f"{path}: unexpected property '{key}'")
            elif isinstance(additional, dict):
                errors += validate(value, additional, root, f"{path}.{key}")

    if isinstance(instance, list) and "items" in schema:
        for i, item in enumerate(instance):
            errors += validate(item, schema["items"], root, f"{path}[{i}]")

    if "oneOf" in schema:
        results = [validate(instance, branch, root, path) for branch in schema["oneOf"]]
        matches = sum(1 for r in results if not r)
        if matches != 1:
            detail = "; ".join(r[0] for r in results if r)[:300]
            errors.append(
                f"{path}: must match exactly one schema in oneOf "
                f"(matched {matches}). {detail}"
            )
    if "anyOf" in schema:
        results = [validate(instance, branch, root, path) for branch in schema["anyOf"]]
        if all(results):
            errors.append(f"{path}: must match at least one schema in anyOf")

    return errors


def validate_payload(payload, schema=None):
    """Validate a company_analysis payload; returns a list of errors."""
    schema = schema or load_schema()
    # Round-trip through JSON so the check sees exactly what consumers see.
    return validate(json.loads(json.dumps(payload, default=str)), schema)
