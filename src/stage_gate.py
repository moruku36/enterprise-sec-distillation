"""Offline preflight shared by legacy generation, training and merge entry points."""

import json
from importlib import metadata
from pathlib import Path

from contracts import ContractError, validate_pilot_config

ROOT = Path(__file__).resolve().parent.parent
LEGACY = {"train.jsonl", "eval.jsonl", "eval_results.json"}


def pilot_config(path):
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ContractError(f"pilot config unreadable: {exc}") from exc
    errors = validate_pilot_config(config)
    model = config.get("model", {})
    for key in ("name", "revision", "tokenizer_revision"):
        if not isinstance(model.get(key), str) or not model[key].strip():
            errors.append(f"pilot: model.{key} must be pinned")
    if model.get("license_checked") is not True:
        errors.append("pilot: model license must be checked")
    budget = config.get("budget", {})
    if not isinstance(budget.get("currency"), str) or not budget["currency"].strip():
        errors.append("pilot: budget.currency must be a currency code")
    for key in ("max_cost", "max_input_tokens", "max_output_tokens", "max_requests", "max_training_gpu_hours",
                "max_models", "max_storage_gb"):
        v = budget.get(key)
        if type(v) not in (int, float) or v <= 0:
            errors.append(f"pilot: budget.{key} must be positive")
    if errors:
        raise ContractError("; ".join(errors))
    return config


def new_output(path):
    target = Path(path).resolve()
    data_dir = (ROOT / "data").resolve()
    if target.name in LEGACY or target == data_dir or data_dir in target.parents or target.exists():
        raise ContractError(f"output must be a new path and must preserve legacy data: {target}")
    return target


def require_version(package, expected):
    try:
        actual = metadata.version(package)
    except metadata.PackageNotFoundError as exc:
        raise ContractError(f"required dependency missing: {package}=={expected}") from exc
    if actual != expected:
        raise ContractError(f"{package}=={expected} required; installed {actual}")


def approved(config, stage, cli_approval):
    if not cli_approval or config.get("approvals", {}).get(stage) is not True:
        raise ContractError(f"{stage} requires config approval and explicit CLI approval")
