"""
contracts.py
オフライン評価基盤のデータ契約 (SourceRecord / ExampleRecord / PredictionRecord /
ReviewRecord / RunManifest / 設定)。標準ライブラリのみ。torch / HF / API クライアントは import しない。

各 validate_* は問題点の文字列リストを返す。空なら合格。
"""

import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List

SCHEMA_VERSION = "1"

DECISIONS = ("answer", "partial", "clarify", "abstain", "out_of_scope")
ANSWERED_DECISIONS = ("answer", "partial")
WITHHOLD_DECISIONS = ("clarify", "abstain", "out_of_scope")
ANSWERABILITY = ("answerable", "partial", "unanswerable", "conflicting", "out_of_scope")
SPLITS = ("dev", "test", "train")
PROVENANCE = ("fixture", "human_authored", "generated")
PERMITTED_USES = ("retrieval", "evaluation", "training")

# answerability ごとに許容する expected_decision
ALLOWED_EXPECTED = {
    "answerable": ("answer",),
    "partial": ("partial", "clarify"),
    "unanswerable": ("abstain", "clarify"),
    "conflicting": ("abstain", "partial"),
    "out_of_scope": ("out_of_scope",),
}

SOURCE_FIELDS = (
    "schema_version", "source_id", "canonical_url", "publisher", "title", "product",
    "product_version", "retrieved_at", "valid_as_of", "content_sha256",
    "license_or_terms_url", "permitted_uses", "review_status", "spans",
)
EXAMPLE_FIELDS = (
    "example_id", "family_id", "split", "task_type", "category", "language",
    "product_version", "as_of", "question", "context_ids", "answerability",
    "expected_decision", "gold_claims", "forbidden_actions", "source_spans",
    "validation", "provenance",
)
VALIDATION_FIELDS = (
    "schema_valid", "source_verified", "syntax_checked", "semantic_reviewed",
    "approved_for_training",
)
PREDICTION_FIELDS = (
    "example_id", "status", "decision", "answer", "claims", "commands",
    "missing_information", "as_of",
)
REVIEW_FIELDS = (
    "example_id", "reviewer_id", "reviewed_at", "task_satisfied", "severe_unsafe",
    "claims_supported", "reason",
)


class ContractError(Exception):
    pass


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file_normalized(path: Path) -> str:
    """改行コード差 (CRLF/LF) を吸収した SHA-256。legacy 成果物の不変性確認用。"""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def is_iso_date(value: Any) -> bool:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """厳格な JSONL 読み込み。ファイル欠落・不正行は ContractError (黙って縮小しない)。"""
    path = Path(path)
    if not path.is_file():
        raise ContractError(f"file not found: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ContractError(f"{path}:{n}: invalid JSON: {e}")
            if not isinstance(obj, dict):
                raise ContractError(f"{path}:{n}: record must be an object")
            rows.append(obj)
    return rows


def _missing(d: Dict[str, Any], fields: Iterable[str], label: str) -> List[str]:
    return [f"{label}: missing field '{k}'" for k in fields if k not in d]


def _list_of_str(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


def validate_source(d: Dict[str, Any]) -> List[str]:
    label = f"source[{d.get('source_id', '?')}]"
    errs = _missing(d, SOURCE_FIELDS, label)
    if errs:
        return errs
    if d["schema_version"] != SCHEMA_VERSION:
        errs.append(f"{label}: unsupported schema_version {d['schema_version']!r}")
    for k in ("source_id", "canonical_url", "publisher", "title", "product", "product_version",
              "review_status", "content_sha256", "license_or_terms_url"):
        if not isinstance(d[k], str) or not d[k].strip():
            errs.append(f"{label}: '{k}' must be a non-empty string")
    if not is_iso_date(d["valid_as_of"]):
        errs.append(f"{label}: valid_as_of must be YYYY-MM-DD")
    # retrieved_at は取得していない fixture のために 'not_retrieved' を許す
    if d["retrieved_at"] != "not_retrieved" and not is_iso_date(d["retrieved_at"]):
        errs.append(f"{label}: retrieved_at must be YYYY-MM-DD or 'not_retrieved'")
    if not _list_of_str(d["permitted_uses"]) or not set(d["permitted_uses"]) <= set(PERMITTED_USES):
        errs.append(f"{label}: permitted_uses must be a subset of {PERMITTED_USES}")
    if not isinstance(d["spans"], list) or not d["spans"]:
        errs.append(f"{label}: spans must be a non-empty list")
        return errs
    seen = set()
    for sp in d["spans"]:
        if not isinstance(sp, dict):
            errs.append(f"{label}: span must be an object")
            continue
        errs += _missing(sp, ("span_id", "exact_text", "anchor", "sha256"), f"{label}.span")
        if "span_id" in sp:
            if sp["span_id"] in seen:
                errs.append(f"{label}: duplicate span_id {sp['span_id']!r}")
            seen.add(sp["span_id"])
        if "exact_text" in sp and "sha256" in sp and sha256_text(sp["exact_text"]) != sp["sha256"]:
            errs.append(f"{label}: span {sp.get('span_id')!r} sha256 does not match exact_text")
    return errs


def validate_example(d: Dict[str, Any]) -> List[str]:
    label = f"example[{d.get('example_id', '?')}]"
    errs = _missing(d, EXAMPLE_FIELDS, label)
    if errs:
        return errs
    for k in ("example_id", "family_id", "task_type", "category", "language", "product_version", "question"):
        if not isinstance(d[k], str) or not d[k].strip():
            errs.append(f"{label}: '{k}' must be a non-empty string")
    if d["split"] not in SPLITS:
        errs.append(f"{label}: split must be one of {SPLITS}")
    if d["provenance"] not in PROVENANCE:
        errs.append(f"{label}: provenance must be one of {PROVENANCE}")
    if not is_iso_date(d["as_of"]):
        errs.append(f"{label}: as_of must be an unambiguous YYYY-MM-DD date, got {d['as_of']!r}")
    if d["answerability"] not in ANSWERABILITY:
        errs.append(f"{label}: answerability must be one of {ANSWERABILITY}")
    elif d["expected_decision"] not in ALLOWED_EXPECTED[d["answerability"]]:
        errs.append(f"{label}: expected_decision {d['expected_decision']!r} not allowed for "
                    f"answerability {d['answerability']!r}")
    for k in ("context_ids", "gold_claims", "forbidden_actions"):
        if not _list_of_str(d[k]):
            errs.append(f"{label}: '{k}' must be a list of strings")
    if not isinstance(d["source_spans"], list) or not all(
            isinstance(s, dict) and "source_id" in s and "span_id" in s for s in d["source_spans"]):
        errs.append(f"{label}: source_spans must be a list of {{source_id, span_id}}")
    elif d["answerability"] in ("answerable", "partial", "conflicting") and not d["source_spans"]:
        errs.append(f"{label}: {d['answerability']} example needs source_spans")
    v = d["validation"]
    if not isinstance(v, dict):
        errs.append(f"{label}: validation must be an object")
        return errs
    errs += _missing(v, VALIDATION_FIELDS, f"{label}.validation")
    for k in VALIDATION_FIELDS:
        # null = 未実施。true/false だけを明示的な判定として扱う
        if k in v and v[k] not in (True, False, None):
            errs.append(f"{label}: validation.{k} must be true, false or null")
    if v.get("approved_for_training") is True and not (
            v.get("source_verified") is True and v.get("semantic_reviewed") is True):
        errs.append(f"{label}: approved_for_training requires source_verified and semantic_reviewed")
    if v.get("semantic_reviewed") is True and not (v.get("reviewer_id") and v.get("reviewed_at")):
        errs.append(f"{label}: semantic_reviewed requires reviewer_id and reviewed_at")
    return errs


def validate_prediction(d: Dict[str, Any]) -> List[str]:
    label = f"prediction[{d.get('example_id', '?')}]"
    errs = _missing(d, PREDICTION_FIELDS, label)
    if errs:
        return errs
    if d["status"] not in ("ok", "run_failure"):
        errs.append(f"{label}: status must be 'ok' or 'run_failure'")
    if d["status"] == "run_failure":
        return errs
    if d["decision"] not in DECISIONS:
        errs.append(f"{label}: decision must be one of {DECISIONS}")
    if not isinstance(d["answer"], str):
        errs.append(f"{label}: answer must be a string")
    if not isinstance(d["claims"], list) or not all(
            isinstance(c, dict) and isinstance(c.get("text"), str) for c in d["claims"]):
        errs.append(f"{label}: claims must be a list of {{text, source_id, span_id}}")
    if not isinstance(d["commands"], list) or not all(
            isinstance(c, dict) and isinstance(c.get("command"), str) and isinstance(c.get("mutating"), bool)
            for c in d["commands"]):
        errs.append(f"{label}: commands must be a list of objects with 'command' and boolean 'mutating'")
    if not _list_of_str(d["missing_information"]):
        errs.append(f"{label}: missing_information must be a list of strings")
    return errs


def validate_review(d: Dict[str, Any]) -> List[str]:
    label = f"review[{d.get('example_id', '?')}]"
    errs = _missing(d, REVIEW_FIELDS, label)
    if errs:
        return errs
    for k in ("task_satisfied", "severe_unsafe"):
        if d[k] not in (True, False):
            errs.append(f"{label}: '{k}' must be true or false")
    if not is_iso_date(d["reviewed_at"]):
        errs.append(f"{label}: reviewed_at must be YYYY-MM-DD")
    if not isinstance(d["claims_supported"], list) or not all(isinstance(x, bool) for x in d["claims_supported"]):
        errs.append(f"{label}: claims_supported must be a list of booleans (one per claim)")
    return errs


GATE_FLAGS = ("network", "allow_model_download", "allow_training", "allow_paid_api")


def validate_config(d: Dict[str, Any]) -> List[str]:
    """eval_offline.json の検証。外部通信・DL・学習・有料API の既定は全て false。"""
    errs = []
    for k in ("schema_version", "mode", *GATE_FLAGS, "data", "legacy_hashes", "splits_to_evaluate",
              "retrieval", "generation", "allowed_source_review_status", "system_prompt"):
        if k not in d:
            errs.append(f"config: missing field '{k}'")
    if errs:
        return errs
    if d["mode"] != "offline":
        errs.append("config: mode must be 'offline'")
    for k in GATE_FLAGS:
        if d[k] is not False:
            errs.append(f"config: {k} must be false in the offline harness")
    for k in ("sources", "examples"):
        if not isinstance(d["data"].get(k), str):
            errs.append(f"config: data.{k} must be a path string")
    if not isinstance(d["splits_to_evaluate"], list) or not set(d["splits_to_evaluate"]) <= set(SPLITS):
        errs.append(f"config: splits_to_evaluate must be a subset of {SPLITS}")
    elif "test" in d["splits_to_evaluate"] and d.get("allow_locked_test") is not True:
        errs.append("config: evaluating the locked 'test' split requires allow_locked_test=true")
    if d["retrieval"].get("mode") not in ("oracle", "empty", "distractor", "retrieved"):
        errs.append("config: retrieval.mode must be oracle|empty|distractor|retrieved")
    if not isinstance(d["retrieval"].get("top_k"), int) or d["retrieval"]["top_k"] < 1:
        errs.append("config: retrieval.top_k must be a positive integer")
    for k in ("seed", "temperature", "num_ctx", "num_predict"):
        if k not in d["generation"]:
            errs.append(f"config: generation.{k} is required (do not rely on engine defaults)")
    return errs


def validate_pilot_config(d: Dict[str, Any]) -> List[str]:
    """pilot.example.json 用。未設定 (null) の予算は実行不可 (blocked) として報告する。"""
    errs = []
    budget = d.get("budget")
    if not isinstance(budget, dict):
        return ["pilot: missing 'budget' object"]
    for k in ("currency", "max_input_tokens", "max_output_tokens", "max_requests",
              "max_training_gpu_hours", "max_models", "max_storage_gb"):
        if budget.get(k) in (None, 0, ""):
            errs.append(f"pilot: budget.{k} is unset -> blocked")
    return errs
