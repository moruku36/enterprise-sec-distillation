"""
validate_data.py
評価データ契約・設定・split・参照整合・legacy 成果物の不変性を検証する (オフライン、標準ライブラリのみ)。
エラーが1件でもあれば終了コード1 (fail-closed)。ニアデュプリケートは警告として人手確認に回す。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from contracts import (
    ContractError, load_jsonl, sha256_file_normalized,
    validate_config, validate_example, validate_review, validate_source,
)

ROOT = Path(__file__).resolve().parent.parent


def resolve(root: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else root / q


def _norm(text: str) -> str:
    return "".join(text.lower().split())


def _ngrams(text: str, n: int = 3) -> set:
    t = _norm(text)
    return {t[i:i + n] for i in range(max(len(t) - n + 1, 1))}


def load_config(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ContractError(f"config not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ContractError(f"{path}: invalid JSON: {e}")


def load_corpus(config: Dict[str, Any], root: Path) -> Tuple[List[dict], List[dict]]:
    return (load_jsonl(resolve(root, config["data"]["sources"])),
            load_jsonl(resolve(root, config["data"]["examples"])))


def check_legacy(config: Dict[str, Any], root: Path) -> List[str]:
    errs = []
    hp = resolve(root, config["legacy_hashes"])
    if not hp.is_file():
        return [f"legacy hash manifest not found: {hp}"]
    for rel, expected in json.loads(hp.read_text(encoding="utf-8")).items():
        f = resolve(root, rel)
        if not f.is_file():
            errs.append(f"legacy artifact missing: {rel}")
        elif sha256_file_normalized(f) != expected:
            errs.append(f"legacy artifact modified: {rel}")
    return errs


def validate_all(config: Dict[str, Any], root: Path = ROOT) -> Tuple[List[str], List[str]]:
    """(errors, warnings) を返す。"""
    errors = validate_config(config)
    warnings: List[str] = []
    if errors:
        return errors, warnings
    try:
        sources, examples = load_corpus(config, root)
    except ContractError as e:
        return [str(e)], warnings

    errors += check_legacy(config, root)
    allowed_status = set(config["allowed_source_review_status"])

    # --- sources ---
    src_spans: Dict[str, set] = {}
    for s in sources:
        errors += validate_source(s)
        sid = s.get("source_id")
        if sid in src_spans:
            errors.append(f"duplicate source_id {sid!r}")
        src_spans[sid] = {sp.get("span_id") for sp in s.get("spans", []) if isinstance(sp, dict)}
        if s.get("review_status") not in allowed_status:
            errors.append(f"source[{sid}]: review_status {s.get('review_status')!r} is not approved "
                          f"(allowed: {sorted(allowed_status)})")

    # --- examples ---
    seen_ids = set()
    family_splits: Dict[str, set] = {}
    for ex in examples:
        errs = validate_example(ex)
        errors += errs
        if errs:
            continue
        eid = ex["example_id"]
        if eid in seen_ids:
            errors.append(f"duplicate example_id {eid!r}")
        seen_ids.add(eid)
        family_splits.setdefault(ex["family_id"], set()).add(ex["split"])
        for sp in ex["source_spans"]:
            if sp["source_id"] not in src_spans:
                errors.append(f"example[{eid}]: unknown source_id {sp['source_id']!r}")
            elif sp["span_id"] not in src_spans[sp["source_id"]]:
                errors.append(f"example[{eid}]: unknown span {sp['source_id']}#{sp['span_id']}")
            elif sp["source_id"] not in ex["context_ids"]:
                errors.append(f"example[{eid}]: span source {sp['source_id']!r} missing from context_ids")
        for cid in ex["context_ids"]:
            if cid not in src_spans:
                errors.append(f"example[{eid}]: unknown context id {cid!r}")

    for fam, splits in sorted(family_splits.items()):
        if len(splits) > 1:
            errors.append(f"family {fam!r} spans multiple splits {sorted(splits)} (family leakage)")

    valid_examples = [e for e in examples if not validate_example(e)]

    # --- 重複・漏洩 ---
    by_norm: Dict[str, dict] = {}
    for ex in valid_examples:
        k = _norm(ex["question"])
        other = by_norm.get(k)
        if other and other["family_id"] != ex["family_id"]:
            errors.append(f"exact duplicate question across families: {other['example_id']} / {ex['example_id']}")
        by_norm.setdefault(k, ex)
    grams = {e["example_id"]: _ngrams(e["question"]) for e in valid_examples}
    for i, a in enumerate(valid_examples):
        for b in valid_examples[i + 1:]:
            if a["family_id"] == b["family_id"]:
                continue
            ga, gb = grams[a["example_id"]], grams[b["example_id"]]
            if len(ga | gb) and len(ga & gb) / len(ga | gb) >= 0.8:
                warnings.append(f"near-duplicate questions (needs human family review): "
                                f"{a['example_id']} / {b['example_id']}")
    span_texts = [sp.get("exact_text", "") for s in sources for sp in s.get("spans", []) if isinstance(sp, dict)]
    for ex in valid_examples:
        if ex["split"] != "test":
            continue
        probes = [ex["question"]] + ex["gold_claims"]
        if any(p and p in t for p in probes for t in span_texts):
            errors.append(f"example[{ex['example_id']}]: test question/gold claim appears verbatim in the "
                          f"retrieval corpus (answer leakage)")

    # --- 任意ファイル: cached responses / reviews ---
    ids = {e.get("example_id") for e in examples}
    data = config["data"]
    if data.get("cached_responses"):
        try:
            for row in load_jsonl(resolve(root, data["cached_responses"])):
                if row.get("example_id") not in ids:
                    errors.append(f"cached response for unknown example {row.get('example_id')!r}")
                if row.get("provenance") not in ("fixture", "cached", "recorded"):
                    errors.append(f"cached response {row.get('example_id')!r}: provenance must be "
                                  f"fixture|cached|recorded")
        except ContractError as e:
            errors.append(str(e))
    if data.get("reviews"):
        try:
            for row in load_jsonl(resolve(root, data["reviews"])):
                errors += validate_review(row)
                if row.get("example_id") not in ids:
                    errors.append(f"review for unknown example {row.get('example_id')!r}")
        except ContractError as e:
            errors.append(str(e))
    return errors, warnings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate offline evaluation data contracts (no network, no models)")
    ap.add_argument("--config", default="configs/eval_offline.json")
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args(argv)
    root = Path(args.root)
    try:
        config = load_config(resolve(root, args.config))
    except ContractError as e:
        print(f"FAIL: {e}")
        return 1
    errors, warnings = validate_all(config, root)
    for w in warnings:
        print(f"WARN: {w}")
    for e in errors:
        print(f"ERROR: {e}")
    print(f"validate_data: {len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
