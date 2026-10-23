"""
evaluate.py
オフライン評価ハーネス (外部通信なし・CPUのみ)。

  python src/evaluate.py --config configs/eval_offline.json --dry_run
  python src/evaluate.py --config configs/eval_offline.json --backend cached --output runs/offline-review/

* 入力が欠けていたり契約違反なら fail-closed (終了コード1)。縮小フォールバックはしない。
* 既存の出力先 (data/eval_results.json 等) は上書きしない。--output は存在しない/空のディレクトリのみ。
* 品質指標は人手レビューが無ければ unknown。fixture 実行から品質は主張しない。
* ollama バックエンドは config.network=true かつ --allow-network の両方が必要 (この基盤では未使用)。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from backends import (AbstainBackend, CachedResponsesBackend, OllamaBackend, RetrievalOnlyBackend,
                      parse_prediction)
from contracts import ContractError, load_jsonl, sha256_text
from metrics import compute_metrics, legacy_flags
from retrieval import build_corpus, get_context
from run_manifest import build_manifest, canonical_hash
from validate_data import ROOT, load_config, resolve, validate_all

# 旧評価器の known-bad 文字列 (補助フラグ専用。unsafe 判定には使わない)
GLOBAL_KNOWN_BAD_PATTERNS = [
    "revoke-instance-profile-credentials-permission", "diskpart /s bitlocker", "0x8037",
    "aws_securityhub_findings", "revoking-instance-profile", "state = \"disabled\"",
    "http-mode none", "rm-mountpoint", "delete-bucket", "--no-mfa-enabled",
]


def evaluate_response_quality(response_text: str, criteria: Dict[str, Any]) -> Dict[str, Any]:
    """legacy 文字列ヒューリスティック (data/eval.jsonl 用)。正しさの判定ではない。

    verified_correct は常に None。キーワードの位置・文脈・引数の整合は見ていない。
    """
    low = response_text.lower()
    checks = criteria.get("required_checks", [])
    passed = [c.get("name", "unnamed") for c in checks
              if all(k.lower() in low for k in c.get("all_of", []))
              and (not c.get("any_of") or any(k.lower() in low for k in c["any_of"]))]
    coverage = round(len(passed) / len(checks), 2) if checks else None
    bad = legacy_flags(response_text, list(set(GLOBAL_KNOWN_BAD_PATTERNS + criteria.get("forbidden_patterns", []))))
    return {
        "legacy_keyword_coverage": coverage,
        "legacy_blacklist_flags": bad,
        "verified_correct": None,
        "note": "string-heuristic only; not hallucination rate, executability or operational correctness",
    }


def _backend(name: str, config: Dict[str, Any], root: Path, allow_network: bool, model: str):
    if name == "abstain":
        return AbstainBackend(), "baseline"
    if name == "retrieval_only":
        return RetrievalOnlyBackend(), "baseline"
    if name == "cached":
        path = config["data"].get("cached_responses")
        if not path:
            raise ContractError("backend 'cached' requires data.cached_responses in config")
        return CachedResponsesBackend(resolve(root, path)), "cached"
    if name == "ollama":
        if config["network"] is not True or not allow_network:
            raise ContractError("ollama backend needs config.network=true AND --allow-network "
                                "(offline harness refuses by default)")
        return OllamaBackend(model, config["generation"]), "recorded"
    raise ContractError(f"unknown backend {name!r}")


def _check_output_dir(out: Path, root: Path):
    out = out.resolve()
    if out.is_file():
        raise ContractError(f"output path is an existing file, refusing to overwrite: {out}")
    if out.exists() and any(out.iterdir()):
        raise ContractError(f"output directory is not empty, refusing to overwrite: {out}")
    data_dir = (root / "data").resolve()
    if data_dir == out or data_dir in out.parents:
        raise ContractError("refusing to write under data/ (legacy artifacts are immutable)")


def run(args) -> int:
    root = Path(args.root)
    config = load_config(resolve(root, args.config))
    errors, warnings = validate_all(config, root)
    for w in warnings:
        print(f"WARN: {w}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        print("evaluate: data/config validation failed (fail-closed); nothing was run")
        return 1

    sources = load_jsonl(resolve(root, config["data"]["sources"]))
    examples = [e for e in load_jsonl(resolve(root, config["data"]["examples"]))
                if e["split"] in config["splits_to_evaluate"]]
    if not examples:
        print("ERROR: no examples in the selected splits; refusing to run")
        return 1
    corpus = build_corpus(sources, config["allowed_source_review_status"])
    corpus_keys = {(d["source_id"], d["span_id"]) for d in corpus}
    reviews = {}
    if config["data"].get("reviews"):
        reviews = {r["example_id"]: r for r in load_jsonl(resolve(root, config["data"]["reviews"]))}

    backend, kind = _backend(args.backend, config, root, args.allow_network, args.model)
    rmode, top_k = config["retrieval"]["mode"], config["retrieval"]["top_k"]
    fixture = any(e["provenance"] == "fixture" for e in examples)
    result_kind = "fixture" if fixture and kind != "baseline" else kind

    print("=" * 50)
    print(" Offline Evaluation Harness")
    print("=" * 50)
    print(f"backend={args.backend} retrieval={rmode}(k={top_k}) splits={config['splits_to_evaluate']} "
          f"examples={len(examples)} result_kind={result_kind}")
    if args.dry_run:
        print("DRY RUN: contracts validated; no backend called, no files written, no metrics computed.")
        print("\nEvaluation Summary (dry run): OK")
        return 0
    if not args.output:
        print("ERROR: --output DIR is required for a real (offline) run")
        return 1
    out = Path(args.output)
    _check_output_dir(out, root)

    predictions, requests = [], []
    for ex in examples:
        ctx = get_context(ex, corpus, rmode, top_k)
        req = {"example": ex, "context": ctx, "system_prompt": config["system_prompt"],
               "generation": config["generation"],
               "prompt": ex["question"] + "".join(f"\n[{c['source_id']}#{c['span_id']}] {c['text']}" for c in ctx)}
        requests.append({"example_id": ex["example_id"], "prompt_hash": sha256_text(req["prompt"])})
        predictions.append(parse_prediction(ex["example_id"], backend.generate(req)))

    metrics = compute_metrics(examples, {p["example_id"]: p for p in predictions}, corpus_keys, reviews,
                              seed=config["generation"]["seed"])
    data_files = {k: resolve(root, v) for k, v in config["data"].items() if v}
    manifest = build_manifest(root, config, args.backend, result_kind, data_files,
                              [e["example_id"] for e in examples], canonical_hash(predictions),
                              canonical_hash(requests))
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n" for p in predictions), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    print("\n" + "=" * 50)
    print(" Evaluation Summary")
    print("=" * 50)
    if result_kind in ("fixture", "baseline"):
        print(f"NOTE: result_kind={result_kind}. Structural metrics only; NOT a quality or accuracy result.")
    print(f"run_failures: {metrics['run_failures']}/{metrics['n_examples']}")
    print(f"answerable_coverage (all requests): {metrics['answerable_coverage']['all_requests']}")
    print(f"false_answer_on_unanswerable (all requests): {metrics['false_answer_on_unanswerable']['all_requests']}")
    print(f"human-reviewed quality: {'unknown (no reviews)' if not metrics['review']['n_reviewed'] else metrics['review']}")
    print(f"saved to: {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Offline evaluation harness (no network, no GPU, no training)")
    ap.add_argument("--config", default="configs/eval_offline.json")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--backend", default="cached",
                    choices=["abstain", "retrieval_only", "cached", "ollama"])
    ap.add_argument("--model", default="", help="Ollama model name (ollama backend only)")
    ap.add_argument("--allow-network", action="store_true", help="required in addition to config.network=true")
    ap.add_argument("--output", default=None, help="new or empty output directory")
    ap.add_argument("--dry_run", "--dry-run", dest="dry_run", action="store_true",
                    help="validate only; writes nothing")
    args = ap.parse_args(argv)
    try:
        return run(args)
    except ContractError as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
