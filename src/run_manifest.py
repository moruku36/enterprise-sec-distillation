"""
run_manifest.py
実行の再現に必要な情報 (git / データ / 設定 / プロンプト / 生成条件のハッシュ) を記録する。
git の HEAD は .git を直接読む (サブプロセスを起動しない)。
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from contracts import SCHEMA_VERSION, sha256_text


def git_head(root: Path) -> str:
    try:
        head = (root / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = root / ".git" / head.split(" ", 1)[1]
            if ref.is_file():
                return ref.read_text(encoding="utf-8").strip()
            for line in (root / ".git" / "packed-refs").read_text(encoding="utf-8").splitlines():
                if line.endswith(head.split(" ", 1)[1]):
                    return line.split(" ", 1)[0]
            return "unknown"
        return head
    except OSError:
        return "unknown"


def canonical_hash(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False))


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def build_manifest(root: Path, config: Dict[str, Any], backend_name: str, result_kind: str,
                   data_files: Dict[str, Path], example_ids: List[str], prediction_hash: str,
                   requests_hash: str, review_run_hash: str, model_name: str) -> Dict[str, Any]:
    """content_hash は created_at を除く全項目から計算する (同一入力 → 同一ハッシュ)。"""
    body = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": git_head(root),
        "backend": backend_name,
        "result_kind": result_kind,  # fixture | cached | baseline | recorded
        "config_hash": canonical_hash(config),
        "data_hashes": {k: file_hash(p) for k, p in sorted(data_files.items()) if p and Path(p).is_file()},
        "system_prompt_hash": sha256_text(config["system_prompt"]),
        "generation": config["generation"],
        "retrieval": config["retrieval"],
        "n_examples": len(example_ids),
        "example_ids": example_ids,
        "requests_hash": requests_hash,
        "predictions_hash": prediction_hash,
        "review_run_hash": review_run_hash,
        "gates": {k: config[k] for k in ("network", "allow_model_download", "allow_training", "allow_paid_api")},
        "model": model_name or None, "tokenizer": None,
        "token_usage": None, "cost": None,
        "quality_claim": ("NONE: fixture/offline run; structural metrics only, not a quality result"
                          if result_kind in ("fixture", "baseline") else "see metrics.review (unknown if unreviewed)"),
    }
    return {**body, "content_hash": canonical_hash(body),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
