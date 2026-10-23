"""
backends.py
推論バックエンド。既定はすべてオフライン (外部通信・モデル取得なし)。

  AbstainBackend         全拒否ベースライン
  RetrievalOnlyBackend   検索スパンをそのまま返す生成なしベースライン (B3 相当)
  CachedResponsesBackend 保存済み応答の再生 (fixture / cached / recorded を provenance で区別)
  OllamaBackend          明示 opt-in のみ。この PR では実行しない

transport error / timeout / 空応答 / 不正 JSON は status="run_failure" とし、
回答テキストに埋め込まない (= 正しい保留として数えない)。
"""

import json
from typing import Any, Dict, Optional

from contracts import DECISIONS, load_jsonl, validate_prediction


class RawResult:
    def __init__(self, status: str, text: str = "", error: Optional[str] = None,
                 provenance: str = "unknown"):
        self.status = status  # "ok" | "error"
        self.text = text
        self.error = error
        self.provenance = provenance


def _output(decision, answer="", claims=None, commands=None, missing=None, as_of="") -> str:
    return json.dumps({"decision": decision, "answer": answer, "claims": claims or [],
                       "commands": commands or [], "missing_information": missing or [],
                       "as_of": as_of}, ensure_ascii=False)


def parse_prediction(example_id: str, raw: RawResult) -> Dict[str, Any]:
    """RawResult -> PredictionRecord。失敗は理由付きの run_failure。"""
    fail = lambda reason: {"example_id": example_id, "status": "run_failure", "decision": None,
                           "answer": "", "claims": [], "commands": [], "missing_information": [],
                           "as_of": "", "failure_reason": reason, "provenance": raw.provenance}
    if raw.status != "ok":
        return fail(raw.error or "backend_error")
    if not raw.text.strip():
        return fail("empty_response")
    try:
        obj = json.loads(raw.text)
    except json.JSONDecodeError:
        return fail("invalid_json")
    if not isinstance(obj, dict):
        return fail("invalid_json")
    rec = {"example_id": example_id, "status": "ok", "provenance": raw.provenance, **obj}
    rec["example_id"] = example_id
    rec["status"] = "ok"
    errs = validate_prediction(rec) if all(k in rec for k in ("decision", "answer")) else ["missing keys"]
    if errs or rec["decision"] not in DECISIONS:
        return fail("schema_invalid")
    # Positional IDs are scoped to this immutable prediction hash, not to an example alone.
    for i, claim in enumerate(rec["claims"], 1):
        claim.setdefault("claim_id", f"c{i}")
    return rec


class AbstainBackend:
    name = "abstain"

    def generate(self, request: Dict[str, Any]) -> RawResult:
        return RawResult("ok", _output("abstain", "根拠が不十分なため回答を保留します。"), provenance="baseline")


class RetrievalOnlyBackend:
    name = "retrieval_only"

    def generate(self, request: Dict[str, Any]) -> RawResult:
        ctx = request["context"]
        if not ctx:
            return RawResult("ok", _output("abstain", "関連する根拠が見つかりません。"), provenance="baseline")
        claims = [{"text": c["text"], "source_id": c["source_id"], "span_id": c["span_id"]} for c in ctx]
        as_of = max(c["valid_as_of"] for c in ctx)
        return RawResult("ok", _output("answer", "\n".join(c["text"] for c in ctx), claims, as_of=as_of),
                         provenance="baseline")


class CachedResponsesBackend:
    name = "cached"

    def __init__(self, path):
        self._rows = {r["example_id"]: r for r in load_jsonl(path)}

    def generate(self, request: Dict[str, Any]) -> RawResult:
        row = self._rows.get(request["example"]["example_id"])
        if row is None:
            return RawResult("error", error="no_cached_response", provenance="cached")
        prov = row.get("provenance", "cached")
        if row.get("status", "ok") != "ok":
            return RawResult("error", error=row.get("error", "backend_error"), provenance=prov)
        if "raw_text" in row:
            return RawResult("ok", row["raw_text"], provenance=prov)
        return RawResult("ok", json.dumps(row["response"], ensure_ascii=False), provenance=prov)


class OllamaBackend:
    """明示 opt-in 専用 (--allow-network かつ config.network=true)。この PR では実行しない。"""
    name = "ollama"

    def __init__(self, model: str, generation: Dict[str, Any], timeout: int = 120):
        self.model, self.generation, self.timeout = model, generation, timeout

    def generate(self, request: Dict[str, Any]) -> RawResult:
        import urllib.request  # 遅延 import: オフライン経路では読み込まない
        g = self.generation
        body = json.dumps({
            "model": self.model, "system": request["system_prompt"], "prompt": request["prompt"],
            "stream": False, "format": "json",
            "options": {"seed": g["seed"], "temperature": g["temperature"],
                        "num_ctx": g["num_ctx"], "num_predict": g["num_predict"],
                        **({"stop": g["stop"]} if g.get("stop") else {})},
        }).encode("utf-8")
        req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return RawResult("ok", json.loads(r.read().decode("utf-8")).get("response", ""),
                                 provenance="recorded")
        except Exception as e:  # noqa: BLE001 - transport 失敗は run_failure に分類する
            return RawResult("error", error=f"transport:{type(e).__name__}", provenance="recorded")
