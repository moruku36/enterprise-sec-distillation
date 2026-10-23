"""
retrieval.py
承認済み snippet だけを対象にした CPU・字句ベース (BM25) の検索。標準ライブラリのみ。

モード:
  oracle     人が選んだ必要スパン (example.source_spans)
  empty      根拠なし
  distractor oracle + 他 example の無関係スパン
  retrieved  質問文だけから BM25 で top-k (test のラベル・gold は一切参照しない)
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List

_WORD = re.compile(r"[A-Za-z0-9_\-\.]+")
_CJK = re.compile(r"[぀-ヿ㐀-鿿]+")


def tokenize(text: str) -> List[str]:
    """ASCII 語 + 日本語の文字 bigram。形態素解析器に依存しない。"""
    text = text.lower()
    toks = _WORD.findall(text)
    for run in _CJK.findall(text):
        toks += [run] if len(run) == 1 else [run[i:i + 2] for i in range(len(run) - 1)]
    return toks


def build_corpus(sources: List[Dict[str, Any]], allowed_status: List[str]) -> List[Dict[str, Any]]:
    """retrieval 利用が許可され、review_status が承認済みの source のスパンだけを索引化する。"""
    corpus = []
    for s in sources:
        if "retrieval" not in s["permitted_uses"] or s["review_status"] not in allowed_status:
            continue
        for sp in s["spans"]:
            corpus.append({"source_id": s["source_id"], "span_id": sp["span_id"],
                           "text": sp["exact_text"], "valid_as_of": s["valid_as_of"]})
    return corpus


def bm25_search(query: str, corpus: List[Dict[str, Any]], top_k: int, k1: float = 1.5, b: float = 0.75):
    docs = [Counter(tokenize(d["text"])) for d in corpus]
    lens = [sum(c.values()) for c in docs]
    avg = (sum(lens) / len(lens)) if lens else 0.0
    df: Counter = Counter()
    for c in docs:
        df.update(c.keys())
    n = len(docs)
    scored = []
    for i, c in enumerate(docs):
        score = 0.0
        for t in set(tokenize(query)):
            if t not in c:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            score += idf * c[t] * (k1 + 1) / (c[t] + k1 * (1 - b + b * lens[i] / (avg or 1)))
        if score > 0:
            scored.append((score, i))
    # 同点は corpus 順で安定化 (決定論的)
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [corpus[i] for _, i in scored[:top_k]]


def get_context(example: Dict[str, Any], corpus: List[Dict[str, Any]], mode: str, top_k: int):
    """モデル/ベースラインへ渡す根拠スパンを返す。retrieved は question のみ使用する。"""
    key = lambda d: (d["source_id"], d["span_id"])
    if mode == "empty":
        return []
    if mode == "retrieved":
        return bm25_search(example["question"], corpus, top_k)
    gold = {(s["source_id"], s["span_id"]) for s in example["source_spans"]}
    oracle = [d for d in corpus if key(d) in gold]
    if mode == "oracle":
        return oracle
    if mode == "distractor":
        others = [d for d in corpus if key(d) not in gold][:top_k]
        return oracle + others
    raise ValueError(f"unknown retrieval mode: {mode}")
