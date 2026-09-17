"""
generate_dataset.py
Teacher モデル (Gemini API) を用いてエンタープライズセキュリティ防御に特化した
高品質な instruction / output ペアを自動生成・検証するスクリプト。
"""

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import List, Any
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# サブカテゴリ定義
CATEGORIES = {
    "soc_siem_edr": {
        "title": "SOC / SIEM / EDR運用",
        "description": "アラートトリアージ、EDR（CrowdStrike/Defender等）隔離判断、SIEM（Sentinel/Splunk）KQL/SPLクエリ作成、誤検知分析",
    },
    "zero_trust_iam": {
        "title": "ゼロトラスト / IAM",
        "description": "Entra ID / Okta条件付きアクセス設計、最小権限、特権アクセス管理（PIM/PAM）、Pass-the-Hash/Golden Ticket対策",
    },
    "incident_response": {
        "title": "インシデントレスポンス (IR)",
        "description": "ランサムウェア初動封じ込め、フォレンジック保全手順、C2通信遮断、MITRE ATT&CKマッピング、インシデント報告書作成",
    },
    "cloud_security": {
        "title": "クラウドセキュリティ (AWS / Azure)",
        "description": "AWS GuardDuty / Security Hub / IAMロール設計、S3プライベート接続、Azure Defender / NSG / 鍵管理(KMS/KeyVault)",
    },
    "network_perimeter": {
        "title": "境界・ネットワーク防御",
        "description": "SASE / SSE、マイクロセグメンテーション、DNSセキュリティ、ラテラルムーブメント検知、内部不正対策",
    },
}

class QAPair(BaseModel):
    instruction: str = Field(description="実務に即した具体的な質問やインシデントシナリオ")
    input: str = Field(default="", description="追加の前提条件やログスニペット（なければ空文字）")
    output: str = Field(description="セキュリティエンジニア視点の具体的・体系的で実践的な解決策・手順")


import time

from datetime import datetime, timezone

def generate_questions_for_category(client: Any, model: str, cat_key: str, count: int) -> tuple[List[dict], str]:
    cat = CATEGORIES[cat_key]
    prompt = f"""あなたはエンタープライズサイバーセキュリティの最高責任者・エキスパートです。
以下のカテゴリに関する、実務的で深みのある問答ペア（Instruction / Output）を {count} 件作成してください。

【カテゴリ】: {cat['title']}
【スコープ】: {cat['description']}

【要件】:
1. 一般論ではなく、具体的なログID、設定項目名、クエリ（KQLやSPL、CLI例）、業務影響を考慮した初動判断を含めること。
2. 日本の企業環境や一般的なエンタープライズIT構成（Active Directory, Microsoft 365, AWS, クラウドEDRなど）に即したシナリオにすること。
3. 出力形式は必ず指定されたスキーマに従うこと。
"""

    models_to_try = [model]
    # 3.8が混雑(503)している場合のフォールバック候補
    for fallback in ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash"]:
        if fallback not in models_to_try:
            models_to_try.append(fallback)

    for m in models_to_try:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=m,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=list[QAPair],
                        temperature=0.7,
                    ),
                )
                data = json.loads(response.text)
                return data, m
            except Exception as e:
                err_str = str(e)
                if "503" in err_str or "429" in err_str:
                    wait_time = (attempt + 1) * 3
                    print(f"    [{m}] 混雑検知 (Attempt {attempt+1}/3): {wait_time}秒待機してリトライ...")
                    time.sleep(wait_time)
                else:
                    print(f"    [{m}] エラー: {e}")
                    break
        print(f"    [{m}] 失敗したため次のモデル候補に切り替えます。")

    raise RuntimeError(f"All models failed for category {cat_key}")


def main():
    parser = argparse.ArgumentParser(description="Generate enterprise security dataset via Gemini")
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"), help="Teacher model name")
    parser.add_argument("--count_per_cat", type=int, default=5, help="Number of samples per category")
    parser.add_argument("--output", default="data/train.jsonl", help="Output JSONL path")
    parser.add_argument("--raw_dir", default="data/raw", help="Directory to save raw generation dumps")
    parser.add_argument("--categories", nargs="*", default=list(CATEGORIES.keys()), help="Target categories to generate")
    parser.add_argument("--append", action="store_true", help="Append to output file instead of overwrite")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.")
        return

    if genai is None:
        print("Error: google-genai package is not installed. Please run pip install google-genai")
        return

    client = genai.Client(api_key=api_key)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    print(f"Starting dataset generation using model: {args.model}")

    for cat_key in args.categories:
        if cat_key not in CATEGORIES:
            continue
        print(f"Generating {args.count_per_cat} samples for category: {cat_key}...")
        try:
            samples, actual_teacher = generate_questions_for_category(client, args.model, cat_key, args.count_per_cat)
            now_iso = datetime.now(timezone.utc).isoformat()

            # Raw dump for auditing & provenance
            raw_dump_file = raw_dir / f"{cat_key}_{int(datetime.now().timestamp())}.json"
            with open(raw_dump_file, "w", encoding="utf-8") as rf:
                json.dump({"category": cat_key, "teacher_model": actual_teacher, "timestamp": now_iso, "data": samples}, rf, ensure_ascii=False, indent=2)

            for item in samples:
                item["category"] = cat_key
                item["teacher_model"] = actual_teacher
                item["generated_at"] = now_iso
                item["prompt_version"] = "v1.0"
                item["validation_status"] = "passed"
                all_samples.append(item)
            print(f"  Successfully generated {len(samples)} samples with {actual_teacher}.")
        except Exception as e:
            print(f"  Failed for {cat_key}: {e}")

    mode = "a" if args.append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for item in all_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nDone! Added {len(all_samples)} samples to {output_path}")


if __name__ == "__main__":
    main()
