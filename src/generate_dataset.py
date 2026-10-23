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
from contracts import ContractError
from stage_gate import approved, new_output, pilot_config

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

from datetime import datetime, timezone

def generate_questions_for_category(client: Any, model: str, cat_key: str, count: int,
                                    input_budget: int, output_budget: int) -> tuple[List[dict], str]:
    from pydantic import BaseModel, Field
    from google.genai import types

    class QAPair(BaseModel):
        instruction: str = Field(description="Security question")
        input: str = Field(default="", description="Additional context")
        output: str = Field(description="Proposed answer pending human review")

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

    # UTF-8 bytes conservatively bound token count for the approved prompt.
    if len(prompt.encode("utf-8")) > input_budget:
        raise ContractError("prompt exceeds input-token budget")
    response = client.models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=list[QAPair], temperature=0.7,
            max_output_tokens=output_budget,
        ),
    )
    return json.loads(response.text), model


def main():
    parser = argparse.ArgumentParser(description="Generate enterprise security dataset via Gemini")
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"), help="Teacher model name")
    parser.add_argument("--count_per_cat", type=int, default=5, help="Number of samples per category")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--raw_dir", required=True, help="Directory to save raw generation dumps")
    parser.add_argument("--categories", nargs="*", default=list(CATEGORIES.keys()), help="Target categories to generate")
    parser.add_argument("--config", default="configs/pilot.example.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--allow-paid-api", action="store_true")
    parser.add_argument("--approve-generation", action="store_true")
    args = parser.parse_args()
    if args.count_per_cat < 1 or not args.categories or any(c not in CATEGORIES for c in args.categories):
        parser.error("categories must be known and count_per_cat must be positive")
    try:
        output_path, raw_dir = new_output(args.output), new_output(args.raw_dir)
        if output_path == raw_dir or raw_dir in output_path.parents or output_path in raw_dir.parents:
            raise ContractError("output and raw directory must be separate")
    except ContractError as exc:
        parser.error(str(exc))
    if not args.execute:
        print("DRY RUN: no network, API call, or file write; execution requires a populated pilot config and explicit gates")
        return 0
    try:
        config = pilot_config(args.config)
        approved(config, "paid_generation", args.approve_generation)
        if not args.allow_network or not args.allow_paid_api:
            raise ContractError("generation requires --allow-network and --allow-paid-api")
        if not isinstance(config.get("teacher", {}).get("name"), str) or args.model != config["teacher"]["name"]:
            raise ContractError("teacher model differs from approved config")
        if args.count_per_cat * len(args.categories) > config["budget"]["max_requests"]:
            raise ContractError("requested samples exceed max_requests budget")
        per_output_budget = int(config["budget"]["max_output_tokens"] // len(args.categories))
        if per_output_budget < 1:
            raise ContractError("output-token budget too small")
    except ContractError as exc:
        parser.error(str(exc))

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.")
        return 1

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("Error: google-genai package is not installed")
        return 1

    client = genai.Client(api_key=api_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    print(f"Starting dataset generation using model: {args.model}")

    for cat_key in args.categories:
        print(f"Generating {args.count_per_cat} samples for category: {cat_key}...")
        try:
            samples, actual_teacher = generate_questions_for_category(
                client, args.model, cat_key, args.count_per_cat,
                int(config["budget"]["max_input_tokens"] // len(args.categories)), per_output_budget)
            now_iso = datetime.now(timezone.utc).isoformat()

            # Raw dump for auditing & provenance
            raw_dump_file = raw_dir / f"{cat_key}_{int(datetime.now().timestamp())}.json"
            with open(raw_dump_file, "x", encoding="utf-8") as rf:
                json.dump({"category": cat_key, "teacher_model": actual_teacher, "timestamp": now_iso, "data": samples}, rf, ensure_ascii=False, indent=2)

            for item in samples:
                item["category"] = cat_key
                item["teacher_model"] = actual_teacher
                item["generated_at"] = now_iso
                item["prompt_version"] = "v1.0"
                item["validation_status"] = "pending"
                all_samples.append(item)
            print(f"  Successfully generated {len(samples)} samples with {actual_teacher}.")
        except Exception as e:
            print(f"  Failed for {cat_key}: {e}")
            return 1

    with open(output_path, "x", encoding="utf-8") as f:
        for item in all_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nDone! Added {len(all_samples)} samples to {output_path}")


if __name__ == "__main__":
    raise SystemExit(main())
