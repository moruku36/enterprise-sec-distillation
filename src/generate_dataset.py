"""
generate_dataset.py
Teacher モデル (Gemini API) を用いてエンタープライズセキュリティ防御に特化した
高品質な instruction / output ペアを自動生成・検証するスクリプト。
"""

import os
import json
import argparse
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

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


def generate_questions_for_category(client: genai.Client, model: str, cat_key: str, count: int) -> List[dict]:
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

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[QAPair],
            temperature=0.7,
        ),
    )

    data = json.loads(response.text)
    return data


def main():
    parser = argparse.ArgumentParser(description="Generate enterprise security dataset via Gemini")
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), help="Teacher model name")
    parser.add_argument("--count_per_cat", type=int, default=10, help="Number of samples per category")
    parser.add_argument("--output", default="data/train.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.")
        return

    client = genai.Client(api_key=api_key)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_samples = []
    print(f"Starting dataset generation using model: {args.model}")

    for cat_key in CATEGORIES:
        print(f"Generating {args.count_per_cat} samples for category: {cat_key}...")
        try:
            samples = generate_questions_for_category(client, args.model, cat_key, args.count_per_cat)
            for item in samples:
                item["category"] = cat_key
                all_samples.append(item)
            print(f"  Successfully generated {len(samples)} samples.")
        except Exception as e:
            print(f"  Failed for {cat_key}: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nDone! Total {len(all_samples)} samples saved to {output_path}")


if __name__ == "__main__":
    main()
