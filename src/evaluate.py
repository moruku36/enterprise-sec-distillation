"""
evaluate.py
蒸留前後のモデル比較ベンチマークスクリプト。
未学習のインシデントシナリオを各モデルに投入し、出力品質・具体性を比較評価する。
"""

import os
import json
import argparse
import urllib.request
from dotenv import load_dotenv

load_dotenv()

TEST_PROMPTS = [
    {
        "id": "eval_01",
        "category": "Incident Response / Active Directory",
        "scenario": "社内LAN内の複数端末でBitLockerの不正暗号化を装うランサムウェアの挙動を検知しました。ドメインコントローラーへの侵入を防ぎつつ、感染拡大を阻止するための最初の30分で実施すべきトリアージ手順と、全社ネットワークレベルでの緊急遮断判断基準を具体的に提示してください。",
    },
    {
        "id": "eval_02",
        "category": "Cloud Security / AWS",
        "scenario": "AWS環境において、GuardDutyから「UnauthorizedAccess:IAMUser/InstanceCredentialExfiltration.OutsideAWS」アラートが発報されました。EC2インスタンスプロファイルの認証情報が外部に漏洩した疑いがあります。被害の局限化手順、侵害された認証情報の無効化コマンド、およびメタデータサービス（IMDSv2）強制適用のTerraformまたはCLIコマンドを提示してください。",
    },
]


def query_ollama(model: str, prompt: str) -> str:
    url = "http://127.0.0.1:11434/api/generate"
    data = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.8,
        }
    }).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            res = json.loads(response.read().decode("utf-8"))
            return res.get("response", "")
    except Exception as e:
        return f"Error querying Ollama ({model}): {e}"


def main():
    parser = argparse.ArgumentParser(description="Evaluate distilled model vs base model")
    parser.add_argument("--model", default="sec-defense", help="Distilled Ollama model name")
    parser.add_argument("--base_model", default="qwen2.5-coder:7b", help="Base Ollama model name for comparison")
    parser.add_argument("--output", default="data/eval_results.json", help="Evaluation output path")
    args = parser.parse_args()

    results = []
    print(f"=== Starting Model Evaluation ===")
    print(f"Target Distilled Model: {args.model}")
    print(f"Comparison Base Model:  {args.base_model}\n")

    for item in TEST_PROMPTS:
        print(f"--- Running Test Scenario: {item['id']} ({item['category']}) ---")
        prompt = item["scenario"]

        print(f"Querying distilled model '{args.model}'...")
        distilled_res = query_ollama(args.model, prompt)

        print(f"Querying baseline model '{args.base_model}'...")
        base_res = query_ollama(args.base_model, prompt)

        results.append({
            "test_id": item["id"],
            "category": item["category"],
            "prompt": prompt,
            "distilled_response": distilled_res,
            "base_response": base_res,
        })
        print("Done.\n")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Evaluation complete! Results saved to {args.output}")


if __name__ == "__main__":
    main()
