"""
evaluate.py
蒸留モデルとベースモデルの比較評価ベンチマークスクリプト。
data/eval.jsonl から未学習インシデントシナリオを読み込み、
「具体性」「実務適合性」「正確性 / ハルシネーション検証」の多軸で評価・比較する。
"""

import os
import json
import argparse
import urllib.request
from pathlib import Path
from typing import List, Dict, Any
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_EVAL_FILE = "data/eval.jsonl"


def load_eval_scenarios(path: str) -> List[Dict[str, Any]]:
    scenarios = []
    eval_path = Path(path)
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    scenarios.append(json.loads(line))
        print(f"Loaded {len(scenarios)} evaluation scenarios from {path}")
    else:
        print(f"Warning: {path} not found. Using fallback scenarios.")
        scenarios = [
            {
                "id": "eval_01",
                "category": "incident_response",
                "instruction": "社内LAN内の複数端末でBitLockerの不正暗号化を装うランサムウェアの挙動を検知しました。最初の30分で実施すべきトリアージ手順と緊急遮断判断基準を提示してください。",
                "ground_truth_criteria": {
                    "required_concepts": ["EDRによる隔離", "ドメインコントローラー保護", "manage-bdeによる状態確認"],
                    "fact_check_notes": "正規コマンドは manage-bde。diskpart等によるキー破棄検証はハルシネーション。",
                },
            },
            {
                "id": "eval_02",
                "category": "cloud_security",
                "instruction": "AWS環境でGuardDuty「UnauthorizedAccess:IAMUser/InstanceCredentialExfiltration.OutsideAWS」アラート対応、認証情報無効化、IMDSv2強制適用のCLIを提示してください。",
                "ground_truth_criteria": {
                    "required_concepts": ["EC2通信遮断", "IAMセッション失効", "modify-instance-metadata-options"],
                    "fact_check_notes": "IMDSv2必須化は aws ec2 modify-instance-metadata-options。aws iam revoke-instance-profile-credentials-permission は存在しないAPI。",
                },
            },
        ]
    return scenarios


def query_ollama(model: str, prompt: str, timeout: int = 120) -> str:
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
        with urllib.request.urlopen(req, timeout=timeout) as response:
            res = json.loads(response.read().decode("utf-8"))
            return res.get("response", "")
    except Exception as e:
        return f"Error querying Ollama ({model}): {e}"


GLOBAL_KNOWN_BAD_PATTERNS = [
    "revoke-instance-profile-credentials-permission",
    "diskpart /s bitlocker",
    "0x8037",
    "aws_securityhub_findings",
    "revoking-instance-profile",
    "state = \"disabled\"",
    "http-mode none",
    "rm-mountpoint",
    "delete-bucket",
    "--no-mfa-enabled",
]


def evaluate_response_quality(response_text: str, criteria: Dict[str, Any]) -> Dict[str, Any]:
    """
    回答テキストに対して、厳格な required_checks (all_of / any_of) の充足度、
    コマンド構文の存在、および known-bad / forbidden パターンの検出を行いスコアリングする。
    """
    resp_lower = response_text.lower()

    # 1. 概念・要件チェック (required_checks)
    required_checks = criteria.get("required_checks", [])
    passed_checks = []
    failed_checks = []

    if required_checks:
        for check in required_checks:
            name = check.get("name", "unnamed_check")
            all_of = check.get("all_of", [])
            any_of = check.get("any_of", [])

            # all_of: すべての単語が含まれている必要がある
            all_ok = all(kw.lower() in resp_lower for kw in all_of) if all_of else True
            # any_of: いずれか1つが含まれていればOK
            any_ok = any(kw.lower() in resp_lower for kw in any_of) if any_of else True

            if all_ok and any_ok:
                passed_checks.append(name)
            else:
                failed_checks.append(name)
        total_checks = len(required_checks)
        coverage_score = round(len(passed_checks) / total_checks, 2) if total_checks else 1.0
    else:
        # 古いスキーマ (required_concepts) のフォールバック
        legacy_req = criteria.get("required_concepts", [])
        hit = [c for c in legacy_req if any(kw.lower() in resp_lower for kw in c.split("/"))]
        coverage_score = round(len(hit) / len(legacy_req), 2) if legacy_req else 1.0
        passed_checks = hit
        failed_checks = [c for c in legacy_req if c not in hit]
        total_checks = len(legacy_req)

    # 2. 禁止パターン・Known-bad パターン検出
    scenario_forbidden = criteria.get("forbidden_patterns", [])
    all_bad_candidates = list(set(GLOBAL_KNOWN_BAD_PATTERNS + scenario_forbidden))
    detected_bad = [bad for bad in all_bad_candidates if bad.lower() in resp_lower]

    # 3. 実コマンド構文の存在
    has_code_or_commands = any(marker in response_text for marker in [
        "```", "aws ", "az ", "Get-", "gpupdate", "manage-bde", "netstat", "ss ", "tcpdump", "sudo ", "Connect-MgGraph", "modify-instance-metadata-options"
    ])

    # 4. 自動判定スコアリング (1〜5点)
    if detected_bad:
        auto_accuracy = 2  # 誤コマンド・非実在APIが含まれる場合は上限2点
        auto_status = "KNOWN_BAD_DETECTED"
    elif coverage_score >= 0.75:
        auto_accuracy = 5 if has_code_or_commands else 4
        auto_status = "PASS_STRICT" if has_code_or_commands else "PASS_GENERIC"
    elif coverage_score >= 0.5:
        auto_accuracy = 3
        auto_status = "PASS_PARTIAL"
    else:
        auto_accuracy = 2
        auto_status = "INSUFFICIENT"

    auto_operational = 4 if (has_code_or_commands and not detected_bad and coverage_score >= 0.5) else (
        2 if detected_bad else (3 if coverage_score >= 0.5 else 2)
    )

    return {
        "coverage_score": coverage_score,
        "command_presence": has_code_or_commands,
        "known_bad_pattern_count": len(detected_bad),
        "detected_bad_patterns": detected_bad,
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "total_checks": total_checks,
        "auto_accuracy_score": auto_accuracy,
        "auto_operational_score": auto_operational,
        "auto_validation_status": auto_status,
        "manual_review": None,
        "fact_check_notes": criteria.get("fact_check_notes", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate distilled model vs base model with fact-checking")
    parser.add_argument("--model", default="sec-defense", help="Distilled Ollama model name (e.g., sec-defense)")
    parser.add_argument("--base_model", default="qwen2.5:3b", help="Base Ollama model name for comparison (default: qwen2.5:3b)")
    parser.add_argument("--eval_file", default=DEFAULT_EVAL_FILE, help="Path to evaluation questions (JSONL)")
    parser.add_argument("--output", default="data/eval_results.json", help="Evaluation output path")
    parser.add_argument("--dry_run", action="store_true", help="Run without Ollama invocation (validates prompt loading and schema)")
    args = parser.parse_args()

    scenarios = load_eval_scenarios(args.eval_file)

    print("==================================================")
    print(" Enterprise Security Model Evaluation Benchmark   ")
    print("==================================================")
    print(f"Target Distilled Model: {args.model}")
    print(f"Baseline Comparison:    {args.base_model}")
    print(f"Scenarios to evaluate:  {len(scenarios)}")
    print(f"Output Destination:     {args.output}\n")

    results = []
    category_stats: Dict[str, Dict[str, int]] = {}

    for i, item in enumerate(scenarios, 1):
        s_id = item.get("id", f"eval_{i:02d}")
        cat = item.get("category", "general")
        prompt = item.get("instruction", "")
        if item.get("input"):
            prompt += f"\n\n【前提・ログ】\n{item['input']}"

        criteria = item.get("ground_truth_criteria", {})

        print(f"[{i}/{len(scenarios)}] Testing {s_id} ({cat})...")

        if args.dry_run:
            distilled_res = f"[DRY_RUN] Simulated distilled output for {s_id}"
            base_res = f"[DRY_RUN] Simulated base output for {s_id}"
        else:
            print(f"  - Querying distilled ({args.model})...")
            distilled_res = query_ollama(args.model, prompt)
            print(f"  - Querying baseline ({args.base_model})...")
            base_res = query_ollama(args.base_model, prompt)

        distilled_eval = evaluate_response_quality(distilled_res, criteria)
        base_eval = evaluate_response_quality(base_res, criteria)

        results.append({
            "test_id": s_id,
            "category": cat,
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "references": item.get("references", []),
            "distilled_model": args.model,
            "distilled_response": distilled_res,
            "distilled_evaluation": distilled_eval,
            "base_model": args.base_model,
            "base_response": base_res,
            "base_evaluation": base_eval,
        })

        if cat not in category_stats:
            category_stats[cat] = {"count": 0}
        category_stats[cat]["count"] += 1

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n==================================================")
    print(" Evaluation Summary")
    print("==================================================")
    print(f"Total Test Cases: {len(results)}")
    print("Categories evaluated:")
    for cat, stat in category_stats.items():
        print(f"  - {cat}: {stat['count']} scenarios")
    print(f"\nDetailed benchmark results saved to: {args.output}")


if __name__ == "__main__":
    main()
