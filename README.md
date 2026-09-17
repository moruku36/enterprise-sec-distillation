# Enterprise Security LLM Distillation (Gemini 3.8 Flash -> Qwen2.5-3B)

![Distillation Concept](docs/distillation-concept.jpg)

エンタープライズ環境におけるサイバーセキュリティ防御に特化した、**Teacher-generated Synthetic Data Distillation（合成データによる指示チューニング / 蒸留）** の技術検証リポジトリです。

フロンティアモデル（Teacher: **Gemini 3.8 Flash**）から実践的な防御手順・ログ解析・設定指示の問答ペアを合成抽出し、ローカルの小型オープンモデル（Student: **Qwen2.5-3B-Instruct**）へ **QLoRA (4-bit)** を用いて知識を転移。マージおよび **GGUF (Q8_0)** 変換を経て、ローカル環境（**Ollama**）での実行および多軸ベンチマーク評価を行います。

> [!NOTE]
> **Inspiration / Acknowledgment**:  
> 本検証は、[@furuhashilab](https://github.com/furuhashilab) 氏による [KnowledgeDistillation2026](https://github.com/furuhashilab/KnowledgeDistillation2026) に着想を得て、エンタープライズセキュリティ防御領域への適用・再現性向上を目的として実施されました。

---

## 1. TL;DR

- **手法**: Teacherモデル（Gemini 3.8 Flash）による高品質合成データ生成 ＋ QLoRA (4-bit) 指示チューニング ＋ GGUF (Q8_0) ローカル配備
- **対象ドメイン**: SOC/SIEM/EDR、ゼロトラスト/IAM、インシデント対応、クラウドセキュリティ、境界防御
- **計算資源**: 単一コンシューマGPU（NVIDIA GeForce RTX 3070 8GB VRAM）で完結
- **主結果**:
  - 小型モデル（3B）でありながら、一般的な概念論にとどまっていたベースモデルに対し、実務コマンド（PowerShell, AWS CLI, KQL等）を積極的に提示する傾向が増加（出力特性の変化）。
  - 一方で、小型モデル特有の「堂々としたハルシネーション（非実在コマンドの生成や設定の誤認）」も観察され、小型モデルの現場適用におけるファクトチェックとプロンプトガードレールの重要性が浮き彫りとなりました。

---

## 2. アーキテクチャ & データフロー

![Enterprise Security Distillation Flow](docs/architecture-flow.jpg)

```text
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 1: 知識抽出 & データセット生成 (Teacher: Gemini 3.8 Flash)        │
│  - 5カテゴリのセキュリティ防御シナリオをスキーマ検証付きで生成           │
│  - プロベナンスメタデータ (モデル・生成日時・検証ステータス) を付与      │
│  - 出力: `data/raw/` (生データ) ➔ `data/train.jsonl`                   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│ Phase 2: QLoRA 指示チューニング (Student: Qwen2.5-3B-Instruct)          │
│  - 4-bit NF4 量子化 + PEFT (rank=16, alpha=32)                         │
│  - ChatML 形式にテンプレート適用 (`src/train_lora.py`)                   │
│  - 出力: `models/lora_adapter`                                         │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│ Phase 3: 重みマージ & GGUF (Q8_0) 量子化                                │
│  - ベースモデルとLoRAをCPU/GPUでマージ (`src/merge_lora.py`)            │
│  - llama.cpp により GGUF (Q8_0) 変換 ➔ Ollama ローカル配備             │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│ Phase 4: 多軸評価ベンチマーク (`src/evaluate.py`)                       │
│  - 未学習シナリオ 10 問 (`data/eval.jsonl`) によるブラインド比較       │
│  - 比較対象: ベースモデル (Qwen2.5-3B) vs 蒸留モデル (sec-defense:3B)  │
│  - 評価軸: ①具体性 ②実務適合性 ③正確性・ハルシネーション検証         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Quick Start

### 前提条件
- Python 3.10+
- NVIDIA GPU（VRAM 8GB以上推奨）※推論・評価のみならCPU環境やOllamaのみでも実行可能
- [Ollama](https://ollama.com/)（ローカル推論検証を行う場合）

### セットアップ
```bash
git clone https://github.com/moruku36/enterprise-sec-distillation.git
cd enterprise-sec-distillation

# 仮想環境の作成と依存関係のインストール
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 環境変数の設定 (データセット生成時のみ必要)
cp .env.example .env
# .env 内の GEMINI_API_KEY を設定
```

### テスト実行 (スキーマ検証 & スモークテスト)
```bash
python -m unittest discover -s tests -v
```

---

## 4. データセット & プロベナンス (Dataset & Provenance)

データセットは以下の5領域から実務的なインシデントシナリオを網羅するように設計されています。

| カテゴリ | 対象スコープ |
| :--- | :--- |
| `soc_siem_edr` | アラートトリアージ、EDR隔離判断、Microsoft Sentinel (KQL) / Splunk (SPL) クエリ |
| `zero_trust_iam` | Entra ID / Okta条件付きアクセス、最小権限、PAM、Pass-the-Hash / Kerberoasting対策 |
| `incident_response` | ランサムウェア初動対応、動的証拠保全 (Live Response)、C2遮断、MITRE ATT&CK連携 |
| `cloud_security` | AWS (GuardDuty, IAM, IMDSv2), Azure (Defender, Storage Private Endpoint, NSG) |
| `network_perimeter` | DNSトンネリング検知、RPZシンクホール化、SSL-VPN脆弱性対応、マイクロセグメンテーション |

### データ構造と監査性 (Provenance)
生成スクリプト（`src/generate_dataset.py`）により、すべてのレコードに生成元モデル・生成日時・バリデーションステータスが付与されます。
```json
{
  "instruction": "社内Active Directory環境に対するKerberoasting攻撃を検知するKQLクエリを作成してください。",
  "input": "ログソース: SecurityEvent (Event ID 4769), 暗号化タイプ: 0x17",
  "output": "【1. 検知用KQLクエリ】\nSecurityEvent | where EventID == 4769 ...",
  "category": "soc_siem_edr",
  "teacher_model": "gemini-3.8-flash",
  "generated_at": "2026-09-17T06:13:00Z",
  "prompt_version": "v1.0",
  "validation_status": "passed"
}
```

- 生成実行コマンド:
  ```bash
  python src/generate_dataset.py --model gemini-3.8-flash --count_per_cat 5 --output data/train.jsonl
  ```

---

## 5. 学習 (QLoRA Fine-Tuning)

VRAM 8GB（RTX 3070）環境で効率的に学習するため、BitsAndBytes 4-bit NF4 量子化と LoRA を組み合わせています。

- **Base Model**: `Qwen/Qwen2.5-3B-Instruct`
- **LoRA設定**: `r=16`, `lora_alpha=32`, `target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
- **シーケンス長**: 1,536 tokens

```bash
python src/train_lora.py \
  --base_model Qwen/Qwen2.5-3B-Instruct \
  --data_path data/train.jsonl \
  --output_dir models/lora_adapter \
  --epochs 3 \
  --batch_size 1 \
  --grad_accum 4 \
  --lr 2e-4
```

---

## 6. GGUF 変換 & Ollama 配備

学習済み LoRA 重みをベースモデルにマージし、`llama.cpp` を利用して `Q8_0` 形式へ量子化して Ollama に登録します。

### 1. LoRA重みのマージ
```bash
python src/merge_lora.py \
  --base_model Qwen/Qwen2.5-3B-Instruct \
  --adapter_dir models/lora_adapter \
  --output_dir models/merged_model
```

### 2. GGUF (FP16 ➔ Q8_0) 変換
```bash
# llama.cpp リポジトリのスクリプトを使用
python convert_hf_to_gguf.py models/merged_model --outfile models/sec-defense-f16.gguf

# Q8_0 量子化 (品質と推論速度のバランスを重視)
llama-quantize models/sec-defense-f16.gguf models/sec-defense-q8_0.gguf Q8_0
```

### 3. Ollama への登録
```bash
ollama create sec-defense -f ollama/Modelfile
```

---

## 7. 評価手法 (Evaluation Methodology)

モデルの評価は、学習データセットに含まれない独立した 10 問の未学習シナリオ（`data/eval.jsonl`）を用いて、**「ルールベース厳格チェック ＋ Known-Bad パターン検知」** による自動評価を実施します。

各シナリオには公式仕様（AWS API リファレンス、Microsoft Learn、IETF RFC、MITRE ATT&CK 等）に基づく根拠情報（`references`）と、以下の検証ロジックを紐づけています。

### 評価項目・メトリクス
1. **概念・要件網羅性 (`coverage_score`)**:
   - 各シナリオで必須となる対応（例: EDR隔離、正規CLI実行、IMDSv2必須化等）について、単語の単一ヒットではなく `all_of`（必須全含有）および `any_of`（いずれか含有）条件で厳格判定（0.0〜1.0）。
2. **実務コマンド含有 (`command_presence`)**:
   - 概念論にとどまらず、現場で実行可能なコマンド構文（CLI, PowerShell, KQL等）を含んでいるか（Boolean）。
3. **Known-Bad パターン検知 (`known_bad_pattern_count`)**:
   - 非実在のAWS CLIオプション（例: `delete-bucket` による無関係な破壊、`rm-mountpoint`, `--no-mfa-enabled`）や誤った復旧コマンド（BitLockerに対する `diskpart /s` 等）を検出。
4. **自動判定スコア (`auto_accuracy_score` / `auto_operational_score`)**:
   - 正確性（1〜5点）：Known-Badパターンが検出された場合は上限2点に減点、厳格チェックの充足率に応じてスコアリング。
   - 運用性（1〜5点）：安全なコマンド提示と実務適合性に基づく自動評価。
5. **判定ステータス (`auto_validation_status`)**:
   - `PASS_STRICT`（厳格要件を満たし実コマンドを含む）
   - `PASS_PARTIAL`（部分的な要件充足）
   - `KNOWN_BAD_DETECTED`（非実在・危険パターンの検知）
   - `INSUFFICIENT`（要件未達）

### 評価スクリプト実行
比較基準（Baseline）は、学習元である同一サイズのベースモデル **`qwen2.5:3b`** に固定しています。
```bash
# 実機評価実行 (Ollama で推論を実行し、data/eval_results.json に保存)
python src/evaluate.py --model sec-defense --base_model qwen2.5:3b --output data/eval_results.json

# ドライラン (モデル呼び出しを行わず、設問ロードとスキーマ検証のみ実施)
python src/evaluate.py --dry_run
```

---

## 8. ベンチマーク結果 (Benchmark Results)

全10問の未学習シナリオに対する実機推論評価（詳細は [`data/eval_results.json`](data/eval_results.json) 参照）の集計サマリーです。

### 全10シナリオ総合集計 (Overall Benchmark Summary)

| 評価指標 | ベースモデル (`qwen2.5:3b`) | 蒸留モデル (`sec-defense:3B`) | 分析・出力特性の変化 |
| :--- | :---: | :---: | :--- |
| **平均要件充足度 (`coverage_score`)** | 0.46 | **0.53** | 必須要件チェックの充足率は微増 |
| **実コマンド提示率 (`command_presence`)** | 50.0% (5/10) | **70.0% (7/10)** | 概念論からコマンドブロック提示への明らかなシフト |
| **Known-Bad（非実在/危険構文）検知率** | **0.0% (0/10)** | 10.0% (1/10) | もっともらしい非実在・破壊的コマンドの混入リスクが増加 |
| **要件合格率 (`PASS_STRICT` + `PASS_PARTIAL` + `PASS_GENERIC`)** | 50.0% (5/10) | **60.0% (6/10)** | 全体的な回答構成は改善傾向 |
| **平均正確性スコア (`auto_accuracy_score` 1〜5)** | 2.80 | **3.00** | 出力形式はプロ的になるが、正確性の絶対値は発展途上 |

> [!NOTE]
> この集計結果は、**「少量SFTにより回答の具体性やコマンド出力意欲は向上するが、セキュリティ特有の正確性・事実性の保証には至らず、非実在コマンドの生成リスクを伴う」** という本PoCの主要発見を裏付けています。

### 代表的なテストケース比較

### Case 1: Active Directory × BitLocker ランサムウェア初動対応 (`eval_01`)
- **質問**: 社内LANでBitLockerの不正暗号化を装うランサムウェアを検知。最初の30分で実施すべきトリアージ手順と緊急遮断判断基準を提示せよ。
- **ベースモデル (`qwen2.5:3b`)**:
  - `coverage_score: 0.25`, `command_presence: False`, `auto_validation_status: INSUFFICIENT`
  - 「エンドポイント間の通信状況確認」「ドメインコントローラーへの不正アクセス確認」など教科書的な概念論に終始。正規コマンド `manage-bde` の提示はなし。
- **蒸留モデル (`sec-defense:3B`)**:
  - `coverage_score: 0.00`, `command_presence: False`, `auto_validation_status: INSUFFICIENT`
  - Event ID（4625, 4760）などの専門用語は出現するものの、正規のBitLocker検証コマンドは提示できず、「如月」といった架空ツールや曖昧な遮断基準に言及。

### Case 2: AWS GuardDuty インスタンス認証情報漏洩 (`eval_02`)
- **質問**: GuardDuty「InstanceCredentialExfiltration.OutsideAWS」検知時の被害局限化、セッション無効化、IMDSv2強制適用手順。
- **ベースモデル (`qwen2.5:3b`)**:
  - `coverage_score: 0.00`, `command_presence: True`, `auto_validation_status: INSUFFICIENT`
  - `modify-instance-metadata-options` を提示するも、`--http-mode none` という非実在オプションを捏造。
- **蒸留モデル (`sec-defense:3B`)**:
  - `coverage_score: 0.00`, `command_presence: True`, `known_bad_pattern_count: 3`, `auto_validation_status: KNOWN_BAD_DETECTED`
  - コマンドラインの出力意欲は極めて高いが、`aws efs rm-mountpoint`, `aws s3api delete-bucket`, `--no-mfa-enabled` といった無関係かつ非実在の危険コマンドを多数生成。IMDSv2でも必須となる `--http-tokens required` が欠落。

---

## 9. 考察と既知の制約 (Findings & Limitations)

本検証により、**「少量の合成データによる小型LLM（3Bクラス）の指示チューニング」** における極めて重要な技術的教訓が浮き彫りとなりました。

### 1. 「具体性の向上」と「堂々とした誤謬」のトレードオフ
- **現象**: わずか 25 件の SFT によって、モデルは「セキュリティエンジニアらしい口調」や「コマンドブロックを積極的に出力する姿勢」を迅速に獲得しました。
- **リスク**: 一方で、専門用語の獲得に対して事実性の裏付け（正確なAPI仕様やオプション構文の記憶）が追いつかず、**「ベースモデル以上に堂々と間違ったコマンド（非実在CLI・危険な削除操作）を出力する」** 傾向が顕著に現れました。

### 2. 評価器（Evaluator）の重要性
- 単純な「キーワード検索（単語のOR一致）」で評価した場合、誤った回答であっても高いカバレッジスコアが付与されてしまう脆弱性が確認されました。
- セキュリティやインフラのような正確性が最優先されるドメインでは、**`all_of` による必須オプションの完全一致検証** や、**Known-Bad パターン（非実在構文・破壊的コマンド）の静的検出** を評価器に組み込むことが不可欠です。

### 3. 本PoCの結論と運用アーキテクチャへの示唆
「少量のSynthetic SFTで小型オープンモデルを即戦力の専門家化できる」という初期仮説は、**正確性の観点では成り立たない** ことが実証されました。小型モデルを実務に組み込む場合は、以下の防衛的アーキテクチャが前提となります：
1. **コマンド生成のホワイトリスト静的構文解析**: 生成された CLI/KQL を実行前にパーサー（AST / OpenAPI仕様）で構文検証する。
2. **フロンティアモデル（Gemini等）による二重検証 (Dual-LLM Guardrail)**: 一次案作成はローカル小型LLM、最終的な構文・リスク検証はTeacher側で行う。
3. **正確性を重視した選別的アライメント**: DPO (Direct Preference Optimization) 等による非実在構文に対するペナルティ学習。

---

## 10. ディレクトリ構成

```text
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions (スキーマ検証 & スモークテスト)
├── README.md                   # プロジェクト概要・検証手順書
├── requirements.txt            # 依存ライブラリ (バージョン固定)
├── .env.example                # 環境変数テンプレート
├── .gitignore
├── LICENSE                     # MIT License
├── data/
│   ├── raw/                    # 教師モデル生成の生ログ保管用 (.gitkeep)
│   ├── train.jsonl             # 学習用合成データセット (Provenance付き)
│   ├── eval.jsonl              # 評価用ベンチマークデータセット (10シナリオ)
│   └── eval_results.json       # 蒸留モデル vs ベースモデル評価結果
├── docs/
│   ├── architecture-flow.jpg   # アーキテクチャ図
│   └── distillation-concept.jpg# 蒸留コンセプト図
├── ollama/
│   └── Modelfile               # Ollama配備用設定 (Q8_0 GGUF指定)
├── src/
│   ├── generate_dataset.py     # 教師モデルからの合成データ生成
│   ├── train_lora.py           # QLoRA (4-bit) 学習スクリプト
│   ├── merge_lora.py           # LoRA重みマージスクリプト
│   └── evaluate.py             # 多軸ベンチマーク & ハルシネーション評価
└── tests/
    ├── test_dataset_schema.py  # データセットスキーマ & プロベナンステスト
    └── test_smoke.py           # スクリプトCLIスモークテスト
```

---

## 11. 今後の課題 (Future Work)

1. **Dual-Teacher 知識蒸留**:
   - 単一のTeacherだけでなく、複数モデルの相互レビュー（Generator-Critic）によるデータセット生成時のハルシネーション事前フィルタリング。
2. **Command Verification Tool-Use (RL / DPO)**:
   - CLI/API構文の正確性を報酬とした DPO (Direct Preference Optimization) によるハルシネーション抑制。
3. **他小型アーキテクチャとの比較**:
   - `Llama-3.2-3B` や `Gemma-2-2.6B` を用いた同一データセットでの汎化性能・ハルシネーション傾向の比較検証。

---

## 12. ライセンス & 免責事項 (License & Disclaimer)

- **License**: 本リポジトリのコードおよびデータセットは [MIT License](LICENSE) の下で公開されています。
- **Disclaimer**: 本リポジトリに含まれるデータセットおよび生成例は、すべて教育・検証を目的とした合成データです。本モデルの出力を事前検証なしに本番環境で実行することによる損害について、作者は一切の責任を負いません。
