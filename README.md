# Enterprise Security LLM Distillation (Gemini 3.8 Flash -> Local OSS LLM)

エンタープライズ環境におけるサイバーセキュリティ防御に特化した知識蒸留（Distillation / Fine-Tuning）検証プロジェクト。
Teacher モデルとして **Gemini 3.8 Flash** を使用して高品質なセキュリティ防御データセット（問答ペア）を生成し、**QLoRA (4-bit)** を用いてローカルの小型 OSS LLM（生徒モデル）にファインチューニングを実施、最終的に **Ollama (GGUF)** でローカル推論・効果検証を行います。

---

## 1. 検証アーキテクチャ

```text
[Teacher: Gemini 3.8 Flash]
         │ (自動生成: 150〜300件)
         ▼
[Dataset: data/train.jsonl (Alpaca/ChatML形式)]
         │
         ▼ (QLoRA 4-bit Fine-Tuning)
[Student: Qwen2.5-3B / Llama-3.2-3B] (ローカル RTX 3070 8GB / Colab)
         │
         ▼ (Merge & GGUF Q4_K_M 変換)
[Ollama Local Engine: `sec-defense`]
         │
         ▼ (ベンチマーク・比較評価)
[Before vs After vs Teacher 評価]
```

---

## 2. 検証フェーズ (Roadmap)

### Phase 1: データセット設計・自動生成 (Teacher)
- **対象領域**:
  1. **SOC / SIEM / EDR 運用**: アラートトリアージ、KQL/SPLクエリ、インシデント検知
  2. **ゼロトラスト / IAM**: 条件付きアクセス、最小権限の原則、特権アクセス管理(PAM)
  3. **インシデント対応 (IR)**: ランサムウェア初動対応、証拠保全、フォレンジック
  4. **クラウドセキュリティ**: AWS/Azure のセキュリティ設計、GuardDuty、ポリシー実装
  5. **ネットワーク・境界防御**: マイクロセグメンテーション、SASE、横展開(Lateral Movement)対策
- **フォーマット**: Alpaca形式 (`instruction`, `input`, `output`)
- **生成スクリプト**: `src/generate_dataset.py` (Gemini API 使用)

### Phase 2: 生徒モデル選定と微調整 (Student Fine-Tuning)
- **ベースモデル候補**:
  - `Qwen/Qwen2.5-3B-Instruct` (日本語・技術系推論に強く、VRAM 8GBで安定動作)
  - `meta-llama/Llama-3.2-3B-Instruct`
- **学習構成**:
  - 手法: QLoRA (BitsAndBytes 4-bit NF4)
  - LoRA パラメータ: rank=16, alpha=32, target_modules=[q_proj, k_proj, v_proj, o_proj]
  - 学習スクリプト: `src/train_lora.py` (Hugging Face `trl` / `peft`)

### Phase 3: GGUF 変換 & Ollama 登録
- LoRA アダプターをベースモデルにマージ
- `llama.cpp` により GGUF 形式 (`q4_k_m`) に量子化変換
- Ollama `Modelfile` を定義し、ローカルモデルとしてビルド (`ollama create sec-defense -f Modelfile`)

### Phase 4: 蒸留効果の検証 (Evaluation)
- 未学習のインシデント対応シナリオ 10 問によるブラインドテスト
- 評価軸:
  1. **具体性**: ログ ID、設定値、クエリ等の出力精度
  2. **実務適合性**: 企業環境の業務影響・保全を考慮した推奨手順か
  3. **ベースモデルとの差異**: 微調整前 vs 微調整後 vs Gemini 3.8 Flash

---

## 3. ディレクトリ構成

```text
├── README.md               # プロジェクト概要・検証手順書
├── requirements.txt        # 依存ライブラリ
├── .gitignore
├── data/
│   ├── raw/                # Gemini API 生成の生データ
│   ├── train.jsonl         # 学習用データセット
│   └── eval.jsonl          # 評価用テストデータセット
├── src/
│   ├── generate_dataset.py # データセット自動生成スクリプト
│   ├── train_lora.py       # QLoRA 学習スクリプト
│   └── evaluate.py         # 評価・ベンチマークスクリプト
└── ollama/
    └── Modelfile           # Ollama 登録用設定ファイル
```

---

## 4. 実行環境
- OS: Windows 11 (WSL2 Ubuntu 推奨)
- GPU: NVIDIA GeForce RTX 3070 (VRAM 8GB)
- Framework: Python 3.10+, PyTorch, Hugging Face Transformers/PEFT/TRL, Ollama
