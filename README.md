# Enterprise Security LLM Distillation (Gemini 3.8 Flash -> Local OSS LLM)

![Distillation Concept](docs/distillation-concept.jpg)

エンタープライズ環境におけるサイバーセキュリティ防御に特化した知識蒸留（Distillation / Fine-Tuning）検証プロジェクト。
Teacher モデルとして **Gemini 3.8 Flash** を使用して高品質なセキュリティ防御データセット（問答ペア）を生成し、**QLoRA (4-bit)** を用いてローカルの小型 OSS LLM（生徒モデル: Qwen 3B）にファインチューニングを実施、最終的に **Ollama (GGUF)** でローカル推論・効果検証を行います。

> [!NOTE]
> **Inspiration / Acknowledgment**:  
> 本検証は、[@furuhashilab](https://github.com/furuhashilab) 氏による [KnowledgeDistillation2026](https://github.com/furuhashilab/KnowledgeDistillation2026) の検証に着想を得て、エンタープライズセキュリティ防御領域への応用として実施されました。

---

## 1. 検証アーキテクチャ & データフロー

![Enterprise Security Distillation Flow](docs/architecture-flow.jpg)

### アーキテクチャ構成の詳細説明

本プロジェクトの蒸留パイプラインは、以下の4つのフェーズを通じて未加工のセキュリティナレッジを現場対応可能なローカルLLMへと集約します：

```text
┌────────────────────────────────────────────────────────────────────────┐
│ フェーズ1: 知識の発見と初期抽出 (未加工データからのコントロール抽出)     │
│  - エンタープライズポリシー / 脅威インテリジェンス (M365, AWS, AD)     │
│  - 抽出エンジン: Gemini 3.8 Flash (`src/generate_dataset.py`)         │
│  - 出力: 高品質問答ペア `data/train.jsonl`                             │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ データフロー
┌──────────────────────────────────▼─────────────────────────────────────┐
│ フェーズ2: フレームワークのマッピングと正規化 (業界標準コントロール)   │
│  - 業界標準 (MITRE ATT&CK, NIST CSF, Zero Trust) に基づく体系化        │
│  - ChatML 形式へのプロンプトテンプレート正規化                         │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│ フェーズ3: 自動蒸留 (矛盾解決ルールによるコントロールの最適化)         │
│  - 生徒モデル: Qwen2.5-3B-Instruct                                    │
│  - 蒸留ロジック: QLoRA 4-bit NF4 (`src/train_lora.py`, RTX 3070 8GB)   │
│  - 重みマージ (`src/merge_and_export.py`) ＋ GGUF Q8_0 量子化          │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│ フェーズ4: 合成と出力生成 (最終的なセキュリティアーキテクチャモデル)   │
│  - 配備: Ollama (`ollama create sec-defense -f ollama/Modelfile`)      │
│  - 評価検証: 未学習シナリオでのベンチマーク (`src/evaluate.py`)         │
│  - 出力: 実務即応型のアクションプラン・コマンドライン出力               │
└────────────────────────────────────────────────────────────────────────┘
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
- OS: Windows 11
- GPU: NVIDIA GeForce RTX 3070 (VRAM 8GB)
- Framework: Python 3.10, PyTorch 2.6 (CUDA 12.4), Hugging Face Transformers/PEFT/TRL, Ollama

---

## 5. 蒸留効果の検証結果 (Benchmark Results)

未学習のインシデントシナリオ（`data/eval_results.json`）を用いた、蒸留モデル（`sec-defense:3B`）とベースモデル（`llama3.1:8b`）の比較結果：

### テストケース 1: Active Directory × ランサムウェア初動対応
- **質問**: 社内LANでBitLockerの不正暗号化を装うランサムウェアを検知。最初の30分で実施すべきトリアージ手順と緊急遮断判断基準を提示せよ。
- **ベースモデル (`llama3.1:8b`)**:
  - 「私はランサムウェア対策の専門家ではありません」という前置きから開始。
  - 「LANのトラフィック監視」「端末隔離」「緊急バックアップ」などの一般的な概念論のみ。
- **蒸留モデル (`sec-defense:3B`)**:
  - セキュリティエンジニアとしての立場を明確化。
  - BitLocker復元キーの抽出・破棄検証（`0x8037`, `diskpart /s`）、GPO強制適用（`gpupdate /force`）による暗号化設定の無効化、ゼロトラストアクセス制御（ZT/MFA）での隔離など、実務で直ちに入力可能なコマンドと手順を提示。

### テストケース 2: AWS GuardDuty インスタンス認証情報漏洩
- **質問**: GuardDuty「UnauthorizedAccess:IAMUser/InstanceCredentialExfiltration.OutsideAWS」アラート対応、認証情報無効化、IMDSv2強制適用の提示。
- **蒸留モデル (`sec-defense:3B`)**:
  - Security Hub連携、特定インスタンスに対する認証情報無効化、TerraformでのIMDSv2強制適用構成を提示。

---

## 6. 総評

### ① 今回は何を達成したのか？（技術の概要）
クラウド上の超巨大で賢いAI（**Gemini 3.8 Flash**）を「名門校の熟練セキュリティ教授」、手元の市販パソコン（RTX 3070）で軽快に動く小さなAI（**Qwen 3B**）を「生徒」に見立て、**知識をギュッと濃縮して移植する技術（知識蒸留）** を手元のPCで完結させました。

### ② 何がすごいのか？（ビジネス・実用上の3つのメリット）
1. **「専門外です」から「プロの即答」への激変**
   - もとの小型AIは「私はセキュリティの専門家ではありません」と逃げてしまい、表面的な教科書通りの回答しかできませんでした。
   - わずか **25問・学習時間85秒** の蒸留を行っただけで、実際の緊急現場で即座に打ち込むべきコマンドや手順（BitLockerキー破棄、GPO強制、ゼロトラスト隔離など）を自信を持って提示できるようになりました。
2. **完全オフライン動作（機密情報が外に漏れない）**
   - 一度学習したAIは、**インターネットを切断したローカル環境（Ollama）** で100%動作します。
   - 外部に出せない社内の脆弱性ログ、漏洩インシデント情報、システム構成図などを安心して入力し、アドバイスを求めることができます。
3. **圧倒的なコストパフォーマンス**
   - かかったクラウドAPI費用は **ほぼ0円（数十円程度）**。
   - 月額数十万円〜数百万円のエンタープライズ向け高額AIサーバーを契約しなくても、一般的なゲーミングPC（GPUメモリ 8GB）1台で社内特化型AIを構築できることが実証されました。

---

## 7. 今後の発展性 (Future Prospects & Roadmap)

本検証で確立した「クラウドTeacher × ローカルStudent（QLoRA/Ollama）」の基盤は、以下のような実務適用・アーキテクチャ拡張へと発展させることが可能です。

### ① 社内固有ナレッジの追加蒸留（プライベートSOCアドバイザー化）
- **社内ドキュメントの統合**: 自社のセキュリティ基本方針、社内ネットワーク構成図、クラウドインフラ設計書、過去のインシデント報告書（SOP）をTeacherモデルにインプット。
- **企業特化の回答**: 一般論ではなく「自社が契約しているEDR製品」「社内IPアドレス帯」「自社のエスカレーション先」を熟知した、完全社内特化のセキュリティアドバイザーを低コストに内製できます。

### ② 脅威インテリジェンスの継続的自動蒸留 (Continuous Distillation)
- **定期的な知識更新**: 日々公開される新規CVE、ゼロデイ攻撃手法、MITRE ATT&CKの新規テクニックをトリガーにして、月次・四半期ごとに差分データセットを自動生成。
- **短時間でのモデル再学習**: 今回の実績通り、差分学習は数分で完了するため、常に最新の脅威傾向をインプットした最新防衛モデルを保守・運用し続けるパイプラインを自動化できます。

### ③ ChatOps / SIEM 自動連携 (自律型インシデント対応)
- **Ollama API 連携**: ローカルで稼働する Ollama（HTTP API）を社内チャットツール（Slack, Microsoft Teams）や SIEM（Microsoft Sentinel, Splunk）と Webhook 連携。
- **即時初動支援**: アラート検知時に自動でトリアージ手順、調査用KQLクエリ、推奨遮断コマンドをチャットルームへ即時投稿し、一次対応エンジニアの初動時間を大幅に短縮します。

### ④ 多様な小型モデルとのアンサンブル / 蒸留先モデルの比較
- Llama-3.2 (3B)、Phi-3.5 (3.8B)、Gemma-2 (2.6B/9B) など複数の生徒モデルへ同一データセットを蒸留し、推論速度・メモリ効率・日本語表現力の最適バランスを追求する検証への拡張。

---

## 8. 謝辞・参考プロジェクト・レポート (Acknowledgments & References)

本プロジェクトは、以下の検証および公開知見・レポートに着想・知見を得て実施されました。有益な知見を公開してくださったことに感謝いたします。

- **着想元リポジトリ**: [furuhashilab/KnowledgeDistillation2026](https://github.com/furuhashilab/KnowledgeDistillation2026) by [@furuhashilab](https://github.com/furuhashilab)
- **脅威インテリジェンス参照**: [Anthropic Threat Intelligence Report: September 2026](https://www.anthropic.com/threat-intelligence-report-september-2026) (最新の脅威動向および防御コントロール設計の参考)


