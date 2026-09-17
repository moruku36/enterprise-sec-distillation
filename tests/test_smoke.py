"""
test_smoke.py
主要スクリプトのCLI引数やモジュール構成のスモークテスト。
pytest / unittest 両対応。
"""

import subprocess
import sys
import unittest
from pathlib import Path


class TestSmoke(unittest.TestCase):
    def test_evaluate_cli_dry_run(self):
        res = subprocess.run(
            [sys.executable, "src/evaluate.py", "--dry_run"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"evaluate.py dry_run failed:\n{res.stderr}")
        self.assertIn("Evaluation Summary", res.stdout)

    def test_merge_lora_help(self):
        res = subprocess.run(
            [sys.executable, "src/merge_lora.py", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"merge_lora.py --help failed:\n{res.stderr}")
        self.assertIn("Merge LoRA weights into base model", res.stdout)

    def test_generate_dataset_help(self):
        res = subprocess.run(
            [sys.executable, "src/generate_dataset.py", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"generate_dataset.py --help failed:\n{res.stderr}")
        self.assertIn("Generate enterprise security dataset", res.stdout)

    def test_train_lora_help(self):
        res = subprocess.run(
            [sys.executable, "src/train_lora.py", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"train_lora.py --help failed:\n{res.stderr}")
        self.assertIn("QLoRA training for Enterprise Security LLM", res.stdout)

    def test_modelfile_consistency(self):
        modelfile_path = Path("ollama/Modelfile")
        self.assertTrue(modelfile_path.exists(), "ollama/Modelfile must exist")
        content = modelfile_path.read_text(encoding="utf-8")
        self.assertIn("sec-defense-q8_0.gguf", content, "Modelfile should reference Q8_0 GGUF")


if __name__ == "__main__":
    unittest.main()
