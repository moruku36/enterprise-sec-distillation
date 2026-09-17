"""
test_dataset_schema.py
データセット (train.jsonl, eval.jsonl, eval_results.json) の構造・整合性・プロベナンスを検証するテスト。
pytest / unittest 両対応。
"""

import json
import unittest
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class TrainSample(BaseModel):
    instruction: str = Field(min_length=10)
    input: Optional[str] = ""
    output: str = Field(min_length=20)
    category: str
    teacher_model: str
    generated_at: str
    prompt_version: str
    validation_status: str


class EvalSample(BaseModel):
    id: str
    category: str
    instruction: str = Field(min_length=10)
    input: Optional[str] = ""
    ground_truth_criteria: Dict[str, Any]
    references: List[Dict[str, str]]


class TestDatasetSchema(unittest.TestCase):
    def test_train_dataset_schema(self):
        train_path = Path("data/train.jsonl")
        self.assertTrue(train_path.exists(), "data/train.jsonl must exist")

        samples = []
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    sample = TrainSample(**data)
                    samples.append(sample)

        self.assertGreaterEqual(len(samples), 10, f"Expected at least 10 training samples, got {len(samples)}")
        for s in samples:
            self.assertIsNotNone(s.teacher_model, "teacher_model provenance is required")
            self.assertIn(
                s.category,
                [
                    "soc_siem_edr",
                    "zero_trust_iam",
                    "incident_response",
                    "cloud_security",
                    "network_perimeter",
                ],
            )

    def test_eval_dataset_schema(self):
        eval_path = Path("data/eval.jsonl")
        self.assertTrue(eval_path.exists(), "data/eval.jsonl must exist")

        samples = []
        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    sample = EvalSample(**data)
                    samples.append(sample)

        self.assertGreaterEqual(len(samples), 10, f"Expected at least 10 eval questions, got {len(samples)}")
        categories = {s.category for s in samples}
        self.assertGreaterEqual(len(categories), 4, f"Expected scenarios across diverse categories, got {categories}")

    def test_eval_results_json_schema(self):
        results_path = Path("data/eval_results.json")
        self.assertTrue(results_path.exists(), "data/eval_results.json must exist")

        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), 10, f"Expected at least 10 eval results, got {len(results)}")
        for item in results:
            self.assertIn("test_id", item)
            self.assertIn("category", item)
            self.assertIn("distilled_response", item)
            self.assertIn("base_response", item)


if __name__ == "__main__":
    unittest.main()
