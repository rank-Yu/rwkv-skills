from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.space.data import load_scores
from src.eval.scheduler.dataset_utils import canonical_slug


class TestSpaceDataCompatibility(unittest.TestCase):
    def _write_score(self, root: Path, relpath: str, payload: dict) -> None:
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_load_scores_handles_legacy_code_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_score(
                root,
                "model_a/human_eval_test.json",
                {
                    "dataset": "human_eval_test",
                    "model": "demo-humaneval",
                    "metrics": {"pass@1": 0.77},
                    "samples": 1,
                    "log_path": "/tmp/human_eval.jsonl",
                    "score": 0.33,
                },
            )
            self._write_score(
                root,
                "model_b/mbpp_test.json",
                {
                    "dataset": "mbpp_test",
                    "model": "demo-mbpp",
                    "metrics": {"score": {"pass1": 0.5}},
                    "samples": 1,
                    "log_path": "/tmp/mbpp.jsonl",
                },
            )
            self._write_score(
                root,
                "model_c/human_eval_dev.json",
                {
                    "dataset": "HumanEval-Dev",
                    "model": "demo-humaneval-dev",
                    "metrics": {"pass1": "0.11"},
                    "samples": 1,
                    "log_path": "/tmp/human_eval_dev.jsonl",
                },
            )

            entries = load_scores(root)
            by_dataset = {entry.dataset: entry for entry in entries}

            self.assertIn("human_eval_test", by_dataset)
            self.assertEqual(by_dataset["human_eval_test"].metrics["pass@1"], 0.77)

            self.assertIn("mbpp_test", by_dataset)
            self.assertEqual(by_dataset["mbpp_test"].metrics["pass@1"], 0.5)

            dev_slug = canonical_slug("HumanEval-Dev")
            self.assertIn(dev_slug, by_dataset)
            self.assertAlmostEqual(by_dataset[dev_slug].metrics["pass@1"], 0.11)
