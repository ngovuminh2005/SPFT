"""Offline checks for media escaping and raw-cache invalidation (stdlib only)."""

import ast
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from escape_wethink_media_tags import escape_sample  # noqa: E402
from smoke_wethink import audit  # noqa: E402
from prepare_wethink import build_response  # noqa: E402


class WeThinkMediaTests(unittest.TestCase):
    def setUp(self):
        self.sample = {
            "conversations": [
                {"from": "human", "value": "<image>\nQuestion <image>example</image>"},
                {"from": "gpt", "value": "<think>HTML `<video>` and `<audio>`</think>\n<answer>HTML5</answer>"},
            ],
            "images": ["image.jpg"], "system": "Explain <audio>",
        }

    def test_escape_preserves_real_image_and_is_idempotent(self):
        sample = copy.deepcopy(self.sample)
        assert escape_sample(sample) == 5
        assert sample["conversations"][0]["value"] == "<image>\nQuestion &lt;image&gt;example&lt;/image&gt;"
        assert sample["images"] == self.sample["images"]
        assert "&lt;video&gt;" in sample["conversations"][1]["value"]
        assert escape_sample(sample) == 0

    def test_repair_smoke_backup_and_mtime(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "image.jpg").write_bytes(b"fixture")
            dataset = directory / "fixture.jsonl"
            original = json.dumps(self.sample) + "\n"
            dataset.write_text(original)
            os.utime(dataset, (1000, 1000))
            with self.assertRaisesRegex(ValueError, "tokens"):
                audit(dataset, directory, 1)
            command = [sys.executable, str(ROOT / "scripts/escape_wethink_media_tags.py"), str(dataset)]
            subprocess.run(command, check=True, capture_output=True)
            assert dataset.stat().st_mtime > 1000
            assert dataset.with_name("fixture.jsonl.bak").read_text() == original
            audit(dataset, directory, 1)
            subprocess.run(command, check=True, capture_output=True)
            assert not dataset.with_name("fixture.jsonl.bak.1").exists()
            dataset.write_text(original)
            subprocess.run(command, check=True, capture_output=True)
            assert dataset.with_name("fixture.jsonl.bak.1").read_text() == original

    def test_prepare_escapes_tags_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "image.jpg").write_bytes(b"fixture")
            raw = directory / "raw.jsonl"
            raw.write_text(json.dumps({
                "problem": "Question <image>example</image>", "refined_cot": "HTML <video>",
                "answer": "<audio>", "image_path": "image.jpg",
            }) + "\n")
            output = directory / "output.jsonl"
            info = directory / "dataset_info.json"
            info.write_text("{}")
            subprocess.run([
                sys.executable, str(ROOT / "scripts/prepare_wethink.py"), "--raw-jsonl", str(raw),
                "--image-root", str(directory), "--output", str(output), "--dataset-info", str(info),
            ], check=True, capture_output=True)
            audit(output, directory, 1)

    def test_normalize_existing_answer_and_think_tags(self):
        result = build_response("<think>Reason <answer>wrong</answer></think>", "<answer>correct</answer>")
        self.assertEqual(result, "<think>\nReason\n</think>\n<answer>\ncorrect\n</answer>")
        self.assertEqual(build_response(result, "correct"), result)

    def test_missing_image_does_not_overwrite_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            raw = directory / "raw.jsonl"
            raw.write_text(json.dumps({"problem": "Q", "refined_cot": "Reason", "answer": "A",
                                       "image_path": "missing.jpg"}) + "\n")
            output = directory / "output.jsonl"
            output.write_text("original")
            info = directory / "dataset_info.json"
            info.write_text("{}")
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/prepare_wethink.py"), "--raw-jsonl", str(raw),
                "--image-root", str(directory), "--output", str(output), "--dataset-info", str(info),
            ], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_text(), "original")

    def test_local_overwrite_requests_raw_cache_rebuild(self):
        # Execute the actual loader function with a stub dataset backend, without
        # importing torch or downloading model files into the active environment.
        tree = ast.parse((ROOT / "src/llamafactory/data/loader.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "_load_single_dataset")
        function.returns = None
        for argument in function.args.args:
            argument.annotation = None
        modes = []
        namespace = {
            "os": os, "FILEEXT2TYPE": {"jsonl": "json"},
            "logger": SimpleNamespace(info_rank0=lambda message: None),
            "load_dataset": lambda **kwargs: modes.append(kwargs["download_mode"]),
            "align_dataset": lambda dataset, *args: dataset,
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), "loader", "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "fixture.jsonl").write_text("{}\n")
            attribute = SimpleNamespace(load_from="file", dataset_name="fixture.jsonl", split="train", num_samples=None)
            data = SimpleNamespace(dataset_dir=directory, overwrite_cache=True, streaming=False,
                                   preprocessing_num_workers=None, max_samples=None)
            model = SimpleNamespace(cache_dir=None, hf_hub_token=None)
            training = SimpleNamespace(local_process_index=0)
            namespace["_load_single_dataset"](attribute, model, data, training)
            assert modes[-1] == "force_redownload"
            data.overwrite_cache = False
            namespace["_load_single_dataset"](attribute, model, data, training)
            assert modes[-1] == "reuse_dataset_if_exists"

    def test_dft_spft_filter_identical_samples_after_causal_shift(self):
        tree = ast.parse((ROOT / "src/llamafactory/train/sft/workflow.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_sft")
        block = next(node for node in function.body if isinstance(node, ast.If)
                     and any(isinstance(child, ast.FunctionDef) and child.name == "has_supervision"
                             for child in node.body))

        class Dataset(list):
            def filter(self, predicate):
                return Dataset(row for row in self if predicate(row))

        rows = Dataset([{"labels": [-100, -100]}, {"labels": [42, -100]},
                        {"labels": [-100, 42]}, {"labels": [42, 43]}])
        results = []
        for dft, spft in ((True, False), (False, True)):
            module = {"train_dataset": rows, "eval_dataset": {"validation": rows}}
            namespace = {"finetuning_args": SimpleNamespace(use_dft_loss=dft, use_spft_loss=spft),
                         "dataset_module": module, "IGNORE_INDEX": -100}
            exec(compile(ast.Module(body=[block], type_ignores=[]), "workflow", "exec"), namespace)
            self.assertEqual(module["train_dataset"], rows[2:])
            self.assertEqual(module["eval_dataset"]["validation"], rows[2:])
            results.append(module)
        self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
