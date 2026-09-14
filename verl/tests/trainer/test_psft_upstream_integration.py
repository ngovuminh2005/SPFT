"""CPU checks for the PSFT launch contract and Numina-format data adapter.

Run directly with Python; no GPU or model download is needed.
"""

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "third_party/psft"
spec = importlib.util.spec_from_file_location("openr1_psft", ROOT / "examples/data_preprocess/openr1_psft.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def launch_args(script="train_psft.sh", overrides=None):
    env = {"PATH": os.environ["PATH"], "DRY_RUN": "1", **(overrides or {})}
    output = subprocess.check_output(["bash", str(ROOT / script)], env=env, text=True, cwd="/tmp")
    return output.splitlines()[2:]


def local_dry_run(script, dataset, epochs="3"):
    env = {"PATH": os.environ["PATH"], "DRY_RUN": "1", "DATASET": dataset, "EPOCHS": epochs}
    return subprocess.check_output(["bash", str(ROOT / script)], env=env, text=True, cwd="/tmp")


def as_config(args):
    return dict(arg.split("=", 1) for arg in args if "=" in arg)


class DatasetContract(unittest.TestCase):
    def setUp(self):
        self.raw = {
            "prompt": [{"role": "system", "content": "original"}, {"role": "user", "content": "Question?"}],
            "target": [{"role": "assistant", "content": r"Reasoning. \boxed{42}"}],
        }

    def test_raw_openr1_preserves_demonstration_without_adding_eos(self):
        result = adapter.train_row(self.raw, 7)
        self.assertEqual(result["demonstration"], self.raw["target"][0]["content"])
        self.assertEqual(result["extra_info"]["answer"], result["demonstration"])
        self.assertEqual(result["extra_info"]["question"], "Question?")
        self.assertEqual(result["prompt"], [{"role": "user", "content": "Question? " + adapter.INSTRUCTION}])

    def test_hosted_openr1_and_raw_openr1_match(self):
        hosted = {"prompt": self.raw["prompt"], "demonstration": self.raw["target"][0]["content"]}
        self.assertEqual(adapter.train_row(hosted, 7), adapter.train_row(self.raw, 7))

    def test_upstream_template_is_available(self):
        result = adapter.train_row(self.raw, 0, "upstream")
        self.assertEqual(result["prompt"][0], {"role": "system", "content": adapter.SYSTEM})
        self.assertEqual(result["prompt"][1]["content"], "Question?")

    def test_validation_cannot_select_demonstration_branch(self):
        row = {"problem": "Question?", "answer": 0, "demonstration": "must be removed"}
        result = adapter.validation_row(row, 3)
        self.assertNotIn("demonstration", result)
        self.assertEqual(result["reward_model"]["ground_truth"], "0")

    def test_existing_reward_routing_is_preserved(self):
        result = adapter.validation_row({"problem": "Q", "answer": 42, "data_source": "aime24"}, 0)
        self.assertEqual(result["data_source"], "aime24")

    def test_numina_conversion_does_not_duplicate_instruction(self):
        row = adapter.train_row(self.raw, 0)
        self.assertEqual(adapter.train_row(row, 0), row)

    def test_missing_demonstration_fails(self):
        with self.assertRaises(ValueError):
            adapter.train_row({"problem": "Question?"}, 0)

    def test_whitespace_demonstration_is_preserved_like_upstream(self):
        row = {"problem": "Question?", "demonstration": "\n"}
        self.assertEqual(adapter.train_row(row, 0)["demonstration"], "\n")


class LauncherContract(unittest.TestCase):
    def test_local_dft_profiles_select_both_datasets_and_epochs(self):
        numina = local_dry_run("train_dft_1gpu.sh", "numina")
        openr1 = local_dry_run("train_dft_1gpu.sh", "openr1")
        self.assertIn("data/numina_cot/train.parquet", numina)
        self.assertIn("Qwen/Qwen2.5-Math-1.5B", numina)
        self.assertIn("epochs=3", numina)
        self.assertIn("data/openr1_psft/train.parquet", openr1)
        self.assertIn("Qwen/Qwen2.5-7B-Instruct", openr1)
        self.assertIn("epochs=3", openr1)

    def test_dft_8gpu_launcher_accepts_dataset_and_epochs(self):
        output = local_dry_run("train_dft.sh", "openr1", epochs="4")
        self.assertIn("DATASET=openr1", output)
        self.assertIn("EPOCHS=4", output)

    def test_runtime_matches_pinned_upstream(self):
        manifest = json.loads((RUNTIME / "UPSTREAM_MANIFEST.json").read_text())
        for name, expected in manifest["sha256"].items():
            with self.subTest(file=name):
                data = (RUNTIME / name).read_text()
                if data and not data.endswith("\n"):
                    data += "\n"
                self.assertEqual(hashlib.sha256(data.encode()).hexdigest(), expected)

    def test_default_overrides_match_upstream_shell(self):
        # Execute only shell argument expansion. The Python function captures
        # argv, so upstream never imports a model or starts a training process.
        capture = 'python3() { printf "%s\\n" "$@"; }; export -f python3; bash "$1"'
        with tempfile.TemporaryDirectory() as tmp:
            original = subprocess.check_output(
                ["bash", "-c", capture, "capture", str(RUNTIME / "recipe/psft/run_psft.sh")],
                cwd=tmp, env={"PATH": os.environ["PATH"]}, text=True, stderr=subprocess.DEVNULL,
            )
        expected = as_config(original.splitlines())
        actual = as_config(launch_args(overrides={"DATASET": "openr1"}))
        for path in ("data.train_files", "data.val_files", "trainer.default_local_dir"):
            del expected[path]
            del actual[path]
        self.assertEqual(actual, expected)

    def test_one_gpu_changes_only_topology(self):
        full = as_config(launch_args())
        single = as_config(launch_args("train_psft_1gpu.sh"))
        self.assertEqual(single["trainer.n_gpus_per_node"], "1")
        self.assertEqual(single["actor_rollout_ref.rollout.tensor_model_parallel_size"], "1")
        for key in ("trainer.n_gpus_per_node", "actor_rollout_ref.rollout.tensor_model_parallel_size"):
            del single[key]
            del full[key]
        self.assertEqual(single, full)

    def test_sweep_uses_upstream_module_and_lr(self):
        args = launch_args("sweep_psft_1gpu.sh")
        self.assertEqual(args[:2], ["-m", "recipe.psft.main_psft"])
        self.assertEqual(as_config(args)["actor_rollout_ref.actor.optim.lr"], "1e-6")

    def test_psft_epoch_override_reaches_upstream_runtime(self):
        config = as_config(launch_args(overrides={"EPOCHS": "4"}))
        self.assertEqual(config["trainer.total_epochs"], "4")

    def test_numina_profile_uses_dft_data_and_model(self):
        config = as_config(launch_args("train_psft_numina_1gpu.sh"))
        self.assertTrue(config["data.train_files"].endswith("data/numina_psft/train.parquet"))
        self.assertTrue(config["data.val_files"].endswith("data/numina_psft/test.parquet"))
        self.assertEqual(config["actor_rollout_ref.model.path"], "Qwen/Qwen2.5-Math-1.5B")

    def test_user_overrides_reach_launcher(self):
        config = as_config(launch_args(overrides={"OPTIM_LR": "3e-6", "MODEL_NAME": "custom/model"}))
        self.assertEqual(config["actor_rollout_ref.actor.optim.lr"], "3e-6")
        self.assertEqual(config["actor_rollout_ref.model.path"], "custom/model")

    def test_invalid_tensor_parallelism_fails(self):
        with self.assertRaises(subprocess.CalledProcessError):
            launch_args(overrides={"NGPUS_PER_NODE": "1", "ROLLOUT_TP": "2"})


if __name__ == "__main__":
    unittest.main()
