#!/usr/bin/env python3
"""Audit every WeThink row; optionally run real multimodal tokenization on selected rows."""

import argparse
import json
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def audit_source(path: Path, raw_path: Path) -> None:
    from prepare_wethink import SYSTEM_PROMPT, build_response
    from escape_wethink_media_tags import escape_text

    matched = rejected = 0
    with raw_path.open(encoding="utf-8") as raw, path.open(encoding="utf-8") as output:
        for line_number, line in enumerate(raw, 1):
            if not line.strip():
                continue
            source = json.loads(line)
            try:
                response = build_response(str(source["refined_cot"]), str(source["answer"]))
            except ValueError:
                rejected += 1
                continue
            converted_line = output.readline()
            if not converted_line:
                raise ValueError(f"Raw line {line_number}: missing converted sample.")
            converted = json.loads(converted_line)
            problem, _ = escape_text(str(source["problem"]))
            expected = {
                "conversations": [{"from": "human", "value": f"<image>\n{problem}"},
                                  {"from": "gpt", "value": response}],
                "images": [Path(str(source["image_path"])).as_posix()], "system": SYSTEM_PROMPT,
            }
            if converted != expected:
                raise ValueError(f"Raw line {line_number}: question/answer/image/system does not match source.")
            matched += 1
        if output.read().strip():
            raise ValueError("Converted dataset contains extra samples.")
    print(f"PASS source alignment: {matched} exact matches; {rejected} empty targets rejected.", flush=True)


def audit(path: Path, media_dir: Path, sample_count: int) -> list[tuple[int, dict]]:
    selected = []
    repaired_samples = []
    row_count = 0
    escaped_count = 0
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            messages = sample.get("conversations", [])
            if not messages or any(not isinstance(message.get("value"), str) for message in messages):
                raise ValueError(f"Line {line_number}: invalid conversations.")
            if len(messages) != 2 or [message.get("from") for message in messages] != ["human", "gpt"]:
                raise ValueError(f"Line {line_number}: expected one human/gpt pair.")
            if not messages[0]["value"].startswith("<image>\n"):
                raise ValueError(f"Line {line_number}: missing leading image placeholder.")
            response = messages[1]["value"]
            if re.findall(r"</?(?:think|answer)>", response) != ["<think>", "</think>", "<answer>", "</answer>"]:
                raise ValueError(f"Line {line_number}: nested or duplicate reasoning/answer tags.")
            if not re.fullmatch(r"<think>\s*\S[\s\S]*?</think>\s*<answer>\s*\S[\s\S]*?</answer>", response):
                raise ValueError(f"Line {line_number}: invalid or empty target.")
            texts = [message["value"] for message in messages]
            texts.append(sample.get("system", ""))
            for field, tag in (("images", "<image>"), ("videos", "<video>"), ("audios", "<audio>")):
                media = sample.get(field) or []
                if not isinstance(media, list):
                    raise ValueError(f"Line {line_number}: {field} must be a list.")
                count = sum(text.count(tag) for text in texts)
                if count != len(media):
                    raise ValueError(f"Line {line_number}: {count} {tag} tokens, but {len(media)} {field}.")
            if sample.get("videos") or sample.get("audios"):
                raise ValueError(f"Line {line_number}: expected image-only WeThink data.")
            for image in sample.get("images") or []:
                image_path = Path(image)
                if not image_path.is_absolute():
                    image_path = media_dir / image_path
                if not image_path.is_file():
                    raise ValueError(f"Line {line_number}: missing image {image_path}")
            row_count += 1
            if row_count <= sample_count:
                selected.append((line_number, sample))
            if any(any(tag in text for tag in ("&lt;video&gt;", "&lt;audio&gt;", "&lt;image&gt;")) for text in texts):
                escaped_count += 1
                if row_count > sample_count and len(repaired_samples) < max(sample_count, 10):
                    repaired_samples.append((line_number, sample))
    if not row_count:
        raise ValueError("The dataset is empty.")
    print(f"PASS: {row_count} rows; media placeholders match; images exist; {escaped_count} escaped rows.", flush=True)
    return selected + repaired_samples


def preprocess(samples: list[tuple[int, dict]], config_path: Path, media_dir: Path) -> None:
    # Use the activated environment and local tokenizer/processor artifacts only.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(ROOT / "src"))
    import yaml

    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.data.converter import get_dataset_converter
    from llamafactory.data.parser import DatasetAttr
    from llamafactory.data.processor import SupervisedDatasetProcessor
    from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq
    from llamafactory.hparams import DataArguments, ModelArguments
    from llamafactory.model import load_tokenizer

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model_args = ModelArguments(
        model_name_or_path=config["model_name_or_path"],
        trust_remote_code=config.get("trust_remote_code", False),
        image_max_pixels=config.get("image_max_pixels", 262144),
        cache_dir=config.get("cache_dir"),
    )
    data_args = DataArguments(
        template=config["template"], cutoff_len=config.get("cutoff_len", 8192), media_dir=str(media_dir)
    )
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    attributes = DatasetAttr("file", "wethink_sft.jsonl", formatting="sharegpt", system="system", images="images")
    converter = get_dataset_converter("sharegpt", attributes, data_args)
    processor = SupervisedDatasetProcessor(template=template, data_args=data_args, **tokenizer_module)
    collator = MultiModalDataCollatorForSeq2Seq(
        tokenizer=tokenizer_module["tokenizer"], processor=tokenizer_module["processor"], template=template
    )
    for line_number, sample in samples:
        aligned = converter(sample)
        # Bypass all Arrow/map caches, and exercise the exact image plugin and SFT tokenizer.
        result = processor.preprocess_dataset({key: [value] for key, value in aligned.items()})
        if not result["input_ids"] or not any(label != -100 for label in result["labels"][0][1:]):
            raise ValueError(f"Line {line_number}: tokenization produced no supervised tokens.")
        import torch

        batch = collator([{key: values[0] for key, values in result.items()}])
        pixels = batch.get("pixel_values")
        grid = batch.get("image_grid_thw")
        if pixels is None or grid is None or grid.shape[0] != 1 or not torch.isfinite(pixels).all():
            raise ValueError(f"Line {line_number}: missing/invalid image tensors.")
        tokenizer = tokenizer_module["tokenizer"]
        image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        image_mask = batch["input_ids"] == image_token_id
        merge_size = tokenizer_module["processor"].image_processor.merge_size
        expected_image_tokens = int(grid.prod(dim=-1).sum()) // (merge_size ** 2)
        if int(image_mask.sum()) != expected_image_tokens or not (batch["labels"][image_mask] == -100).all():
            raise ValueError(f"Line {line_number}: image tokens do not match grid or are not masked.")
        target = tokenizer.decode(batch["labels"][batch["labels"] != -100].tolist())
        if any(tag not in target for tag in ("<think>", "</think>", "<answer>", "</answer>")):
            raise ValueError(f"Line {line_number}: supervised target is truncated or missing tags.")
        print(f"PASS preprocessing + collator line {line_number}: {len(result['input_ids'][0])} tokens; "
              f"pixels={tuple(pixels.shape)}; grid={grid.tolist()}.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/wethink_sft.jsonl")
    parser.add_argument("--media-dir", type=Path, default=ROOT / "data/wethink_images")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--preprocess", action="store_true", help="Also tokenize real images, using cached model artifacts.")
    parser.add_argument("--raw-jsonl", type=Path, help="Verify every converted row against the original dataset.")
    parser.add_argument("--config", type=Path, default=ROOT / "examples/extras/spft/qwen2_5vl_wethink_full_sft.yaml")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    print(f"Python: {sys.executable}", flush=True)
    print(f"Dataset: {args.dataset.resolve()}", flush=True)
    try:
        samples = audit(args.dataset.resolve(), args.media_dir.resolve(), args.samples)
        if args.raw_jsonl:
            audit_source(args.dataset.resolve(), args.raw_jsonl.resolve())
        if args.preprocess:
            preprocess(samples, args.config.resolve(), args.media_dir.resolve())
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
