#!/usr/bin/env python3
"""Convert the raw WeThink JSONL dataset to LLaMA-Factory ShareGPT JSONL."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from escape_wethink_media_tags import escape_text


SYSTEM_PROMPT = (
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\n"
    "The reasoning process MUST BE enclosed within <think> </think> tags. "
    "The final answer MUST BE enclosed within <answer> </answer> tags."
)


def build_response(cot: str, answer: str) -> str:
    """Normalize existing wrappers; the original answer field is authoritative."""
    cot = re.sub(r"<answer>.*?</answer>", "", cot, flags=re.DOTALL)
    cot = re.sub(r"</?(?:think|answer)>", "", cot).strip()
    answer = re.sub(r"</?(?:think|answer)>", "", answer).strip()
    if not cot or not answer:
        raise ValueError("Empty reasoning or answer after normalization.")
    cot, _ = escape_text(cot)
    answer, _ = escape_text(answer)
    return f"<think>\n{cot}\n</think>\n<answer>\n{answer}\n</answer>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-info", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def update_dataset_info(path: Path, output: Path) -> None:
    with path.open(encoding="utf-8") as handle:
        dataset_info = json.load(handle)

    dataset_info["wethink_sft"] = {
        "file_name": output.name,
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "system": "system", "images": "images"},
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
        },
    }

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(dataset_info, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    if not args.raw_jsonl.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {args.raw_jsonl}")
    if not args.image_root.is_dir():
        raise FileNotFoundError(f"Image root not found: {args.image_root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted = 0
    rejected: list[dict] = []
    missing_images: list[str] = []

    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    with args.raw_jsonl.open(encoding="utf-8") as source, temporary_output.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            required = ("problem", "answer", "refined_cot", "image_path")
            missing_fields = [field for field in required if not sample.get(field)]
            if missing_fields:
                raise ValueError(f"Line {line_number} is missing fields: {missing_fields}")

            image_path = Path(str(sample["image_path"]))
            if image_path.is_absolute():
                image_file = image_path
                image_value = str(image_path)
            else:
                image_file = args.image_root / image_path
                image_value = image_path.as_posix()

            if not image_file.is_file():
                missing_images.append(str(image_file))
                continue

            try:
                response = build_response(str(sample["refined_cot"]), str(sample["answer"]))
            except ValueError as error:
                rejected.append({"line": line_number, "reason": str(error), "image_path": str(image_path)})
                continue
            problem, _ = escape_text(str(sample["problem"]))
            converted_sample = {
                "conversations": [
                    {"from": "human", "value": f"<image>\n{problem}"},
                    {"from": "gpt", "value": response},
                ],
                "images": [image_value],
                "system": SYSTEM_PROMPT,
            }
            target.write(json.dumps(converted_sample, ensure_ascii=False) + "\n")
            converted += 1
            if args.max_samples is not None and converted >= args.max_samples:
                break

    if missing_images:
        preview = "\n".join(missing_images[:10])
        raise FileNotFoundError(
            f"{len(missing_images)} image files are missing. First files:\n{preview}\n"
            "Check --image-root and the extracted image directory."
        )

    if not converted:
        raise ValueError("No samples converted.")
    temporary_output.replace(args.output)
    update_dataset_info(args.dataset_info, args.output)
    report_path = args.output.with_suffix(".conversion_report.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump({"raw_jsonl": str(args.raw_jsonl), "converted": converted, "rejected": rejected}, handle, indent=2)
    print(f"Converted samples: {converted}")
    print(f"Rejected samples: {len(rejected)} (details: {report_path})")
    print(f"Output: {args.output}")
    print("Registered dataset: wethink_sft")


if __name__ == "__main__":
    main()
