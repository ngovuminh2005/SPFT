"""Convert OpenR1/AIME to the Numina parquet schema used in this repository.

The additional train-only `demonstration` column selects upstream PSFT's
fixed-demonstration branch. Validation deliberately has no such column.
"""

import argparse
import json
from pathlib import Path


INSTRUCTION = r"Let's think step by step and output the final answer within \boxed{}."
SYSTEM = r"Please reason step by step, and put your final answer within \boxed{}."


def strip_numina_instruction(question):
    """Return the raw problem when a Numina parquet row already has the suffix."""
    suffix = " " + INSTRUCTION
    if isinstance(question, str) and question.endswith(suffix):
        return question[: -len(suffix)]
    return question


def question_from_row(row):
    extra = row.get("extra_info") or {}
    if isinstance(extra.get("question"), str) and extra["question"].strip():
        return strip_numina_instruction(extra["question"])
    messages = row.get("prompt")
    if messages is not None and not isinstance(messages, str):
        for message in reversed(list(messages)):
            if message["role"] == "user":
                return strip_numina_instruction(message["content"])
    question = row.get("problem") or row.get("question") or extra.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Missing nonempty user question")
    return strip_numina_instruction(question)


def prompt_for(question, template):
    if template == "upstream":
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    if template != "numina":
        raise ValueError(f"Unknown prompt template: {template}")
    return [{"role": "user", "content": question + " " + INSTRUCTION}]


def train_row(row, index, template="numina"):
    question = question_from_row(row)
    answer = row.get("demonstration")
    if answer is None and row.get("target") is not None:
        answer = row["target"][0]["content"]
    if answer is None:
        answer = (row.get("extra_info") or {}).get("answer") or row.get("solution")
    if not isinstance(answer, str):
        raise ValueError(f"Row {index}: missing string demonstration")
    return {
        "data_source": row.get("data_source") or "openr1",
        "prompt": prompt_for(question, template),
        "ability": "math",
        "reward_model": row.get("reward_model") or {"style": "rule", "ground_truth": ""},
        "extra_info": {"split": "train", "index": index, "question": question, "answer": answer},
        "demonstration": answer,
    }


def validation_row(row, index, template="numina"):
    question = question_from_row(row)
    ground_truth = (row.get("reward_model") or {}).get("ground_truth")
    if ground_truth is None:
        ground_truth = row.get("answer")
    if ground_truth is None:
        raise ValueError(f"Validation row {index}: missing ground truth")
    return {
        "data_source": row.get("data_source") or "lighteval/MATH",
        "prompt": prompt_for(question, template),
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(ground_truth)},
        "extra_info": {"split": "test", "index": index, "question": question, "answer": str(ground_truth)},
    }


def main():
    import datasets

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", default="wh-zhu/train_openr1_4k")
    parser.add_argument("--val-source", default="wh-zhu/aime-24")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="train")
    parser.add_argument("--local-dir", type=Path, default=Path("data/openr1_psft"))
    parser.add_argument("--prompt-template", choices=("numina", "upstream"), default="numina")
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-end", type=int, default=0)
    args = parser.parse_args()

    def load(source, split):
        if Path(source).is_file():
            return datasets.load_dataset("parquet", data_files=source, split="train")
        return datasets.load_dataset(source, split=split)

    args.local_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train.parquet", "test.parquet", "manifest.json"):
        if (args.local_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite {args.local_dir / name}; choose a new --local-dir")
    outputs = []
    manifest = {"prompt_template": args.prompt_template, "datasets": {}}
    for source, split, name, convert in (
        (args.train_source, args.train_split, "train.parquet", train_row),
        (args.val_source, args.val_split, "test.parquet", validation_row),
    ):
        destination = args.local_dir / name
        data = load(source, split)
        if name == "train.parquet" and (args.train_start or args.train_end):
            end = args.train_end or len(data)
            data = data.select(range(args.train_start, min(end, len(data))))
        converted = data.map(
            lambda row, index: convert(row, index, args.prompt_template),
            with_indices=True,
            remove_columns=data.column_names,
        )
        outputs.append((destination, converted))
        manifest["datasets"][name] = {
            "source": source, "split": split, "rows": len(data),
            "source_fingerprint": data._fingerprint,
        }
    for destination, converted in outputs:
        converted.to_parquet(str(destination))
        print(f"{destination}: {len(converted)} rows, template={args.prompt_template}")
    (args.local_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
