"""Generate and Math-Verify the rejection-sampling data used by offline DFT."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_manifest(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing offline-data manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def source_rows(path: Path, limit: int):
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["prompt", "reward_model", "extra_info"])
    if table.num_rows < limit:
        raise ValueError(f"Requested {limit} questions, but {path} contains {table.num_rows}")
    return table.slice(0, limit).to_pylist()


def generation_config(args) -> dict:
    return {
        "generator_model": args.model,
        "source_file": str(args.source_file.resolve()),
        "source_sha256": file_sha256(args.source_file),
        "num_questions": args.num_questions,
        "responses_per_question": args.responses_per_question,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }


def check_config(expected: dict, actual: dict) -> None:
    mismatches = [
        f"{key}: expected={value!r}, manifest={actual.get(key)!r}"
        for key, value in expected.items()
        if actual.get(key) != value
    ]
    if mismatches:
        raise ValueError("Offline dataset configuration mismatch:\n  " + "\n  ".join(mismatches))


def count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def generate(args, manifest_path: Path, raw_path: Path) -> None:
    expected = generation_config(args)
    if manifest_path.is_file():
        manifest = load_manifest(manifest_path)
        check_config(expected, manifest)
    else:
        manifest = {**expected, "generation_completed": False, "filter_completed": False}
        write_manifest(manifest_path, manifest)

    if manifest.get("generation_completed"):
        print(f"Generation already complete: {raw_path}")
        return

    completed = count_lines(raw_path)
    if completed > args.num_questions:
        raise ValueError(f"{raw_path} has {completed} rows; expected at most {args.num_questions}")

    rows = source_rows(args.source_file, args.num_questions)
    if completed == args.num_questions:
        manifest["generation_completed"] = True
        write_manifest(manifest_path, manifest)
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=True,
        disable_custom_all_reduce=True,
    )
    sampling = SamplingParams(
        n=args.responses_per_question,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    with raw_path.open("a", encoding="utf-8") as output:
        for start in range(completed, args.num_questions, args.batch_size):
            batch = rows[start : min(start + args.batch_size, args.num_questions)]
            prompts = [
                tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
                for row in batch
            ]
            generations = llm.generate(prompts, sampling, use_tqdm=True)
            for offset, (row, generated) in enumerate(zip(batch, generations)):
                extra = row["extra_info"]
                record = {
                    "source_index": start + offset,
                    "question": extra["question"],
                    "prompt": row["prompt"],
                    "gold_answer": row["reward_model"]["ground_truth"],
                    "gold_solution": extra["answer"],
                    "responses": [candidate.text for candidate in generated.outputs],
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
            print(f"Generated {min(start + len(batch), args.num_questions)}/{args.num_questions} questions")

    manifest["generation_completed"] = True
    write_manifest(manifest_path, manifest)


def verify_record(line: str):
    from math_verify import parse, verify

    record = json.loads(line)
    gold_text = record.get("gold_solution") or record["gold_answer"]
    try:
        gold = parse(gold_text)
    except Exception:
        gold = []
    accepted = []
    if gold:
        for response_index, response in enumerate(record["responses"]):
            try:
                prediction = parse(response)
                correct = bool(prediction) and verify(gold, prediction)
            except Exception:
                correct = False
            if correct:
                accepted.append((response_index, response))
    return record, accepted


def parquet_schema():
    import pyarrow as pa

    message = pa.struct([("role", pa.string()), ("content", pa.string())])
    return pa.schema(
        [
            ("data_source", pa.string()),
            ("prompt", pa.list_(message)),
            ("ability", pa.string()),
            ("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
            (
                "extra_info",
                pa.struct(
                    [
                        ("split", pa.string()),
                        ("index", pa.int64()),
                        ("question", pa.string()),
                        ("answer", pa.string()),
                        ("source_index", pa.int64()),
                        ("response_index", pa.int64()),
                        ("generator_model", pa.string()),
                    ]
                ),
            ),
        ]
    )


def filter_generations(args, manifest_path: Path, raw_path: Path, train_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    manifest = load_manifest(manifest_path)
    check_config(generation_config(args), manifest)
    if not manifest.get("generation_completed"):
        raise RuntimeError("Generation is incomplete; resume the generate stage first")
    if count_lines(raw_path) != args.num_questions:
        raise ValueError(f"Expected {args.num_questions} generation rows in {raw_path}")
    if manifest.get("filter_completed") and train_path.is_file():
        print(f"Filtering already complete: {train_path}")
        return

    temporary = train_path.with_suffix(".parquet.incomplete")
    if temporary.exists():
        temporary.unlink()
    schema = parquet_schema()
    writer = pq.ParquetWriter(temporary, schema=schema, compression="zstd")
    buffer = []
    accepted_count = 0
    processed = 0

    def flush():
        nonlocal buffer
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
            buffer = []

    try:
        with raw_path.open(encoding="utf-8") as source, ProcessPoolExecutor(max_workers=args.verify_workers) as pool:
            for record, accepted in pool.map(verify_record, source, chunksize=8):
                processed += 1
                for response_index, response in accepted:
                    buffer.append(
                        {
                            "data_source": "offline_math_rejection_sampling",
                            "prompt": record["prompt"],
                            "ability": "math",
                            "reward_model": {"style": "rule", "ground_truth": record["gold_answer"]},
                            "extra_info": {
                                "split": "train",
                                "index": accepted_count,
                                "question": record["question"],
                                "answer": response,
                                "source_index": record["source_index"],
                                "response_index": response_index,
                                "generator_model": args.model,
                            },
                        }
                    )
                    accepted_count += 1
                if len(buffer) >= 1024:
                    flush()
                if processed % 1000 == 0:
                    print(f"Verified {processed}/{args.num_questions}; retained {accepted_count}")
        flush()
    finally:
        writer.close()

    if accepted_count == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Math-Verify retained zero responses")
    os.replace(temporary, train_path)
    manifest.update({"filter_completed": True, "retained_responses": accepted_count})
    write_manifest(manifest_path, manifest)
    print(f"Saved {accepted_count} verified responses to {train_path}")


def validate(args, manifest_path: Path, raw_path: Path, train_path: Path) -> None:
    manifest = load_manifest(manifest_path)
    check_config(generation_config(args), manifest)
    if not manifest.get("generation_completed") or not manifest.get("filter_completed"):
        raise RuntimeError("Offline dataset generation/filtering is incomplete")
    if not raw_path.is_file() or not train_path.is_file():
        raise FileNotFoundError("Offline generation JSONL or train Parquet is missing")
    import pyarrow.parquet as pq

    rows = pq.ParquetFile(train_path).metadata.num_rows
    if rows != manifest.get("retained_responses"):
        raise ValueError(f"Parquet has {rows} rows; manifest records {manifest.get('retained_responses')}")
    print(f"Validated offline dataset: model={args.model}, questions={args.num_questions}, retained={rows}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["generate", "filter", "all", "validate"], default="all")
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-questions", type=int, default=100_000)
    parser.add_argument("--responses-per-question", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--verify-workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.source_file.is_file():
        raise FileNotFoundError(f"Missing source questions: {args.source_file}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    raw_path = args.output_dir / "generations.jsonl"
    train_path = args.output_dir / "train.parquet"
    if args.stage in {"generate", "all"}:
        generate(args, manifest_path, raw_path)
    if args.stage in {"filter", "all"}:
        filter_generations(args, manifest_path, raw_path, train_path)
    if args.stage == "validate":
        validate(args, manifest_path, raw_path, train_path)


if __name__ == "__main__":
    main()
