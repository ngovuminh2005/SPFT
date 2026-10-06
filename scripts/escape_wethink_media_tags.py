#!/usr/bin/env python3
"""Escape literal media tags in WeThink, preserving the leading real image placeholder."""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path


REPLACEMENTS = {
    "<image>": "&lt;image&gt;",
    "</image>": "&lt;/image&gt;",
    "<video>": "&lt;video&gt;",
    "</video>": "&lt;/video&gt;",
    "<audio>": "&lt;audio&gt;",
    "</audio>": "&lt;/audio&gt;",
}


def escape_text(text: str) -> tuple[str, int]:
    count = 0
    for tag, escaped in REPLACEMENTS.items():
        count += text.count(tag)
        text = text.replace(tag, escaped)
    return text, count


def escape_sample(sample: dict) -> int:
    if sample.get("videos") or sample.get("audios"):
        raise ValueError("Only image-only WeThink data is supported.")
    messages = sample["conversations"]
    if len(sample.get("images") or []) != 1 or not messages or messages[0].get("from") != "human":
        raise ValueError("Expected one image and a leading human message.")
    if not messages[0]["value"].startswith("<image>"):
        raise ValueError("The first human message must begin with the real <image> placeholder.")
    total = 0
    for index, message in enumerate(messages):
        prefix = "<image>" if index == 0 else ""
        text, count = escape_text(message["value"][len(prefix):])
        message["value"] = prefix + text
        total += count
    if isinstance(sample.get("system"), str):
        sample["system"], count = escape_text(sample["system"])
        total += count
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=Path("data/wethink_sft.jsonl"))
    args = parser.parse_args()
    path = args.path.resolve(strict=True)
    backup = path.with_name(path.name + ".bak")
    changed_rows = 0
    changed_tags = 0

    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as target:
        temporary_path = Path(target.name)
        try:
            with path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        target.write(line)
                        continue
                    sample = json.loads(line)
                    try:
                        row_count = escape_sample(sample)
                    except (ValueError, KeyError, TypeError) as error:
                        raise ValueError(f"Line {line_number}: {error}") from error
                    if row_count:
                        changed_rows += 1
                        changed_tags += row_count
                        target.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    else:
                        target.write(line)
            target.close()
            if changed_rows:
                # Keep old backups if the dataset was prepared again.
                suffix = 1
                while backup.exists():
                    backup = path.with_name(path.name + f".bak.{suffix}")
                    suffix += 1
                with backup.open("xb") as backup_file, path.open("rb") as source:
                    shutil.copyfileobj(source, backup_file)
                # Preserve permissions, but NOT mtime: datasets hashes local
                # source files by mtime and would otherwise reuse stale Arrow data.
                os.chmod(temporary_path, path.stat().st_mode)
                os.replace(temporary_path, path)
                print(f"Backup: {backup}")
            print(f"Changed {changed_tags} tags in {changed_rows} rows: {path}")
        finally:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
