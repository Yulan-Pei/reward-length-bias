"""Download a deterministic seven-candidate Nectar subset through the HF rows API."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path


DATASET = "berkeley-nest/Nectar"
CONFIG = "default"
SPLIT = "train"
TOTAL_ROWS = 182_954
ROWS_PER_PAGE = 100
API = "https://datasets-server.huggingface.co/rows"


def valid_seven_candidate_row(row: dict) -> bool:
    answers = row.get("answers", [])
    ranks = [answer.get("rank") for answer in answers]
    return (
        row.get("num_responses") == 7
        and len(answers) == 7
        and sorted(ranks) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        and all(answer.get("answer", "").strip() for answer in answers)
    )


def fetch_page(offset: int, timeout: int = 90, attempts: int = 5) -> list[dict]:
    query = urllib.parse.urlencode({
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "offset": offset,
        "length": ROWS_PER_PAGE,
    })
    error = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(f"{API}?{query}", timeout=timeout) as response:
                payload = json.load(response)
            return [item["row"] | {"nectar_row_idx": item["row_idx"]} for item in payload["rows"]]
        except urllib.error.HTTPError as exc:
            error = exc
            if exc.code == 429:
                time.sleep(int(exc.headers.get("Retry-After", 60)))
            else:
                time.sleep(2 ** attempt)
        except Exception as exc:
            error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to fetch offset {offset}") from error


def canonical_hash(rows: list[dict]) -> str:
    content = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/nectar/nectar_7way.jsonl"))
    parser.add_argument("--metadata", type=Path, default=Path("data/nectar/metadata.json"))
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args()
    if args.out.exists() and args.metadata.exists():
        print(f"using existing subset: {args.out}")
        return

    pages = list(range(0, TOTAL_ROWS - ROWS_PER_PAGE + 1, ROWS_PER_PAGE))
    random.Random(args.seed).shuffle(pages)
    rows: list[dict] = []
    used_offsets: list[int] = []
    for offset in pages:
        page = fetch_page(offset)
        used_offsets.append(offset)
        rows.extend(row for row in page if valid_seven_candidate_row(row))
        print(f"offset={offset:6d} valid_rows={len(rows):5d}/{args.rows}")
        if len(rows) >= args.rows:
            break
    rows = rows[: args.rows]
    if len(rows) < args.rows:
        raise RuntimeError(f"only found {len(rows)} valid rows")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metadata = {
        "dataset": DATASET,
        "dataset_url": "https://huggingface.co/datasets/berkeley-nest/Nectar",
        "config": CONFIG,
        "split": SPLIT,
        "total_upstream_rows": TOTAL_ROWS,
        "selection": "seeded shuffle of 100-row API page offsets, then valid seven-candidate filter",
        "seed": args.seed,
        "rows": len(rows),
        "page_offsets": used_offsets,
        "sha256_canonical_rows": canonical_hash(rows),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
