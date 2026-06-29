#!/usr/bin/env python3
import argparse
import collections
import json
from pathlib import Path


def load_rows(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"event": "json_decode_error", "raw": line[:500]})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize OpenAI/TRAPI problem-call JSONL logs.")
    parser.add_argument("path", nargs="?", default="outputs/openai_monitor/problem_calls.jsonl")
    parser.add_argument("--recent", type=int, default=5)
    args = parser.parse_args()

    rows = load_rows(Path(args.path))
    print(f"rows: {len(rows)}")
    if not rows:
        return 0

    for key in ("event", "model", "resolved_model", "error_type"):
        counts = collections.Counter(row.get(key, "<missing>") for row in rows)
        print(f"\n{key}:")
        for value, count in counts.most_common(12):
            print(f"  {count:5d}  {value}")

    print("\nrecent:")
    for row in rows[-args.recent :]:
        print(
            json.dumps(
                {
                    "event": row.get("event"),
                    "model": row.get("model"),
                    "resolved_model": row.get("resolved_model"),
                    "attempt": row.get("attempt"),
                    "fallback": row.get("fallback"),
                    "elapsed_s": row.get("elapsed_s"),
                    "error_type": row.get("error_type"),
                    "error": row.get("error"),
                },
                ensure_ascii=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
