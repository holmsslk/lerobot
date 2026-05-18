import argparse
from pathlib import Path
import shutil
import json
import pandas as pd
import pyarrow as pa

from lerobot.datasets.io_utils import (
    load_info,
    write_info,
    write_tasks,
    write_episodes,
    write_stats,
    aggregate_stats,
)

LEGACY_EPISODES = "meta/episodes.jsonl"
LEGACY_TASKS = "meta/tasks.jsonl"
LEGACY_STATS = "meta/episodes_stats.jsonl"


def safe_load_jsonl(path):
    if not Path(path).exists():
        print(f"[WARN] missing file: {path}, skipping")
        return []
    import jsonlines
    with jsonlines.open(path) as f:
        return list(f)


def convert_tasks(root, out):
    path = root / LEGACY_TASKS
    tasks = safe_load_jsonl(path)

    if not tasks:
        return

    df = pd.DataFrame(
        {"task_index": [t["task_index"] for t in tasks]},
        index=[t["task"] for t in tasks],
    )
    write_tasks(df, out)


def convert_info(root, out):
    info = load_info(root)
    info["codebase_version"] = "v3.0"
    write_info(info, out)


def convert_dataset(root, out):
    data = root / "data"
    ep_files = sorted(data.glob("*/*.parquet"))

    episodes_meta = []
    for i, f in enumerate(ep_files):
        episodes_meta.append({
            "episode_index": i,
            "data_file": str(f)
        })

    return episodes_meta


def convert_stats(root):
    stats = safe_load_jsonl(root / LEGACY_STATS)
    if not stats:
        print("[WARN] no episode stats found, skipping stats conversion")
        return None

    return aggregate_stats([s["stats"] for s in stats])


def main(args):
    root = Path(args.root)
    out = Path(args.output)

    if out.exists():
        shutil.rmtree(out)

    out.mkdir(parents=True)

    print("== convert info ==")
    convert_info(root, out)

    print("== convert tasks ==")
    convert_tasks(root, out)

    print("== convert data ==")
    episodes = convert_dataset(root, out)

    print("== convert stats ==")
    stats = convert_stats(root)

    print("== write episodes ==")
    write_episodes(
        pd.DataFrame(episodes),
        out
    )

    if stats:
        write_stats(stats, out)

    print("DONE ->", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args)
