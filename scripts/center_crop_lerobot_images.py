#!/usr/bin/env python3
"""Correct 16:9 camera frames that were squeezed into 640x480 LeRobot images.

The stored 640x480 image is center-cropped to 480x480 (x=80:560) and resized
back to 640x480. This is geometrically equivalent to center-cropping the
original 1280x720 frame to 960x720 and then resizing it to 640x480.
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


IMAGE_COLUMNS = ("image", "wrist_image", "side_image")
EXPECTED_HEIGHT = 480
EXPECTED_WIDTH = 640
CROP_WIDTH = 480
CROP_LEFT = (EXPECTED_WIDTH - CROP_WIDTH) // 2


def _transform_encoded_image(value: dict, jpeg_quality: int) -> dict:
    encoded = value.get("bytes")
    if encoded is None:
        raise ValueError(f"Image entry has no inline bytes: {value.get('path')!r}")

    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode JPEG image")
    if image.shape[:2] != (EXPECTED_HEIGHT, EXPECTED_WIDTH):
        raise ValueError(
            f"Expected {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}, got "
            f"{image.shape[1]}x{image.shape[0]}"
        )

    cropped = image[:, CROP_LEFT : CROP_LEFT + CROP_WIDTH]
    corrected = cv2.resize(
        cropped,
        (EXPECTED_WIDTH, EXPECTED_HEIGHT),
        interpolation=cv2.INTER_LANCZOS4,
    )
    ok, output = cv2.imencode(
        ".jpg",
        corrected,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    )
    if not ok:
        raise ValueError("Failed to encode corrected JPEG image")
    return {"bytes": output.tobytes(), "path": value.get("path")}


def _process_episode(
    source_path_str: str,
    source_root_str: str,
    output_root_str: str,
    jpeg_quality: int,
) -> tuple[str, int]:
    source_path = Path(source_path_str)
    source_root = Path(source_root_str)
    output_root = Path(output_root_str)
    relative_path = source_path.relative_to(source_root)
    output_path = output_root / relative_path

    table = pq.read_table(source_path)
    original_non_image = table.drop(list(IMAGE_COLUMNS))

    for column_name in IMAGE_COLUMNS:
        index = table.schema.get_field_index(column_name)
        if index < 0:
            raise ValueError(f"{source_path}: missing image column {column_name!r}")
        field = table.schema.field(index)
        values = [
            _transform_encoded_image(value.as_py(), jpeg_quality)
            for value in table[column_name]
        ]
        table = table.set_column(index, field, pa.array(values, type=field.type))

    if not table.drop(list(IMAGE_COLUMNS)).equals(original_non_image):
        raise RuntimeError(f"{source_path}: a non-image column changed")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary_path, compression="snappy")
    temporary_path.replace(output_path)
    return relative_path.as_posix(), table.num_rows


def _copy_metadata(source_root: Path, output_root: Path, episode_count: int | None) -> None:
    shutil.copytree(source_root / "meta", output_root / "meta")
    if episode_count is None:
        return

    for filename in ("episodes.jsonl", "episodes_stats.jsonl"):
        path = output_root / "meta" / filename
        if not path.exists():
            continue
        rows = path.read_text(encoding="utf-8").splitlines()[:episode_count]
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    info_path = output_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    episode_rows = [
        json.loads(line)
        for line in (output_root / "meta" / "episodes.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]
    info["total_episodes"] = episode_count
    info["total_frames"] = sum(int(row["length"]) for row in episode_rows)
    info["splits"] = {"train": f"0:{episode_count}"}
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=4) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Only process the first N episodes (for validation).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source.resolve()
    output_root = args.output.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    episode_paths = sorted((source_root / "data").rglob("episode_*.parquet"))
    if args.max_episodes is not None:
        episode_paths = episode_paths[: args.max_episodes]
    if not episode_paths:
        raise ValueError("No episode parquet files found")

    output_root.mkdir(parents=True)
    _copy_metadata(source_root, output_root, args.max_episodes)

    total_frames = 0
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _process_episode,
                str(path),
                str(source_root),
                str(output_root),
                args.jpeg_quality,
            ): path
            for path in episode_paths
        }
        for future in as_completed(futures):
            relative_path, frame_count = future.result()
            completed += 1
            total_frames += frame_count
            print(
                f"[{completed}/{len(episode_paths)}] {relative_path}: "
                f"{frame_count} frames",
                flush=True,
            )

    print(
        f"[ok] corrected {completed} episodes / {total_frames} frames -> {output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
