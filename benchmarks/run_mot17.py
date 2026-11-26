
import argparse
import csv
import json
import re
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import motmetrics as mm
import numpy as np

# Compatibility fix for motmetrics with NumPy 2.x
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

from backend.app.core.argus_engine import ArgusEngine
SEQUENCE = ROOT / "datasets" / "MOT17" / "MOT17" / "train" / "MOT17-02-FRCNN"
IMAGE_DIR = SEQUENCE / "img1"
GT_FILE = SEQUENCE / "gt" / "gt.txt"
MODEL = ROOT / "models" / "yolo26n.pt"
OUT_DIR = ROOT / "outputs" / "mot17_benchmark"


def load_ground_truth(path, max_frames):
    """Load marked pedestrian annotations from MOT17 GT."""
    rows = []

    with path.open("r", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 9:
                continue

            frame = int(float(row[0]))
            identity = int(float(row[1]))
            x, y, w, h = map(float, row[2:6])
            mark = float(row[6])
            cls = int(float(row[7]))

            # MOT17 pedestrian class, marked as valid.
            if mark != 1 or cls != 1:
                continue

            if max_frames and frame > max_frames:
                continue

            rows.append((frame, identity, x, y, w, h))

    return rows


def save_predictions(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        for frame, identity, x, y, w, h in sorted(rows):
            writer.writerow([
                frame, identity, x, y, w, h, 1, -1, -1, -1
            ])


def evaluate(gt_rows, pred_rows, frame_count):
    """Calculate MOT metrics using IoU matching at 0.5."""
    gt_by_frame = {}
    pred_by_frame = {}

    for frame, identity, x, y, w, h in gt_rows:
        gt_by_frame.setdefault(frame, []).append(
            (identity, [x, y, w, h])
        )

    for frame, identity, x, y, w, h in pred_rows:
        pred_by_frame.setdefault(frame, []).append(
            (identity, [x, y, w, h])
        )

    accumulator = mm.MOTAccumulator(auto_id=True)

    for frame in range(1, frame_count + 1):
        gt = gt_by_frame.get(frame, [])
        pred = pred_by_frame.get(frame, [])

        gt_ids = [item[0] for item in gt]
        pred_ids = [item[0] for item in pred]

        gt_boxes = np.asarray(
            [item[1] for item in gt], dtype=float
        ).reshape((-1, 4))

        pred_boxes = np.asarray(
            [item[1] for item in pred], dtype=float
        ).reshape((-1, 4))

        distances = mm.distances.iou_matrix(
            gt_boxes, pred_boxes, max_iou=0.5
        )

        accumulator.update(gt_ids, pred_ids, distances)

    metrics = [
        "num_frames",
        "num_objects",
        "num_predictions",
        "num_matches",
        "num_switches",
        "num_false_positives",
        "num_misses",
        "mota",
        "motp",
        "idf1",
    ]

    summary = mm.metrics.create().compute(
        accumulator,
        metrics=metrics,
        name="ARGUS-X",
    )

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-frames",
        type=int,
        default=20,
        help="Frames to process; use 0 for the entire sequence.",
    )
    args = parser.parse_args()

    max_frames = max(0, args.max_frames)

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    if not GT_FILE.is_file():
        raise FileNotFoundError(f"Ground truth not found: {GT_FILE}")

    if not MODEL.is_file():
        raise FileNotFoundError(f"YOLO model not found: {MODEL}")

    images = sorted(
        IMAGE_DIR.glob("*.jpg"),
        key=lambda p: int(p.stem),
    )

    if max_frames:
        images = images[:max_frames]

    if not images:
        raise RuntimeError("No image frames found.")

    frame_count = len(images)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("ARGUS-X MOT17 benchmark")
    print(f"Sequence: {SEQUENCE.name}")
    print(f"Frames: {frame_count}")
    print(f"Model: {MODEL.name}")
    print("Device: auto")
    print("Tracker: ByteTrack")
    print("Person class: 0")
    print("ReID: colour, threshold 0.80")
    print()

    gt_rows = load_ground_truth(GT_FILE, frame_count)

    engine = ArgusEngine(
        model_path=str(MODEL),
        conf=0.20,
        imgsz=960,
        classes=[0],
        reid=True,
        reid_thr=0.80,
        missing_after=5,
        archive_after=300,
        tracker="bytetrack.yaml",
        appearance="colour",
        gate=True,
        device="auto",
        start_no=1,
        confirm_frames=4,
        reid_window=90,
        track_buffer=60,
    )

    predictions = set()
    start = time.time()

    for index, image_path in enumerate(images, start=1):
        frame = cv2.imread(str(image_path))

        if frame is None:
            print(f"WARNING: Could not read {image_path.name}")
            continue

        engine.process(frame)

        # The engine can add earlier confirmation frames as well
        # as the current frame. A set prevents duplicate rows.
        for f, argus_id, x1, y1, x2, y2 in engine.tracks:
            match = re.fullmatch(r"ARGUS-(\d+)", argus_id)
            if not match:
                continue

            identity = int(match.group(1))
            width = max(0.0, float(x2 - x1))
            height = max(0.0, float(y2 - y1))

            if f <= frame_count and width > 0 and height > 0:
                predictions.add((
                    int(f),
                    identity,
                    float(x1),
                    float(y1),
                    width,
                    height,
                ))

        if index == 1 or index % 5 == 0 or index == frame_count:
            elapsed = time.time() - start
            fps = index / elapsed if elapsed else 0
            print(
                f"Processed {index}/{frame_count} frames "
                f"| {fps:.2f} FPS "
                f"| identities: {len(engine.ids)}"
            )

    elapsed = time.time() - start
    pred_rows = sorted(predictions)

    pred_file = OUT_DIR / "argus_mot17_02_frcnn.txt"
    save_predictions(pred_file, pred_rows)

    summary = evaluate(gt_rows, pred_rows, frame_count)

    summary_file = OUT_DIR / "argus_mot17_02_frcnn_metrics.csv"
    summary.to_csv(summary_file)

    settings = {
        "sequence": SEQUENCE.name,
        "frames_processed": frame_count,
        "model": str(MODEL),
        "confidence": 0.20,
        "image_size": 960,
        "classes": [0],
        "tracker": "bytetrack.yaml",
        "reid": True,
        "appearance": "colour",
        "reid_threshold": 0.80,
        "location_gate": True,
        "reid_window": 90,
        "confirm_frames": 4,
        "device": engine.device,
        "processing_seconds": round(elapsed, 2),
        "processing_fps": round(frame_count / elapsed, 2) if elapsed else 0,
        "ground_truth_pedestrian_rows": len(gt_rows),
        "prediction_rows": len(pred_rows),
        "evaluation_note": (
            "Preliminary marked-pedestrian evaluation. "
            "MOTChallenge distractor/ignore-region handling "
            "has not been applied."
        ),
    }

    settings_file = OUT_DIR / "benchmark_settings.json"
    settings_file.write_text(json.dumps(settings, indent=2))

    print("\n--- RESULTS ---")
    print(summary.to_string())
    print(f"\nProcessing time: {elapsed:.2f} seconds")
    print(f"Predictions: {len(pred_rows)}")
    print(f"Ground-truth pedestrian rows: {len(gt_rows)}")
    print(f"\nSaved predictions: {pred_file}")
    print(f"Saved metrics: {summary_file}")
    print(f"Saved settings: {settings_file}")
    print("\nBenchmark run finished.")


if __name__ == "__main__":
    main()
