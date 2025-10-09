#!/usr/bin/env python3
"""
A script to visualize tracking errors (False Negatives and ID Switches)
by extracting and annotating the corresponding image frames.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict
from dataclasses import dataclass

# --- Constants for Visualization ---
GT_COLOR = (0, 255, 0)  # Green for Ground Truth
PRED_COLOR = (0, 0, 255)  # Red for Predictions
FN_COLOR = (255, 165, 0)  # Orange for False Negatives
IDSW_COLOR = (255, 0, 255)  # Magenta for ID Switches
TEXT_COLOR = (255, 255, 255)  # White
IOU_THRESHOLD = 0.5


@dataclass
class Bbox:
    x: float
    y: float
    w: float
    h: float

    def to_xyxy(self):
        return int(self.x), int(self.y), int(self.x + self.w), int(self.y + self.h)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize tracking errors.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to the experiment run directory.",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        required=True,
        help="Sequence name to process (e.g., SNGS-024).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default="data/SoccerNetGS",
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="outputs/error_visualizations",
        help="Directory to save visualizations.",
    )
    parser.add_argument(
        "--top-k-errors",
        type=int,
        default=10,
        help="Number of error frames to save for each category.",
    )
    return parser.parse_args()


def load_predictions(run_dir: Path, sequence: str):
    """Loads tracker predictions and groups them by frame."""
    pred_file = run_dir / "exports" / f"{sequence}.json"
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")
    with pred_file.open() as f:
        data = json.load(f)

    preds_by_frame = defaultdict(list)
    for pred in data["predictions"]:
        frame_idx = int(pred["image_id"][-6:])
        preds_by_frame[frame_idx].append(pred)
    return preds_by_frame


def find_sequence_split(data_root: Path, sequence: str):
    """Finds which split (train/valid/test) a sequence belongs to."""
    for split in ["train", "valid", "test"]:
        if (data_root / split / sequence).exists():
            return split
    raise FileNotFoundError(
        f"Sequence '{sequence}' not found in {data_root}/{{train,valid,test}}"
    )


def load_ground_truth(data_root: Path, sequence: str, split: str):
    """Loads ground truth annotations and groups them by frame."""
    gt_file = data_root / split / sequence / "Labels-GameState.json"
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
    with gt_file.open() as f:
        data = json.load(f)

    gt_by_frame = defaultdict(list)
    for ann in data["annotations"]:
        frame_idx = int(ann["image_id"][-6:])
        gt_by_frame[frame_idx].append(ann)
    return gt_by_frame, data["info"]


def calculate_iou(boxA, boxB):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA.x, boxB.x)
    yA = max(boxA.y, boxB.y)
    xB = min(boxA.x + boxA.w, boxB.x + boxB.w)
    yB = min(boxA.y + boxA.h, boxB.y + boxB.h)

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = boxA.w * boxA.h
    boxBArea = boxB.w * boxB.h

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def find_errors(gt_by_frame, preds_by_frame):
    """Identify ID switches and False Negatives."""
    # First pass: create a global mapping from GT ID to the most likely Pred ID
    gt_to_pred_iou = defaultdict(lambda: defaultdict(float))
    for frame_idx, gts in gt_by_frame.items():
        preds = preds_by_frame.get(frame_idx, [])
        for gt_ann in gts:
            gt_box = Bbox(**gt_ann["bbox"])
            gt_id = gt_ann["track_id"]
            for pred in preds:
                pred_box = Bbox(**pred["bbox_image"])
                iou = calculate_iou(gt_box, pred_box)
                if iou > 0.1:  # Use a low threshold to accumulate evidence
                    gt_to_pred_iou[gt_id][pred["track_id"]] += iou

    track_map = {}
    for gt_id, pred_ious in gt_to_pred_iou.items():
        if pred_ious:
            # Assign the pred_id with the highest total IoU to this gt_id
            best_pred_id = max(pred_ious.keys(), key=lambda k: pred_ious[k])
            track_map[gt_id] = best_pred_id

    # Second pass: find errors frame by frame
    id_switches = []
    false_negatives = []
    last_pred_id_for_gt = {}

    sorted_frames = sorted(gt_by_frame.keys())
    for frame_idx in sorted_frames:
        gts = gt_by_frame.get(frame_idx, [])
        preds = preds_by_frame.get(frame_idx, [])

        # Find false negatives
        for gt_ann in gts:
            gt_box = Bbox(**gt_ann["bbox"])
            is_matched = False
            for pred in preds:
                pred_box = Bbox(**pred["bbox_image"])
                if calculate_iou(gt_box, pred_box) > IOU_THRESHOLD:
                    is_matched = True
                    break
            if not is_matched:
                false_negatives.append((frame_idx, gt_ann))

        # Find ID switches
        for gt_id, correct_pred_id in track_map.items():
            # Find the GT annotation for this track in this frame
            gt_ann = next((g for g in gts if g["track_id"] == gt_id), None)
            if not gt_ann:
                continue

            gt_box = Bbox(**gt_ann["bbox"])
            # Find the best matching prediction in this frame
            best_match_pred = None
            max_iou = IOU_THRESHOLD
            for pred in preds:
                pred_box = Bbox(**pred["bbox_image"])
                iou = calculate_iou(gt_box, pred_box)
                if iou > max_iou:
                    max_iou = iou
                    best_match_pred = pred

            if best_match_pred:
                current_pred_id = best_match_pred["track_id"]
                last_pred_id = last_pred_id_for_gt.get(gt_id)

                # An ID switch occurs if the current pred_id is different from the last one we saw for this GT track
                if last_pred_id is not None and current_pred_id != last_pred_id:
                    id_switches.append(
                        (frame_idx, gt_ann, last_pred_id, current_pred_id)
                    )

                last_pred_id_for_gt[gt_id] = current_pred_id
            else:
                # The track is lost, so clear the last seen pred_id
                if gt_id in last_pred_id_for_gt:
                    del last_pred_id_for_gt[gt_id]

    return id_switches, false_negatives, track_map


def draw_visualizations(image_path, frame_idx, gts, preds, errors, output_dir):
    """Draw bounding boxes and error info on an image and save it."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return

    # Draw GT boxes
    for gt in gts:
        box = Bbox(**gt["bbox"]).to_xyxy()
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), GT_COLOR, 2)
        cv2.putText(
            img,
            f'GT-{gt["track_id"]}',
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            GT_COLOR,
            2,
        )

    # Draw Pred boxes
    for pred in preds:
        box = Bbox(**pred["bbox_image"]).to_xyxy()
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), PRED_COLOR, 2)
        cv2.putText(
            img,
            f'P-{pred["track_id"]}',
            (box[0], box[3] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            PRED_COLOR,
            2,
        )

    # Highlight errors
    for error_type, ann, details in errors:
        if error_type == "FN":
            box = Bbox(**ann["bbox"]).to_xyxy()
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), FN_COLOR, 3)
            cv2.putText(
                img,
                "FN",
                (box[0], box[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                FN_COLOR,
                2,
            )
        elif error_type == "IDSW":
            box = Bbox(**ann["bbox"]).to_xyxy()
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), IDSW_COLOR, 3)
            msg = f"IDSW: {details[0]}->{details[1]}"
            cv2.putText(
                img,
                msg,
                (box[0], box[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                IDSW_COLOR,
                2,
            )

    filename = f"{frame_idx:06d}_{'_'.join(e[0] for e in errors)}.jpg"
    output_path = output_dir / filename
    cv2.imwrite(str(output_path), img)


def main():
    args = parse_args()
    seq_output_dir = args.output_dir / args.sequence
    seq_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Visualizing errors for sequence: {args.sequence}")
    print(f"Run directory: {args.run_dir}")
    print(f"Outputting visualizations to: {seq_output_dir}")

    try:
        # 1. Load data
        split = find_sequence_split(args.data_root, args.sequence)
        gt_by_frame, seq_info = load_ground_truth(args.data_root, args.sequence, split)
        preds_by_frame = load_predictions(args.run_dir, args.sequence)

        image_dir = args.data_root / split / args.sequence / seq_info["im_dir"]

        # 2. Find errors
        id_switches, false_negatives, track_map = find_errors(
            gt_by_frame, preds_by_frame
        )
        print(f"Found {len(track_map)} stable track matches.")
        print(f"Found {len(id_switches)} ID switch events.")
        print(f"Found {len(false_negatives)} false negative events.")

        # 3. Group errors by frame
        errors_by_frame = defaultdict(list)
        for frame_idx, gt_ann in false_negatives:
            errors_by_frame[frame_idx].append(("FN", gt_ann, None))
        for frame_idx, gt_ann, old_id, new_id in id_switches:
            errors_by_frame[frame_idx].append(("IDSW", gt_ann, (old_id, new_id)))

        # 4. Visualize top K error frames (sorted by number of errors)
        sorted_error_frames = sorted(
            errors_by_frame.items(), key=lambda item: len(item[1]), reverse=True
        )

        print(f"\nSaving top {args.top_k_errors} error frames...")
        for i, (frame_idx, errors) in enumerate(
            sorted_error_frames[: args.top_k_errors]
        ):
            image_file = image_dir / f"{frame_idx:06d}.jpg"
            gts_in_frame = gt_by_frame.get(frame_idx, [])
            preds_in_frame = preds_by_frame.get(frame_idx, [])

            draw_visualizations(
                image_file,
                frame_idx,
                gts_in_frame,
                preds_in_frame,
                errors,
                seq_output_dir,
            )

        print(f"Visualization complete. Check the '{seq_output_dir}' directory.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()
