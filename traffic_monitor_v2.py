#!/usr/bin/env python3
"""
traffic_monitor.py
Moving-camera traffic monitoring V2 for Raspberry Pi 4.

Design goals:
- CPU-only, low memory, low latency.
- One lightweight YOLO ONNX detector.
- Motion-compensated tracker with constant-velocity prediction and IoU/center association.
- Optional ROI filtering; no fixed-camera motion assumption.
- Persistent IDs, vehicle counting, direction, optional speed estimation.
- Latest-frame-wins capture so processing never queues stale frames.
- Optional annotated video recording.
- No Faster R-CNN, no multiprocessing detector, no heavy appearance model.

Tested conceptually against standard Ultralytics YOLOv8 ONNX output:
    [1, 84, N] or [1, N, 84]
where each row is [cx, cy, w, h, class scores...].

Recommended Pi 4 operating point:
    640x480 camera/video
    YOLO input 320x320
    confidence ~0.25 (motorcycles benefit from the lower threshold)
    1-2 ONNX Runtime threads
    ROI covering only the road
    process every frame if fast enough, otherwise --process-every 2

Dependencies:
    pip install numpy opencv-python onnxruntime

Example:
    python3 traffic_monitor.py --model yolov8n.onnx --source 0

Video:
    python3 traffic_monitor.py --model yolov8n.onnx --source traffic.mp4

Road ROI:
    --roi "0,120 640,120 640,480 0,480"

Counting line:
    --line 360

Save annotated stream:
    --save-video traffic_record.mp4

Useful keys:
    q / ESC : quit
    p       : pause/resume
    r       : reset counts and tracks
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, Thread
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# COCO names. YOLO uses 0-based class IDs.
# ---------------------------------------------------------------------------

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# For a traffic monitor, only these COCO classes are useful by default.
DEFAULT_VEHICLE_CLASSES = {
    "car", "motorcycle", "bus", "truck", "bicycle", "person"
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def parse_source(value: str):
    """Turn a numeric camera source into int; otherwise keep it as a path/URL."""
    try:
        return int(value)
    except ValueError:
        return value


def parse_points(text: str) -> np.ndarray:
    """
    Parse:
        "0,120 640,120 640,480 0,480"
    into an Nx2 float32 polygon.
    """
    pts = []
    for token in text.replace(";", " ").split():
        x, y = token.split(",")
        pts.append((float(x), float(y)))
    if len(pts) < 3:
        raise ValueError("ROI needs at least 3 points")
    return np.asarray(pts, dtype=np.float32)


def parse_class_list(text: str) -> set[str]:
    names = {x.strip().lower() for x in text.split(",") if x.strip()}
    unknown = names - set(COCO_CLASSES)
    if unknown:
        raise ValueError(
            f"Unknown COCO classes: {sorted(unknown)}"
        )
    return names


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU for xyxy boxes."""
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def bbox_center(box: np.ndarray) -> Tuple[float, float]:
    return (
        (float(box[0]) + float(box[2])) * 0.5,
        (float(box[1]) + float(box[3])) * 0.5,
    )


def point_in_polygon(point: Tuple[float, float], polygon: Optional[np.ndarray]) -> bool:
    if polygon is None:
        return True
    return cv2.pointPolygonTest(polygon.reshape((-1, 1, 2)), point, False) >= 0


def signed_side_of_line(x: float, line_x: float) -> int:
    """-1 left, +1 right, 0 on line."""
    if x < line_x - 2:
        return -1
    if x > line_x + 2:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Latest-frame-wins camera
# ---------------------------------------------------------------------------

class LatestFrameCamera:
    def __init__(self, source, width: int, height: int, camera_fps: float):
        self.source = source
        self.width = width
        self.height = height
        self.camera_fps = camera_fps

        self.cap: Optional[cv2.VideoCapture] = None
        self.lock = Lock()
        self.latest: Optional[np.ndarray] = None
        self.frame_index = 0
        self.running = False
        self.thread: Optional[Thread] = None
        self.error: Optional[str] = None

    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open source: {self.source}")

        # CAP_PROP_BUFFERSIZE is not honored by every backend, but is harmless.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.camera_fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)

        self.running = True
        self.thread = Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        try:
            while self.running:
                ok, frame = self.cap.read()
                if not ok:
                    self.running = False
                    break
                with self.lock:
                    self.latest = frame
                    self.frame_index += 1
        except Exception as exc:
            self.error = str(exc)
            self.running = False

    def read(self) -> Tuple[Optional[np.ndarray], int]:
        with self.lock:
            if self.latest is None:
                return None, self.frame_index
            return self.latest.copy(), self.frame_index

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


# ---------------------------------------------------------------------------
# YOLO ONNX detector
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    box: np.ndarray
    score: float
    class_id: int
    class_name: str


class YOLODetector:
    """
    Lightweight YOLO ONNX wrapper.

    Handles common Ultralytics YOLOv8/v9-style exports with:
        [1, C, N]
    or:
        [1, N, C]

    It intentionally does not depend on the ultralytics Python package.
    """

    def __init__(
        self,
        model_path: str,
        input_size: int = 320,
        conf_threshold: float = 0.40,
        nms_threshold: float = 0.45,
        threads: int = 2,
        allowed_classes: Optional[set[str]] = None,
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required. Activate your Pi venv and run "
                "'pip install onnxruntime'."
            ) from exc

        self.ort = ort
        self.model_path = str(model_path)
        self.input_size = int(input_size)
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.allowed_classes = allowed_classes or DEFAULT_VEHICLE_CLASSES

        available = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if "CPUExecutionProvider" not in available:
            raise RuntimeError(f"CPUExecutionProvider unavailable: {available}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, int(threads))
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_mem_pattern = True
        opts.enable_cpu_mem_arena = True
        opts.log_severity_level = 3

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=opts,
            providers=providers,
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        shape = self.session.get_inputs()[0].shape

        # Most exported YOLO models are NCHW.
        self.model_height = self._static_dim(shape[2], self.input_size)
        self.model_width = self._static_dim(shape[3], self.input_size)

        if self.model_height != self.model_width:
            raise ValueError(
                f"Non-square YOLO input is not supported by this lightweight wrapper: "
                f"{self.model_width}x{self.model_height}"
            )

        self.input_size = self.model_width
        self.model_name = Path(model_path).name

        print(
            f"[YOLO] {self.model_name} | input={self.input_size}x{self.input_size} "
            f"| threads={threads} | providers={self.session.get_providers()}"
        )

    @staticmethod
    def _static_dim(value, fallback: int) -> int:
        return int(value) if isinstance(value, (int, np.integer)) else int(fallback)

    def _letterbox(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, float, float, float]:
        h, w = frame.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        nw = int(round(w * scale))
        nh = int(round(h * scale))

        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        pad_x = (self.input_size - nw) / 2.0
        pad_y = (self.input_size - nh) / 2.0

        # uint8 canvas is enough; conversion to float happens once below.
        canvas = np.full(
            (self.input_size, self.input_size, 3), 114, dtype=np.uint8
        )
        left = int(round(pad_x))
        top = int(round(pad_y))
        canvas[top:top + nh, left:left + nw] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

        return tensor, scale, pad_x, pad_y

    def _decode(self, raw: np.ndarray, frame_shape) -> List[Detection]:
        h, w = frame_shape[:2]
        out = np.asarray(raw)

        if out.ndim == 3:
            out = out[0]

        # Convert [C,N] -> [N,C] when necessary.
        if out.ndim != 2:
            raise RuntimeError(f"Unsupported YOLO output shape: {raw.shape}")

        # Standard YOLOv8 COCO has 84 channels for [1,84,N].
        if out.shape[0] < out.shape[1] and out.shape[0] <= 512:
            out = out.T

        if out.shape[1] <= 4:
            raise RuntimeError(f"YOLO output has too few columns: {out.shape}")

        n_classes = out.shape[1] - 4
        if n_classes > len(COCO_CLASSES):
            # This is safer than silently assigning wrong names.
            n_classes = len(COCO_CLASSES)

        boxes = []
        scores = []
        class_ids = []

        for row in out:
            cx, cy, bw, bh = map(float, row[:4])
            cls_scores = row[4:4 + n_classes]

            cls_id = int(np.argmax(cls_scores))
            cls_score = float(cls_scores[cls_id])

            # For YOLOv8 exports, class scores are already confidence scores.
            score = cls_score

            name = COCO_CLASSES[cls_id]
            if name not in self.allowed_classes:
                continue

            # Small/low-contrast motorcycles are commonly scored lower than
            # cars in a 320x320 CPU model. Keep a class-specific floor while
            # retaining the user's global threshold for other classes.
            class_floor = {
                "motorcycle": min(self.conf_threshold, 0.22),
                "bicycle": min(self.conf_threshold, 0.25),
                "person": min(self.conf_threshold, 0.25),
            }.get(name, self.conf_threshold)
            if score < class_floor:
                continue

            # Ultralytics YOLOv8 ONNX exports normally use model-input pixels.
            x1 = (cx - bw * 0.5 - 0.0)
            y1 = (cy - bh * 0.5 - 0.0)
            x2 = (cx + bw * 0.5)
            y2 = (cy + bh * 0.5)

            # Undo centered letterbox.
            # Padding is applied after resize, so subtract it first.
            # These values are still in model-input coordinates.
            # pad_x/pad_y are supplied by infer() after this decoder call.
            boxes.append((x1, y1, x2, y2))
            scores.append(score)
            class_ids.append(cls_id)

        if not boxes:
            return []

        return self._map_boxes(
            np.asarray(boxes, dtype=np.float32),
            np.asarray(scores, dtype=np.float32),
            np.asarray(class_ids, dtype=np.int32),
            w,
            h,
            self._last_scale,
            self._last_pad_x,
            self._last_pad_y,
        )

    def _map_boxes(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        frame_w: int,
        frame_h: int,
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> List[Detection]:
        detections: List[Detection] = []

        for box, score, cls_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box

            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

            x1 = max(0.0, min(float(frame_w - 1), x1))
            y1 = max(0.0, min(float(frame_h - 1), y1))
            x2 = max(0.0, min(float(frame_w - 1), x2))
            y2 = max(0.0, min(float(frame_h - 1), y2))

            if x2 <= x1 or y2 <= y1:
                continue

            area = (x2 - x1) * (y2 - y1)
            # Small motorcycles can occupy only a few dozen pixels.
            if area < 24:
                continue

            detections.append(
                Detection(
                    box=np.asarray([x1, y1, x2, y2], dtype=np.float32),
                    score=float(score),
                    class_id=int(cls_id),
                    class_name=COCO_CLASSES[int(cls_id)],
                )
            )

        return self._class_aware_nms(detections)

    def _class_aware_nms(self, detections: List[Detection]) -> List[Detection]:
        if len(detections) <= 1:
            return detections

        kept: List[Detection] = []

        # Per-class NMS keeps a car from suppressing a motorcycle, etc.
        by_class: Dict[int, List[Detection]] = {}
        for d in detections:
            by_class.setdefault(d.class_id, []).append(d)

        for cls_dets in by_class.values():
            cls_dets.sort(key=lambda d: d.score, reverse=True)
            while cls_dets:
                best = cls_dets.pop(0)
                kept.append(best)
                cls_dets = [
                    d for d in cls_dets
                    if box_iou(best.box, d.box) < self.nms_threshold
                ]

        return kept

    def infer(self, frame: np.ndarray) -> List[Detection]:
        tensor, scale, pad_x, pad_y = self._letterbox(frame)
        self._last_scale = scale
        self._last_pad_x = pad_x
        self._last_pad_y = pad_y

        raw = self.session.run(
            [self.output_name],
            {self.input_name: tensor},
        )[0]

        return self._decode(raw, frame.shape)


# ---------------------------------------------------------------------------
# Lightweight tracker
# ---------------------------------------------------------------------------

@dataclass
class Track:
    track_id: int
    box: np.ndarray
    class_id: int
    class_name: str
    score: float

    cx: float
    cy: float
    vx: float = 0.0
    vy: float = 0.0

    age: int = 0
    hits: int = 1
    missed: int = 0
    confirmed: bool = False

    last_side: int = 0
    previous_cx: Optional[float] = None
    counted: bool = False

    trail: List[Tuple[int, int]] = field(default_factory=list)


class LightweightTracker:
    """
    Moving-camera friendly tracker for CPU-only Raspberry Pi use.

    It combines:
      - global camera-motion compensation from background optical flow
      - constant-velocity prediction
      - class-aware IoU + normalized center-distance association
      - high/low confidence two-stage matching
      - short occlusion tolerance

    No Re-ID network is used, so the tracker remains practical on a Pi 4.
    """

    def __init__(
        self,
        iou_threshold: float = 0.12,
        max_missed: int = 12,
        min_hits: int = 2,
        velocity_smoothing: float = 0.55,
    ):
        self.iou_threshold = float(iou_threshold)
        self.max_missed = int(max_missed)
        self.min_hits = int(min_hits)
        self.velocity_smoothing = float(velocity_smoothing)
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.prev_gray: Optional[np.ndarray] = None
        self.camera_dx = 0.0
        self.camera_dy = 0.0
        self.camera_scale = 1.0

    def reset(self):
        self.tracks.clear()
        self.next_id = 1
        self.prev_gray = None
        self.camera_dx = self.camera_dy = 0.0
        self.camera_scale = 1.0

    def _estimate_camera_motion(self, frame: np.ndarray) -> None:
        # Downsample heavily: optical flow is only for global camera motion,
        # not object tracking. This keeps the Pi CPU cost small.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 240), interpolation=cv2.INTER_AREA)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.camera_dx = self.camera_dy = 0.0
            self.camera_scale = 1.0
            return

        pts0 = cv2.goodFeaturesToTrack(
            self.prev_gray, maxCorners=80, qualityLevel=0.01,
            minDistance=8, blockSize=7
        )
        if pts0 is None or len(pts0) < 8:
            self.prev_gray = gray
            self.camera_dx = self.camera_dy = 0.0
            self.camera_scale = 1.0
            return

        pts1, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, pts0, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )
        if pts1 is None or status is None:
            self.prev_gray = gray
            self.camera_dx = self.camera_dy = 0.0
            self.camera_scale = 1.0
            return

        good0 = pts0[status.ravel() == 1]
        good1 = pts1[status.ravel() == 1]
        if len(good0) < 8:
            self.prev_gray = gray
            self.camera_dx = self.camera_dy = 0.0
            self.camera_scale = 1.0
            return

        # Partial affine gives translation + small rotation/scale, which is
        # a useful approximation for a vehicle-mounted camera.
        M, inliers = cv2.estimateAffinePartial2D(
            good0, good1, method=cv2.RANSAC,
            ransacReprojThreshold=2.5, maxIters=80, confidence=0.95
        )
        if M is not None:
            dx = float(M[0, 2]) * 2.0
            dy = float(M[1, 2]) * 2.0
            scale = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
            # Reject absurd estimates caused by a moving object dominating
            # the sparse feature set.
            if abs(dx) < 80 and abs(dy) < 80 and 0.85 < scale < 1.18:
                self.camera_dx = 0.7 * self.camera_dx + 0.3 * dx
                self.camera_dy = 0.7 * self.camera_dy + 0.3 * dy
                self.camera_scale = 0.85 * self.camera_scale + 0.15 * scale
            else:
                self.camera_dx *= 0.5
                self.camera_dy *= 0.5
                self.camera_scale = 1.0
        else:
            self.camera_dx *= 0.5
            self.camera_dy *= 0.5
            self.camera_scale = 1.0

        self.prev_gray = gray

    def _predicted_box(self, t: Track) -> np.ndarray:
        p = t.box.astype(np.float32).copy()
        cx, cy = bbox_center(p)
        # Apply estimated camera motion first, then the object's residual
        # velocity. Scale around the current box center.
        s = self.camera_scale
        w = (p[2] - p[0]) * s
        h = (p[3] - p[1]) * s
        pcx = cx + self.camera_dx + t.vx
        pcy = cy + self.camera_dy + t.vy
        return np.asarray([pcx - w/2, pcy - h/2, pcx + w/2, pcy + h/2], dtype=np.float32)

    def _match(self, track_ids: List[int], detections: List[Detection]):
        if not track_ids or not detections:
            return [], track_ids[:], list(range(len(detections)))

        candidates = []
        for tid in track_ids:
            t = self.tracks[tid]
            pred = self._predicted_box(t)
            tcx, tcy = bbox_center(pred)
            pw = max(12.0, float(pred[2] - pred[0]))
            ph = max(12.0, float(pred[3] - pred[1]))
            diag = max(20.0, float(np.hypot(pw, ph)))

            for di, d in enumerate(detections):
                if d.class_id != t.class_id:
                    continue
                dcx, dcy = bbox_center(d.box)
                iou = box_iou(pred, d.box)
                dist = float(np.hypot(dcx - tcx, dcy - tcy)) / diag
                dw = max(1.0, float(d.box[2] - d.box[0]))
                dh = max(1.0, float(d.box[3] - d.box[1]))
                size_ratio = max(dw / pw, pw / dw) * max(dh / ph, ph / dh)

                # Accept either reasonable overlap or a plausible center jump.
                max_dist = 2.4 if t.missed <= 2 else 3.2
                if (iou >= self.iou_threshold or dist <= max_dist) and size_ratio < 5.0:
                    # Higher is better. IoU dominates when available; center
                    # distance prevents ID fragmentation when boxes shift.
                    score = 2.2 * iou + 0.55 / (1.0 + dist) - 0.04 * min(size_ratio, 5.0)
                    candidates.append((score, tid, di))

        candidates.sort(reverse=True)
        used_t, used_d, matches = set(), set(), []
        for score, tid, di in candidates:
            if tid in used_t or di in used_d:
                continue
            used_t.add(tid); used_d.add(di); matches.append((tid, di))
        return (matches,
                [tid for tid in track_ids if tid not in used_t],
                [i for i in range(len(detections)) if i not in used_d])

    def update(self, detections: List[Detection], frame_width: int, frame_height: int, frame: Optional[np.ndarray] = None) -> List[Track]:
        if frame is not None:
            self._estimate_camera_motion(frame)

        for t in self.tracks.values():
            t.age += 1
            t.missed += 1

        high = [d for d in detections if d.score >= 0.45]
        low = [d for d in detections if 0.22 <= d.score < 0.45]
        active_ids = list(self.tracks.keys())

        matches, unmatched_tracks, unmatched_high = self._match(active_ids, high)
        low_matches, _, _ = self._match(unmatched_tracks, low)
        all_matches = [(tid, i, high) for tid, i in matches] + [(tid, i, low) for tid, i in low_matches]

        for tid, di, dets in all_matches:
            t = self.tracks[tid]; d = dets[di]
            old_cx, old_cy = t.cx, t.cy
            new_cx, new_cy = bbox_center(d.box)
            measured_vx = (new_cx - old_cx) - self.camera_dx
            measured_vy = (new_cy - old_cy) - self.camera_dy
            a = self.velocity_smoothing
            t.vx = a * t.vx + (1.0 - a) * measured_vx
            t.vy = a * t.vy + (1.0 - a) * measured_vy
            t.box = d.box.copy(); t.class_id = d.class_id; t.class_name = d.class_name; t.score = d.score
            t.cx, t.cy = new_cx, new_cy; t.missed = 0; t.hits += 1
            t.confirmed = t.confirmed or t.hits >= self.min_hits
            t.trail.append((int(new_cx), int(new_cy)))
            if len(t.trail) > 24: del t.trail[:-24]

        for di in unmatched_high:
            d = high[di]; cx, cy = bbox_center(d.box); tid = self.next_id; self.next_id += 1
            self.tracks[tid] = Track(track_id=tid, box=d.box.copy(), class_id=d.class_id, class_name=d.class_name,
                                     score=d.score, cx=cx, cy=cy, confirmed=(self.min_hits <= 1),
                                     trail=[(int(cx), int(cy))])

        for tid in [tid for tid, t in self.tracks.items() if t.missed > self.max_missed]:
            del self.tracks[tid]

        for t in self.tracks.values():
            t.cx = max(0.0, min(frame_width - 1.0, t.cx)); t.cy = max(0.0, min(frame_height - 1.0, t.cy))
        return list(self.tracks.values())


# ---------------------------------------------------------------------------
# Traffic analytics
# ---------------------------------------------------------------------------

@dataclass
class TrafficStats:
    total_count: int = 0
    up_count: int = 0
    down_count: int = 0
    current_tracks: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)

    def reset(self):
        self.total_count = 0
        self.up_count = 0
        self.down_count = 0
        self.current_tracks = 0
        self.class_counts.clear()


class TrafficAnalyzer:
    def __init__(
        self,
        line_y: Optional[float] = None,
        meters_per_pixel: Optional[float] = None,
        fps_for_speed: float = 0.0,
    ):
        self.line_y = line_y
        self.meters_per_pixel = meters_per_pixel
        self.fps_for_speed = fps_for_speed
        self.stats = TrafficStats()
        self.speed_history: Dict[int, float] = {}

    def reset(self):
        self.stats.reset()
        self.speed_history.clear()

    def update(self, tracks: List[Track], dt: float):
        self.stats.current_tracks = sum(1 for t in tracks if t.confirmed)

        for t in tracks:
            if not t.confirmed:
                continue

            if self.line_y is not None:
                # Count an object when its tracked CENTER actually crosses the
                # vertical line between two consecutive tracker updates. This
                # is more reliable than waiting for last_side to be initialized
                # on both sides of the line, especially with a moving camera.
                current_x = float(t.cx)
                previous_x = t.previous_cx

                if previous_x is not None and not t.counted:
                    crossed = (
                        (previous_x < self.line_y <= current_x)
                        or (previous_x > self.line_y >= current_x)
                    )

                    if crossed:
                        self.stats.total_count += 1
                        self.stats.class_counts[t.class_name] = (
                            self.stats.class_counts.get(t.class_name, 0) + 1
                        )

                        # Vertical line: left -> right is DOWN, right -> left is UP
                        if current_x > previous_x:
                            self.stats.down_count += 1
                        else:
                            self.stats.up_count += 1

                        t.counted = True

                t.previous_cx = current_x

                # Keep side information for compatibility/debugging.
                side = signed_side_of_line(current_x, self.line_y)
                if side != 0:
                    t.last_side = side

            if self.meters_per_pixel and dt > 1e-4:
                px_per_sec = float(np.hypot(t.vx, t.vy)) / dt
                speed_mps = px_per_sec * self.meters_per_pixel
                speed_kmh = speed_mps * 3.6

                # Reject obvious numerical spikes.
                speed_kmh = min(200.0, max(0.0, speed_kmh))
                old = self.speed_history.get(t.track_id, speed_kmh)
                speed_kmh = 0.75 * old + 0.25 * speed_kmh
                self.speed_history[t.track_id] = speed_kmh

    def speed(self, track_id: int) -> Optional[float]:
        return self.speed_history.get(track_id)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class TrafficMonitor:
    def __init__(self, args):
        self.args = args

        allowed = parse_class_list(args.classes)

        self.detector = YOLODetector(
            model_path=args.model,
            input_size=args.input_size,
            conf_threshold=args.conf,
            nms_threshold=args.nms,
            threads=args.threads,
            allowed_classes=allowed,
        )

        self.tracker = LightweightTracker(
            iou_threshold=args.track_iou,
            max_missed=args.max_missed,
            min_hits=args.min_hits,
        )

        roi = parse_points(args.roi) if args.roi else None
        self.roi = roi

        self.analyzer = TrafficAnalyzer(
            line_y=args.line,
            meters_per_pixel=args.meters_per_pixel,
        )

        self.camera = LatestFrameCamera(
            source=parse_source(args.source),
            width=args.width,
            height=args.height,
            camera_fps=args.camera_fps,
        )

        self.writer: Optional[cv2.VideoWriter] = None
        self.paused = False
        self.last_processed_index = -1

        self.proc_times: List[float] = []
        self.fps_ema = 0.0
        self.last_time = time.perf_counter()

    def _roi_mask_filter(self, detections: List[Detection]) -> List[Detection]:
        if self.roi is None:
            return detections

        out = []
        for d in detections:
            cx, cy = bbox_center(d.box)
            if point_in_polygon((cx, cy), self.roi):
                out.append(d)
        return out

    def _draw_roi(self, frame):
        if self.roi is not None:
            pts = self.roi.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (255, 200, 0), 2)

    def _draw_line(self, frame):
        if self.args.line is not None:
            x = int(self.args.line)
            cv2.line(
                frame,
                (x, 0),
                (x, frame.shape[0] - 1),
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "COUNT LINE",
                (max(5, x + 8), 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _draw_track(self, frame, t: Track):
        x1, y1, x2, y2 = map(int, t.box)

        # Confirmed tracks are drawn thicker.
        thickness = 2 if t.confirmed else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), thickness)

        label = f"ID {t.track_id} {t.class_name} {t.score:.2f}"

        speed = self.analyzer.speed(t.track_id)
        if speed is not None:
            label += f" {speed:.0f}km/h"

        cv2.putText(
            frame,
            label,
            (x1, max(16, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )

        if self.args.trails and len(t.trail) >= 2:
            pts = np.asarray(t.trail, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, (0, 220, 255), 1)

    def _overlay(self, frame, detections, tracks, latency_ms):
        h, w = frame.shape[:2]

        panel_h = 92
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (min(w, 570), panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

        fps_text = f"FPS {self.fps_ema:.1f}"
        cv2.putText(
            frame, fps_text, (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            frame,
            f"YOLO {latency_ms:.0f}ms | DET {len(detections)} | TRACKS {len(tracks)}",
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame,
            f"COUNT {self.analyzer.stats.total_count} | "
            f"UP {self.analyzer.stats.up_count} | "
            f"DOWN {self.analyzer.stats.down_count}",
            (10, 67),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame,
            f"MODEL {self.detector.model_name} | ROI {'ON' if self.roi is not None else 'FULL'}",
            (10, 88),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA
        )

    def _open_writer(self, frame):
        if not self.args.save_video:
            return

        h, w = frame.shape[:2]
        fps = self.args.output_fps if self.args.output_fps > 0 else 15.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            self.args.save_video,
            fourcc,
            fps,
            (w, h),
        )
        if not self.writer.isOpened():
            print("[WARN] Could not open output video writer")
            self.writer = None

    def run(self):
        self.camera.start()
        print("[INFO] Traffic monitor started.")
        print("[INFO] Press q/ESC to quit, p to pause, r to reset.")

        displayed_index = -1

        try:
            while True:
                if self.paused:
                    time.sleep(0.05)
                    key = cv2.waitKey(20) & 0xFF
                    if key in (27, ord("q")):
                        break
                    if key == ord("p"):
                        self.paused = False
                    continue

                frame, index = self.camera.read()
                if frame is None:
                    if not self.camera.running:
                        break
                    time.sleep(0.005)
                    continue

                if index == displayed_index:
                    # For a video file, this is the final frame after EOF.
                    if not self.camera.running:
                        break
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    continue

                displayed_index = index

                # Optional frame skipping. For a Pi 4, processing every 2nd
                # frame can dramatically reduce latency when the detector
                # cannot sustain the camera rate.
                if self.args.process_every > 1:
                    if index % self.args.process_every != 0:
                        continue

                start = time.perf_counter()

                detections = self.detector.infer(frame)
                detections = self._roi_mask_filter(detections)

                tracks = self.tracker.update(
                    detections,
                    frame.shape[1],
                    frame.shape[0],
                    frame,
                )

                now = time.perf_counter()
                dt = max(1e-3, now - self.last_time)
                self.last_time = now

                self.analyzer.update(tracks, dt)

                latency_ms = (now - start) * 1000.0
                self.proc_times.append(latency_ms)
                if len(self.proc_times) > 30:
                    self.proc_times.pop(0)

                inst_fps = 1.0 / max(1e-6, now - start)
                self.fps_ema = (
                    inst_fps if self.fps_ema <= 0
                    else 0.85 * self.fps_ema + 0.15 * inst_fps
                )

                # Draw.
                self._draw_roi(frame)
                self._draw_line(frame)

                for t in tracks:
                    # Do not display stale/unconfirmed objects by default.
                    if t.confirmed or t.missed <= 1:
                        self._draw_track(frame, t)

                self._overlay(frame, detections, tracks, latency_ms)

                if self.writer is None:
                    self._open_writer(frame)

                if self.writer is not None:
                    self.writer.write(frame)

                if self.args.display:
                    cv2.imshow("Raspberry Pi Traffic Monitor", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    elif key == ord("p"):
                        self.paused = True
                    elif key == ord("r"):
                        self.tracker.reset()
                        self.analyzer.reset()

        finally:
            self.camera.stop()
            if self.writer is not None:
                self.writer.release()
            cv2.destroyAllWindows()

            if self.proc_times:
                arr = np.asarray(self.proc_times, dtype=np.float32)
                print(
                    f"[INFO] Processing latency: "
                    f"mean={arr.mean():.1f}ms "
                    f"p95={np.percentile(arr, 95):.1f}ms"
                )

            s = self.analyzer.stats
            print(
                f"[INFO] Final count={s.total_count} "
                f"up={s.up_count} down={s.down_count} "
                f"classes={s.class_counts}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Moving-camera traffic monitoring V2 for Raspberry Pi 4"
    )

    p.add_argument("--model", required=True, help="YOLO ONNX model, e.g. yolov8n.onnx")
    p.add_argument("--source", default="0", help="Camera index, video file, or URL")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--camera-fps", type=float, default=20.0)

    # Pi 4 inference settings.
    p.add_argument("--input-size", type=int, default=320)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--process-every", type=int, default=1,
                   help="Run inference every N captured frames")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--nms", type=float, default=0.50)

    # Traffic classes.
    p.add_argument(
        "--classes",
        default="car,motorcycle,bus,truck,bicycle,person",
        help="Comma-separated COCO classes to keep",
    )

    # Optional ROI and vertical counting line.
    p.add_argument(
        "--roi",
        default=None,
        help='Polygon, e.g. "0,120 640,120 640,480 0,480"',
    )
    p.add_argument(
        "--line",
        type=float,
        default=None,
        help="Vertical counting line x-coordinate in pixels",
    )

    p.add_argument(
        "--track-iou",
        type=float,
        default=0.12,
        help="IoU threshold for motion-compensated tracker association",
    )
    p.add_argument("--max-missed", type=int, default=12)
    p.add_argument("--min-hits", type=int, default=2)

    # Optional speed estimation.
    p.add_argument(
        "--meters-per-pixel",
        type=float,
        default=None,
        help="Calibration scale. Do not use arbitrary values.",
    )

    # Output.
    p.add_argument(
        "--save-video",
        default=None,
        help="Save annotated output video",
    )
    p.add_argument("--output-fps", type=float, default=15.0)
    p.add_argument("--trails", action="store_true")

    p.add_argument(
        "--no-display",
        dest="display",
        action="store_false",
        help="Run headless",
    )
    p.set_defaults(display=True)

    return p


def main():
    args = build_parser().parse_args()

    if not Path(args.model).exists():
        print(f"[ERROR] Model not found: {args.model}", file=sys.stderr)
        return 2

    if args.input_size < 160 or args.input_size > 640:
        print("[ERROR] --input-size should normally be between 160 and 640")
        return 2

    if args.process_every < 1:
        print("[ERROR] --process-every must be >= 1")
        return 2

    try:
        app = TrafficMonitor(args)
        app.run()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
