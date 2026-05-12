"""Inference worker – runs YOLO ONNX/OpenVINO on frames, outputs detections."""

from __future__ import annotations

import os
import platform
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import List, Optional

import cv2
import numpy as np
from loguru import logger

from safetyvision.config import SafetyVisionConfig
from safetyvision.types import Detection, DetectionEvent, FramePacket
from safetyvision.zones import ZoneStrategy, create_zone_strategy


def _normalize_runtime(requested_runtime: str, machine: Optional[str] = None) -> str:
    """Map unsupported runtime selections to compatible ones per architecture."""
    machine_name = (machine or platform.machine() or "").lower()
    is_arm = machine_name.startswith("arm") or machine_name.startswith("aarch64")

    # OpenVINO on Raspberry Pi/ARM is commonly unavailable in this stack.
    if requested_runtime == "openvino" and is_arm:
        return "onnxruntime"
    return requested_runtime


def _pin_to_cores(cores: list[int]) -> None:
    try:
        os.sched_setaffinity(0, set(cores))
        logger.debug("Inference thread pinned to cores {}", cores)
    except (AttributeError, OSError):
        pass


def _letterbox(frame: np.ndarray, target: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize with letterbox padding, return (padded, scale, (pad_w, pad_h))."""
    h, w = frame.shape[:2]
    scale = min(target / h, target / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target - new_w) // 2
    pad_h = (target - new_h) // 2
    padded = np.full((target, target, 3), 114, dtype=np.uint8)
    padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
    return padded, scale, (pad_w, pad_h)


def _preprocess(frame: np.ndarray, input_size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """BGR frame -> NCHW float32 blob."""
    padded, scale, pad = _letterbox(frame, input_size)
    blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(blob, 0), scale, pad


def _nms(detections: List[Detection], iou_threshold: float) -> List[Detection]:
    """Simple NMS over a list of Detection objects."""
    if not detections:
        return []

    dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        remaining = []
        for d in dets:
            iou = _compute_iou(best, d)
            if iou < iou_threshold:
                remaining.append(d)
        dets = remaining
    return keep


def _compute_iou(a: Detection, b: Detection) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _postprocess(
    output: np.ndarray,
    conf_thresh: float,
    iou_thresh: float,
    person_class_id: int,
    scale: float,
    pad: tuple[int, int],
) -> List[Detection]:
    """Parse YOLO output tensor into filtered person detections.

    Supports two common YOLO output formats:
      - (1, 84, N) – YOLOv8/YOLO11 style: 4 box coords + 80 class scores
      - (1, N, 85) – YOLOv5 style: 4 box coords + objectness + 80 class scores
    """
    if output.ndim == 3:
        output = output[0]
    if output.ndim != 2:
        logger.warning("Unexpected model output rank: {}", output.ndim)
        return []

    # Export with built-in NMS can produce (N, 6):
    # [x1, y1, x2, y2, score, class_id]
    if output.shape[1] == 6:
        detections: List[Detection] = []
        for row in output:
            conf = float(row[4])
            cls_id = int(row[5])
            if cls_id != person_class_id or conf < conf_thresh:
                continue
            # These coordinates are already in original image space.
            detections.append(
                Detection(
                    x1=float(row[0]),
                    y1=float(row[1]),
                    x2=float(row[2]),
                    y2=float(row[3]),
                    confidence=conf,
                    class_id=cls_id,
                )
            )
        return detections

    # Detect format: (84|85, N) vs (N, 84|85)
    # Use feature-dimension matching instead of shape ordering to avoid false transposes.
    feature_dims = {84, 85}
    rows, cols = output.shape
    if cols in feature_dims and rows not in feature_dims:
        parsed = output
    elif rows in feature_dims and cols not in feature_dims:
        parsed = output.T
    elif cols in feature_dims:
        parsed = output
    elif rows in feature_dims:
        parsed = output.T
    else:
        logger.warning("Unrecognized YOLO output shape: {}", output.shape)
        return []

    has_objectness = parsed.shape[1] > 84

    # Vectorized scoring: avoids a per-row Python loop over thousands of
    # anchors, which was the dominant per-frame cost (12.8→15+ FPS target).
    if has_objectness:
        # YOLOv5: [cx, cy, w, h, obj_conf, cls0, cls1, ...]
        obj_conf = parsed[:, 4]
        class_scores = parsed[:, 5:]
        cls_ids = np.argmax(class_scores, axis=1)
        confs = obj_conf * class_scores[np.arange(class_scores.shape[0]), cls_ids]
    else:
        # YOLOv8/YOLO11/YOLO26: [cx, cy, w, h, cls0, cls1, ...]
        class_scores = parsed[:, 4:]
        cls_ids = np.argmax(class_scores, axis=1)
        confs = class_scores[np.arange(class_scores.shape[0]), cls_ids]

    mask = (cls_ids == person_class_id) & (confs >= conf_thresh)
    if not np.any(mask):
        return []

    kept = parsed[mask]
    kept_confs = confs[mask]
    kept_cls = cls_ids[mask]

    cx = kept[:, 0]
    cy = kept[:, 1]
    w = kept[:, 2]
    h = kept[:, 3]
    x1 = (cx - w / 2 - pad[0]) / scale
    y1 = (cy - h / 2 - pad[1]) / scale
    x2 = (cx + w / 2 - pad[0]) / scale
    y2 = (cy + h / 2 - pad[1]) / scale

    detections: List[Detection] = [
        Detection(
            x1=float(x1[i]),
            y1=float(y1[i]),
            x2=float(x2[i]),
            y2=float(y2[i]),
            confidence=float(kept_confs[i]),
            class_id=int(kept_cls[i]),
        )
        for i in range(kept.shape[0])
    ]

    return _nms(detections, iou_thresh)


class InferenceWorker:
    """Runs YOLO inference on frames from input queue, pushes DetectionEvents."""

    def __init__(
        self,
        cfg: SafetyVisionConfig,
        in_queue: Queue,
        out_queue: Queue,
        stop_event: threading.Event,
        latency_cb=None,
        frame_cb=None,
    ):
        self._cfg = cfg
        self._in_queue = in_queue
        self._out_queue = out_queue
        self._stop = stop_event
        self._latency_cb = latency_cb
        self._frame_cb = frame_cb
        self._session = None
        self._pt_model = None
        self._runtime_type = ""
        self._thread: Optional[threading.Thread] = None
        # Temporal smoothing buffer for person-detected gating.
        self._recent_detections: list[bool] = []
        # Zone strategy (band or distance) chosen by config; raises here on
        # missing/corrupt calibration so systemd notices a misconfigured unit.
        self._zone_strategy: ZoneStrategy = create_zone_strategy(cfg)

    def start(self) -> None:
        self._load_model()
        self._thread = threading.Thread(
            target=self._run, name="inference_worker", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _load_model(self) -> None:
        requested_runtime = self._cfg.model.runtime
        runtime = _normalize_runtime(requested_runtime)

        if runtime != requested_runtime:
            logger.warning(
                "Runtime '{}' is not supported on this architecture; using '{}' instead",
                requested_runtime,
                runtime,
            )

        if runtime == "openvino":
            self._load_openvino()
        elif runtime == "ultralytics":
            self._load_ultralytics(self._cfg.model.path_pt)
        else:
            self._load_onnxruntime(self._cfg.model.path_onnx)

    def _load_onnxruntime(self, model_path: str) -> None:
        try:
            import onnxruntime as ort

            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_opts.intra_op_num_threads = self._cfg.perf.inference_threads
            sess_opts.inter_op_num_threads = 1
            sess_opts.enable_cpu_mem_arena = True

            self._session = ort.InferenceSession(
                model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            self._runtime_type = "onnxruntime"
            logger.info("ONNX Runtime model loaded: {}", model_path)
        except Exception as e:
            pt_path = Path(self._cfg.model.path_pt)
            if pt_path.exists():
                logger.warning(
                    "ONNX Runtime unavailable ({}); falling back to Ultralytics PT model: {}",
                    e,
                    pt_path,
                )
                self._load_ultralytics(str(pt_path))
                return
            raise RuntimeError(
                f"Failed to load ONNX Runtime model '{model_path}': {e}. "
                "Install onnxruntime or provide model.path_pt for ultralytics fallback."
            ) from e

    def _load_openvino(self) -> None:
        try:
            from openvino.runtime import Core

            ov_path = Path(self._cfg.model.path_openvino)
            model_path = str(ov_path if ov_path.exists() else Path(self._cfg.model.path_onnx))
            core = Core()
            model = core.read_model(model_path)
            config = {"INFERENCE_NUM_THREADS": str(self._cfg.perf.inference_threads)}
            self._session = core.compile_model(model, "CPU", config)
            self._infer_request = self._session.create_infer_request()
            self._runtime_type = "openvino"
            logger.info("OpenVINO model loaded: {}", model_path)
        except Exception as e:
            fallback_path = self._cfg.model.path_onnx
            logger.warning(
                "OpenVINO load failed ({}), falling back to ONNX Runtime with {}",
                e,
                fallback_path,
            )
            self._load_onnxruntime(fallback_path)

    def _load_ultralytics(self, model_path: str) -> None:
        try:
            from ultralytics import YOLO

            self._pt_model = YOLO(model_path)
            self._runtime_type = "ultralytics"
            logger.info("Ultralytics PT model loaded: {}", model_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Ultralytics PT model '{model_path}': {e}"
            ) from e

    def _infer(self, blob: np.ndarray) -> np.ndarray:
        if self._runtime_type == "openvino":
            self._infer_request.infer({0: blob})
            return self._infer_request.get_output_tensor(0).data.copy()
        else:
            return self._session.run(None, {self._input_name: blob})[0]

    def _infer_pt(self, frame: np.ndarray) -> np.ndarray:
        """Run Ultralytics PT inference and return Nx6 [x1,y1,x2,y2,conf,cls]."""
        results = self._pt_model.predict(
            source=frame,
            imgsz=self._cfg.model.input_size,
            conf=self._cfg.model.conf_threshold,
            iou=self._cfg.model.iou_threshold,
            verbose=False,
            device="cpu",
            classes=[self._cfg.model.person_class_id],
        )
        if not results:
            return np.empty((0, 6), dtype=np.float32)

        boxes = results[0].boxes
        if boxes is None or boxes.data is None:
            return np.empty((0, 6), dtype=np.float32)
        data = boxes.data
        if hasattr(data, "detach"):
            data = data.detach().cpu().numpy()
        else:
            data = np.asarray(data)
        return data.astype(np.float32, copy=False)

    def _apply_temporal_smoothing(self, person_detected: bool) -> bool:
        """Require N consecutive frames of detection to confirm presence."""
        n = self._cfg.perf.temporal_smoothing_frames
        self._recent_detections.append(person_detected)
        if len(self._recent_detections) > n:
            self._recent_detections.pop(0)
        # Person confirmed only if majority of recent frames agree
        return sum(self._recent_detections) >= (n + 1) // 2

    def _run(self) -> None:
        _pin_to_cores(self._cfg.perf.inference_cpu_cores)
        input_size = self._cfg.model.input_size

        while not self._stop.is_set():
            try:
                pkt: FramePacket = self._in_queue.get(timeout=0.1)
            except Empty:
                continue

            t0 = time.time_ns()
            if self._runtime_type == "ultralytics":
                raw_out = self._infer_pt(pkt.frame)
                scale, pad = 1.0, (0, 0)
            else:
                blob, scale, pad = _preprocess(pkt.frame, input_size)
                raw_out = self._infer(blob)
            dets = _postprocess(
                raw_out,
                self._cfg.model.conf_threshold,
                self._cfg.model.iou_threshold,
                self._cfg.model.person_class_id,
                scale,
                pad,
            )
            t1 = time.time_ns()

            raw_detected = len(dets) > 0
            smoothed = self._apply_temporal_smoothing(raw_detected)
            max_conf = max((d.confidence for d in dets), default=0.0)
            frame_h, frame_w = pkt.frame.shape[:2]

            zone = self._zone_strategy.classify(dets, frame_h, frame_w)

            event = DetectionEvent(
                timestamp_ns=pkt.timestamp_ns,
                person_detected=smoothed,
                confidence_max=max_conf,
                bbox_count=len(dets),
                zone_level=zone.zone_level,
                source_id=pkt.source_id,
                distance_m=zone.distance_m,
            )

            # Non-blocking push to decision queue
            while not self._out_queue.empty():
                try:
                    self._out_queue.get_nowait()
                except Exception:
                    break
            try:
                self._out_queue.put_nowait(event)
            except Exception:
                pass

            if self._latency_cb:
                self._latency_cb((t1 - t0) / 1e6)
            if self._frame_cb:
                self._frame_cb()

        logger.info("Inference worker stopped")
