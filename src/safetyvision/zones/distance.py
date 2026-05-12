"""Distance-based zone strategy: pixel footpoints -> meters via homography."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from safetyvision.types import Detection
from safetyvision.zones.base import ZoneResult


class _Homography:
    """Thin cv2 wrapper standing in for ``supervision.ViewTransformer``.

    Holds the 3x3 homography matrix mapping pixel coords to forklift-relative
    meters (X = lateral, Y = longitudinal, origin = forklift center).
    """

    def __init__(self, source: np.ndarray, target: np.ndarray):
        m, _ = cv2.findHomography(source, target)
        if m is None:
            raise ValueError(
                "cv2.findHomography returned None — calibration points are degenerate"
            )
        self._m = m.astype(np.float64)

    def transform_points(self, pts: np.ndarray) -> np.ndarray:
        """Map (N,2) pixel points to (N,2) forklift-frame meters."""
        if len(pts) == 0:
            return pts.reshape(-1, 2).astype(np.float64)
        return cv2.perspectiveTransform(
            pts.reshape(-1, 1, 2).astype(np.float64), self._m
        ).reshape(-1, 2)


class DistanceZoneStrategy:
    """Classify a frame's detections by metric distance to the forklift.

    Footpoint = ((x1 + x2) / 2, y2). Each detection's footpoint is projected
    through the homography into the forklift coordinate frame; distance is
    the Euclidean norm to the origin (the forklift). The closest person's
    distance is what the strategy reports.

    Temporal smoothing: rolling median over the last N min-distance values
    to dampen bbox-edge jitter. Buffer is cleared when no detections are
    present so the next person to appear gets an immediate (un-smoothed)
    reading.
    """

    def __init__(
        self,
        calibration_path: str,
        danger_m: float,
        warning_m: float,
        smoothing_frames: int = 3,
    ):
        self._calibration_path = Path(calibration_path)
        if not self._calibration_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

        self._danger_m = float(danger_m)
        self._warning_m = float(warning_m)
        self._smoothing = max(1, int(smoothing_frames))
        self._buffer: list[float] = []
        self._calibration_mtime: float = 0.0
        # Load initial homography (raises on bad calibration, so the worker
        # fails fast at startup rather than silently using stale data).
        self._load_calibration()

    def _load_calibration(self) -> None:
        """Read the JSON, build a fresh homography, snapshot mtime."""
        with open(self._calibration_path) as f:
            data = json.load(f)

        source = np.array(data["source_points"], dtype=np.float32)
        target = np.array(data["target_points"], dtype=np.float32)
        if source.shape != (4, 2) or target.shape != (4, 2):
            raise ValueError(
                f"Calibration must have exactly 4 source and 4 target points; "
                f"got source={source.shape}, target={target.shape}"
            )

        self._homography = _Homography(source, target)
        self._calibration_mtime = self._calibration_path.stat().st_mtime
        # Smoothing buffer holds distances from the previous homography;
        # clear it so the dashboard reflects the new calibration immediately.
        self._buffer.clear()

    def _maybe_reload(self) -> None:
        """Reload homography if the calibration file changed on disk.

        Called on every ``classify()``; one stat() per frame is negligible
        and lets the Calibration UI's POST be visible on the Dashboard
        without restarting the service.
        """
        try:
            mtime = self._calibration_path.stat().st_mtime
        except OSError:
            return
        if mtime == self._calibration_mtime:
            return
        try:
            self._load_calibration()
            logger.info(
                "Distance calibration reloaded from {}", self._calibration_path
            )
        except (OSError, ValueError, KeyError) as e:
            # Keep the previous (valid) homography; bump the mtime so we
            # don't retry on every frame until the file changes again.
            try:
                self._calibration_mtime = self._calibration_path.stat().st_mtime
            except OSError:
                pass
            logger.warning(
                "Calibration reload failed, keeping previous homography: {}", e
            )

    def classify(
        self,
        detections: list[Detection],
        frame_h: int,
        frame_w: int,
    ) -> ZoneResult:
        self._maybe_reload()
        if not detections:
            self._buffer.clear()
            return ZoneResult(zone_level="", distance_m=None)

        footpoints = np.array(
            [((d.x1 + d.x2) / 2.0, d.y2) for d in detections],
            dtype=np.float32,
        )
        world_pts = self._homography.transform_points(footpoints)
        distances = np.linalg.norm(world_pts, axis=1)
        min_dist = float(distances.min())

        self._buffer.append(min_dist)
        if len(self._buffer) > self._smoothing:
            self._buffer.pop(0)
        smoothed = float(np.median(self._buffer))

        if smoothed <= self._danger_m:
            zone = "danger"
        elif smoothed <= self._warning_m:
            zone = "medium"
        else:
            zone = ""

        return ZoneResult(zone_level=zone, distance_m=smoothed)
