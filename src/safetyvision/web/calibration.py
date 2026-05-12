"""Calibration API: save/load homography points and toggle distance mode.

Single-camera (back) scope. The router is constructed by ``app.py`` via
``create_calibration_router`` so the auth dependency and config path can
be injected without circular imports.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import yaml
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from safetyvision.workers.capture import CALIBRATION_FRAME_PATH

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
REPROJ_ERR_PX = 2.0
DET_MIN = 1e-6


class CalibrationPayload(BaseModel):
    source_points: list[list[float]]
    target_points: list[list[float]]
    frame_width: int
    frame_height: int


class CalibrationError(ValueError):
    """Raised for invalid calibration submissions."""


def _validate(payload: CalibrationPayload) -> np.ndarray:
    """Validate calibration. Returns the 3x3 homography matrix."""
    if len(payload.source_points) != 4 or len(payload.target_points) != 4:
        raise CalibrationError("Exactly 4 source and 4 target points required")

    for i, p in enumerate(payload.source_points):
        if len(p) != 2:
            raise CalibrationError(f"source_points[{i}] must be [x, y]")
        x, y = p
        if not (0 <= x <= payload.frame_width and 0 <= y <= payload.frame_height):
            raise CalibrationError(
                f"source_points[{i}] = ({x:.0f}, {y:.0f}) is outside "
                f"the {payload.frame_width}x{payload.frame_height} frame"
            )

    for i, p in enumerate(payload.target_points):
        if len(p) != 2:
            raise CalibrationError(f"target_points[{i}] must be [x, y]")

    src = np.array(payload.source_points, dtype=np.float32)
    tgt = np.array(payload.target_points, dtype=np.float32)

    h, _ = cv2.findHomography(src, tgt)
    if h is None:
        raise CalibrationError(
            "Homography is degenerate — points may be collinear or duplicated"
        )
    if abs(float(np.linalg.det(h))) < DET_MIN:
        raise CalibrationError(
            f"Homography determinant near zero (|det| < {DET_MIN}) — degenerate calibration"
        )

    # Round-trip reprojection: src -> tgt -> src must be < 2 px on average.
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), h).reshape(-1, 2)
    h_inv = np.linalg.inv(h)
    back = cv2.perspectiveTransform(
        projected.reshape(-1, 1, 2).astype(np.float64), h_inv
    ).reshape(-1, 2)
    err = float(np.linalg.norm(back - src, axis=1).mean())
    if err > REPROJ_ERR_PX:
        raise CalibrationError(
            f"Reprojection error {err:.2f}px exceeds {REPROJ_ERR_PX}px threshold"
        )
    return h


# ---------------------------------------------------------------------------
# YAML helpers (kept local to avoid a circular import with app.py)
# ---------------------------------------------------------------------------
def _load_yaml(config_path: str) -> dict:
    p = Path(config_path)
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _write_yaml(config_path: str, data: dict) -> None:
    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=None, sort_keys=False)


def _alert_section(raw: dict) -> dict:
    return raw.get("alert", {}) or {}


def _calibration_path(raw: dict) -> str:
    return _alert_section(raw).get("calibration_path", "config/calibration_back.json")


def _history_dir(active_path: Path) -> Path:
    """Directory of timestamped calibration archives (sibling of active file)."""
    return active_path.parent / "calibrations"


# Safe filename pattern for /history/load — restricts to files we generated
# (stem + ISO-compact timestamp + .json) so the endpoint can't be coerced
# into reading arbitrary paths.
_HISTORY_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+\.json$")


def _archive_calibration(active_path: Path, record: dict) -> Path:
    """Write a timestamped copy of *record* under the history dir.

    Returns the archive path. The active file is the source of truth for
    the runtime; the archive is purely for "load previous" UX.
    """
    hist_dir = _history_dir(active_path)
    hist_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{active_path.stem}_{ts}{active_path.suffix}"
    archive_path = hist_dir / name
    with open(archive_path, "w") as f:
        json.dump(record, f, indent=2)
    return archive_path


def _restart_service() -> tuple[bool, str]:
    """Restart safetyvision via systemctl. Returns (ok, message)."""
    try:
        result = subprocess.run(
            ["sudo", "systemctl", "restart", "safetyvision"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return False, result.stderr.strip() or "systemctl restart failed"
    except subprocess.TimeoutExpired:
        return False, "systemctl restart timed out"
    except Exception as e:
        return False, str(e)
    return True, "service restarted"


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------
def create_calibration_router(
    check_session: Callable,
    get_config_path: Callable[[], str],
) -> APIRouter:
    """Build the calibration APIRouter with injected auth + config-path deps.

    ``get_config_path`` is a callable so the router picks up the live value
    of the app's CONFIG_PATH (which can be overridden by ``--config`` after
    the router has been constructed).
    """

    router = APIRouter(prefix="/api/calibration")
    _last_toggle: dict[str, float] = {"ts": 0.0}
    TOGGLE_COOLDOWN = 5.0  # seconds

    @router.get("/frame")
    async def get_frame(_t: str = Depends(check_session)):
        """Latest decoded frame from the back camera (tmpfs)."""
        path = Path(CALIBRATION_FRAME_PATH)
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail="No frame available yet — is the capture worker running?",
            )
        return FileResponse(str(path), media_type="image/jpeg")

    @router.get("/status")
    async def get_status(_t: str = Depends(check_session)):
        raw = _load_yaml(get_config_path())
        cal_path = Path(_calibration_path(raw))
        return {
            "zone_mode": _alert_section(raw).get("zone_mode", "bands"),
            "calibrated": cal_path.exists(),
            "calibration_path": str(cal_path),
        }

    @router.get("")
    async def get_calibration(_t: str = Depends(check_session)):
        raw = _load_yaml(get_config_path())
        cal_path = Path(_calibration_path(raw))
        if not cal_path.exists():
            raise HTTPException(status_code=404, detail="No calibration saved")
        with open(cal_path) as f:
            return json.load(f)

    @router.post("")
    async def save_calibration(
        payload: CalibrationPayload, _t: str = Depends(check_session)
    ):
        try:
            _validate(payload)
        except CalibrationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        raw = _load_yaml(get_config_path())
        cal_path = Path(_calibration_path(raw))
        cal_path.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "camera_id": "back",
            "source_points": payload.source_points,
            "target_points": payload.target_points,
            "frame_width": payload.frame_width,
            "frame_height": payload.frame_height,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(cal_path, "w") as f:
            json.dump(record, f, indent=2)
        # Archive a timestamped copy so the user can browse / restore
        # previous calibrations from the UI.
        archive_path: Optional[Path] = None
        try:
            archive_path = _archive_calibration(cal_path, record)
        except OSError as e:
            # Archive failures are non-fatal — the active calibration is
            # already saved and reload-on-mtime will pick it up.
            archive_path = None
        return {
            "ok": True,
            "path": str(cal_path),
            "archive": str(archive_path) if archive_path else None,
        }

    @router.delete("")
    async def delete_calibration(_t: str = Depends(check_session)):
        raw = _load_yaml(get_config_path())
        cal_path = Path(_calibration_path(raw))
        if not cal_path.exists():
            raise HTTPException(status_code=404, detail="No calibration to delete")
        cal_path.unlink()
        return {"ok": True}

    @router.get("/history")
    async def list_history(_t: str = Depends(check_session)):
        """List archived calibrations, newest first."""
        raw = _load_yaml(get_config_path())
        cal_path = Path(_calibration_path(raw))
        hist_dir = _history_dir(cal_path)
        if not hist_dir.exists():
            return {"items": []}

        try:
            active_mtime = cal_path.stat().st_mtime if cal_path.exists() else None
        except OSError:
            active_mtime = None

        items: list[dict] = []
        for p in hist_dir.glob(f"{cal_path.stem}_*{cal_path.suffix}"):
            try:
                with open(p) as f:
                    data = json.load(f)
                created_at = data.get("created_at") or ""
                frame_w = data.get("frame_width")
                frame_h = data.get("frame_height")
            except (OSError, json.JSONDecodeError):
                continue
            try:
                mtime = p.stat().st_mtime
            except OSError:
                mtime = 0.0
            items.append({
                "filename": p.name,
                "created_at": created_at,
                "mtime": mtime,
                "frame_width": frame_w,
                "frame_height": frame_h,
                "active": (active_mtime is not None and abs(mtime - active_mtime) < 0.5),
            })
        items.sort(key=lambda x: x["mtime"], reverse=True)
        return {"items": items}

    class _LoadHistoryBody(BaseModel):
        filename: str

    @router.post("/history/load")
    async def load_history(
        body: _LoadHistoryBody, _t: str = Depends(check_session)
    ):
        """Promote an archived calibration to the active file.

        The inference worker's DistanceZoneStrategy watches the active
        file's mtime and reloads, so no service restart is needed.
        """
        if not _HISTORY_NAME_RE.match(body.filename):
            raise HTTPException(status_code=400, detail="Invalid filename")

        raw = _load_yaml(get_config_path())
        cal_path = Path(_calibration_path(raw))
        hist_dir = _history_dir(cal_path)
        src = hist_dir / body.filename
        # Defence in depth: resolve and confirm src is still under hist_dir.
        try:
            src_resolved = src.resolve(strict=True)
            hist_resolved = hist_dir.resolve(strict=True)
        except (OSError, FileNotFoundError):
            raise HTTPException(status_code=404, detail="Calibration not found")
        if hist_resolved not in src_resolved.parents:
            raise HTTPException(status_code=400, detail="Invalid path")

        # Validate the archived calibration before promoting it.
        try:
            with open(src_resolved) as f:
                data = json.load(f)
            payload = CalibrationPayload(
                source_points=data["source_points"],
                target_points=data["target_points"],
                frame_width=int(data["frame_width"]),
                frame_height=int(data["frame_height"]),
            )
            _validate(payload)
        except (OSError, KeyError, ValueError, CalibrationError) as e:
            raise HTTPException(status_code=400, detail=f"Archive invalid: {e}")

        cal_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replace so a partial copy can never be visible to the
        # inference worker between reload polls.
        tmp = cal_path.with_suffix(cal_path.suffix + ".tmp")
        shutil.copy2(src_resolved, tmp)
        tmp.replace(cal_path)
        return {"ok": True, "loaded": body.filename, "path": str(cal_path)}

    def _toggle(target_mode: str) -> JSONResponse:
        now = time.time()
        if now - _last_toggle["ts"] < TOGGLE_COOLDOWN:
            raise HTTPException(
                status_code=429, detail="Please wait before toggling mode again"
            )
        _last_toggle["ts"] = now

        raw = _load_yaml(get_config_path())
        if target_mode == "distance":
            cal_path = Path(_calibration_path(raw))
            if not cal_path.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot enable distance mode: calibration file not found at {cal_path}",
                )

        raw.setdefault("alert", {})["zone_mode"] = target_mode
        _write_yaml(get_config_path(), raw)

        ok, msg = _restart_service()
        if not ok:
            return JSONResponse({"ok": False, "error": msg}, status_code=500)
        return JSONResponse({"ok": True, "zone_mode": target_mode, "message": msg})

    @router.post("/enable")
    async def enable(_t: str = Depends(check_session)):
        return _toggle("distance")

    @router.post("/disable")
    async def disable(_t: str = Depends(check_session)):
        return _toggle("bands")

    return router
