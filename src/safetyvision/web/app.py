"""SafetyVision Web UI – FastAPI backend."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from safetyvision import runtime_state
from safetyvision.config import SafetyVisionConfig, load_config, validate, ConfigError
from safetyvision.web.calibration import create_calibration_router

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_TEMPLATES_DIR = _HERE / "templates"
_STATIC_DIR = _HERE / "static"

app = FastAPI(title="SafetyVision UI", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
CONFIG_PATH: str = os.environ.get("SAFETYVISION_CONFIG", "config/safetyvision.yaml")
BACKUP_SUFFIX = ".last-good"
SESSION_TOKENS: dict[str, float] = {}  # token -> expiry timestamp
SESSION_TTL = 3600 * 8  # 8 hours

# Default admin credentials (override via env)
ADMIN_USER = os.environ.get("SAFETYVISION_UI_USER", "admin")
ADMIN_PASS_HASH = hashlib.sha256(
    os.environ.get("SAFETYVISION_UI_PASS", "safetyvision").encode()
).hexdigest()

# Rate limiter for apply
_last_apply_ts: float = 0.0
APPLY_COOLDOWN = 5.0  # seconds

# Web preview tuning (override via env if needed)
WEB_RTSP_TRANSPORT = os.environ.get("SAFETYVISION_WEB_RTSP_TRANSPORT", "tcp").lower()
WEB_PREVIEW_WIDTH = int(os.environ.get("SAFETYVISION_WEB_PREVIEW_WIDTH", "960"))
WEB_PREVIEW_FPS = float(os.environ.get("SAFETYVISION_WEB_PREVIEW_FPS", "12"))
WEB_JPEG_QUALITY = int(os.environ.get("SAFETYVISION_WEB_JPEG_QUALITY", "45"))


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
class LoginRequest(BaseModel):
    username: str
    password: str


def _check_session(request: Request) -> str:
    token = request.cookies.get("sv_session")
    if not token or token not in SESSION_TOKENS:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if time.time() > SESSION_TOKENS[token]:
        SESSION_TOKENS.pop(token, None)
        raise HTTPException(status_code=401, detail="Session expired")
    return token


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------
@app.post("/api/auth/login")
async def login(body: LoginRequest):
    pw_hash = hashlib.sha256(body.password.encode()).hexdigest()
    if body.username != ADMIN_USER or pw_hash != ADMIN_PASS_HASH:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = secrets.token_urlsafe(32)
    SESSION_TOKENS[token] = time.time() + SESSION_TTL
    resp = JSONResponse({"ok": True})
    resp.set_cookie("sv_session", token, httponly=True, max_age=SESSION_TTL, samesite="strict")
    return resp


@app.post("/api/auth/logout")
async def logout(request: Request):
    token = request.cookies.get("sv_session")
    SESSION_TOKENS.pop(token, None)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("sv_session")
    return resp


@app.get("/api/auth/check")
async def auth_check(request: Request):
    token = request.cookies.get("sv_session")
    ok = token in SESSION_TOKENS and time.time() < SESSION_TOKENS.get(token, 0)
    return {"authenticated": ok}


# ---------------------------------------------------------------------------
# Mute toggle (cross-process; AlertWorker reads the same flag)
# ---------------------------------------------------------------------------
class MuteRequest(BaseModel):
    # Optional: when omitted, POST /api/mute toggles the current state.
    muted: Optional[bool] = None


def _runtime_log_dir() -> str:
    """Resolve ``logging.log_dir`` from the active config, with a sane default.

    Matches the supervisor's resolution so the mute flag file path is the
    same in both processes.
    """
    raw = _load_raw_config()
    log_dir = raw.get("logging", {}).get("log_dir") or "./logs"
    return str(log_dir)


@app.get("/api/mute")
async def get_mute(_token: str = Depends(_check_session)):
    return {"muted": runtime_state.is_muted(_runtime_log_dir())}


@app.post("/api/mute")
async def set_mute(body: MuteRequest, _token: str = Depends(_check_session)):
    log_dir = _runtime_log_dir()
    if body.muted is None:
        new_state = runtime_state.toggle_muted(log_dir)
    else:
        new_state = runtime_state.set_muted(log_dir, bool(body.muted))
    return {"muted": new_state}


# ---------------------------------------------------------------------------
# Config endpoints
# ---------------------------------------------------------------------------
class ZoneConfig(BaseModel):
    yellow_start_y: float
    red_start_y: float


class AlertTimingConfig(BaseModel):
    repeat_interval_sec: float
    min_clear_sec: float
    min_alert_confidence: float


@app.get("/api/config")
async def get_config(_token: str = Depends(_check_session)):
    raw = _load_raw_config()
    return raw


@app.post("/api/config/zones")
async def set_zones(body: ZoneConfig, _token: str = Depends(_check_session)):
    """Validate and save zone cut lines."""
    try:
        cfg = _build_config_with_overrides({"alert": {
            "yellow_start_y": body.yellow_start_y,
            "red_start_y": body.red_start_y,
        }})
        validate(cfg)
    except ConfigError as e:
        raise HTTPException(status_code=400, detail=str(e))

    _update_config_section("alert", {
        "yellow_start_y": round(body.yellow_start_y, 4),
        "red_start_y": round(body.red_start_y, 4),
    })
    return {"ok": True, "yellow_start_y": body.yellow_start_y, "red_start_y": body.red_start_y}


@app.post("/api/config/timing")
async def set_timing(body: AlertTimingConfig, _token: str = Depends(_check_session)):
    try:
        cfg = _build_config_with_overrides({"alert": {
            "repeat_interval_sec": body.repeat_interval_sec,
            "min_clear_sec": body.min_clear_sec,
            "min_alert_confidence": body.min_alert_confidence,
        }})
        validate(cfg)
    except ConfigError as e:
        raise HTTPException(status_code=400, detail=str(e))

    _update_config_section("alert", {
        "repeat_interval_sec": body.repeat_interval_sec,
        "min_clear_sec": body.min_clear_sec,
        "min_alert_confidence": body.min_alert_confidence,
    })
    return {"ok": True}


@app.post("/api/config/validate")
async def validate_config(_token: str = Depends(_check_session)):
    try:
        load_config(CONFIG_PATH)
        return {"valid": True}
    except (ConfigError, Exception) as e:
        return {"valid": False, "error": str(e)}


@app.post("/api/config/restore")
async def restore_config(_token: str = Depends(_check_session)):
    """Restore last-known-good config."""
    backup = Path(CONFIG_PATH + BACKUP_SUFFIX)
    if not backup.exists():
        raise HTTPException(status_code=404, detail="No backup config found")
    shutil.copy2(str(backup), CONFIG_PATH)
    return {"ok": True, "message": "Previous config restored"}


# ---------------------------------------------------------------------------
# Service control
# ---------------------------------------------------------------------------
@app.get("/api/status")
async def get_status(_token: str = Depends(_check_session)):
    try:
        result = subprocess.run(
            ["sudo", "systemctl", "is-active", "safetyvision"],
            capture_output=True, text=True, timeout=5,
        )
        state = result.stdout.strip()
    except Exception:
        state = "unknown"
    return {"service": state}


def _metrics_log_candidates() -> list[Path]:
    """Return likely paths for the SafetyVision metrics log."""
    candidates: list[Path] = []
    env_path = os.environ.get("SAFETYVISION_METRICS_LOG", "").strip()
    if env_path:
        candidates.append(Path(env_path))

    try:
        raw = _load_raw_config()
    except Exception:
        raw = {}

    log_dir_raw = str(raw.get("logging", {}).get("log_dir", "./logs"))
    log_dir = Path(log_dir_raw)
    cfg_path = Path(CONFIG_PATH)

    if log_dir.is_absolute():
        candidates.append(log_dir / "safetyvision.log")
    else:
        candidates.append((Path.cwd() / log_dir / "safetyvision.log").resolve())
        candidates.append((cfg_path.parent / log_dir / "safetyvision.log").resolve())
        candidates.append((cfg_path.parent.parent / log_dir / "safetyvision.log").resolve())

    candidates.append(Path("/var/log/safetyvision/safetyvision.log"))

    # Stable dedupe while preserving order
    seen: set[str] = set()
    unique: list[Path] = []
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def _coerce_metric(payload: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(payload.get(key, default))
    except (TypeError, ValueError):
        return default


def _normalize_metrics(payload: dict) -> dict:
    """Normalize metrics payload shape for frontend consumption."""
    raw_distance = payload.get("last_distance_m")
    try:
        last_distance_m = float(raw_distance) if raw_distance is not None else None
    except (TypeError, ValueError):
        last_distance_m = None

    last_zone_level = payload.get("last_zone_level") or ""
    if not isinstance(last_zone_level, str):
        last_zone_level = ""

    return {
        "fps": _coerce_metric(payload, "fps"),
        "capture_fps": _coerce_metric(payload, "capture_fps"),
        "inference_fps": _coerce_metric(payload, "inference_fps"),
        "decision_fps": _coerce_metric(payload, "decision_fps"),
        "latency_capture_ms": _coerce_metric(payload, "latency_capture_ms"),
        "latency_inference_ms": _coerce_metric(payload, "latency_inference_ms"),
        "latency_decision_ms": _coerce_metric(payload, "latency_decision_ms"),
        "latency_total_ms": _coerce_metric(payload, "latency_total_ms"),
        "frames_dropped": int(_coerce_metric(payload, "frames_dropped")),
        "alerts": int(_coerce_metric(payload, "alerts")),
        "uptime_s": _coerce_metric(payload, "uptime_s"),
        "last_distance_m": last_distance_m,
        "last_zone_level": last_zone_level,
    }


def _try_parse_metrics_candidate(candidate: str) -> Optional[dict]:
    """Parse a metrics JSON candidate, including escaped-quote variants."""
    try:
        parsed = json.loads(candidate)
    except Exception:
        parsed = None
    if isinstance(parsed, dict) and parsed.get("type") == "metrics":
        return _normalize_metrics(parsed)

    if '\\"' in candidate:
        try:
            parsed = json.loads(candidate.replace('\\"', '"'))
        except Exception:
            parsed = None
        if isinstance(parsed, dict) and parsed.get("type") == "metrics":
            return _normalize_metrics(parsed)
    return None


def _extract_metrics_from_text(text: str) -> Optional[dict]:
    """Extract metrics payload from a log line."""
    if not text:
        return None

    # Case 1: direct metrics JSON line.
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        if parsed.get("type") == "metrics":
            return _normalize_metrics(parsed)
        msg = parsed.get("msg")
        if isinstance(msg, str):
            try:
                msg_json = json.loads(msg)
            except Exception:
                msg_json = None
            if isinstance(msg_json, dict) and msg_json.get("type") == "metrics":
                return _normalize_metrics(msg_json)

    # Case 2: extract a metrics object from mixed/invalid log wrappers.
    # This handles lines like:
    #   {"msg":"{"type": "metrics", ...}"}
    # where the outer line is not valid JSON.
    patterns = (
        r'\{[^{}]*"type"\s*:\s*"metrics"[^{}]*\}',
        r'\{[^{}]*\\"type\\"\s*:\s*\\"metrics\\"[^{}]*\}',
    )
    for pattern in patterns:
        matches = list(re.finditer(pattern, text))
        for match in reversed(matches):
            data = _try_parse_metrics_candidate(match.group(0))
            if data is not None:
                return data

    return None


def _read_latest_metrics(max_lines: int = 2000) -> Optional[dict]:
    """Read latest metrics snapshot from SafetyVision log file."""
    for path in _metrics_log_candidates():
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                tail = deque(f, maxlen=max_lines)
        except Exception:
            continue

        for line in reversed(tail):
            data = _extract_metrics_from_text(line.strip())
            if data is not None:
                return data
    return None


@app.get("/api/metrics")
async def get_metrics(_token: str = Depends(_check_session)):
    data = _read_latest_metrics()
    raw = _load_raw_config()
    zone_mode = raw.get("alert", {}).get("zone_mode", "bands")
    if data is None:
        return {"available": False, "zone_mode": zone_mode}
    return {"available": True, "zone_mode": zone_mode, **data}


@app.post("/api/apply")
async def apply_config(_token: str = Depends(_check_session)):
    """Validate config, backup current, restart safetyvision service."""
    global _last_apply_ts
    now = time.time()
    if now - _last_apply_ts < APPLY_COOLDOWN:
        raise HTTPException(status_code=429, detail="Please wait before applying again")
    _last_apply_ts = now

    # Validate first
    try:
        load_config(CONFIG_PATH)
    except (ConfigError, Exception) as e:
        raise HTTPException(status_code=400, detail=f"Config invalid: {e}")

    # Backup current config
    backup = Path(CONFIG_PATH + BACKUP_SUFFIX)
    shutil.copy2(CONFIG_PATH, str(backup))

    # Restart service
    try:
        result = subprocess.run(
            ["sudo", "systemctl", "restart", "safetyvision"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return {"ok": False, "error": result.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Restart timed out"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

    return {"ok": True, "message": "Config applied and service restarted"}


# ---------------------------------------------------------------------------
# Live MJPEG stream
# ---------------------------------------------------------------------------
import threading

_stream_frame: Optional[bytes] = None
_stream_lock = threading.Lock()
_stream_thread: Optional[threading.Thread] = None


def _open_stream_camera() -> cv2.VideoCapture:
    """Open camera for the web UI live view with low-latency settings.

    Uses the main stream (rtsp_url_main) for higher resolution preview.
    Falls back to sub stream (rtsp_url) or USB if main stream unavailable.
    """
    raw = _load_raw_config()
    inp = raw.get("input", {})
    mode = inp.get("mode", "usb")
    dev = inp.get("usb_device", "/dev/video0")
    rtsp_main = inp.get("rtsp_url_main", "")
    rtsp_sub = inp.get("rtsp_url", "")
    width = int(inp.get("width", 640))
    height = int(inp.get("height", 480))
    fps = int(inp.get("fps", 25))

    # For RTSP: prefer main stream, then sub stream.
    for url in ([rtsp_main, rtsp_sub] if mode in ("rtsp", "auto") else []):
        if not url:
            continue
        transports = [WEB_RTSP_TRANSPORT, "tcp", "udp"]
        tried = set()
        for transport in transports:
            if transport in tried:
                continue
            tried.add(transport)
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{transport}|fflags;nobuffer|flags;low_delay|framedrop;1|"
                "probesize;32|analyzeduration;0"
            )
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                return cap

    # USB
    if mode in ("usb", "auto") and os.path.exists(dev):
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            return cap

    return cv2.VideoCapture()


def _draw_zone_overlay(frame: np.ndarray, yellow_y: float, red_y: float) -> np.ndarray:
    """Draw semi-transparent zone bands on frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    y_yel = int(yellow_y * h)
    y_red = int(red_y * h)

    cv2.rectangle(overlay, (0, 0), (w, y_yel), (0, 180, 0), -1)
    cv2.rectangle(overlay, (0, y_yel), (w, y_red), (0, 220, 255), -1)
    cv2.rectangle(overlay, (0, y_red), (w, h), (0, 0, 255), -1)

    frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
    cv2.line(frame, (0, y_yel), (w, y_yel), (0, 220, 255), 2)
    cv2.line(frame, (0, y_red), (w, y_red), (0, 0, 255), 2)
    return frame


def _stream_capture_loop():
    """Background thread: grabs frames from the main stream for live view.

    Low-latency strategy:
    - Grab one buffered frame to reduce stale output.
    - Cache zone config (reload every 2 seconds, not every frame).
    - Throttle encode rate and keep JPEG quality/resolution bounded.
    """
    global _stream_frame
    cap = None
    yellow_y, red_y = 0.33, 0.66
    config_refresh_ns = 0

    target_interval = 1.0 / max(WEB_PREVIEW_FPS, 1.0)
    next_frame_time = time.monotonic()
    read_failures = 0

    while True:
        if cap is None or not cap.isOpened():
            cap = _open_stream_camera()
            if not cap.isOpened():
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Camera unavailable",
                            (160, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                _, buf = cv2.imencode(".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, WEB_JPEG_QUALITY])
                with _stream_lock:
                    _stream_frame = buf.tobytes()
                time.sleep(2.0)
                continue

        now_mono = time.monotonic()
        if now_mono < next_frame_time:
            time.sleep(next_frame_time - now_mono)
        next_frame_time = time.monotonic() + target_interval

        # Drain one buffered frame to reduce stale output.
        cap.grab()
        ret, frame = cap.read()
        if not ret:
            read_failures += 1
            if read_failures >= 5:
                cap.release()
                cap = None
                read_failures = 0
                time.sleep(0.5)
            continue
        read_failures = 0

        # Refresh zone config every 2 seconds
        now = time.time_ns()
        if now - config_refresh_ns > 2_000_000_000:
            raw = _load_raw_config()
            alert = raw.get("alert", {})
            yellow_y = alert.get("yellow_start_y", 0.33)
            red_y = alert.get("red_start_y", 0.66)
            config_refresh_ns = now

        frame = _draw_zone_overlay(frame, yellow_y, red_y)

        # Keep web preview resolution bounded to reduce encode CPU and jitter.
        if WEB_PREVIEW_WIDTH > 0 and frame.shape[1] > WEB_PREVIEW_WIDTH:
            scale = WEB_PREVIEW_WIDTH / float(frame.shape[1])
            out_h = max(1, int(frame.shape[0] * scale))
            frame = cv2.resize(frame, (WEB_PREVIEW_WIDTH, out_h), interpolation=cv2.INTER_AREA)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, WEB_JPEG_QUALITY])
        with _stream_lock:
            _stream_frame = buf.tobytes()


def _ensure_stream_thread():
    """Start the background capture thread if not running."""
    global _stream_thread
    if _stream_thread is not None and _stream_thread.is_alive():
        return
    _stream_thread = threading.Thread(target=_stream_capture_loop, daemon=True)
    _stream_thread.start()


async def _mjpeg_generator():
    _ensure_stream_thread()
    last_frame = None
    while True:
        with _stream_lock:
            frame = _stream_frame
        if frame is not None and frame is not last_frame:
            last_frame = frame
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )
        await asyncio.sleep(0.033)


@app.get("/api/stream.mjpg")
async def stream(request: Request):
    token = request.cookies.get("sv_session")
    if not token or token not in SESSION_TOKENS:
        raise HTTPException(status_code=401)
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# Frontend pages
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/calibration", response_class=HTMLResponse)
async def calibration_page(request: Request):
    return templates.TemplateResponse("calibration.html", {"request": request})


# Mount the calibration API router. ``CONFIG_PATH`` is read lazily so the
# ``--config`` CLI override applied in ``main()`` is honored.
app.include_router(
    create_calibration_router(
        check_session=_check_session,
        get_config_path=lambda: CONFIG_PATH,
    )
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _load_raw_config() -> dict:
    p = Path(CONFIG_PATH)
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f) or {}
    return {}


def _build_config_with_overrides(overrides: dict) -> SafetyVisionConfig:
    """Load current config, apply overrides, and return for validation.

    Does NOT validate — caller is responsible for calling validate().
    """
    from safetyvision.config import _merge, InputConfig, ModelConfig, AlertConfig, PerfConfig, LoggingConfig, HealthConfig

    raw = _load_raw_config()
    for section, values in overrides.items():
        raw.setdefault(section, {}).update(values)

    return SafetyVisionConfig(
        input=_merge(InputConfig, raw.get("input")),
        model=_merge(ModelConfig, raw.get("model")),
        alert=_merge(AlertConfig, raw.get("alert")),
        perf=_merge(PerfConfig, raw.get("perf")),
        logging=_merge(LoggingConfig, raw.get("logging")),
        health=_merge(HealthConfig, raw.get("health")),
    )


def _update_config_section(section: str, values: dict) -> None:
    """Merge values into a section of the config YAML."""
    raw = _load_raw_config()
    raw.setdefault(section, {}).update(values)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(raw, f, default_flow_style=None, sort_keys=False)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="SafetyVision Web UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("-c", "--config", default=None)
    args = parser.parse_args()

    global CONFIG_PATH
    if args.config:
        CONFIG_PATH = args.config

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
