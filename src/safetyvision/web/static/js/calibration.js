// Distance-mode calibration UI for the back camera.

const canvas = document.getElementById('frameCanvas');
const ctx = canvas.getContext('2d');
const placeholder = document.getElementById('canvasPlaceholder');
const captureBtn = document.getElementById('captureBtn');
const resetBtn = document.getElementById('resetPointsBtn');
const saveBtn = document.getElementById('saveBtn');
const enableBtn = document.getElementById('enableBtn');
const disableBtn = document.getElementById('disableBtn');
const pointInputs = document.getElementById('pointInputs');
const frameStatus = document.getElementById('frameStatus');
const actionStatus = document.getElementById('actionStatus');
const modeBadge = document.getElementById('modeBadge');
const toast = document.getElementById('toast');
const logoutBtn = document.getElementById('logoutBtn');
const historyList = document.getElementById('historyList');

const POINT_COLORS = ['#00c853', '#ffd600', '#448aff', '#ff1744'];
const REDUCED_MOTION = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const state = {
  frameLoaded: false,
  frameImg: null,
  frameWidth: 640,
  frameHeight: 480,
  points: [], // [{px, py, xm, ym}]
};
// Map filename -> last-seen item from /api/calibration/history. Filled by
// refreshHistory(); read by previewHistoryItem() and loadHistoryItem() so
// they can show the archived frame + points without re-fetching the list.
let historyItemsByName = {};

function showToast(msg, kind = 'info') {
  toast.textContent = msg;
  toast.className = 'toast ' + kind;
  toast.style.display = 'block';
  setTimeout(() => { toast.style.display = 'none'; }, 2800);
}

async function refreshStatus() {
  const r = await fetch('/api/calibration/status');
  if (!r.ok) {
    modeBadge.textContent = 'auth?';
    if (r.status === 401) window.location.href = '/login';
    return;
  }
  const data = await r.json();
  modeBadge.textContent = data.zone_mode === 'distance' ? 'DISTANCE' : 'BANDS';
  modeBadge.className = 'mode-badge ' + (data.zone_mode === 'distance' ? 'distance' : 'bands');

  enableBtn.disabled = !(data.calibrated && data.zone_mode !== 'distance');
  disableBtn.disabled = data.zone_mode !== 'distance';
}

function redraw(animIdx = -1, animScale = 1) {
  if (!state.frameImg) return;
  ctx.drawImage(state.frameImg, 0, 0, canvas.width, canvas.height);
  state.points.forEach((p, i) => {
    const r = (i === animIdx) ? 8 * animScale : 8;
    if (r < 0.5) return;
    ctx.fillStyle = POINT_COLORS[i];
    ctx.beginPath();
    ctx.arc(p.px, p.py, r, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 2;
    ctx.stroke();
    if (r > 4) {
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(i + 1), p.px, p.py);
    }
  });
}

// Pop-in animation for the most recently placed marker. Ease-out-back
// gives a small overshoot so the dot lands with a tactile "tap" feel.
function animateMarkerIn(idx) {
  if (REDUCED_MOTION) { redraw(); return; }
  const start = performance.now();
  const DUR = 260;
  const c1 = 1.7, c3 = c1 + 1;
  function frame(now) {
    const t = Math.min(1, (now - start) / DUR);
    const eased = 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
    redraw(idx, eased);
    if (t < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

function rebuildPointInputs() {
  if (state.points.length === 0) {
    pointInputs.innerHTML =
      '<div class="point-empty">Click 4 points on the captured frame to begin.</div>';
    saveBtn.disabled = true;
    return;
  }
  pointInputs.innerHTML = '';
  state.points.forEach((p, i) => {
    const placed = Number.isFinite(p.xm) && Number.isFinite(p.ym);
    const row = document.createElement('div');
    row.className = 'point-row' + (placed ? ' placed' : '');
    row.innerHTML = `
      <div class="point-row-top">
        <span class="point-badge" style="background:${POINT_COLORS[i]}">${i + 1}</span>
        <span class="point-label">Point ${i + 1}</span>
        <span class="${placed ? 'placed-badge' : 'waiting-badge'}">${placed ? 'Placed' : 'Awaiting input'}</span>
      </div>
      <div class="input-pair">
        <label>
          <span><i data-lucide="move-horizontal" style="width:10px;height:10px;"></i> X (m)</span>
          <input type="number" step="0.01" data-idx="${i}" data-axis="xm" value="${p.xm ?? ''}">
        </label>
        <label>
          <span><i data-lucide="move-vertical" style="width:10px;height:10px;"></i> Y (m)</span>
          <input type="number" step="0.01" data-idx="${i}" data-axis="ym" value="${p.ym ?? ''}">
        </label>
      </div>
      <div class="point-pixel">
        <i data-lucide="crosshair" style="width:10px;height:10px;"></i>
        px (${Math.round(p.px)}, ${Math.round(p.py)})
      </div>
    `;
    pointInputs.appendChild(row);
  });

  pointInputs.querySelectorAll('input[type="number"]').forEach(el => {
    el.addEventListener('input', e => {
      const idx = Number(e.target.dataset.idx);
      const axis = e.target.dataset.axis;
      const v = e.target.value === '' ? null : parseFloat(e.target.value);
      state.points[idx][axis] = Number.isFinite(v) ? v : null;
      // Toggle placed/waiting badge live without a full rebuild
      const row = el.closest('.point-row');
      const p = state.points[idx];
      const nowPlaced = Number.isFinite(p.xm) && Number.isFinite(p.ym);
      row.classList.toggle('placed', nowPlaced);
      const badge = row.querySelector('.placed-badge, .waiting-badge');
      if (badge) {
        badge.className = nowPlaced ? 'placed-badge' : 'waiting-badge';
        badge.textContent = nowPlaced ? 'Placed' : 'Awaiting input';
      }
      saveBtn.disabled = !canSave();
    });
  });

  saveBtn.disabled = !canSave();

  // Render any new lucide icons inserted by the row template
  if (window.lucide && typeof window.lucide.createIcons === 'function') {
    window.lucide.createIcons();
  }
}

function canSave() {
  if (state.points.length !== 4) return false;
  return state.points.every(p =>
    Number.isFinite(p.xm) && Number.isFinite(p.ym)
  );
}

canvas.addEventListener('click', (e) => {
  if (!state.frameLoaded) return;
  if (state.points.length >= 4) {
    showToast('4 points already placed. Reset to start over.', 'warn');
    return;
  }
  const rect = canvas.getBoundingClientRect();
  const sx = canvas.width / rect.width;
  const sy = canvas.height / rect.height;
  const px = (e.clientX - rect.left) * sx;
  const py = (e.clientY - rect.top) * sy;
  state.points.push({ px, py, xm: null, ym: null });
  animateMarkerIn(state.points.length - 1);
  rebuildPointInputs();
});

captureBtn.addEventListener('click', async () => {
  frameStatus.textContent = 'Loading...';
  try {
    const r = await fetch('/api/calibration/frame', { cache: 'no-store' });
    if (!r.ok) {
      const detail = (await r.json().catch(() => ({}))).detail || r.statusText;
      frameStatus.textContent = 'Error: ' + detail;
      return;
    }
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      state.frameImg = img;
      state.frameWidth = img.naturalWidth;
      state.frameHeight = img.naturalHeight;
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      state.frameLoaded = true;
      placeholder.style.display = 'none';
      redraw();
      frameStatus.textContent = `Frame ${img.naturalWidth}×${img.naturalHeight}`;
      URL.revokeObjectURL(url);
    };
    img.src = url;
  } catch (err) {
    frameStatus.textContent = 'Error: ' + err.message;
  }
});

resetBtn.addEventListener('click', () => {
  state.points = [];
  redraw();
  rebuildPointInputs();
});

saveBtn.addEventListener('click', async () => {
  if (!canSave()) return;
  actionStatus.textContent = 'Saving...';
  const body = {
    source_points: state.points.map(p => [p.px, p.py]),
    target_points: state.points.map(p => [p.xm, p.ym]),
    frame_width: state.frameWidth,
    frame_height: state.frameHeight,
  };
  const r = await fetch('/api/calibration', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    actionStatus.textContent = 'Save failed: ' + (data.detail || r.statusText);
    showToast('Calibration rejected: ' + (data.detail || r.statusText), 'error');
    return;
  }
  actionStatus.textContent = 'Saved to ' + data.path;
  showToast('Calibration saved', 'success');
  refreshStatus();
  refreshHistory();
});

function formatCreatedAt(iso) {
  if (!iso) return '(unknown time)';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString();
}

async function refreshHistory() {
  if (!historyList) return;
  try {
    const r = await fetch('/api/calibration/history');
    if (!r.ok) {
      if (r.status === 401) window.location.href = '/login';
      historyList.innerHTML =
        '<div class="point-empty">Could not load history.</div>';
      return;
    }
    const data = await r.json();
    const items = (data.items || []);
    if (items.length === 0) {
      historyList.innerHTML =
        '<div class="point-empty">No previous calibrations yet.</div>';
      return;
    }
    historyList.innerHTML = '';
    // Cache the most recent item-by-filename for the Load handler so the
    // archived points + frame flag don't need a second round-trip.
    historyItemsByName = {};
    items.forEach(item => {
      historyItemsByName[item.filename] = item;
      const row = document.createElement('div');
      row.className = 'history-row' + (item.active ? ' active' : '');
      const dims = (item.frame_width && item.frame_height)
        ? `${item.frame_width}×${item.frame_height}` : '';
      row.innerHTML = `
        <div class="history-meta">
          <div class="history-when">${formatCreatedAt(item.created_at)}</div>
          <div class="history-sub">
            <span class="history-name">${item.filename}</span>
            ${dims ? `<span class="dim">· ${dims}</span>` : ''}
            <span class="dim">· ${item.has_frame ? 'frame ✓' : 'no frame'}</span>
            ${item.active ? '<span class="active-badge">active</span>' : ''}
          </div>
        </div>
        <div class="history-actions">
          <button class="btn-secondary history-preview" data-filename="${item.filename}" title="Show this calibration's frame and points without activating it">
            <i data-lucide="eye" style="width:12px;height:12px;"></i>
            Preview
          </button>
          <button class="btn-secondary history-load" data-filename="${item.filename}" ${item.active ? 'disabled' : ''}>
            <i data-lucide="upload" style="width:12px;height:12px;"></i>
            Load
          </button>
        </div>
      `;
      historyList.appendChild(row);
    });
    historyList.querySelectorAll('.history-load').forEach(btn => {
      btn.addEventListener('click', () => loadHistoryItem(btn.dataset.filename));
    });
    historyList.querySelectorAll('.history-preview').forEach(btn => {
      btn.addEventListener('click', () => previewHistoryItem(btn.dataset.filename));
    });
    if (window.lucide && typeof window.lucide.createIcons === 'function') {
      window.lucide.createIcons();
    }
  } catch (err) {
    historyList.innerHTML =
      '<div class="point-empty">Error: ' + err.message + '</div>';
  }
}

// Load an image into the canvas from a URL. Returns true on success.
async function loadCanvasFromUrl(url, statusLabel) {
  try {
    const r = await fetch(url, { cache: 'no-store' });
    if (!r.ok) return false;
    const blob = await r.blob();
    const objUrl = URL.createObjectURL(blob);
    return await new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        state.frameImg = img;
        state.frameWidth = img.naturalWidth;
        state.frameHeight = img.naturalHeight;
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        state.frameLoaded = true;
        placeholder.style.display = 'none';
        frameStatus.textContent =
          `${statusLabel} ${img.naturalWidth}×${img.naturalHeight}`;
        URL.revokeObjectURL(objUrl);
        resolve(true);
      };
      img.onerror = () => { URL.revokeObjectURL(objUrl); resolve(false); };
      img.src = objUrl;
    });
  } catch (_err) {
    return false;
  }
}

// Paint the archive's saved frame on the canvas (or the current live
// frame if no snapshot was archived) and pre-fill the 4 points from
// its source/target so the user can see exactly what was clicked.
// Returns the kind of frame shown: 'archived', 'live', or 'none'.
async function showHistoryFrame(item) {
  if (!item) return 'none';
  const srcPts = item.source_points || [];
  const tgtPts = item.target_points || [];
  if (srcPts.length !== 4 || tgtPts.length !== 4) {
    showToast('Archive missing points', 'warn');
    return 'none';
  }

  let frameKind = 'none';
  if (item.has_frame) {
    const url = '/api/calibration/history/frame?filename='
      + encodeURIComponent(item.filename);
    if (await loadCanvasFromUrl(url, 'Archived frame')) frameKind = 'archived';
  }
  // Fallback: no snapshot in the archive (older save) — show the live
  // frame so the points have some visual reference. The points are still
  // in the same pixel coordinates the user clicked, so they line up if
  // the camera hasn't moved.
  if (frameKind === 'none') {
    if (await loadCanvasFromUrl('/api/calibration/frame', 'Live frame')) {
      frameKind = 'live';
    }
  }

  state.points = srcPts.map((p, i) => ({
    px: p[0],
    py: p[1],
    xm: tgtPts[i][0],
    ym: tgtPts[i][1],
  }));
  redraw();
  rebuildPointInputs();
  return frameKind;
}

async function previewHistoryItem(filename) {
  const item = historyItemsByName[filename];
  if (!item) return;
  const kind = await showHistoryFrame(item);
  const frameNote = {
    archived: 'showing the frame captured at calibration time',
    live: 'no snapshot was archived — showing the current live frame',
    none: 'no frame available — Capture a frame to see the points',
  }[kind] || '';
  actionStatus.textContent =
    `Previewing ${filename} — ${frameNote}. Press Load to apply.`;
}

async function loadHistoryItem(filename) {
  if (!confirm(`Load "${filename}" as the active calibration?`)) return;
  actionStatus.textContent = 'Loading ' + filename + '...';
  const url = '/api/calibration/history/load?filename='
    + encodeURIComponent(filename);
  const r = await fetch(url, { method: 'POST' });
  const data = await r.json().catch(() => ({}));
  if (!r.ok || data.ok === false) {
    actionStatus.textContent = 'Load failed: ' + (data.detail || data.error || r.statusText);
    showToast('Load failed', 'error');
    return;
  }
  // Show the archived frame + points on the canvas, then refresh the
  // history listing so the "active" badge moves to this row.
  await showHistoryFrame(historyItemsByName[filename]);
  actionStatus.textContent = 'Loaded ' + filename + ' — Dashboard will reflect within ~1 frame.';
  showToast('Calibration activated', 'success');
  refreshHistory();
}

async function postToggle(path, label) {
  if (!confirm(`${label}? This will restart the SafetyVision service.`)) return;
  actionStatus.textContent = label + '...';
  const r = await fetch(path, { method: 'POST' });
  const data = await r.json().catch(() => ({}));
  if (!r.ok || data.ok === false) {
    actionStatus.textContent = 'Failed: ' + (data.detail || data.error || r.statusText);
    showToast('Mode change failed', 'error');
    return;
  }
  actionStatus.textContent = `Mode set to ${data.zone_mode}. ${data.message || ''}`;
  showToast('Mode changed — service restarting', 'success');
  setTimeout(refreshStatus, 1200);
}

enableBtn.addEventListener('click', () => postToggle('/api/calibration/enable', 'Enable distance mode'));
disableBtn.addEventListener('click', () => postToggle('/api/calibration/disable', 'Switch to band mode'));

logoutBtn.addEventListener('click', async () => {
  await fetch('/api/auth/logout', { method: 'POST' });
  window.location.href = '/login';
});

// Render initial set of lucide icons
if (window.lucide && typeof window.lucide.createIcons === 'function') {
  window.lucide.createIcons();
}

refreshStatus();
refreshHistory();
