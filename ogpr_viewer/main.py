"""
Main GUI Application for OGPR Radar Viewer.

Layout:
  LEFT   : Swath sidebar  (checkbox + per-swath channel selector)
  CENTER : Multi-panel canvas  (auto grid)
  RIGHT  : Global processing + display controls

Y-axis modes:
  - Time (ns)
  - Relative depth (m)   depth = t * v / 2
  - Absolute depth (m)   depth = (t - t0) * v / 2   (t0 from time-zero correction)

Velocity is a single global parameter shared between:
  - Y-axis depth conversion
  - Kirchhoff migration
"""

import sys
import logging
import traceback
import json
import zipfile
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing  import Any, Dict, List, Optional, Tuple

import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog,
    QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QMessageBox, QStatusBar, QScrollArea, QSizePolicy,
    QSplitter, QFrame, QDialog, QButtonGroup, QRadioButton, QInputDialog,
)
from PyQt6.QtCore import Qt, QSettings, QThread, pyqtSignal
from PyQt6.QtGui  import QAction, QKeySequence, QFont

from .ogpr_parser       import OGPRParser
from .signal_processing import SignalProcessor
from .visualization     import MultiPanelCanvas


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logger() -> logging.Logger:
    log_path = Path(sys.argv[0]).parent / 'ogpr_viewer_debug.log'
    fmt = '%(asctime)s  %(levelname)-8s  %(message)s'
    logging.basicConfig(
        level=logging.DEBUG,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8', mode='a'),
            logging.StreamHandler(sys.stderr),
        ]
    )
    log = logging.getLogger('ogpr_viewer')
    log.info('=' * 60)
    log.info('OGPR Viewer started')
    log.info(f'Log: {log_path}')
    return log


LOG = _setup_logger()
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


PROJECT_STAGE_DIRS = [
    'raw',
    '00_stack',
    '01_time0',
    '02_dewow',
    '03_background',
    '04_bandpass',
    '05_gain',
]


def _sanitize_project_name(name: str) -> str:
    clean = ''.join(ch if (ch.isalnum() or ch in '-_') else '_' for ch in str(name).strip())
    clean = clean.strip('_')
    return clean or datetime.now().strftime('project_%Y%m%d_%H%M%S')


def _read_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return dict(default)


def _write_json(path: Path, payload: Dict[str, Any]):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def _create_project_structure(root_dir: str, project_name: str) -> Path:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f'Root directory not found: {root}')

    clean_name = _sanitize_project_name(project_name)
    project_path = root / f'Project_{clean_name}'
    project_path.mkdir(parents=True, exist_ok=True)
    for folder in PROJECT_STAGE_DIRS:
        (project_path / folder).mkdir(parents=True, exist_ok=True)

    meta_path = project_path / 'metadata.json'
    if not meta_path.exists():
        _write_json(meta_path, {
            'project_name': clean_name,
            'created_at': datetime.now().isoformat(),
            'acquisition_params': {},
            'processing_log': [],
            'stage_dirs': PROJECT_STAGE_DIRS,
        })
    return project_path


# ---------------------------------------------------------------------------
# File loader thread
# ---------------------------------------------------------------------------

class FileLoaderThread(QThread):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, filepath: str, lazy: bool = False):
        super().__init__()
        self.filepath = filepath
        self.lazy     = lazy

    def run(self):
        try:
            LOG.debug(f'Loading: {self.filepath}  lazy={self.lazy}')
            self.progress.emit('Parsing header…')
            parser = OGPRParser(self.filepath)
            self.progress.emit('Loading radar volume…')
            d = parser.load_data(lazy=self.lazy)
            LOG.info(
                f'OK {Path(self.filepath).name}  '
                f'shape={d["radar_volume"].shape}  '
                f'dtype={d["metadata"]["dtype_name"]}'
            )
            self.finished.emit(d)
        except Exception as e:
            LOG.error(f'Load error {self.filepath}: {e}\n{traceback.format_exc()}')
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Processing helpers / worker thread
# ---------------------------------------------------------------------------

def _extract_trace_elevations(
    geolocations: Optional[np.ndarray],
    channel: int,
    n_trc: int,
) -> Optional[np.ndarray]:
    """
    Extract one elevation value per trace from OGPR geolocation blocks.
    Supports common shapes:
      - (slices, channels, samples, 3)
      - (slices, samples, 3)
      - (slices, 3)
    """
    if geolocations is None:
        return None

    g = np.asarray(geolocations)
    if g.size == 0:
        return None

    z = None
    try:
        if g.ndim == 4 and g.shape[-1] >= 3:
            ch = int(np.clip(channel, 0, g.shape[1] - 1))
            z = np.nanmedian(g[:, ch, :, 2], axis=1)
        elif g.ndim == 3 and g.shape[-1] >= 3:
            z = np.nanmedian(g[:, :, 2], axis=1)
        elif g.ndim == 2 and g.shape[-1] >= 3:
            z = g[:, 2]
    except Exception:
        z = None

    if z is None:
        return None

    z = np.asarray(z, dtype=np.float64).ravel()
    if z.size < 2:
        return None

    if z.size != n_trc:
        x_src = np.linspace(0.0, 1.0, z.size, dtype=np.float64)
        x_dst = np.linspace(0.0, 1.0, n_trc, dtype=np.float64)
        good = np.isfinite(z)
        if int(np.sum(good)) < 2:
            return None
        z = np.interp(x_dst, x_src[good], z[good])

    if int(np.sum(np.isfinite(z))) < 2:
        return None
    return z.astype(np.float64, copy=False)


def _safe_tz_threshold(requested: float) -> float:
    thr = float(requested)
    if not np.isfinite(thr):
        return 0.08
    thr = float(np.clip(thr, 0.02, 0.50))
    if thr > 0.10:
        LOG.warning(
            f'Time-zero threshold {requested:.3f} too high for first-break picking; '
            'using 0.10'
        )
        thr = 0.10
    return thr


def _safe_dewow_window_samples(
    requested_samples: int,
    dt_ns: float,
    center_freq_mhz: float,
) -> int:
    req = max(1, int(requested_samples))
    dt = max(float(dt_ns), 1e-9)
    f0 = float(center_freq_mhz) if np.isfinite(center_freq_mhz) else 0.0

    # Physically safe minimum:
    #  - at least 2.5 ns
    #  - and at least 1.5 periods of center frequency when known.
    min_ns = 2.5
    if f0 > 0.0:
        period_ns = 1000.0 / f0
        min_ns = max(min_ns, 1.5 * period_ns)
    min_smp = max(3, int(np.ceil(min_ns / dt)))

    if req < min_smp:
        LOG.warning(
            f'Dewow window too short ({req} smp, dt={dt_ns:.3f} ns, f0={f0:.1f} MHz). '
            f'Using safe minimum {min_smp} smp (~{min_smp * dt_ns:.2f} ns).'
        )
        return min_smp
    return req


def _resolve_stack_params(params: Dict[str, Any]) -> Tuple[int, int]:
    w = max(1, int(params.get('stack_win_trc', 5)))
    d = max(1, int(params.get('stack_decim', 1)))
    auto_decim = bool(params.get('stack_auto_decim', True))
    if auto_decim and (w > 1) and (d <= 1):
        d = w
        LOG.debug(f'Stack auto-decimation: decimation set to window ({w})')
    return w, d


def _safe_background_params(
    requested_method: str,
    rolling: bool,
    requested_win_trc: int,
    dx_m: float,
) -> Tuple[str, int]:
    method = str(requested_method).lower().strip()
    win = max(5, int(requested_win_trc))
    dx = max(float(dx_m), 1e-9)

    if not rolling:
        return method, win

    span_m = win * dx
    max_span_m = 4.0
    if span_m > max_span_m:
        win_max = max(5, int(np.floor(max_span_m / dx)))
        if win > win_max:
            LOG.warning(
                f'Background rolling window too wide ({span_m:.2f} m). '
                f'Using {win_max} traces (~{win_max * dx:.2f} m).'
            )
            win = win_max
            span_m = win * dx

    if method == 'mean' and span_m >= 3.0:
        LOG.warning(
            f'Background method "mean" on wide rolling span ({span_m:.2f} m) may cause shadowing. '
            'Using "median".'
        )
        method = 'median'

    return method, win


def _safe_migration_aperture_traces(requested_ap: int, dx_m: float) -> int:
    ap = max(3, int(requested_ap))
    dx = max(float(dx_m), 1e-9)
    span_m = ap * dx
    max_span_m = 3.0
    if span_m > max_span_m:
        ap_max = max(5, int(np.floor(max_span_m / dx)))
        if ap > ap_max:
            LOG.warning(
                f'Migration aperture too wide ({span_m:.2f} m). '
                f'Using {ap_max} traces (~{ap_max * dx:.2f} m).'
            )
            ap = ap_max
    return ap


def _process_matrix_with_pipeline(
    data: np.ndarray,
    meta: Dict[str, Any],
    geolocations: Optional[np.ndarray],
    channel: int,
    p: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, float]:
    dt_ns = float(meta['sampling_time_ns'])
    dx_m = float(meta.get('sampling_step_m', 0.05))
    f0_mhz = float(meta.get('frequency_mhz', 0.0) or 0.0)
    tz_thr = _safe_tz_threshold(float(p.get('tz_thresh', 0.1)))
    tz_mode = str(p.get('tz_mode', 'line')).lower().strip()
    if tz_mode not in ('scan', 'line'):
        tz_mode = 'line'
    dewow_win = _safe_dewow_window_samples(
        requested_samples=int(p.get('dewow_win', 50)),
        dt_ns=dt_ns,
        center_freq_mhz=f0_mhz,
    )

    proc = SignalProcessor(
        data,
        sampling_time_ns=dt_ns,
        trace_spacing_m=dx_m,
    )

    if p.get('stack', False):
        stack_w, stack_d = _resolve_stack_params(p)
        proc.apply_spatial_stacking(
            window_traces=stack_w,
            decimation=stack_d,
        )

    # Per high-density or decimated geometry, line-wise time-zero is more
    # stable and preserves micro-topography better than per-trace scan mode.
    if tz_mode == 'scan' and float(proc.trace_spacing_m) >= 0.10:
        LOG.warning(
            f'Time-zero scan mode with dx={proc.trace_spacing_m:.3f} m can induce jitter. '
            'Using line mode.'
        )
        tz_mode = 'line'

    if p.get('dewow_first', False):
        if p.get('dewow', False):
            proc.dewow(window_size=dewow_win)
        if p.get('t0', False):
            proc.correct_time_zero(
                method=str(p.get('tz_method', 'threshold')),
                threshold=tz_thr,
                offset_mode=tz_mode,
            )
    else:
        if p.get('t0', False):
            proc.correct_time_zero(
                method=str(p.get('tz_method', 'threshold')),
                threshold=tz_thr,
                offset_mode=tz_mode,
            )
        if p.get('dewow', False):
            proc.dewow(window_size=dewow_win)

    if float(p.get('top_mute_ns', 0.0)) > 0.0:
        proc.apply_top_mute(mute_ns=float(p.get('top_mute_ns', 0.0)))

    if p.get('static_corr', False):
        elev = _extract_trace_elevations(
            geolocations=geolocations,
            channel=channel,
            n_trc=int(proc.processed_data.shape[1]),
        )
        if elev is not None:
            proc.apply_static_correction(
                elevations_m=elev,
                velocity_m_ns=float(p.get('velocity', 0.10)),
                reference=str(p.get('static_ref', 'max')),
                two_way=bool(p.get('static_two_way', False)),
            )
        else:
            LOG.debug('Static correction: no usable elevation vector for this swath')

    if p.get('bg', False):
        bg_method, bg_win = _safe_background_params(
            requested_method=str(p.get('bg_method', 'mean')),
            rolling=bool(p.get('bg_rolling', False)),
            requested_win_trc=int(p.get('bg_win_trc', 50)),
            dx_m=float(proc.trace_spacing_m),
        )
        proc.remove_background(
            method=bg_method,
            rolling=bool(p.get('bg_rolling', False)),
            window_traces=bg_win,
            trim_percent=float(p.get('bg_trim_pct', 10.0)),
        )
    if p.get('bp', False):
        bp_low = float(p.get('bp_low', 100.0))
        bp_high = float(p.get('bp_high', 800.0))
        if p.get('bp_auto', False):
            f0 = float(meta.get('frequency_mhz', 0.0) or 0.0)
            nyq = 0.5 * (1000.0 / max(dt_ns, 1e-9))
            if f0 >= 500.0:
                bp_low = 200.0
                bp_high = min(600.0, 0.95 * nyq)
            elif f0 > 0.0:
                bp_low = max(20.0, 0.33 * f0)
                bp_high = min(0.95 * nyq, 1.20 * f0)
            LOG.debug(
                f'Bandpass auto-tune from metadata: f0={f0:.1f} MHz '
                f'-> {bp_low:.1f}-{bp_high:.1f} MHz'
            )
        proc.apply_bandpass(
            low_freq=bp_low,
            high_freq=bp_high,
            order=int(p.get('bp_order', 4)),
            mode=str(p.get('bp_mode', 'gpr')),
        )
    if p.get('notch', False):
        proc.apply_notch(
            freq_mhz=float(p.get('notch_freq', 50.0)),
            bandwidth_mhz=float(p.get('notch_bw', 5.0)),
        )
    if p.get('sw', False):
        proc.apply_spectral_whitening(
            bp_low=float(p.get('bp_low')) if p.get('sw_bp', False) else None,
            bp_high=float(p.get('bp_high')) if p.get('sw_bp', False) else None,
        )
    if p.get('gain', False):
        proc.apply_gain(
            gain_type=str(p.get('gain_type', 'agc')),
            factor=float(p.get('gain_factor', 20.0)),
            alpha=float(p.get('gain_alpha', 0.5)),
            t_start_ns=float(p.get('gain_tstart', 5.0)),
            window_ns=float(p.get('gain_agcwin', 25.0)),
            agc_start_ns=float(p.get('gain_agc_start', 0.0)),
        )
    if p.get('hilbert', False):
        proc.apply_hilbert()
    if p.get('norm', False):
        proc.normalize(method=str(p.get('norm_method', 'minmax')))
    if p.get('mig', False):
        ap = _safe_migration_aperture_traces(
            requested_ap=int(p.get('mig_ap', 30)),
            dx_m=float(proc.trace_spacing_m),
        )
        proc.apply_migration(
            velocity_m_ns=float(p.get('mig_vel', p.get('velocity', 0.10))),
            aperture_traces=ap,
        )

    return proc.get_processed_data(), proc.get_time_axis(), float(proc.trace_spacing_m)


class ProcessingThread(QThread):
    finished = pyqtSignal(int, list)
    error = pyqtSignal(int, str)
    progress = pyqtSignal(int, str)

    def __init__(self, request_id: int, jobs: List[Dict[str, Any]], params: Dict[str, Any]):
        super().__init__()
        self.request_id = int(request_id)
        self.jobs = jobs
        self.params = params

    def run(self):
        try:
            panels = []
            n_jobs = len(self.jobs)
            p = self.params

            for i, job in enumerate(self.jobs):
                if self.isInterruptionRequested():
                    self.finished.emit(self.request_id, [])
                    return

                rv = job['radar_volume']
                ch = int(job['channel'])
                meta = job['metadata']
                data = rv[:, ch, :]

                proc_data, time_axis, dx_m = _process_matrix_with_pipeline(
                    data=data,
                    meta=meta,
                    geolocations=job.get('geolocations'),
                    channel=ch,
                    p=p,
                )

                title = f"{meta['swath_name']} · Ch {ch} · {meta['dtype_name']}"
                n_trc = int(proc_data.shape[1])
                total_len_m = max(0.0, (n_trc - 1) * dx_m)
                x_start_m = 0.0
                x_disp_w_m = float(p.get('x_display_width_m')) if p.get('lock_x_scale', False) else None

                if p.get('limit_view', False):
                    start_m = max(0.0, float(p.get('view_start_m', 0.0)))
                    width_m = max(dx_m, float(p.get('view_width_m', dx_m)))
                    vs = int(np.floor(start_m / max(dx_m, 1e-9)))
                    vs = max(0, min(vs, n_trc - 1))
                    vw = int(np.floor(width_m / max(dx_m, 1e-9))) + 1
                    vw = min(max(2, vw), n_trc - vs)
                    proc_data = proc_data[:, vs:vs + vw]
                    x_start_m = vs * dx_m
                    x1_m = x_start_m + (proc_data.shape[1] - 1) * dx_m
                    x_disp_w_m = width_m
                    title += f'  [{x_start_m:.1f}-{x1_m:.1f} m]'
                else:
                    max_display_traces = int(max(200, p.get('max_display_traces', 2500)))
                    n_trc_total = int(proc_data.shape[1])
                    if n_trc_total > max_display_traces:
                        step_dec = max(1, int(np.ceil(n_trc_total / max_display_traces)))
                        proc_data = proc_data[:, ::step_dec]
                        dx_m = float(dx_m) * float(step_dec)
                        LOG.debug(
                            f'Spatial decimation applied: 1/{step_dec} '
                            f'(from {n_trc_total} to {proc_data.shape[1]} traces)'
                        )

                panels.append({
                    'data': proc_data,
                    'time_axis': time_axis,
                    'title': title,
                    'trace_spacing_m': dx_m,
                    'x_start_m': x_start_m,
                    'x_display_width_m': x_disp_w_m,
                    'total_length_m': total_len_m,
                })
                self.progress.emit(
                    self.request_id,
                    f'Processed {i + 1}/{n_jobs}: {meta["swath_name"]} ch{ch}',
                )

            if self.isInterruptionRequested():
                self.finished.emit(self.request_id, [])
                return
            self.finished.emit(self.request_id, panels)
        except Exception as e:
            self.error.emit(self.request_id, str(e))


def _build_batch_stage_plan(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    stages: List[Dict[str, Any]] = []

    if bool(params.get('stack', False)):
        stages.append({
            'name': 'stack',
            'folder': '00_stack',
        })

    dewow_enabled = bool(params.get('dewow', False))
    dewow_first = bool(params.get('dewow_first', False))
    has_time0_stage = bool(params.get('t0', False))
    has_time0_stage = has_time0_stage or (float(params.get('top_mute_ns', 0.0)) > 0.0)
    has_time0_stage = has_time0_stage or bool(params.get('static_corr', False))

    if dewow_enabled and dewow_first:
        stages.append({
            'name': 'dewow',
            'folder': '02_dewow',
        })
    if has_time0_stage:
        stages.append({
            'name': 'time0',
            'folder': '01_time0',
        })
    if dewow_enabled and (not dewow_first):
        stages.append({
            'name': 'dewow',
            'folder': '02_dewow',
        })
    if bool(params.get('bg', False)):
        stages.append({
            'name': 'background',
            'folder': '03_background',
        })
    if bool(params.get('bp', False)):
        stages.append({
            'name': 'bandpass',
            'folder': '04_bandpass',
        })
    if bool(params.get('gain', False)):
        stages.append({
            'name': 'gain',
            'folder': '05_gain',
        })

    return stages


class BatchProcessingThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, project_path: str, pipeline_params: Dict[str, Any]):
        super().__init__()
        self.project_path = Path(project_path)
        self.pipeline_params = dict(pipeline_params)

    def _apply_stage(
        self,
        stage_name: str,
        proc: SignalProcessor,
        meta: Dict[str, Any],
        geolocations: Optional[np.ndarray],
        channel: int,
    ):
        p = self.pipeline_params
        dt_ns = float(meta.get('sampling_time_ns', proc.sampling_time_ns))
        f0_mhz = float(meta.get('frequency_mhz', 0.0) or 0.0)

        if stage_name == 'stack':
            stack_w, stack_d = _resolve_stack_params(p)
            proc.apply_spatial_stacking(
                window_traces=stack_w,
                decimation=stack_d,
            )
            return

        if stage_name == 'time0':
            if p.get('t0', False):
                tz_thr = _safe_tz_threshold(float(p.get('tz_thresh', 0.1)))
                tz_mode = str(p.get('tz_mode', 'line')).lower().strip()
                if tz_mode not in ('scan', 'line'):
                    tz_mode = 'line'
                if tz_mode == 'scan' and float(proc.trace_spacing_m) >= 0.10:
                    LOG.warning(
                        f'Batch time-zero scan mode with dx={proc.trace_spacing_m:.3f} m '
                        'can induce jitter. Using line mode.'
                    )
                    tz_mode = 'line'
                proc.correct_time_zero(
                    method=str(p.get('tz_method', 'threshold')),
                    threshold=tz_thr,
                    offset_mode=tz_mode,
                )
            if float(p.get('top_mute_ns', 0.0)) > 0.0:
                proc.apply_top_mute(float(p.get('top_mute_ns', 0.0)))
            if p.get('static_corr', False):
                elev = _extract_trace_elevations(
                    geolocations=geolocations,
                    channel=channel,
                    n_trc=int(proc.processed_data.shape[1]),
                )
                if elev is not None:
                    proc.apply_static_correction(
                        elevations_m=elev,
                        velocity_m_ns=float(p.get('velocity', 0.10)),
                        reference=str(p.get('static_ref', 'max')),
                        two_way=bool(p.get('static_two_way', False)),
                    )
            return

        if stage_name == 'dewow':
            dewow_win = _safe_dewow_window_samples(
                requested_samples=int(p.get('dewow_win', 50)),
                dt_ns=dt_ns,
                center_freq_mhz=f0_mhz,
            )
            proc.dewow(window_size=dewow_win)
            return

        if stage_name == 'background':
            bg_method, bg_win = _safe_background_params(
                requested_method=str(p.get('bg_method', 'mean')),
                rolling=bool(p.get('bg_rolling', False)),
                requested_win_trc=int(p.get('bg_win_trc', 50)),
                dx_m=float(proc.trace_spacing_m),
            )
            proc.remove_background(
                method=bg_method,
                rolling=bool(p.get('bg_rolling', False)),
                window_traces=bg_win,
                trim_percent=float(p.get('bg_trim_pct', 10.0)),
            )
            return

        if stage_name == 'bandpass':
            bp_low = float(p.get('bp_low', 100.0))
            bp_high = float(p.get('bp_high', 800.0))
            if p.get('bp_auto', False):
                f0 = float(meta.get('frequency_mhz', 0.0) or 0.0)
                nyq = 0.5 * (1000.0 / max(dt_ns, 1e-9))
                if f0 >= 500.0:
                    bp_low = 200.0
                    bp_high = min(600.0, 0.95 * nyq)
                elif f0 > 0.0:
                    bp_low = max(20.0, 0.33 * f0)
                    bp_high = min(0.95 * nyq, 1.20 * f0)
            proc.apply_bandpass(
                low_freq=bp_low,
                high_freq=bp_high,
                order=int(p.get('bp_order', 4)),
                mode=str(p.get('bp_mode', 'gpr')),
            )
            return

        if stage_name == 'gain':
            proc.apply_gain(
                gain_type=str(p.get('gain_type', 'agc')),
                factor=float(p.get('gain_factor', 20.0)),
                alpha=float(p.get('gain_alpha', 0.5)),
                t_start_ns=float(p.get('gain_tstart', 5.0)),
                window_ns=float(p.get('gain_agcwin', 25.0)),
                agc_start_ns=float(p.get('gain_agc_start', 0.0)),
            )

    def run(self):
        try:
            project = self.project_path
            raw_dir = project / 'raw'
            if not raw_dir.exists():
                raise RuntimeError(f'Project raw directory not found: {raw_dir}')

            stage_plan = _build_batch_stage_plan(self.pipeline_params)
            if not stage_plan:
                raise RuntimeError('No batch stage enabled (Stack/Time0/Dewow/Background/Bandpass/Gain).')
            for stage in stage_plan:
                (project / stage['folder']).mkdir(parents=True, exist_ok=True)

            raw_files = sorted(raw_dir.glob('*.ogpr'))
            if not raw_files:
                raise RuntimeError(f'No OGPR files found in: {raw_dir}')

            meta_path = project / 'metadata.json'
            meta_doc = _read_json(meta_path, default={})
            if 'processing_log' not in meta_doc or not isinstance(meta_doc['processing_log'], list):
                meta_doc['processing_log'] = []

            run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_log: Dict[str, Any] = {
                'run_id': run_id,
                'started_at': datetime.now().isoformat(),
                'pipeline': self.pipeline_params,
                'stages': [s['name'] for s in stage_plan],
                'inputs': [f.name for f in raw_files],
                'outputs': [],
            }

            self.progress.emit(f'Batch start: {len(raw_files)} file(s), stages={run_log["stages"]}')

            for file_idx, ogpr_file in enumerate(raw_files, start=1):
                if self.isInterruptionRequested():
                    raise RuntimeError('Batch processing interrupted.')

                size_mb = ogpr_file.stat().st_size / (1024 * 1024)
                parser = OGPRParser(str(ogpr_file))
                data_dict = parser.load_data(lazy=(size_mb > 150))
                rv = data_dict['radar_volume']
                geolocations = data_dict.get('geolocations')
                meta = data_dict['metadata']
                dt_ns = float(meta.get('sampling_time_ns', np.nan))
                dx_m = float(meta.get('sampling_step_m', 0.05))
                n_ch = int(meta.get('channels_count', rv.shape[1]))

                self.progress.emit(
                    f'[{file_idx}/{len(raw_files)}] {ogpr_file.name}: {n_ch} channel(s)'
                )

                for ch in range(n_ch):
                    if self.isInterruptionRequested():
                        raise RuntimeError('Batch processing interrupted.')

                    proc = SignalProcessor(
                        rv[:, ch, :],
                        sampling_time_ns=dt_ns,
                        trace_spacing_m=dx_m,
                    )
                    base_name = f'{ogpr_file.stem}__ch{ch:02d}'

                    for stage_idx, stage in enumerate(stage_plan, start=1):
                        self._apply_stage(
                            stage_name=stage['name'],
                            proc=proc,
                            meta=meta,
                            geolocations=geolocations,
                            channel=ch,
                        )
                        out = proc.get_processed_data()
                        t = proc.get_time_axis()

                        out_file = project / stage['folder'] / f'{base_name}.npz'
                        np.savez_compressed(
                            out_file,
                            data=np.asarray(out),
                            time_ns=np.asarray(t, dtype=np.float32),
                            trace_spacing_m=float(proc.trace_spacing_m),
                            sampling_time_ns=float(dt_ns),
                            source_file=ogpr_file.name,
                            swath_name=str(meta.get('swath_name', 'Unknown')),
                            channel=int(ch),
                            stage=str(stage['name']),
                            stage_index=int(stage_idx),
                            run_id=str(run_id),
                        )
                        run_log['outputs'].append(str(out_file.relative_to(project)))

                self.progress.emit(
                    f'Completed {ogpr_file.name} ({file_idx}/{len(raw_files)})'
                )

            run_log['completed_at'] = datetime.now().isoformat()
            run_log['status'] = 'ok'
            meta_doc['processing_log'].append(run_log)
            _write_json(meta_path, meta_doc)

            self.finished.emit(
                f'Batch completed: {len(raw_files)} file(s), {len(run_log["outputs"])} stage file(s)'
            )

        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Swath sidebar entry
# ---------------------------------------------------------------------------

class SwathSidebarEntry(QFrame):
    visibility_changed = pyqtSignal()
    channel_changed    = pyqtSignal()

    def __init__(self, data_dict: dict, parent=None):
        super().__init__(parent)
        self.data_dict = data_dict
        self.setFrameShape(QFrame.Shape.StyledPanel)

        meta = data_dict['metadata']
        fp   = Path(data_dict['filepath'])

        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(4)

        self.check = QCheckBox()
        self.check.setChecked(True)
        self.check.stateChanged.connect(lambda _: self.visibility_changed.emit())
        lay.addWidget(self.check)

        lbl = QLabel(
            f"<b>{meta['swath_name']}</b> "
            f"<small>[{meta['dtype_name']}]</small><br>"
            f"<small>{fp.name}</small>"
        )
        lbl.setWordWrap(False)
        lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        lay.addWidget(lbl)

        lay.addWidget(QLabel('Ch:'))
        self.ch_spin = QSpinBox()
        self.ch_spin.setMinimum(0)
        self.ch_spin.setMaximum(meta['channels_count'] - 1)
        self.ch_spin.setFixedWidth(48)
        self.ch_spin.valueChanged.connect(lambda _: self.channel_changed.emit())
        lay.addWidget(self.ch_spin)

    @property
    def visible(self) -> bool:
        return self.check.isChecked()

    @property
    def channel(self) -> int:
        return self.ch_spin.value()


# ---------------------------------------------------------------------------
# Power spectrum dialog
# ---------------------------------------------------------------------------

class PowerSpectrumDialog(QDialog):
    def __init__(self, freqs, power_db, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Power Spectrum  (first visible swath)')
        self.resize(720, 400)

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        fig = Figure(figsize=(7, 3.6), dpi=100, facecolor='#1e1e1e')
        ax  = fig.add_subplot(111)
        ax.set_facecolor('#111111')
        ax.plot(freqs, power_db, color='#00aaff', linewidth=1.2)
        ax.set_xlabel('Frequency (MHz)', color='#cccccc', fontsize=9)
        ax.set_ylabel('Power (dB)',      color='#cccccc', fontsize=9)
        ax.set_title('Mean Trace Power Spectrum – use to set Bandpass cutoffs',
                     color='white', fontsize=9, fontweight='bold')
        ax.tick_params(colors='#aaaaaa', labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor('#555555')
        ax.grid(True, color='#333333', linewidth=0.5)
        fig.tight_layout()

        canvas = FigureCanvasQTAgg(fig)
        lay    = QVBoxLayout(self)
        lay.addWidget(canvas)
        btn = QPushButton('Close')
        btn.clicked.connect(self.accept)
        lay.addWidget(btn)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class OGPRViewerMainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('OGPR Radar Viewer')
        self.setGeometry(60, 60, 1700, 980)

        self.settings       = QSettings('OGPR', 'RadarViewer')
        self.last_directory = self.settings.value('last_directory', str(Path.home()))
        self._loaders: List[FileLoaderThread] = []
        self._swath_entries: List[SwathSidebarEntry] = []
        self._proc_worker: Optional[ProcessingThread] = None
        self._proc_workers: List[ProcessingThread] = []
        self._batch_worker: Optional[BatchProcessingThread] = None
        self._batch_workers: List[BatchProcessingThread] = []
        self.project_path: Optional[Path] = None
        self._render_request_id: int = 0

        self._build_ui()
        self._build_menu()
        self._build_statusbar()
        LOG.debug('Main window ready')

    # ------------------------------------------------------------------
    # UI skeleton
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_sidebar())
        self.canvas = MultiPanelCanvas(
            parent=self,
            dpi=96,
            # Keep clipping conservative by default: overly aggressive values
            # make high-dynamic-range float32 data appear saturated.
            clip_pct=99.0,
            dw_skip_frac=0.05,
        )
        splitter.addWidget(self.canvas)
        splitter.addWidget(self._build_controls())
        splitter.setSizes([230, 1120, 350])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        root.addWidget(splitter)

    # ---- Sidebar --------------------------------------------------

    def _build_sidebar(self) -> QWidget:
        c = QWidget()
        c.setMinimumWidth(200)
        c.setMaximumWidth(270)
        v = QVBoxLayout(c)
        v.setContentsMargins(2, 2, 2, 2)

        lbl = QLabel('Swaths')
        lbl.setFont(QFont('', 10, QFont.Weight.Bold))
        v.addWidget(lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._sidebar_inner  = QWidget()
        self._sidebar_layout = QVBoxLayout(self._sidebar_inner)
        self._sidebar_layout.setSpacing(2)
        self._sidebar_layout.setContentsMargins(2, 2, 2, 2)
        self._sidebar_layout.addStretch()
        scroll.setWidget(self._sidebar_inner)
        v.addWidget(scroll)

        btn = QPushButton('\u2795  Open file(s)\u2026')
        btn.clicked.connect(self.open_files)
        v.addWidget(btn)
        return c

    # ---- Controls -------------------------------------------------

    def _build_controls(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(290)
        scroll.setMaximumWidth(360)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        panel  = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(6)
        layout.setContentsMargins(4, 4, 4, 4)

        hdr = QLabel('Processing  (global)')
        hdr.setFont(QFont('', 9, QFont.Weight.Bold))
        layout.addWidget(hdr)

        layout.addWidget(self._grp_pipeline())
        layout.addWidget(self._grp_spatial_stack())
        layout.addWidget(self._grp_timezero())
        layout.addWidget(self._grp_topography())
        layout.addWidget(self._grp_dewow())
        layout.addWidget(self._grp_background())
        layout.addWidget(self._grp_bandpass())
        layout.addWidget(self._grp_notch())
        layout.addWidget(self._grp_spectral_whitening())
        layout.addWidget(self._grp_gain())
        layout.addWidget(self._grp_hilbert_norm())
        layout.addWidget(self._grp_migration())
        layout.addWidget(self._grp_view())
        layout.addWidget(self._grp_display())
        layout.addWidget(self._grp_export())
        layout.addStretch()

        scroll.setWidget(panel)
        return scroll

    # ------------------------------------------------------------------
    # Group-builder helpers
    # ------------------------------------------------------------------

    def _cb(self, v, label, attr):
        cb = QCheckBox(label)
        cb.stateChanged.connect(self._apply_and_render)
        v.addWidget(cb)
        setattr(self, attr, cb)

    def _combo(self, v, label, attr, items):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        c = QComboBox(); c.addItems(items)
        c.currentTextChanged.connect(self._apply_and_render)
        row.addWidget(c)
        v.addLayout(row)
        setattr(self, attr, c)

    def _dspin(self, v, label, attr, lo, hi, val, step=0.1, suffix=''):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        s = QDoubleSpinBox()
        s.setRange(lo, hi); s.setValue(val); s.setSingleStep(step)
        if suffix: s.setSuffix(suffix)
        s.valueChanged.connect(self._apply_and_render)
        row.addWidget(s)
        v.addLayout(row)
        setattr(self, attr, s)

    def _ispin(self, v, label, attr, lo, hi, val, suffix=''):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        s = QSpinBox()
        s.setRange(lo, hi); s.setValue(val)
        if suffix: s.setSuffix(suffix)
        s.valueChanged.connect(self._apply_and_render)
        row.addWidget(s)
        v.addLayout(row)
        setattr(self, attr, s)

    # ------------------------------------------------------------------
    # Processing groups
    # ------------------------------------------------------------------

    def _grp_pipeline(self):
        g = QGroupBox('Pipeline'); v = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel('Preset:'))
        self.pipeline_preset = QComboBox()
        self.pipeline_preset.addItems([
            'Raw view',
            'Basic clean (Matlab-like)',
            'Archaeo detail',
            'Noisy ground',
            'Stream UP stable',
        ])
        row.addWidget(self.pipeline_preset)
        v.addLayout(row)

        btn = QPushButton('Apply preset')
        btn.clicked.connect(self._apply_pipeline_preset)
        v.addWidget(btn)

        self.cb_dewow_first = QCheckBox('Dewow before Time0')
        self.cb_dewow_first.stateChanged.connect(self._apply_and_render)
        self.cb_dewow_first.blockSignals(True)
        self.cb_dewow_first.setChecked(True)
        self.cb_dewow_first.blockSignals(False)
        v.addWidget(self.cb_dewow_first)

        self.pipeline_summary = QLabel('RAW only')
        self.pipeline_summary.setWordWrap(True)
        self.pipeline_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.pipeline_summary.setStyleSheet('color:#b8b8b8;')
        v.addWidget(self.pipeline_summary)

        g.setLayout(v); return g

    def _grp_spatial_stack(self):
        g = QGroupBox('Spatial Stacking / Smoothing'); v = QVBoxLayout()
        self._cb(v, 'Enable', 'cb_stack')
        self._ispin(v, '  Window:', 'stack_win_traces', 1, 401, 5, ' trc')
        self._ispin(v, '  Decimation:', 'stack_decim', 1, 128, 1, ' x')
        self._cb(v, '  Auto decimation = window when decim=1', 'cb_stack_auto_decim')
        self.cb_stack_auto_decim.blockSignals(True)
        self.cb_stack_auto_decim.setChecked(True)
        self.cb_stack_auto_decim.blockSignals(False)
        v.addWidget(QLabel(
            '<small><i>Use on ultra-dense profiles before background removal.</i></small>'
        ))
        g.setLayout(v); return g

    def _grp_timezero(self):
        g = QGroupBox('Time-Zero Correction'); v = QVBoxLayout()
        self._cb(v, 'Enable', 'cb_t0')
        self._combo(v, '  Correction:', 'tz_mode', ['line', 'scan'])
        self._combo(v, '  Method:', 'tz_method', ['threshold', 'max', 'xcorr'])
        self._dspin(v, '  Threshold:', 'tz_thresh', 0.02, 0.10, 0.08, 0.01)
        self._dspin(v, '  Top mute:', 'top_mute_ns', 0.0, 60.0, 0.0, 0.5, ' ns')
        g.setLayout(v); return g

    def _grp_topography(self):
        g = QGroupBox('Topography / Static Correction'); v = QVBoxLayout()
        self._cb(v, 'Enable', 'cb_static_corr')
        self._combo(v, '  Datum:', 'static_ref_mode', ['max', 'median', 'mean'])
        self._cb(v, '  Two-way time (2*dz/v)', 'cb_static_two_way')
        v.addWidget(QLabel(
            '<small><i>Uses Z from geolocations; aligns traces to a common topographic datum.</i></small>'
        ))
        g.setLayout(v); return g

    def _grp_dewow(self):
        g = QGroupBox('Dewow  (DC drift)'); v = QVBoxLayout()
        self._cb(v, 'Enable', 'cb_dewow')
        self._ispin(v, '  Window:', 'dewow_win', 5, 500, 50, ' smp')
        g.setLayout(v); return g

    def _grp_background(self):
        g = QGroupBox('Background Removal'); v = QVBoxLayout()
        self._cb(v, 'Enable', 'cb_bg')
        self._combo(v, '  Method:', 'bg_method', ['median', 'trimmed', 'mean'])
        self._dspin(v, '  Trim (%):', 'bg_trim_pct', 0.0, 30.0, 10.0, 1.0, ' %')
        self._cb(v, '  Rolling window', 'cb_bg_rolling')
        self._ispin(v, '  Win traces:', 'bg_win_traces', 5, 500, 50, ' trc')
        g.setLayout(v); return g

    def _grp_bandpass(self):
        g = QGroupBox('Bandpass Filter'); v = QVBoxLayout()
        self._cb(v, 'Enable', 'cb_bp')
        self._combo(v, '  Type:',  'bp_mode',  ['gpr', 'butter'])
        self._cb(v, '  Auto from antenna metadata', 'cb_bp_auto')
        self.cb_bp_auto.blockSignals(True)
        self.cb_bp_auto.setChecked(True)
        self.cb_bp_auto.blockSignals(False)
        self._dspin(v, '  Low:',   'bp_low',   1.0, 4000.0, 100.0, 10.0, ' MHz')
        self._dspin(v, '  High:',  'bp_high',  1.0, 4000.0, 800.0, 10.0, ' MHz')
        self._ispin(v, '  Order:', 'bp_order', 1, 8, 4)
        btn = QPushButton('\U0001f4c8  Power spectrum\u2026')
        btn.clicked.connect(self._show_power_spectrum)
        v.addWidget(btn)
        g.setLayout(v); return g

    def _grp_notch(self):
        g = QGroupBox('Notch Filter'); v = QVBoxLayout()
        self._cb(v, 'Enable', 'cb_notch')
        self._dspin(v, '  Freq:', 'notch_freq', 1.0, 4000.0,  50.0, 5.0, ' MHz')
        self._dspin(v, '  BW:',  'notch_bw',   1.0,  500.0,   5.0, 1.0, ' MHz')
        g.setLayout(v); return g

    def _grp_spectral_whitening(self):
        g = QGroupBox('Spectral Whitening'); v = QVBoxLayout()
        self._cb(v, 'Enable', 'cb_sw')
        self._cb(v, '  Use bandpass in spectral domain', 'cb_sw_bp')
        g.setLayout(v); return g

    def _grp_gain(self):
        g = QGroupBox('Gain'); v = QVBoxLayout()
        self._cb(v, 'Enable', 'cb_gain')
        self._combo(v, '  Type:', 'gain_type', ['agc', 'sec', 'exp', 'linear'])
        v.addWidget(QLabel(
            '<small><i>SEC = Spreading &amp; Exponential Compensation</i></small>'
        ))
        # Factor: for SEC/exp/lin = max gain multiplier at t_max.
        # Range 1-500 to cover from gentle (10x) to aggressive (500x) gains.
        self._dspin(v, '  Factor (SEC/exp/lin):',
                    'gain_factor', 1.0, 500.0, 20.0, 5.0)
        v.addWidget(QLabel(
            '<small><i>Factor = max gain multiplier at t_max</i></small>'
        ))
        self._dspin(v, '  \u03b1 (SEC) [1/ns]:', 'gain_alpha',  0.0, 5.0, 0.5, 0.05)
        self._dspin(v, '  t-start [ns]:', 'gain_tstart', 0.0, 100.0, 5.0, 0.5, ' ns')
        self._dspin(v, '  AGC window:',   'gain_agc_win', 1.0, 500.0, 25.0, 5.0, ' ns')
        self._dspin(v, '  AGC start gate:', 'gain_agc_start', 0.0, 60.0, 0.0, 0.5, ' ns')
        g.setLayout(v); return g

    def _grp_hilbert_norm(self):
        g = QGroupBox('Envelope & Normalisation'); v = QVBoxLayout()
        self._cb(v, 'Hilbert (Envelope)', 'cb_hilbert')
        self._cb(v, 'Normalize', 'cb_norm')
        self._combo(v, '  Method:', 'norm_method', ['minmax', 'zscore', 'robust'])
        g.setLayout(v); return g

    def _grp_migration(self):
        g = QGroupBox('Kirchhoff Migration'); v = QVBoxLayout()
        self._cb(v, 'Enable', 'cb_mig')
        v.addWidget(QLabel(
            '<small><i>Uses the velocity set in Display \u2193</i></small>'
        ))
        self._ispin(v, '  Aperture:', 'mig_aperture', 5, 200, 12, ' trc')
        g.setLayout(v); return g

    # ------------------------------------------------------------------
    # View / metric-range group
    # ------------------------------------------------------------------

    def _grp_view(self):
        g = QGroupBox('View  (metric window)'); v = QVBoxLayout()

        self.cb_limit_view = QCheckBox('Limit to range')
        self.cb_limit_view.stateChanged.connect(self._apply_and_render)
        v.addWidget(self.cb_limit_view)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel('  Start:'))
        self.view_start_m = QDoubleSpinBox()
        self.view_start_m.setRange(0.0, 1_000_000.0)
        self.view_start_m.setValue(0.0)
        self.view_start_m.setSingleStep(5.0)
        self.view_start_m.setDecimals(2)
        self.view_start_m.setSuffix(' m')
        self.view_start_m.valueChanged.connect(self._on_view_range_changed)
        row1.addWidget(self.view_start_m)
        v.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel('  Width:'))
        self.view_width_m = QDoubleSpinBox()
        self.view_width_m.setRange(0.1, 1_000_000.0)
        self.view_width_m.setValue(200.0)
        self.view_width_m.setSingleStep(10.0)
        self.view_width_m.setDecimals(2)
        self.view_width_m.setSuffix(' m')
        self.view_width_m.valueChanged.connect(self._on_view_range_changed)
        row2.addWidget(self.view_width_m)
        v.addLayout(row2)

        self.view_info = QLabel(
            '<small><i>Window in meters; enable "Limit to range" to pan along long profiles.</i></small>'
        )
        v.addWidget(self.view_info)

        row3 = QHBoxLayout()
        self.btn_prev_window = QPushButton('Prev')
        self.btn_prev_window.clicked.connect(lambda: self._nudge_view_window(-1))
        row3.addWidget(self.btn_prev_window)
        self.btn_next_window = QPushButton('Next')
        self.btn_next_window.clicked.connect(lambda: self._nudge_view_window(+1))
        row3.addWidget(self.btn_next_window)
        v.addLayout(row3)

        g.setLayout(v); return g

    def _nudge_view_window(self, direction: int):
        step = float(self.view_width_m.value())
        new_start = max(0.0, float(self.view_start_m.value()) + direction * step)
        self.view_start_m.blockSignals(True)
        self.view_start_m.setValue(new_start)
        self.view_start_m.blockSignals(False)
        self._apply_and_render()

    def _update_view_limits(self):
        if not self._swath_entries:
            return
        max_len_m = 0.0
        min_dx = None
        for e in self._swath_entries:
            m = e.data_dict['metadata']
            dx = float(m.get('sampling_step_m', 0.05))
            n_trc = int(m.get('slices_count', 0))
            length_m = max(0.0, (n_trc - 1) * dx)
            max_len_m = max(max_len_m, length_m)
            min_dx = dx if (min_dx is None) else min(min_dx, dx)

        self.view_start_m.setMaximum(max(0.0, max_len_m))
        self.view_width_m.setMaximum(max(1.0, max_len_m if max_len_m > 0 else 1.0))
        if self.view_start_m.value() > max_len_m:
            self.view_start_m.setValue(max_len_m)
        if min_dx is not None:
            self.view_info.setText(
                f'<small><i>Loaded profiles: 0 - {max_len_m:.2f} m; min dx {min_dx:.4f} m.</i></small>'
            )

    def _on_view_range_changed(self):
        if self.cb_limit_view.isChecked():
            self._apply_and_render()

    # ------------------------------------------------------------------
    # Display group
    # ------------------------------------------------------------------

    def _grp_display(self):
        g = QGroupBox('Display'); v = QVBoxLayout()

        self._combo(v, 'Colormap:', 'cmap_combo',
                    ['gray', 'seismic', 'RdBu_r', 'viridis', 'jet', 'hot', 'bwr'])

        row = QHBoxLayout()
        row.addWidget(QLabel('Contrast preset:'))
        self.contrast_preset = QComboBox()
        self.contrast_preset.addItems(['Auto', '1 % clip', '3 % clip', 'Manual'])
        self.contrast_preset.currentTextChanged.connect(self._on_contrast_preset_changed)
        row.addWidget(self.contrast_preset)
        v.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel('Clip percentile:'))
        self.clip_pct_spin = QDoubleSpinBox()
        self.clip_pct_spin.setRange(90.0, 99.9)
        self.clip_pct_spin.setSingleStep(0.5)
        self.clip_pct_spin.setDecimals(1)
        self.clip_pct_spin.setSuffix(' %')
        self.clip_pct_spin.setValue(self.canvas.get_clip_pct())
        self.clip_pct_spin.valueChanged.connect(self.canvas.set_clip_pct)
        row.addWidget(self.clip_pct_spin)
        v.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel('DW skip for clip:'))
        self.dw_skip_spin = QDoubleSpinBox()
        self.dw_skip_spin.setRange(0.0, 90.0)
        self.dw_skip_spin.setSingleStep(5.0)
        self.dw_skip_spin.setDecimals(0)
        self.dw_skip_spin.setSuffix(' %')
        self.dw_skip_spin.setValue(self.canvas.get_dw_skip_frac() * 100.0)
        self.dw_skip_spin.valueChanged.connect(
            lambda pct: self.canvas.set_dw_skip_frac(pct / 100.0)
        )
        row.addWidget(self.dw_skip_spin)
        v.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel('Amplitude scale:'))
        self.norm_scale_combo = QComboBox()
        self.norm_scale_combo.addItems(['linear', 'symlog'])
        self.norm_scale_combo.currentTextChanged.connect(self._apply_and_render)
        row.addWidget(self.norm_scale_combo)
        v.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel('SymLog lin zone:'))
        self.symlog_linthresh_pct = QDoubleSpinBox()
        self.symlog_linthresh_pct.setRange(0.1, 50.0)
        self.symlog_linthresh_pct.setSingleStep(0.5)
        self.symlog_linthresh_pct.setDecimals(1)
        self.symlog_linthresh_pct.setSuffix(' %')
        self.symlog_linthresh_pct.setValue(3.0)
        self.symlog_linthresh_pct.valueChanged.connect(self._apply_and_render)
        row.addWidget(self.symlog_linthresh_pct)
        v.addLayout(row)

        row = QHBoxLayout()
        self.cb_wiggle_overlay = QCheckBox('Wiggle overlay')
        self.cb_wiggle_overlay.stateChanged.connect(self._on_wiggle_toggle)
        row.addWidget(self.cb_wiggle_overlay)
        v.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel('Wiggle scale:'))
        self.wiggle_scale = QDoubleSpinBox()
        self.wiggle_scale.setRange(0.2, 4.0)
        self.wiggle_scale.setValue(1.0)
        self.wiggle_scale.setSingleStep(0.1)
        self.wiggle_scale.valueChanged.connect(
            lambda val: self.canvas.set_wiggle_scale(val)
        )
        self.wiggle_scale.setEnabled(False)
        row.addWidget(self.wiggle_scale)
        v.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel('Wiggle step:'))
        self.wiggle_stride = QSpinBox()
        self.wiggle_stride.setRange(1, 80)
        self.wiggle_stride.setValue(8)
        self.wiggle_stride.valueChanged.connect(
            lambda val: self.canvas.set_wiggle_stride(val)
        )
        self.wiggle_stride.setEnabled(False)
        row.addWidget(self.wiggle_stride)
        v.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel('Vertical aspect:'))
        self.aspect_factor_spin = QDoubleSpinBox()
        self.aspect_factor_spin.setRange(0.05, 5.00)
        self.aspect_factor_spin.setSingleStep(0.05)
        self.aspect_factor_spin.setDecimals(2)
        self.aspect_factor_spin.setValue(0.50)
        # Geometry-preserving mode: numeric aspect override disabled.
        self.aspect_factor_spin.setEnabled(False)
        self.aspect_factor_spin.setToolTip(
            'Disabled: aspect is automatic to preserve radargram geometry.'
        )
        row.addWidget(self.aspect_factor_spin)
        v.addLayout(row)

        row = QHBoxLayout()
        self.cb_lock_x_scale = QCheckBox('Lock X scale')
        self.cb_lock_x_scale.setChecked(True)
        self.cb_lock_x_scale.stateChanged.connect(self._on_lock_x_scale_toggle)
        row.addWidget(self.cb_lock_x_scale)
        row.addWidget(QLabel('Width:'))
        self.x_display_width_m = QDoubleSpinBox()
        self.x_display_width_m.setRange(1.0, 1_000_000.0)
        self.x_display_width_m.setValue(200.0)
        self.x_display_width_m.setSingleStep(10.0)
        self.x_display_width_m.setDecimals(2)
        self.x_display_width_m.setSuffix(' m')
        self.x_display_width_m.valueChanged.connect(self._apply_and_render)
        row.addWidget(self.x_display_width_m)
        v.addLayout(row)
        v.addWidget(QLabel('<small><i>Mouse on panel: wheel=zoom, drag left=pan, middle click=reset.</i></small>'))

        v.addWidget(_hsep())

        v.addWidget(QLabel('Y axis:'))
        self.ymode_grp = QButtonGroup(self)
        for label, mode in [
            ('Time (ns)',           'time'),
            ('Depth \u2013 relative (m)', 'depth_rel'),
            ('Depth \u2013 absolute (m)', 'depth_abs'),
        ]:
            rb = QRadioButton(label)
            rb.setProperty('ymode', mode)
            rb.toggled.connect(self._apply_and_render)
            self.ymode_grp.addButton(rb)
            v.addWidget(rb)
            if mode == 'time':
                rb.setChecked(True)
                self._ymode_default = rb

        v.addWidget(_hsep())

        v.addWidget(QLabel('EM Velocity  [m/ns]:'))
        v.addWidget(QLabel(
            '<small>dry sand 0.12\u20130.15 \u00b7 moist soil 0.08\u20130.10<br>'
            'wet clay 0.05\u20130.07 \u00b7 concrete 0.10\u20130.12</small>'
        ))
        row = QHBoxLayout()
        self.velocity_spin = QDoubleSpinBox()
        self.velocity_spin.setRange(0.01, 0.30)
        self.velocity_spin.setValue(0.10)
        self.velocity_spin.setSingleStep(0.005)
        self.velocity_spin.setDecimals(3)
        self.velocity_spin.setSuffix(' m/ns')
        self.velocity_spin.valueChanged.connect(self._apply_and_render)
        row.addWidget(self.velocity_spin)
        v.addLayout(row)

        v.addWidget(_hsep())

        reset_btn = QPushButton('Reset all filters')
        reset_btn.clicked.connect(self._reset_filters)
        v.addWidget(reset_btn)

        g.setLayout(v); return g

    def _grp_export(self):
        g = QGroupBox('Export'); v = QVBoxLayout()
        btn = QPushButton('Export current view\u2026')
        btn.clicked.connect(self._export)
        v.addWidget(btn)
        b2 = QPushButton('Batch process project\u2026')
        b2.clicked.connect(self._run_project_batch)
        v.addWidget(b2)
        dbg = QPushButton('Export debug bundle (.zip)\u2026')
        dbg.clicked.connect(self._export_debug_bundle)
        v.addWidget(dbg)
        g.setLayout(v); return g

    # ------------------------------------------------------------------
    # Menu / Status
    # ------------------------------------------------------------------

    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu('&File')
        npj = QAction('&New Project\u2026', self)
        npj.setShortcut('Ctrl+Shift+N')
        npj.triggered.connect(self._new_project)
        fm.addAction(npj)
        opj = QAction('&Open Project\u2026', self)
        opj.triggered.connect(self._open_project)
        fm.addAction(opj)
        addraw = QAction('Add Loaded Files To Project &Raw', self)
        addraw.triggered.connect(self._add_loaded_files_to_project_raw)
        fm.addAction(addraw)
        runbatch = QAction('&Run Project Batch Processing', self)
        runbatch.triggered.connect(self._run_project_batch)
        fm.addAction(runbatch)
        fm.addSeparator()
        a = QAction('&Open OGPR file(s)\u2026', self)
        a.setShortcut(QKeySequence.StandardKey.Open)
        a.triggered.connect(self.open_files)
        fm.addAction(a)
        fm.addSeparator()
        a2 = QAction('&Export view\u2026', self); a2.setShortcut('Ctrl+E')
        a2.triggered.connect(self._export); fm.addAction(a2)
        fm.addSeparator()
        a3 = QAction('E&xit', self)
        a3.setShortcut(QKeySequence.StandardKey.Quit)
        a3.triggered.connect(self.close); fm.addAction(a3)

        hm = mb.addMenu('&Help')
        ab = QAction('&About', self); ab.triggered.connect(self._about)
        hm.addAction(ab)
        la = QAction('Open log file\u2026', self); la.triggered.connect(self._open_log)
        hm.addAction(la)

    def _build_statusbar(self):
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage('Ready \u2013 open OGPR files to start.')

    # ------------------------------------------------------------------
    # Project management
    # ------------------------------------------------------------------

    def _set_project_path(self, project_path: Optional[Path]):
        self.project_path = project_path
        base_title = 'OGPR Radar Viewer'
        if project_path is None:
            self.setWindowTitle(base_title)
            return
        self.setWindowTitle(f'{base_title}  [{project_path.name}]')

    def _new_project(self):
        root_dir = QFileDialog.getExistingDirectory(
            self, 'Select project root folder', self.last_directory
        )
        if not root_dir:
            return
        name, ok = QInputDialog.getText(self, 'New Project', 'Project name:')
        if not ok:
            return
        project_name = _sanitize_project_name(name)
        if not project_name:
            QMessageBox.warning(self, 'New Project', 'Invalid project name.')
            return

        try:
            p = _create_project_structure(root_dir, project_name)
            self._set_project_path(p)
            self.status.showMessage(f'Project created: {p}', 5000)
            self.last_directory = str(p)
            self.settings.setValue('last_directory', self.last_directory)

            if self._swath_entries:
                self._add_loaded_files_to_project_raw()
        except Exception as e:
            LOG.error(f'Project creation error: {e}\n{traceback.format_exc()}')
            QMessageBox.critical(self, 'New Project error', str(e))

    def _open_project(self):
        folder = QFileDialog.getExistingDirectory(
            self, 'Open project folder', self.last_directory
        )
        if not folder:
            return
        p = Path(folder)
        if not (p / 'metadata.json').exists() or not (p / 'raw').exists():
            QMessageBox.warning(
                self, 'Open Project',
                f'Not a valid OGPR project folder:\n{p}'
            )
            return
        self._set_project_path(p)
        self.status.showMessage(f'Project opened: {p}', 5000)
        self.last_directory = str(p)
        self.settings.setValue('last_directory', self.last_directory)

    def _add_loaded_files_to_project_raw(self):
        if self.project_path is None:
            QMessageBox.information(self, 'Project', 'Create or open a project first.')
            return
        if not self._swath_entries:
            QMessageBox.information(self, 'Project', 'No loaded swaths to copy.')
            return

        raw_dir = self.project_path / 'raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        sources = sorted({
            str(e.data_dict.get('filepath', ''))
            for e in self._swath_entries
            if str(e.data_dict.get('filepath', '')).lower().endswith('.ogpr')
        })
        copied = 0
        skipped = 0
        for src in sources:
            sp = Path(src)
            if not sp.exists():
                skipped += 1
                continue
            dst = raw_dir / sp.name
            if dst.exists() and (dst.stat().st_size == sp.stat().st_size):
                skipped += 1
                continue
            shutil.copy2(sp, dst)
            copied += 1

        self.status.showMessage(
            f'Project raw updated: copied={copied}, skipped={skipped}',
            5000,
        )

    def _run_project_batch(self):
        if self.project_path is None:
            QMessageBox.information(self, 'Batch processing', 'Create or open a project first.')
            return
        if self._batch_worker is not None and self._batch_worker.isRunning():
            QMessageBox.information(self, 'Batch processing', 'Batch is already running.')
            return

        p = self._params()
        self._batch_worker = BatchProcessingThread(
            project_path=str(self.project_path),
            pipeline_params=p,
        )
        self._batch_workers.append(self._batch_worker)
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.finished.connect(self._on_batch_finished)
        self._batch_worker.error.connect(self._on_batch_error)
        self.status.showMessage('Starting project batch processing…')
        self._batch_worker.start()

    def _cancel_batch_worker(self):
        w = self._batch_worker
        if w is None:
            return
        if w.isRunning():
            w.requestInterruption()
        self._batch_worker = None

    def _on_batch_progress(self, msg: str):
        self.status.showMessage(msg)

    def _on_batch_finished(self, msg: str):
        w = self.sender()
        if isinstance(w, BatchProcessingThread):
            if w.isRunning():
                w.requestInterruption()
                if not w.wait(2000):
                    LOG.warning('Batch thread still running during finish callback; cleanup deferred.')
                    return
            if w in self._batch_workers:
                self._batch_workers.remove(w)
            if w is self._batch_worker:
                self._batch_worker = None
            w.deleteLater()
        self.status.showMessage(msg, 8000)

    def _on_batch_error(self, msg: str):
        w = self.sender()
        if isinstance(w, BatchProcessingThread):
            if w.isRunning():
                w.requestInterruption()
                if not w.wait(2000):
                    LOG.warning('Batch thread still running during error callback; cleanup deferred.')
                    return
            if w in self._batch_workers:
                self._batch_workers.remove(w)
            if w is self._batch_worker:
                self._batch_worker = None
            w.deleteLater()
        LOG.error(f'Batch processing error: {msg}')
        QMessageBox.warning(self, 'Batch processing error', msg)
        self.status.showMessage('Batch processing failed.', 5000)

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def open_files(self):
        filenames, _ = QFileDialog.getOpenFileNames(
            self, 'Open OGPR file(s)', self.last_directory,
            'OGPR Files (*.ogpr);;All Files (*)'
        )
        for fp in filenames:
            self._load_async(fp)
        if filenames:
            self.last_directory = str(Path(filenames[-1]).parent)
            self.settings.setValue('last_directory', self.last_directory)

    def _load_async(self, filepath: str):
        size_mb = Path(filepath).stat().st_size / (1024 * 1024)
        self.status.showMessage(f'Loading {Path(filepath).name}\u2026')
        loader = FileLoaderThread(filepath, lazy=(size_mb > 150))
        loader.progress.connect(self.status.showMessage)
        loader.finished.connect(self._on_loaded)
        loader.error.connect(self._on_error)
        self._loaders.append(loader)
        loader.start()

    def _on_loaded(self, data_dict: dict):
        loader = self.sender()
        if isinstance(loader, FileLoaderThread):
            if loader in self._loaders:
                self._loaders.remove(loader)
            loader.deleteLater()

        entry = SwathSidebarEntry(data_dict)
        entry.visibility_changed.connect(self._apply_and_render)
        entry.channel_changed.connect(self._apply_and_render)
        self._swath_entries.append(entry)
        idx = self._sidebar_layout.count() - 1
        self._sidebar_layout.insertWidget(idx, entry)

        meta = data_dict['metadata']
        self.status.showMessage(
            f"{Path(data_dict['filepath']).name}  \u00b7  "
            f"{meta['channels_count']} ch  \u00b7  "
            f"{meta['slices_count']} slices  \u00b7  {meta['dtype_name']}",
            5000,
        )
        LOG.info(f"Swath added: {meta['swath_name']}")

        self._update_view_limits()
        if not self.cb_limit_view.isChecked():
            # Keep a practical default visible window for long profiles.
            self.view_width_m.setValue(min(200.0, self.view_width_m.maximum()))

        self._apply_and_render()

    def _on_error(self, msg: str):
        loader = self.sender()
        if isinstance(loader, FileLoaderThread):
            if loader in self._loaders:
                self._loaders.remove(loader)
            loader.deleteLater()

        LOG.error(f'Load error: {msg}')
        QMessageBox.critical(self, 'Load error', msg)
        self.status.showMessage('Error loading file.')

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def _ymode(self) -> str:
        for btn in self.ymode_grp.buttons():
            if btn.isChecked():
                return btn.property('ymode')
        return 'time'

    def _params(self) -> dict:
        v = self.velocity_spin.value()
        return {
            'dewow_first': self.cb_dewow_first.isChecked(),
            'stack':       self.cb_stack.isChecked(),
            'stack_win_trc': self.stack_win_traces.value(),
            'stack_decim': self.stack_decim.value(),
            'stack_auto_decim': self.cb_stack_auto_decim.isChecked(),
            't0':          self.cb_t0.isChecked(),
            'tz_mode':     self.tz_mode.currentText(),
            'tz_method':   self.tz_method.currentText(),
            'tz_thresh':   self.tz_thresh.value(),
            'top_mute_ns': self.top_mute_ns.value(),
            'static_corr': self.cb_static_corr.isChecked(),
            'static_ref':  self.static_ref_mode.currentText(),
            'static_two_way': self.cb_static_two_way.isChecked(),
            'dewow':       self.cb_dewow.isChecked(),
            'dewow_win':   self.dewow_win.value(),
            'bg':          self.cb_bg.isChecked(),
            'bg_method':   self.bg_method.currentText(),
            'bg_trim_pct': self.bg_trim_pct.value(),
            'bg_rolling':  self.cb_bg_rolling.isChecked(),
            'bg_win_trc':  self.bg_win_traces.value(),
            'bp':          self.cb_bp.isChecked(),
            'bp_mode':     self.bp_mode.currentText(),
            'bp_auto':     self.cb_bp_auto.isChecked(),
            'bp_low':      self.bp_low.value(),
            'bp_high':     self.bp_high.value(),
            'bp_order':    self.bp_order.value(),
            'notch':       self.cb_notch.isChecked(),
            'notch_freq':  self.notch_freq.value(),
            'notch_bw':    self.notch_bw.value(),
            'sw':          self.cb_sw.isChecked(),
            'sw_bp':       self.cb_sw_bp.isChecked(),
            'gain':        self.cb_gain.isChecked(),
            'gain_type':   self.gain_type.currentText(),
            'gain_factor': self.gain_factor.value(),
            'gain_alpha':  self.gain_alpha.value(),
            'gain_tstart': self.gain_tstart.value(),
            'gain_agcwin': self.gain_agc_win.value(),
            'gain_agc_start': self.gain_agc_start.value(),
            'hilbert':     self.cb_hilbert.isChecked(),
            'norm':        self.cb_norm.isChecked(),
            'norm_method': self.norm_method.currentText(),
            'mig':         self.cb_mig.isChecked(),
            'mig_vel':     v,
            'mig_ap':      self.mig_aperture.value(),
            'cmap':        self.cmap_combo.currentText(),
            'y_mode':      self._ymode(),
            'velocity':    v,
            'amp_scale':   self.norm_scale_combo.currentText(),
            'symlog_linthresh_pct': self.symlog_linthresh_pct.value(),
            'view_start_m': self.view_start_m.value(),
            'view_width_m': self.view_width_m.value(),
            'limit_view':  self.cb_limit_view.isChecked(),
            'max_display_traces': 2500,
            'aspect_factor': self.aspect_factor_spin.value(),
            'lock_x_scale': self.cb_lock_x_scale.isChecked(),
            'x_display_width_m': self.x_display_width_m.value(),
        }

    @staticmethod
    def _set_check(cb: QCheckBox, value: bool):
        cb.blockSignals(True)
        cb.setChecked(bool(value))
        cb.blockSignals(False)

    @staticmethod
    def _set_combo(combo: QComboBox, text: str):
        combo.blockSignals(True)
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    @staticmethod
    def _set_dspin(spin: QDoubleSpinBox, value: float):
        spin.blockSignals(True)
        spin.setValue(float(value))
        spin.blockSignals(False)

    @staticmethod
    def _set_ispin(spin: QSpinBox, value: int):
        spin.blockSignals(True)
        spin.setValue(int(value))
        spin.blockSignals(False)

    def _on_contrast_preset_changed(self, name: str):
        if name == 'Manual':
            return
        if name == 'Auto':
            clip, skip = 99.0, 5.0
        elif name == '1 % clip':
            clip, skip = 99.0, 3.0
        else:  # 3 % clip
            clip, skip = 97.0, 3.0
        self._set_dspin(self.clip_pct_spin, clip)
        self.canvas.set_clip_pct(clip)
        self._set_dspin(self.dw_skip_spin, skip)
        self.canvas.set_dw_skip_frac(skip / 100.0)

    def _on_wiggle_toggle(self, *_):
        enabled = self.cb_wiggle_overlay.isChecked()
        self.wiggle_scale.setEnabled(enabled)
        self.wiggle_stride.setEnabled(enabled)
        self.canvas.set_wiggle_overlay(enabled)

    def _on_lock_x_scale_toggle(self, *_):
        enabled = self.cb_lock_x_scale.isChecked()
        self.x_display_width_m.setEnabled(enabled)
        self._apply_and_render()

    def _update_pipeline_summary(self, p: dict):
        steps = []
        if p['stack']:
            if p.get('stack_auto_decim', False) and int(p['stack_decim']) <= 1 and int(p['stack_win_trc']) > 1:
                steps.append(f"Stack(w{p['stack_win_trc']},d=auto->{p['stack_win_trc']})")
            else:
                steps.append(f"Stack(w{p['stack_win_trc']},d{p['stack_decim']})")
        if p['dewow'] and p['t0'] and p['dewow_first']:
            steps.append('Order(DW->T0)')
        if p['t0']:
            steps.append(f"T0({p['tz_method']}/{p['tz_mode']})")
        if p['top_mute_ns'] > 0.0:
            steps.append(f"Mute({p['top_mute_ns']:.1f}ns)")
        if p['static_corr']:
            steps.append(f"Static({p['static_ref']})")
        if p['dewow']:
            steps.append(f"Dewow({p['dewow_win']})")
        if p['bg']:
            mode = 'roll' if p['bg_rolling'] else 'global'
            if p['bg_method'] == 'trimmed':
                steps.append(f"BG(trim {p['bg_trim_pct']:.0f}%,{mode})")
            else:
                steps.append(f"BG({p['bg_method']},{mode})")
        if p['bp']:
            if p.get('bp_auto', False):
                steps.append(f"BP({p['bp_mode']} auto)")
            else:
                steps.append(f"BP({p['bp_mode']} {p['bp_low']:.0f}-{p['bp_high']:.0f} MHz)")
        if p['notch']:
            steps.append(f"Notch({p['notch_freq']:.0f} MHz)")
        if p['sw']:
            steps.append('SpecWhite')
        if p['gain']:
            if p['gain_type'] == 'agc' and p['gain_agc_start'] > 0.0:
                steps.append(f"Gain(agc@gate {p['gain_agc_start']:.1f}ns)")
            else:
                steps.append(f"Gain({p['gain_type']})")
        if p['hilbert']:
            steps.append('Hilbert')
        if p['norm']:
            steps.append(f"Norm({p['norm_method']})")
        if p['mig']:
            steps.append('Migration')
        self.pipeline_summary.setText('  ->  '.join(steps) if steps else 'RAW only')

    def _apply_pipeline_preset(self):
        preset = self.pipeline_preset.currentText()

        if preset == 'Raw view':
            vals = {
                'dewow_first': False,
                'stack': False, 'stack_win': 5, 'stack_decim': 1, 'stack_auto_decim': True,
                't0': False, 'dewow': False, 'bg': False, 'bp': False,
                'bp_auto': False,
                'notch': False, 'sw': False, 'gain': False,
                'hilbert': False, 'norm': False, 'mig': False,
                'top_mute_ns': 0.0,
                'static_corr': False, 'static_ref': 'max', 'static_two_way': False,
                'gain_agc_start': 0.0,
            }
        elif preset == 'Basic clean (Matlab-like)':
            vals = {
                'dewow_first': True,
                'stack': False, 'stack_win': 5, 'stack_decim': 1, 'stack_auto_decim': True,
                't0': True, 'tz_mode': 'line', 'tz_method': 'threshold', 'tz_thresh': 0.08,
                'top_mute_ns': 0.0,
                'static_corr': False, 'static_ref': 'max', 'static_two_way': False,
                'dewow': False,
                'bg': True, 'bg_method': 'median', 'bg_trim_pct': 10.0, 'bg_roll': False, 'bg_win': 50,
                'bp': True, 'bp_mode': 'gpr', 'bp_auto': True, 'bp_low': 200.0, 'bp_high': 600.0, 'bp_order': 4,
                'notch': False, 'sw': False,
                'gain': True, 'gain_type': 'agc', 'gain_factor': 10.0,
                'gain_agcwin': 25.0, 'gain_agc_start': 0.0,
                'hilbert': False, 'norm': False, 'mig': False,
            }
        elif preset == 'Archaeo detail':
            vals = {
                'dewow_first': True,
                'stack': False, 'stack_win': 5, 'stack_decim': 1, 'stack_auto_decim': True,
                't0': True, 'tz_mode': 'line', 'tz_method': 'threshold', 'tz_thresh': 0.08,
                'top_mute_ns': 0.0,
                'static_corr': False, 'static_ref': 'max', 'static_two_way': False,
                'dewow': True, 'dewow_win': 60,
                'bg': True, 'bg_method': 'trimmed', 'bg_trim_pct': 12.0, 'bg_roll': True, 'bg_win': 80,
                'bp': True, 'bp_mode': 'gpr', 'bp_auto': False, 'bp_low': 100.0, 'bp_high': 700.0, 'bp_order': 4,
                'notch': False, 'sw': False,
                'gain': True, 'gain_type': 'sec', 'gain_factor': 20.0,
                'gain_alpha': 0.4, 'gain_tstart': 5.0,
                'gain_agcwin': 25.0, 'gain_agc_start': 0.0,
                'hilbert': False, 'norm': True, 'norm_method': 'robust', 'mig': False,
            }
        elif preset == 'Noisy ground':
            vals = {
                'dewow_first': True,
                'stack': True, 'stack_win': 5, 'stack_decim': 1, 'stack_auto_decim': True,
                't0': True, 'tz_mode': 'line', 'tz_method': 'xcorr', 'tz_thresh': 0.08,
                'top_mute_ns': 0.0,
                'static_corr': False, 'static_ref': 'max', 'static_two_way': False,
                'dewow': True, 'dewow_win': 80,
                'bg': True, 'bg_method': 'trimmed', 'bg_trim_pct': 15.0, 'bg_roll': True, 'bg_win': 60,
                'bp': True, 'bp_mode': 'butter', 'bp_auto': False, 'bp_low': 120.0, 'bp_high': 550.0, 'bp_order': 4,
                'notch': True, 'notch_freq': 50.0, 'notch_bw': 5.0,
                'sw': False,
                'gain': True, 'gain_type': 'agc', 'gain_factor': 8.0,
                'gain_agcwin': 30.0, 'gain_agc_start': 6.0,
                'hilbert': False, 'norm': True, 'norm_method': 'robust', 'mig': False,
            }
        else:  # Stream UP stable
            vals = {
                'dewow_first': True,
                'stack': True, 'stack_win': 7, 'stack_decim': 2, 'stack_auto_decim': True,
                't0': True, 'tz_mode': 'line', 'tz_method': 'threshold', 'tz_thresh': 0.08,
                'top_mute_ns': 3.0,
                'static_corr': False, 'static_ref': 'max', 'static_two_way': False,
                'dewow': True, 'dewow_win': 80,
                'bg': True, 'bg_method': 'trimmed', 'bg_trim_pct': 12.0, 'bg_roll': True, 'bg_win': 250,
                'bp': True, 'bp_mode': 'gpr', 'bp_auto': True, 'bp_low': 200.0, 'bp_high': 600.0, 'bp_order': 4,
                'notch': False, 'sw': False,
                'gain': True, 'gain_type': 'sec', 'gain_factor': 18.0,
                'gain_alpha': 0.35, 'gain_tstart': 6.0,
                'gain_agcwin': 30.0, 'gain_agc_start': 8.0,
                'hilbert': False, 'norm': False, 'mig': False,
            }

        self._set_check(self.cb_dewow_first, vals.get('dewow_first', self.cb_dewow_first.isChecked()))
        self._set_check(self.cb_stack, vals.get('stack', self.cb_stack.isChecked()))
        self._set_ispin(self.stack_win_traces, vals.get('stack_win', self.stack_win_traces.value()))
        self._set_ispin(self.stack_decim, vals.get('stack_decim', self.stack_decim.value()))
        self._set_check(self.cb_stack_auto_decim, vals.get('stack_auto_decim', self.cb_stack_auto_decim.isChecked()))
        self._set_check(self.cb_t0, vals.get('t0', self.cb_t0.isChecked()))
        self._set_combo(self.tz_mode, vals.get('tz_mode', self.tz_mode.currentText()))
        self._set_combo(self.tz_method, vals.get('tz_method', self.tz_method.currentText()))
        self._set_dspin(self.tz_thresh, vals.get('tz_thresh', self.tz_thresh.value()))
        self._set_dspin(self.top_mute_ns, vals.get('top_mute_ns', self.top_mute_ns.value()))
        self._set_check(self.cb_static_corr, vals.get('static_corr', self.cb_static_corr.isChecked()))
        self._set_combo(self.static_ref_mode, vals.get('static_ref', self.static_ref_mode.currentText()))
        self._set_check(self.cb_static_two_way, vals.get('static_two_way', self.cb_static_two_way.isChecked()))
        self._set_check(self.cb_dewow, vals.get('dewow', self.cb_dewow.isChecked()))
        self._set_ispin(self.dewow_win, vals.get('dewow_win', self.dewow_win.value()))
        self._set_check(self.cb_bg, vals.get('bg', self.cb_bg.isChecked()))
        self._set_combo(self.bg_method, vals.get('bg_method', self.bg_method.currentText()))
        self._set_dspin(self.bg_trim_pct, vals.get('bg_trim_pct', self.bg_trim_pct.value()))
        self._set_check(self.cb_bg_rolling, vals.get('bg_roll', self.cb_bg_rolling.isChecked()))
        self._set_ispin(self.bg_win_traces, vals.get('bg_win', self.bg_win_traces.value()))
        self._set_check(self.cb_bp, vals.get('bp', self.cb_bp.isChecked()))
        self._set_combo(self.bp_mode, vals.get('bp_mode', self.bp_mode.currentText()))
        self._set_check(self.cb_bp_auto, vals.get('bp_auto', self.cb_bp_auto.isChecked()))
        self._set_dspin(self.bp_low, vals.get('bp_low', self.bp_low.value()))
        self._set_dspin(self.bp_high, vals.get('bp_high', self.bp_high.value()))
        self._set_ispin(self.bp_order, vals.get('bp_order', self.bp_order.value()))
        self._set_check(self.cb_notch, vals.get('notch', self.cb_notch.isChecked()))
        self._set_dspin(self.notch_freq, vals.get('notch_freq', self.notch_freq.value()))
        self._set_dspin(self.notch_bw, vals.get('notch_bw', self.notch_bw.value()))
        self._set_check(self.cb_sw, vals.get('sw', self.cb_sw.isChecked()))
        self._set_check(self.cb_gain, vals.get('gain', self.cb_gain.isChecked()))
        self._set_combo(self.gain_type, vals.get('gain_type', self.gain_type.currentText()))
        self._set_dspin(self.gain_factor, vals.get('gain_factor', self.gain_factor.value()))
        self._set_dspin(self.gain_alpha, vals.get('gain_alpha', self.gain_alpha.value()))
        self._set_dspin(self.gain_tstart, vals.get('gain_tstart', self.gain_tstart.value()))
        self._set_dspin(self.gain_agc_win, vals.get('gain_agcwin', self.gain_agc_win.value()))
        self._set_dspin(self.gain_agc_start, vals.get('gain_agc_start', self.gain_agc_start.value()))
        self._set_check(self.cb_hilbert, vals.get('hilbert', self.cb_hilbert.isChecked()))
        self._set_check(self.cb_norm, vals.get('norm', self.cb_norm.isChecked()))
        self._set_combo(self.norm_method, vals.get('norm_method', self.norm_method.currentText()))
        self._set_check(self.cb_mig, vals.get('mig', self.cb_mig.isChecked()))

        self._apply_and_render()

    # ------------------------------------------------------------------
    # Processing pipeline
    # ------------------------------------------------------------------

    def _process_swath(self, entry: SwathSidebarEntry, p: dict):
        rv = entry.data_dict['radar_volume']
        meta = entry.data_dict['metadata']
        geo = entry.data_dict.get('geolocations')
        data = rv[:, entry.channel, :]
        return _process_matrix_with_pipeline(
            data=data,
            meta=meta,
            geolocations=geo,
            channel=entry.channel,
            p=p,
        )

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _apply_and_render(self, *_):
        visible = [e for e in self._swath_entries if e.visible]
        if not visible:
            self._cancel_processing_worker()
            self.canvas.clear()
            return

        p = self._params()
        self._update_pipeline_summary(p)
        self._render_request_id += 1
        req_id = self._render_request_id
        LOG.debug(
            f'Render request#{req_id}: {len(visible)} panel(s) '
            f'y_mode={p["y_mode"]} v={p["velocity"]}'
        )

        jobs: List[Dict[str, Any]] = []
        for entry in visible:
            d = entry.data_dict
            jobs.append({
                'radar_volume': d['radar_volume'],
                'metadata': d['metadata'],
                'geolocations': d.get('geolocations'),
                'channel': entry.channel,
            })

        self._cancel_processing_worker()
        self._proc_worker = ProcessingThread(request_id=req_id, jobs=jobs, params=p)
        self._proc_workers.append(self._proc_worker)
        self._proc_worker.progress.connect(self._on_processing_progress)
        self._proc_worker.finished.connect(self._on_processing_finished)
        self._proc_worker.error.connect(self._on_processing_error)
        self.status.showMessage(f'Processing {len(jobs)} panel(s)…')
        self._proc_worker.start()

    def _cancel_processing_worker(self):
        w = self._proc_worker
        if w is None:
            return
        if w.isRunning():
            w.requestInterruption()
        self._proc_worker = None

    def _on_processing_progress(self, req_id: int, msg: str):
        if int(req_id) != int(self._render_request_id):
            return
        self.status.showMessage(msg)

    def _on_processing_finished(self, req_id: int, panels: list):
        w = self.sender()
        if isinstance(w, ProcessingThread):
            if w.isRunning():
                w.requestInterruption()
                if not w.wait(2000):
                    LOG.warning('Processing thread still running during finish callback; cleanup deferred.')
                    return
            if w in self._proc_workers:
                self._proc_workers.remove(w)
            if w is self._proc_worker:
                self._proc_worker = None
            w.deleteLater()

        if int(req_id) != int(self._render_request_id):
            return
        p = self._params()
        self.canvas.render_panels(
            panels,
            cmap=p['cmap'],
            y_mode=p['y_mode'],
            velocity_m_ns=p['velocity'],
            aspect_factor=p['aspect_factor'],
            norm_mode=p.get('amp_scale', 'linear'),
            symlog_linthresh_pct=p.get('symlog_linthresh_pct', 3.0),
        )
        self.status.showMessage(f'Ready · rendered {len(panels)} panel(s)', 2500)

    def _on_processing_error(self, req_id: int, msg: str):
        w = self.sender()
        if isinstance(w, ProcessingThread):
            if w.isRunning():
                w.requestInterruption()
                if not w.wait(2000):
                    LOG.warning('Processing thread still running during error callback; cleanup deferred.')
                    return
            if w in self._proc_workers:
                self._proc_workers.remove(w)
            if w is self._proc_worker:
                self._proc_worker = None
            w.deleteLater()

        if int(req_id) != int(self._render_request_id):
            return
        LOG.error(f'Processing error on request#{req_id}: {msg}')
        QMessageBox.warning(self, 'Processing error', msg)
        self.status.showMessage('Processing error.')

    # ------------------------------------------------------------------
    # Power spectrum dialog
    # ------------------------------------------------------------------

    def _show_power_spectrum(self):
        visible = [e for e in self._swath_entries if e.visible]
        if not visible:
            QMessageBox.information(self, 'Power Spectrum', 'No swaths loaded.')
            return
        entry = visible[0]
        rv    = entry.data_dict['radar_volume']
        meta  = entry.data_dict['metadata']
        data  = rv[:, entry.channel, :]
        proc  = SignalProcessor(data, sampling_time_ns=meta['sampling_time_ns'])
        freqs, pdb = proc.get_power_spectrum()
        PowerSpectrumDialog(freqs, pdb, self).exec()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, 'Export view', self.last_directory,
            'PNG (*.png);;PDF (*.pdf);;All Files (*)'
        )
        if filename:
            try:
                self.canvas.save_figure(filename)
                LOG.info(f'Exported: {filename}')
                self.status.showMessage(f'Exported: {Path(filename).name}', 4000)
            except Exception as e:
                LOG.error(f'Export error: {e}\n{traceback.format_exc()}')
                QMessageBox.critical(self, 'Export error', str(e))

    def _export_debug_bundle(self):
        visible = [e for e in self._swath_entries if e.visible]
        if not visible:
            QMessageBox.information(self, 'Debug export', 'No swaths visible.')
            return

        p = self._params()
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_name = str(Path(self.last_directory) / f'ogpr_debug_{ts}.zip')

        zip_path, _ = QFileDialog.getSaveFileName(
            self, 'Export debug bundle', default_name, 'ZIP (*.zip);;All Files (*)'
        )
        if not zip_path:
            return

        try:
            def _json_default(o):
                if isinstance(o, (np.integer, np.floating)): return o.item()
                if isinstance(o, np.ndarray):               return o.tolist()
                if isinstance(o, Path):                     return str(o)
                if isinstance(o, np.dtype):                 return str(o)
                if isinstance(o, type):                     return o.__name__
                return str(o)

            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                manifest = {'created': ts, 'params': p, 'swaths': []}

                for entry in visible:
                    meta       = entry.data_dict['metadata']
                    swath_name = meta.get('swath_name', 'swath')
                    ch         = entry.channel
                    rv         = entry.data_dict['radar_volume']
                    raw        = np.asarray(rv[:, ch, :], dtype=np.float32)
                    processed, time_axis, dx_m = self._process_swath(entry, p)

                    base     = f'{swath_name}_ch{ch}'
                    npz_path = tmp / f'{base}.npz'
                    np.savez_compressed(
                        npz_path,
                        raw=raw,
                        processed=np.asarray(processed, dtype=np.float32),
                        time_ns=np.asarray(time_axis, dtype=np.float32),
                        trace_spacing_m=float(dx_m),
                        sampling_time_ns=float(meta.get('sampling_time_ns', np.nan)),
                    )

                    json_path = tmp / f'{base}_meta.json'
                    json_path.write_text(
                        json.dumps(
                            {'filepath': entry.data_dict.get('filepath', ''),
                             'swath_name': swath_name, 'channel': ch,
                             'metadata': meta, 'params': p},
                            ensure_ascii=False, indent=2, default=_json_default,
                        ),
                        encoding='utf-8',
                    )

                    try:
                        from matplotlib.figure import Figure
                        fig = Figure(figsize=(10, 4), dpi=150)
                        ax1 = fig.add_subplot(1, 2, 1)
                        ax2 = fig.add_subplot(1, 2, 2)

                        def _lims(a):
                            vmin = float(np.percentile(a, 2))
                            vmax = float(np.percentile(a, 98))
                            return vmin, vmax if vmax > vmin else (vmin, vmin + 1.0)

                        ax1.imshow(raw, aspect='auto', cmap='gray', vmin=_lims(raw)[0], vmax=_lims(raw)[1])
                        ax1.set_title('RAW'); ax1.set_xlabel('Trace'); ax1.set_ylabel('Sample')
                        ax2.imshow(processed, aspect='auto', cmap='gray',
                                   vmin=_lims(processed)[0], vmax=_lims(processed)[1])
                        ax2.set_title('PROCESSED'); ax2.set_xlabel('Trace'); ax2.set_ylabel('Sample')
                        fig.tight_layout()
                        fig.savefig(tmp / f'{base}_quicklook.png')
                    except Exception:
                        pass

                    manifest['swaths'].append({
                        'swath': swath_name, 'channel': ch,
                        'npz': npz_path.name, 'meta': json_path.name,
                        'shape_raw': list(raw.shape),
                        'shape_processed': list(np.asarray(processed).shape),
                    })

                (tmp / 'manifest.json').write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default),
                    encoding='utf-8',
                )

                log_path = Path(sys.argv[0]).parent / 'ogpr_viewer_debug.log'
                if log_path.exists():
                    try: (tmp / 'ogpr_viewer_debug.log').write_bytes(log_path.read_bytes())
                    except Exception: pass

                with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
                    for f in tmp.rglob('*'):
                        if f.is_file(): z.write(f, arcname=f.name)

            self.status.showMessage(f'Debug bundle exported: {Path(zip_path).name}', 6000)

        except Exception as e:
            LOG.error(f'Debug export error: {e}\n{traceback.format_exc()}')
            QMessageBox.critical(self, 'Debug export error', str(e))

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _reset_filters(self):
        for cb in [
            self.cb_dewow_first,
            self.cb_stack, self.cb_stack_auto_decim,
            self.cb_static_corr, self.cb_static_two_way,
            self.cb_t0, self.cb_dewow, self.cb_bg, self.cb_bg_rolling,
            self.cb_bp, self.cb_bp_auto, self.cb_notch, self.cb_sw, self.cb_sw_bp,
            self.cb_gain, self.cb_hilbert, self.cb_norm, self.cb_mig,
            self.cb_limit_view,
        ]:
            cb.blockSignals(True); cb.setChecked(False); cb.blockSignals(False)

        self.clip_pct_spin.blockSignals(True)
        self.clip_pct_spin.setValue(99.0)
        self.clip_pct_spin.blockSignals(False)
        self.canvas.set_clip_pct(99.0)

        self.dw_skip_spin.blockSignals(True)
        self.dw_skip_spin.setValue(5.0)
        self.dw_skip_spin.blockSignals(False)
        self.canvas.set_dw_skip_frac(0.05)

        self._set_combo(self.contrast_preset, 'Auto')
        self._set_combo(self.norm_scale_combo, 'linear')
        self._set_dspin(self.symlog_linthresh_pct, 3.0)
        self._set_combo(self.static_ref_mode, 'max')
        self._set_combo(self.tz_mode, 'line')
        self._set_combo(self.bg_method, 'median')
        self._set_ispin(self.stack_win_traces, 5)
        self._set_ispin(self.stack_decim, 1)
        self._set_check(self.cb_stack_auto_decim, True)
        self._set_check(self.cb_wiggle_overlay, False)
        self.wiggle_scale.setEnabled(False)
        self.wiggle_stride.setEnabled(False)
        self.canvas.set_wiggle_overlay(False)
        self.canvas.set_wiggle_scale(1.0)
        self.canvas.set_wiggle_stride(8)
        self._set_dspin(self.bg_trim_pct, 10.0)
        self._set_dspin(self.top_mute_ns, 0.0)
        self._set_dspin(self.gain_agc_start, 0.0)
        self._set_dspin(self.aspect_factor_spin, 0.50)
        self._set_check(self.cb_dewow_first, True)
        self._set_check(self.cb_bp_auto, True)
        self._set_dspin(self.tz_thresh, 0.08)
        self._set_ispin(self.mig_aperture, 12)
        self._set_check(self.cb_lock_x_scale, True)
        self._set_dspin(self.x_display_width_m, 200.0)
        self.x_display_width_m.setEnabled(True)
        self._set_dspin(self.view_start_m, 0.0)
        self._set_dspin(self.view_width_m, min(200.0, self.view_width_m.maximum()))

        self._ymode_default.setChecked(True)
        self._apply_and_render()

    def _about(self):
        QMessageBox.about(
            self, 'About OGPR Radar Viewer',
            '<h3>OGPR Radar Viewer v2.5</h3>'
            '<p>Processing based on:<br>'
            '<i>Goodman &amp; Piro (2013) \u2013 GPR Remote Sensing in Archaeology, Ch.3</i></p>'
        )

    def _open_log(self):
        import subprocess, os
        log_path = Path(sys.argv[0]).parent / 'ogpr_viewer_debug.log'
        if log_path.exists():
            if sys.platform == 'win32': os.startfile(str(log_path))
            else: subprocess.Popen(['xdg-open', str(log_path)])
        else:
            QMessageBox.information(self, 'Log', f'Not found:\n{log_path}')

    def closeEvent(self, event):
        busy_threads = []
        loaders_running: List[FileLoaderThread] = []
        for w in list(self._loaders):
            try:
                if w.isRunning():
                    if not w.wait(5000):
                        busy_threads.append('loader')
                        loaders_running.append(w)
                        continue
                w.deleteLater()
            except Exception:
                pass
        self._loaders = loaders_running

        self._cancel_processing_worker()
        proc_running: List[ProcessingThread] = []
        for w in list(self._proc_workers):
            try:
                if w.isRunning():
                    w.requestInterruption()
                    if not w.wait(5000):
                        busy_threads.append('processing')
                        proc_running.append(w)
                        continue
                w.deleteLater()
            except Exception:
                pass
        self._proc_workers = proc_running
        self._cancel_batch_worker()
        batch_running: List[BatchProcessingThread] = []
        for w in list(self._batch_workers):
            try:
                if w.isRunning():
                    w.requestInterruption()
                    if not w.wait(5000):
                        busy_threads.append('batch')
                        batch_running.append(w)
                        continue
                w.deleteLater()
            except Exception:
                pass
        self._batch_workers = batch_running
        if busy_threads:
            QMessageBox.warning(
                self,
                'Shutdown in progress',
                'There are still active worker threads. Please wait a few seconds and close again.',
            )
            event.ignore()
            return
        self.settings.setValue('last_directory', self.last_directory)
        LOG.info('Closed')
        event.accept()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _hsep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setFrameShadow(QFrame.Shadow.Sunken)
    return f


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setApplicationName('OGPR Radar Viewer')
    try:
        w = OGPRViewerMainWindow()
        w.show()
        sys.exit(app.exec())
    except Exception as e:
        LOG.critical(f'Fatal: {e}\n{traceback.format_exc()}')
        raise


if __name__ == '__main__':
    main()
