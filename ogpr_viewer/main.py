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

import csv
import sys
import logging
import traceback
from pathlib import Path
from typing  import List

import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog,
    QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QMessageBox, QStatusBar, QScrollArea, QSizePolicy,
    QSplitter, QFrame, QDialog, QButtonGroup, QRadioButton,
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
            self.progress.emit('Parsing header\u2026')
            parser = OGPRParser(self.filepath)
            self.progress.emit('Loading radar volume\u2026')
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
        ax.set_title('Mean Trace Power Spectrum \u2013 use to set Bandpass cutoffs',
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
        self.canvas = MultiPanelCanvas()
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

        layout.addWidget(self._grp_timezero())
        layout.addWidget(self._grp_dewow())
        layout.addWidget(self._grp_background())
        layout.addWidget(self._grp_bandpass())
        layout.addWidget(self._grp_notch())
        layout.addWidget(self._grp_spectral_whitening())
        layout.addWidget(self._grp_gain())
        layout.addWidget(self._grp_hilbert_norm())
        layout.addWidget(self._grp_migration())

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

    def _grp_timezero(self):
        g = QGroupBox('Time-Zero Correction'); v = QVBoxLayout()
        self._cb(v, 'Enable', 'cb_t0')
        self._combo(v, '  Method:', 'tz_method', ['threshold', 'max'])
        self._dspin(v, '  Threshold:', 'tz_thresh', 0.01, 0.99, 0.10, 0.01)
        g.setLayout(v); return g

    def _grp_dewow(self):
        g = QGroupBox('Dewow  (DC drift)'); v = QVBoxLayout()
        self._cb(v, 'Enable', 'cb_dewow')
        self._ispin(v, '  Window:', 'dewow_win', 5, 500, 50, ' smp')
        g.setLayout(v); return g

    def _grp_background(self):
        g = QGroupBox('Background Removal'); v = QVBoxLayout()
        self._cb(v, 'Enable', 'cb_bg')
        self._combo(v, '  Method:', 'bg_method', ['mean', 'median'])
        self._cb(v, '  Rolling window', 'cb_bg_rolling')
        self._ispin(v, '  Win traces:', 'bg_win_traces', 5, 500, 50, ' trc')
        # Per-trace equalization: divides each trace by its RMS to remove
        # vertical stripes caused by amplitude heterogeneity between traces.
        self._cb(v, '  Per-trace equalization', 'cb_bg_pertrace')
        g.setLayout(v); return g

    def _grp_bandpass(self):
        g = QGroupBox('Bandpass Filter'); v = QVBoxLayout()
        self._cb(v, 'Enable', 'cb_bp')
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
        self._combo(v, '  Type:', 'gain_type', ['sec', 'exp', 'linear', 'agc'])
        v.addWidget(QLabel(
            '<small><i>SEC = Spreading &amp; Exponential Compensation<br>'
            'Factor controls max gain (factor\u00d7200) for SEC</i></small>'
        ))
        self._dspin(v, '  Factor (exp/lin/sec):',  'gain_factor', 0.1, 20.0, 2.0, 0.1)
        # alpha default changed from 0.5 to 0.05 ns^-1.
        # 0.5 ns^-1 caused g(60ns) ~ 10^16 -> saturation even after normalisation.
        # Typical GPR soil attenuation: 0.01 – 0.10 ns^-1.
        self._dspin(v, '  \u03b1 (SEC) [1/ns]:',      'gain_alpha',  0.0,  2.0, 0.05, 0.01)
        self._dspin(v, '  t-start (SEC) [ns]:', 'gain_tstart', 0.0, 100.0, 0.0, 0.5, ' ns')
        self._dspin(v, '  AGC window:',          'gain_agc_win',1.0, 500.0, 50.0, 5.0, ' ns')
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
        self._ispin(v, '  Aperture:', 'mig_aperture', 5, 200, 30, ' trc')
        g.setLayout(v); return g

    # ------------------------------------------------------------------
    # Display group  (colormap + Y-axis + velocity)
    # ------------------------------------------------------------------

    def _grp_display(self):
        g = QGroupBox('Display'); v = QVBoxLayout()

        self._combo(v, 'Colormap:', 'cmap_combo',
                    ['gray', 'seismic', 'RdBu_r', 'viridis', 'jet', 'hot', 'bwr'])

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

        btn_dbg = QPushButton('\U0001f41b  Debug: export trace stats\u2026')
        btn_dbg.setToolTip(
            'Export a CSV with per-trace statistics (rms, max, mean, std)\n'
            'for every visible swath after the full processing pipeline.\n'
            'Use this to diagnose amplitude heterogeneity / vertical stripes.'
        )
        btn_dbg.clicked.connect(self._export_debug)
        v.addWidget(btn_dbg)

        g.setLayout(v); return g

    # ------------------------------------------------------------------
    # Menu / Status
    # ------------------------------------------------------------------

    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu('&File')
        a = QAction('&Open OGPR file(s)\u2026', self)
        a.setShortcut(QKeySequence.StandardKey.Open)
        a.triggered.connect(self.open_files)
        fm.addAction(a)
        fm.addSeparator()
        a2 = QAction('&Export view\u2026', self); a2.setShortcut('Ctrl+E')
        a2.triggered.connect(self._export); fm.addAction(a2)
        a3 = QAction('Export &debug stats\u2026', self); a3.setShortcut('Ctrl+D')
        a3.triggered.connect(self._export_debug); fm.addAction(a3)
        fm.addSeparator()
        a4 = QAction('E&xit', self)
        a4.setShortcut(QKeySequence.StandardKey.Quit)
        a4.triggered.connect(self.close); fm.addAction(a4)

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
        self._apply_and_render()

    def _on_error(self, msg: str):
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
            't0':           self.cb_t0.isChecked(),
            'tz_method':    self.tz_method.currentText(),
            'tz_thresh':    self.tz_thresh.value(),
            'dewow':        self.cb_dewow.isChecked(),
            'dewow_win':    self.dewow_win.value(),
            'bg':           self.cb_bg.isChecked(),
            'bg_method':    self.bg_method.currentText(),
            'bg_rolling':   self.cb_bg_rolling.isChecked(),
            'bg_win_trc':   self.bg_win_traces.value(),
            'bg_pertrace':  self.cb_bg_pertrace.isChecked(),
            'bp':           self.cb_bp.isChecked(),
            'bp_low':       self.bp_low.value(),
            'bp_high':      self.bp_high.value(),
            'bp_order':     self.bp_order.value(),
            'notch':        self.cb_notch.isChecked(),
            'notch_freq':   self.notch_freq.value(),
            'notch_bw':     self.notch_bw.value(),
            'sw':           self.cb_sw.isChecked(),
            'sw_bp':        self.cb_sw_bp.isChecked(),
            'gain':         self.cb_gain.isChecked(),
            'gain_type':    self.gain_type.currentText(),
            'gain_factor':  self.gain_factor.value(),
            'gain_alpha':   self.gain_alpha.value(),
            'gain_tstart':  self.gain_tstart.value(),
            'gain_agcwin':  self.gain_agc_win.value(),
            'hilbert':      self.cb_hilbert.isChecked(),
            'norm':         self.cb_norm.isChecked(),
            'norm_method':  self.norm_method.currentText(),
            'mig':          self.cb_mig.isChecked(),
            'mig_vel':      v,
            'mig_ap':       self.mig_aperture.value(),
            'cmap':         self.cmap_combo.currentText(),
            'y_mode':       self._ymode(),
            'velocity':     v,
        }

    # ------------------------------------------------------------------
    # Processing pipeline
    # ------------------------------------------------------------------

    def _process_swath(self, entry: SwathSidebarEntry, p: dict):
        rv   = entry.data_dict['radar_volume']
        meta = entry.data_dict['metadata']
        data = rv[:, entry.channel, :]  # (samples, slices)

        dt_ns = meta['sampling_time_ns']
        dx_m  = meta.get('sampling_step_m', 0.05)

        proc = SignalProcessor(
            data,
            sampling_time_ns=dt_ns,
            trace_spacing_m=dx_m,
        )

        if p['t0']:     proc.correct_time_zero(method=p['tz_method'], threshold=p['tz_thresh'])
        if p['dewow']:  proc.dewow(window_size=p['dewow_win'])
        if p['bg']:     proc.remove_background(
                            method=p['bg_method'],
                            rolling=p['bg_rolling'],
                            window_traces=p['bg_win_trc'],
                        )
        # Per-trace equalization: applied after background removal and before gain
        # so that each trace is amplitude-normalised before the gain ramps up.
        if p.get('bg_pertrace', False):
            proc.per_trace_normalize()

        if p['bp']:     proc.apply_bandpass(
                            low_freq=p['bp_low'],
                            high_freq=p['bp_high'],
                            order=p['bp_order'],
                        )
        if p['notch']:  proc.apply_notch(freq_mhz=p['notch_freq'], bandwidth_mhz=p['notch_bw'])
        if p['sw']:
            proc.apply_spectral_whitening(
                bp_low=p['bp_low']  if p['sw_bp'] else None,
                bp_high=p['bp_high'] if p['sw_bp'] else None,
            )
        if p['gain']:   proc.apply_gain(
                            gain_type=p['gain_type'],
                            factor=p['gain_factor'],
                            alpha=p['gain_alpha'],
                            t_start_ns=p['gain_tstart'],
                            window_ns=p['gain_agcwin'],
                        )
        if p['hilbert']: proc.apply_hilbert()
        if p['norm']:    proc.normalize(method=p['norm_method'])
        if p['mig']:     proc.apply_migration(
                            velocity_m_ns=p['mig_vel'],
                            aperture_traces=p['mig_ap'],
                        )

        return proc.get_processed_data(), proc.get_time_axis(), dx_m

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _apply_and_render(self, *_):
        visible = [e for e in self._swath_entries if e.visible]
        if not visible:
            self.canvas.clear()
            return

        p = self._params()
        LOG.debug(f'Render: {len(visible)} panel(s) y_mode={p["y_mode"]} v={p["velocity"]}')

        panels = []
        for entry in visible:
            try:
                data, time_axis, dx_m = self._process_swath(entry, p)
                meta  = entry.data_dict['metadata']
                title = (
                    f"{meta['swath_name']} \u00b7 "
                    f"Ch {entry.channel} \u00b7 "
                    f"{meta['dtype_name']}"
                )
                panels.append({
                    'data':             data,
                    'time_axis':        time_axis,
                    'title':            title,
                    'trace_spacing_m':  dx_m,
                })
            except Exception as e:
                LOG.error(
                    f"Processing error "
                    f"{entry.data_dict['metadata']['swath_name']}: "
                    f"{e}\n{traceback.format_exc()}"
                )
                QMessageBox.warning(self, 'Processing error', str(e))

        self.canvas.render_panels(
            panels,
            cmap=p['cmap'],
            y_mode=p['y_mode'],
            velocity_m_ns=p['velocity'],
        )

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
    # Misc
    # ------------------------------------------------------------------

    def _reset_filters(self):
        for cb in [
            self.cb_t0, self.cb_dewow, self.cb_bg, self.cb_bg_rolling,
            self.cb_bg_pertrace,
            self.cb_bp, self.cb_notch, self.cb_sw, self.cb_sw_bp,
            self.cb_gain, self.cb_hilbert, self.cb_norm, self.cb_mig,
        ]:
            cb.blockSignals(True); cb.setChecked(False); cb.blockSignals(False)
        self._ymode_default.setChecked(True)
        self._apply_and_render()

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

    def _export_debug(self):
        """
        Export per-trace statistics (rms, max_abs, mean, std) for every
        visible swath after the full processing pipeline to a CSV file.

        Use this to diagnose amplitude heterogeneity between traces:
        if rms varies by >10x across a swath you almost certainly have
        vertical-stripe artefacts that per-trace equalization will fix.
        """
        visible = [e for e in self._swath_entries if e.visible]
        if not visible:
            QMessageBox.information(self, 'Debug export', 'No swaths loaded.')
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, 'Export debug trace stats (CSV)', self.last_directory,
            'CSV Files (*.csv);;All Files (*)'
        )
        if not filename:
            return

        try:
            p = self._params()
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['swath', 'trace_index', 'rms', 'max_abs', 'mean', 'std'])
                for entry in visible:
                    data, _, _ = self._process_swath(entry, p)
                    sp   = SignalProcessor(
                        data,
                        sampling_time_ns=entry.data_dict['metadata']['sampling_time_ns'],
                    )
                    # get_trace_stats operates on whatever data is passed in;
                    # we wrap in a fresh processor so we can call it cleanly.
                    stats = sp.get_trace_stats()
                    name  = entry.data_dict['metadata']['swath_name']
                    n_trc = data.shape[1]
                    for i in range(n_trc):
                        writer.writerow([
                            name, i,
                            f'{stats["rms"][i]:.6g}',
                            f'{stats["max"][i]:.6g}',
                            f'{stats["mean"][i]:.6g}',
                            f'{stats["std"][i]:.6g}',
                        ])
            self.status.showMessage(
                f'Debug stats exported: {Path(filename).name}', 5000
            )
            LOG.info(f'Debug export: {filename}')
        except Exception as e:
            LOG.error(f'Debug export error: {e}\n{traceback.format_exc()}')
            QMessageBox.critical(self, 'Export error', str(e))

    def _about(self):
        QMessageBox.about(
            self, 'About OGPR Radar Viewer',
            '<h3>OGPR Radar Viewer v2.3</h3>'
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
