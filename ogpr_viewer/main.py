"""
Main GUI Application for OGPR Radar Viewer.

Layout:
  LEFT   : Swath sidebar  (checkbox list + per-swath channel selector)
  CENTER : Multi-panel canvas  (auto grid: 1x1, 1x2, 2x2 ...)
  RIGHT  : Global processing controls  (applied to all visible swaths)

Features:
  - Multiple OGPR files open simultaneously
  - Checkbox to show/hide each swath
  - Per-swath independent channel selector
  - Global signal processing (all visible swaths share same filters)
  - Debug log file written to <exe_dir>/ogpr_viewer_debug.log
"""

import sys
import math
import logging
import traceback
from pathlib import Path
from typing  import Dict, List, Optional

import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QFileDialog,
    QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QMessageBox, QStatusBar, QScrollArea, QSizePolicy,
    QSplitter, QFrame,
)
from PyQt6.QtCore import Qt, QSettings, QThread, pyqtSignal
from PyQt6.QtGui  import QAction, QKeySequence, QFont

from .ogpr_parser       import OGPRParser
from .signal_processing import SignalProcessor
from .visualization     import MultiPanelCanvas


# ---------------------------------------------------------------------------
# Logging setup  –  writes to file AND to stderr
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
    log.info(f'Log file: {log_path}')
    return log


LOG = _setup_logger()


# ---------------------------------------------------------------------------
# Background file-loading thread
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
            LOG.debug(f'Loading file: {self.filepath}  lazy={self.lazy}')
            self.progress.emit('Parsing header…')
            parser = OGPRParser(self.filepath)
            self.progress.emit('Loading radar volume…')
            data_dict = parser.load_data(lazy=self.lazy)
            LOG.info(
                f'Loaded {Path(self.filepath).name}  '
                f'shape={data_dict["radar_volume"].shape}  '
                f'dtype={data_dict["metadata"]["dtype_name"]}'
            )
            self.finished.emit(data_dict)
        except Exception as e:
            LOG.error(f'Error loading {self.filepath}: {e}\n{traceback.format_exc()}')
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# One entry in the swath sidebar
# ---------------------------------------------------------------------------

class SwathSidebarEntry(QFrame):
    """
    A row in the swath list:
      [checkbox]  SwathName [dtype]   Ch [spinbox]
    """
    visibility_changed = pyqtSignal()   # re-render all panels
    channel_changed    = pyqtSignal()   # re-render all panels

    def __init__(self, data_dict: dict, parent=None):
        super().__init__(parent)
        self.data_dict = data_dict
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setLineWidth(1)

        meta = data_dict['metadata']
        fp   = Path(data_dict['filepath'])

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        # Checkbox + name
        self.check = QCheckBox()
        self.check.setChecked(True)
        self.check.stateChanged.connect(self._on_visibility)
        layout.addWidget(self.check)

        name_lbl = QLabel(
            f"<b>{meta['swath_name']}</b> "
            f"<small>[{meta['dtype_name']}]</small><br>"
            f"<small>{fp.name}</small>"
        )
        name_lbl.setWordWrap(False)
        name_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        layout.addWidget(name_lbl)

        # Channel selector
        layout.addWidget(QLabel('Ch:'))
        self.ch_spin = QSpinBox()
        self.ch_spin.setMinimum(0)
        self.ch_spin.setMaximum(meta['channels_count'] - 1)
        self.ch_spin.setFixedWidth(48)
        self.ch_spin.valueChanged.connect(self._on_channel)
        layout.addWidget(self.ch_spin)

    # ------------------------------------------------------------------
    @property
    def visible(self) -> bool:
        return self.check.isChecked()

    @property
    def channel(self) -> int:
        return self.ch_spin.value()

    def _on_visibility(self):
        LOG.debug(
            f"Swath {self.data_dict['metadata']['swath_name']} "
            f"visible={self.visible}"
        )
        self.visibility_changed.emit()

    def _on_channel(self, value: int):
        LOG.debug(
            f"Swath {self.data_dict['metadata']['swath_name']} "
            f"channel={value}"
        )
        self.channel_changed.emit()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class OGPRViewerMainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('OGPR Radar Viewer')
        self.setGeometry(80, 80, 1600, 950)

        self.settings       = QSettings('OGPR', 'RadarViewer')
        self.last_directory = self.settings.value('last_directory', str(Path.home()))
        self._loaders: List[FileLoaderThread] = []

        # All loaded swaths (in order)
        self._swath_entries: List[SwathSidebarEntry] = []

        self._build_ui()
        self._build_menu()
        self._build_statusbar()
        LOG.debug('Main window initialised')

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

        # ---- LEFT: swath sidebar ----
        splitter.addWidget(self._build_sidebar())

        # ---- CENTER: multi-panel canvas ----
        self.canvas = MultiPanelCanvas()
        splitter.addWidget(self.canvas)

        # ---- RIGHT: global controls ----
        splitter.addWidget(self._build_controls())

        splitter.setSizes([240, 1060, 300])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        root.addWidget(splitter)

    # ---- Sidebar ------------------------------------------------

    def _build_sidebar(self) -> QWidget:
        container = QWidget()
        container.setMinimumWidth(200)
        container.setMaximumWidth(280)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        title = QLabel('Swaths')
        title.setFont(QFont('', 10, QFont.Weight.Bold))
        layout.addWidget(title)

        # Scrollable list of swath entries
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._sidebar_inner = QWidget()
        self._sidebar_layout = QVBoxLayout(self._sidebar_inner)
        self._sidebar_layout.setSpacing(2)
        self._sidebar_layout.setContentsMargins(2, 2, 2, 2)
        self._sidebar_layout.addStretch()
        scroll.setWidget(self._sidebar_inner)
        layout.addWidget(scroll)

        # Open file button at the bottom of sidebar
        open_btn = QPushButton('\u2795  Open file(s)…')
        open_btn.clicked.connect(self.open_files)
        layout.addWidget(open_btn)

        return container

    # ---- Global controls ----------------------------------------

    def _build_controls(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(260)
        scroll.setMaximumWidth(320)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        panel  = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(6)
        layout.setContentsMargins(4, 4, 4, 4)

        lbl = QLabel('Processing  (global)')
        lbl.setFont(QFont('', 9, QFont.Weight.Bold))
        layout.addWidget(lbl)

        layout.addWidget(self._grp_processing())
        layout.addWidget(self._grp_display())
        layout.addWidget(self._grp_export())
        layout.addStretch()

        scroll.setWidget(panel)
        return scroll

    def _grp_processing(self) -> QGroupBox:
        g = QGroupBox('Signal Processing')
        v = QVBoxLayout()
        v.setSpacing(3)

        def _cb(label, attr):
            cb = QCheckBox(label)
            cb.stateChanged.connect(self._apply_and_render)
            v.addWidget(cb)
            setattr(self, attr, cb)

        def _combo(label, attr, items):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            c = QComboBox()
            c.addItems(items)
            c.currentTextChanged.connect(self._apply_and_render)
            row.addWidget(c)
            v.addLayout(row)
            setattr(self, attr, c)

        def _dspin(label, attr, lo, hi, val, step=0.1, suffix=''):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSingleStep(step)
            if suffix:
                s.setSuffix(suffix)
            s.valueChanged.connect(self._apply_and_render)
            row.addWidget(s)
            v.addLayout(row)
            setattr(self, attr, s)

        def _ispin(label, attr, lo, hi, val, suffix=''):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            if suffix:
                s.setSuffix(suffix)
            s.valueChanged.connect(self._apply_and_render)
            row.addWidget(s)
            v.addLayout(row)
            setattr(self, attr, s)

        # Background removal
        _cb('Background Removal', 'cb_bg')
        _combo('  Method:', 'bg_method', ['mean', 'median'])

        v.addWidget(_hsep())

        # Dewow
        _cb('Dewow', 'cb_dewow')
        _ispin('  Window:', 'dewow_win', 5, 500, 50, ' smp')

        v.addWidget(_hsep())

        # Bandpass
        _cb('Bandpass Filter', 'cb_bp')
        _dspin('  Low:', 'bp_low',   1.0, 4000.0, 100.0, 10.0, ' MHz')
        _dspin('  High:', 'bp_high', 1.0, 4000.0, 800.0, 10.0, ' MHz')

        v.addWidget(_hsep())

        # Gain
        _cb('Gain', 'cb_gain')
        _combo('  Type:', 'gain_type', ['exp', 'linear', 'agc'])
        _dspin('  Factor:', 'gain_factor', 0.1, 20.0, 2.0, 0.1)

        v.addWidget(_hsep())

        # Time-zero
        _cb('Time-Zero Correction', 'cb_t0')

        v.addWidget(_hsep())

        # Hilbert
        _cb('Hilbert (Envelope)', 'cb_hilbert')

        v.addWidget(_hsep())

        # Normalize
        _cb('Normalize', 'cb_norm')
        _combo('  Method:', 'norm_method', ['minmax', 'zscore', 'robust'])

        v.addWidget(_hsep())

        reset_btn = QPushButton('Reset All Filters')
        reset_btn.clicked.connect(self._reset_filters)
        v.addWidget(reset_btn)

        g.setLayout(v)
        return g

    def _grp_display(self) -> QGroupBox:
        g = QGroupBox('Display')
        v = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel('Colormap:'))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(['gray', 'seismic', 'RdBu_r', 'viridis', 'jet', 'hot'])
        self.cmap_combo.currentTextChanged.connect(self._apply_and_render)
        row.addWidget(self.cmap_combo)
        v.addLayout(row)

        g.setLayout(v)
        return g

    def _grp_export(self) -> QGroupBox:
        g = QGroupBox('Export')
        v = QVBoxLayout()
        btn = QPushButton('Export current view…')
        btn.clicked.connect(self._export)
        v.addWidget(btn)
        g.setLayout(v)
        return g

    # ------------------------------------------------------------------
    # Menu / Status
    # ------------------------------------------------------------------

    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu('&File')

        a = QAction('&Open OGPR file(s)…', self)
        a.setShortcut(QKeySequence.StandardKey.Open)
        a.triggered.connect(self.open_files)
        fm.addAction(a)

        fm.addSeparator()

        a2 = QAction('&Export view…', self)
        a2.setShortcut('Ctrl+E')
        a2.triggered.connect(self._export)
        fm.addAction(a2)

        fm.addSeparator()

        a3 = QAction('E&xit', self)
        a3.setShortcut(QKeySequence.StandardKey.Quit)
        a3.triggered.connect(self.close)
        fm.addAction(a3)

        hm = mb.addMenu('&Help')
        ab = QAction('&About', self)
        ab.triggered.connect(self._about)
        hm.addAction(ab)

        log_act = QAction('Open log file…', self)
        log_act.triggered.connect(self._open_log)
        hm.addAction(log_act)

    def _build_statusbar(self):
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage('Ready – open OGPR files to start.')

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
        lazy    = size_mb > 150
        self.status.showMessage(f'Loading {Path(filepath).name}…')
        loader = FileLoaderThread(filepath, lazy=lazy)
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

        # Insert before the stretch at the end
        idx = self._sidebar_layout.count() - 1
        self._sidebar_layout.insertWidget(idx, entry)

        meta = data_dict['metadata']
        self.status.showMessage(
            f"Loaded {Path(data_dict['filepath']).name}  ·  "
            f"{meta['channels_count']} ch  ·  "
            f"{meta['slices_count']} slices  ·  "
            f"{meta['dtype_name']}",
            5000
        )
        LOG.info(f"Swath entry added: {meta['swath_name']}")
        self._apply_and_render()

    def _on_error(self, msg: str):
        LOG.error(f'Load error shown to user: {msg}')
        QMessageBox.critical(self, 'Load error', msg)
        self.status.showMessage('Error loading file.')

    # ------------------------------------------------------------------
    # Processing + rendering
    # ------------------------------------------------------------------

    def _collect_processing_params(self) -> dict:
        """Collect current filter settings into a plain dict."""
        return {
            'bg':          self.cb_bg.isChecked(),
            'bg_method':   self.bg_method.currentText(),
            'dewow':       self.cb_dewow.isChecked(),
            'dewow_win':   self.dewow_win.value(),
            'bp':          self.cb_bp.isChecked(),
            'bp_low':      self.bp_low.value(),
            'bp_high':     self.bp_high.value(),
            'gain':        self.cb_gain.isChecked(),
            'gain_type':   self.gain_type.currentText(),
            'gain_factor': self.gain_factor.value(),
            't0':          self.cb_t0.isChecked(),
            'hilbert':     self.cb_hilbert.isChecked(),
            'norm':        self.cb_norm.isChecked(),
            'norm_method': self.norm_method.currentText(),
            'cmap':        self.cmap_combo.currentText(),
        }

    def _process_swath(self, entry: SwathSidebarEntry, params: dict) -> np.ndarray:
        """Run the processing pipeline for one swath entry."""
        rv   = entry.data_dict['radar_volume']
        meta = entry.data_dict['metadata']
        ch   = entry.channel
        data = rv[:, ch, :]   # (samples, slices)

        dt_ns = meta['sampling_time_ns']
        proc  = SignalProcessor(data, dt_ns)

        if params['bg']:
            proc.remove_background(method=params['bg_method'])
        if params['dewow']:
            proc.dewow(window_size=params['dewow_win'])
        if params['bp']:
            proc.apply_bandpass(params['bp_low'], params['bp_high'])
        if params['gain']:
            proc.apply_gain(params['gain_type'], params['gain_factor'])
        if params['t0']:
            proc.correct_time_zero()
        if params['hilbert']:
            proc.apply_hilbert()
        if params['norm']:
            proc.normalize(method=params['norm_method'])

        return proc.get_processed_data(), proc.get_time_axis()

    def _apply_and_render(self, *_):
        """Process all visible swaths and update the multi-panel canvas."""
        visible = [e for e in self._swath_entries if e.visible]
        if not visible:
            self.canvas.clear()
            return

        params = self._collect_processing_params()
        LOG.debug(f'Rendering {len(visible)} swath(s)  params={params}')

        panels = []
        for entry in visible:
            try:
                data, time_axis = self._process_swath(entry, params)
                meta  = entry.data_dict['metadata']
                title = (
                    f"{meta['swath_name']} · Ch {entry.channel} "
                    f"· {meta['dtype_name']}"
                )
                panels.append({
                    'data':      data,
                    'time_axis': time_axis,
                    'title':     title,
                })
            except Exception as e:
                LOG.error(
                    f"Processing error for "
                    f"{entry.data_dict['metadata']['swath_name']}: "
                    f"{e}\n{traceback.format_exc()}"
                )

        self.canvas.render_panels(
            panels,
            cmap=params['cmap'],
        )

    # ------------------------------------------------------------------
    # Filter reset
    # ------------------------------------------------------------------

    def _reset_filters(self):
        LOG.debug('Reset all filters')
        for cb in [self.cb_bg, self.cb_dewow, self.cb_bp,
                   self.cb_gain, self.cb_t0, self.cb_hilbert, self.cb_norm]:
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        self._apply_and_render()

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
                LOG.info(f'Exported to {filename}')
                self.status.showMessage(f'Exported to {Path(filename).name}', 4000)
            except Exception as e:
                LOG.error(f'Export error: {e}\n{traceback.format_exc()}')
                QMessageBox.critical(self, 'Export error', str(e))

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _about(self):
        QMessageBox.about(
            self, 'About OGPR Radar Viewer',
            '<h3>OGPR Radar Viewer v2.1</h3>'
            '<p>Multi-swath GPR visualization &amp; processing.</p>'
            '<ul>'
            '<li>IDS Stream UP — float32</li>'
            '<li>IDS Stream DP — int16</li>'
            '<li>OGPR v1.x / v2.x</li>'
            '</ul>'
            '<p><b>License:</b> MIT</p>'
        )

    def _open_log(self):
        import subprocess, os
        log_path = Path(sys.argv[0]).parent / 'ogpr_viewer_debug.log'
        if log_path.exists():
            if sys.platform == 'win32':
                os.startfile(str(log_path))
            else:
                subprocess.Popen(['xdg-open', str(log_path)])
        else:
            QMessageBox.information(self, 'Log', f'Log file not found:\n{log_path}')

    def closeEvent(self, event):
        self.settings.setValue('last_directory', self.last_directory)
        LOG.info('Application closed')
        event.accept()


# ---------------------------------------------------------------------------
# Helper widgets
# ---------------------------------------------------------------------------

def _hsep() -> QFrame:
    """Thin horizontal separator line."""
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
    app.setOrganizationName('OGPR')
    try:
        window = OGPRViewerMainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        LOG.critical(f'Fatal error: {e}\n{traceback.format_exc()}')
        raise


if __name__ == '__main__':
    main()
