"""
Main GUI Application for OGPR Radar Viewer.

Supports:
  - Multiple OGPR files open simultaneously (one tab per file/swath)
  - IDS Stream UP (float32) and IDS Stream DP (int16) antenna data
  - OGPR format versions 1.x and 2.x
  - Full signal processing pipeline per tab
"""

import sys
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog,
    QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox, QMessageBox,
    QSplitter, QStatusBar, QTabWidget, QTabBar, QToolButton,
    QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, QSettings, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence, QIcon

from .ogpr_parser import OGPRParser
from .signal_processing import SignalProcessor
from .visualization import RadarCanvas


# ---------------------------------------------------------------------------
# Background file-loading thread (keeps GUI responsive for large files)
# ---------------------------------------------------------------------------

class FileLoaderThread(QThread):
    finished  = pyqtSignal(dict)          # emits data_dict
    error     = pyqtSignal(str)           # emits error message
    progress  = pyqtSignal(str)           # emits status string

    def __init__(self, filepath: str, lazy: bool = False):
        super().__init__()
        self.filepath = filepath
        self.lazy     = lazy

    def run(self):
        try:
            self.progress.emit(f'Parsing header…')
            parser = OGPRParser(self.filepath)
            self.progress.emit(f'Loading radar volume…')
            data_dict = parser.load_data(lazy=self.lazy)
            self.finished.emit(data_dict)
        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Single swath tab: canvas + controls
# ---------------------------------------------------------------------------

class SwathTab(QWidget):
    """
    One tab = one OGPR file (one swath / one survey line).
    Contains its own canvas and full processing control panel.
    """

    def __init__(self, data_dict: dict, parent=None):
        super().__init__(parent)
        self.data_dict   = data_dict
        self.processor   = None
        self.current_channel = 0

        self._build_ui()
        self._populate_controls()
        self.update_visualization()

    # ------------------------------------------------------------------
    # UI build
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: canvas
        self.canvas = RadarCanvas(self, width=10, height=8)
        splitter.addWidget(self.canvas)

        # Right: scrollable control panel
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(360)
        ctrl = self._build_controls()
        scroll.setWidget(ctrl)
        splitter.addWidget(scroll)

        splitter.setSizes([1000, 340])
        layout.addWidget(splitter)

    def _build_controls(self) -> QWidget:
        panel  = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(6)

        layout.addWidget(self._group_info())
        layout.addWidget(self._group_navigation())
        layout.addWidget(self._group_processing())
        layout.addWidget(self._group_display())

        btn_row = QHBoxLayout()
        self.reset_btn  = QPushButton('Reset')
        self.export_btn = QPushButton('Export…')
        self.reset_btn.clicked.connect(self.reset_processing)
        self.export_btn.clicked.connect(self.export_image)
        btn_row.addWidget(self.reset_btn)
        btn_row.addWidget(self.export_btn)
        layout.addLayout(btn_row)
        layout.addStretch()
        return panel

    def _group_info(self) -> QGroupBox:
        g = QGroupBox('File Info')
        v = QVBoxLayout()
        self.info_label = QLabel('Loading…')
        self.info_label.setWordWrap(True)
        v.addWidget(self.info_label)
        g.setLayout(v)
        return g

    def _group_navigation(self) -> QGroupBox:
        g = QGroupBox('Navigation')
        v = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel('Channel:'))
        self.channel_spin = QSpinBox()
        self.channel_spin.setMinimum(0)
        self.channel_spin.valueChanged.connect(self._on_channel_changed)
        row.addWidget(self.channel_spin)
        v.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel('Slices:'))
        self.slice_start = QSpinBox()
        self.slice_start.setMinimum(0)
        row2.addWidget(self.slice_start)
        row2.addWidget(QLabel('–'))
        self.slice_end = QSpinBox()
        self.slice_end.setMinimum(0)
        row2.addWidget(self.slice_end)
        v.addLayout(row2)

        self.update_btn = QPushButton('Refresh View')
        self.update_btn.clicked.connect(self.update_visualization)
        v.addWidget(self.update_btn)

        g.setLayout(v)
        return g

    def _group_processing(self) -> QGroupBox:
        g = QGroupBox('Signal Processing')
        v = QVBoxLayout()

        # Background removal
        self.bg_check  = QCheckBox('Background Removal')
        self.bg_check.stateChanged.connect(self.apply_processing)
        v.addWidget(self.bg_check)
        row = QHBoxLayout()
        row.addWidget(QLabel('  Method:'))
        self.bg_method = QComboBox()
        self.bg_method.addItems(['mean', 'median'])
        self.bg_method.currentTextChanged.connect(self.apply_processing)
        row.addWidget(self.bg_method)
        v.addLayout(row)

        # Dewow
        self.dewow_check = QCheckBox('Dewow')
        self.dewow_check.stateChanged.connect(self.apply_processing)
        v.addWidget(self.dewow_check)
        row = QHBoxLayout()
        row.addWidget(QLabel('  Window:'))
        self.dewow_win = QSpinBox()
        self.dewow_win.setRange(5, 500)
        self.dewow_win.setValue(50)
        self.dewow_win.setSuffix(' smp')
        self.dewow_win.valueChanged.connect(self.apply_processing)
        row.addWidget(self.dewow_win)
        v.addLayout(row)

        # Bandpass
        self.bp_check = QCheckBox('Bandpass Filter')
        self.bp_check.stateChanged.connect(self.apply_processing)
        v.addWidget(self.bp_check)
        for label, attr, default in [
            ('  Low:', 'bp_low',  100.0),
            ('  High:', 'bp_high', 800.0),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin = QDoubleSpinBox()
            spin.setRange(1.0, 4000.0)
            spin.setValue(default)
            spin.setSuffix(' MHz')
            spin.valueChanged.connect(self.apply_processing)
            row.addWidget(spin)
            v.addLayout(row)
            setattr(self, attr, spin)

        # Gain
        self.gain_check = QCheckBox('Gain')
        self.gain_check.stateChanged.connect(self.apply_processing)
        v.addWidget(self.gain_check)
        row = QHBoxLayout()
        row.addWidget(QLabel('  Type:'))
        self.gain_type = QComboBox()
        self.gain_type.addItems(['exp', 'linear', 'agc'])
        self.gain_type.currentTextChanged.connect(self.apply_processing)
        row.addWidget(self.gain_type)
        v.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel('  Factor:'))
        self.gain_factor = QDoubleSpinBox()
        self.gain_factor.setRange(0.1, 20.0)
        self.gain_factor.setValue(2.0)
        self.gain_factor.setSingleStep(0.1)
        self.gain_factor.valueChanged.connect(self.apply_processing)
        row.addWidget(self.gain_factor)
        v.addLayout(row)

        # Time-zero
        self.t0_check = QCheckBox('Time-Zero Correction')
        self.t0_check.stateChanged.connect(self.apply_processing)
        v.addWidget(self.t0_check)

        # Hilbert
        self.hilbert_check = QCheckBox('Hilbert (Envelope)')
        self.hilbert_check.stateChanged.connect(self.apply_processing)
        v.addWidget(self.hilbert_check)

        # Normalize
        self.norm_check = QCheckBox('Normalize')
        self.norm_check.stateChanged.connect(self.apply_processing)
        v.addWidget(self.norm_check)
        row = QHBoxLayout()
        row.addWidget(QLabel('  Method:'))
        self.norm_method = QComboBox()
        self.norm_method.addItems(['minmax', 'zscore', 'robust'])
        self.norm_method.currentTextChanged.connect(self.apply_processing)
        row.addWidget(self.norm_method)
        v.addLayout(row)

        g.setLayout(v)
        return g

    def _group_display(self) -> QGroupBox:
        g = QGroupBox('Display')
        v = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(QLabel('Colormap:'))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(['gray', 'seismic', 'RdBu_r', 'viridis', 'jet', 'hot'])
        self.cmap_combo.currentTextChanged.connect(lambda c: self.canvas.update_colormap(c))
        row.addWidget(self.cmap_combo)
        v.addLayout(row)
        g.setLayout(v)
        return g

    # ------------------------------------------------------------------
    # Populate once data is loaded
    # ------------------------------------------------------------------

    def _populate_controls(self):
        meta = self.data_dict['metadata']
        fp   = Path(self.data_dict['filepath'])
        v    = meta['version']

        self.info_label.setText(
            f"<b>{fp.name}</b><br>"
            f"Format v{v['major']}.{v['minor']} · "
            f"<b>{meta['dtype_name']}</b><br>"
            f"Swath: {meta['swath_name']} · Array {meta['array_id']}<br>"
            f"Samples: {meta['samples_count']} · "
            f"Ch: {meta['channels_count']} · "
            f"Slices: {meta['slices_count']}<br>"
            f"Freq: {meta['frequency_mhz']} MHz · "
            f"{meta['sampling_time_ns']} ns<br>"
            f"Step: {meta['sampling_step_m']:.4f} m"
        )

        self.channel_spin.setMaximum(meta['channels_count'] - 1)
        self.slice_start.setMaximum(meta['slices_count'] - 1)
        self.slice_start.setValue(0)
        self.slice_end.setMaximum(meta['slices_count'] - 1)
        self.slice_end.setValue(meta['slices_count'] - 1)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _on_channel_changed(self, value: int):
        self.current_channel = value
        self.update_visualization()

    def update_visualization(self):
        try:
            s0 = self.slice_start.value()
            s1 = self.slice_end.value() + 1
            rv = self.data_dict['radar_volume']
            data_2d = rv[:, self.current_channel, s0:s1]

            dt_ns = self.data_dict['metadata']['sampling_time_ns']
            self.processor = SignalProcessor(data_2d, dt_ns)
            self.apply_processing()
        except Exception as e:
            QMessageBox.warning(self, 'Visualization error', str(e))

    def apply_processing(self):
        if self.processor is None:
            return
        try:
            self.processor.reset()

            if self.bg_check.isChecked():
                self.processor.remove_background(method=self.bg_method.currentText())
            if self.dewow_check.isChecked():
                self.processor.dewow(window_size=self.dewow_win.value())
            if self.bp_check.isChecked():
                self.processor.apply_bandpass(self.bp_low.value(), self.bp_high.value())
            if self.gain_check.isChecked():
                self.processor.apply_gain(self.gain_type.currentText(),
                                          self.gain_factor.value())
            if self.t0_check.isChecked():
                self.processor.correct_time_zero()
            if self.hilbert_check.isChecked():
                self.processor.apply_hilbert()
            if self.norm_check.isChecked():
                self.processor.normalize(method=self.norm_method.currentText())

            self._plot()
        except Exception as e:
            QMessageBox.warning(self, 'Processing error', str(e))

    def _plot(self):
        if self.processor is None:
            return
        data      = self.processor.get_processed_data()
        time_axis = self.processor.get_time_axis()
        meta      = self.data_dict['metadata']
        title = (
            f"{meta['swath_name']} · Array {meta['array_id']} "
            f"· Ch {self.current_channel} · {meta['dtype_name']}"
        )
        self.canvas.plot_bscan(data, time_axis=time_axis,
                               cmap=self.cmap_combo.currentText(), title=title)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def reset_processing(self):
        for cb in [self.bg_check, self.dewow_check, self.bp_check,
                   self.gain_check, self.t0_check,
                   self.hilbert_check, self.norm_check]:
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        if self.processor:
            self.processor.reset()
            self._plot()

    def export_image(self):
        if not hasattr(self.canvas, 'image') or self.canvas.image is None:
            QMessageBox.warning(self, 'Export', 'Nothing to export yet.')
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, 'Export Image', '',
            'PNG Image (*.png);;PDF (*.pdf);;All Files (*)'
        )
        if filename:
            self.canvas.save_figure(filename)


# ---------------------------------------------------------------------------
# Main window with tab manager
# ---------------------------------------------------------------------------

class OGPRViewerMainWindow(QMainWindow):
    """
    Main window. Each opened OGPR file gets its own tab.
    Tabs can be closed individually.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle('OGPR Radar Viewer')
        self.setGeometry(100, 100, 1500, 920)

        self.settings      = QSettings('OGPR', 'RadarViewer')
        self.last_directory = self.settings.value('last_directory', str(Path.home()))
        self._loaders: list[FileLoaderThread] = []   # keep refs alive

        self._build_ui()
        self._build_menu()
        self._build_statusbar()

    # ------------------------------------------------------------------
    # UI skeleton
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._close_tab)
        self.setCentralWidget(self.tabs)

        # Welcome tab
        welcome = QLabel(
            '<h2>OGPR Radar Viewer</h2>'
            '<p>Open one or more <b>.ogpr</b> files via '
            '<b>File → Open</b> or drag &amp; drop.</p>'
            '<p>Each swath/survey line opens in its own tab.</p>'
            '<p>Supports IDS Stream UP (float32) and Stream DP (int16).</p>'
        )
        welcome.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tabs.addTab(welcome, '🏠 Welcome')
        self.tabs.tabBar().setTabButton(0, QTabBar.ButtonPosition.RightSide, None)

    def _build_menu(self):
        mb = self.menuBar()

        file_m = mb.addMenu('&File')

        open_act = QAction('&Open OGPR file(s)…', self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self.open_files)
        file_m.addAction(open_act)

        close_act = QAction('Close current tab', self)
        close_act.setShortcut('Ctrl+W')
        close_act.triggered.connect(lambda: self._close_tab(self.tabs.currentIndex()))
        file_m.addAction(close_act)

        file_m.addSeparator()

        export_act = QAction('&Export image…', self)
        export_act.setShortcut('Ctrl+E')
        export_act.triggered.connect(self._export_current)
        file_m.addAction(export_act)

        file_m.addSeparator()

        quit_act = QAction('E&xit', self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)
        file_m.addAction(quit_act)

        help_m = mb.addMenu('&Help')
        about_act = QAction('&About', self)
        about_act.triggered.connect(self._show_about)
        help_m.addAction(about_act)

    def _build_statusbar(self):
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage('Ready – open an OGPR file to start.')

    # ------------------------------------------------------------------
    # File opening (multi-select)
    # ------------------------------------------------------------------

    def open_files(self):
        """Open multiple OGPR files at once."""
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            'Open OGPR file(s)',
            self.last_directory,
            'OGPR Files (*.ogpr);;All Files (*)'
        )
        for fp in filenames:
            self._load_file_async(fp)
        if filenames:
            self.last_directory = str(Path(filenames[-1]).parent)
            self.settings.setValue('last_directory', self.last_directory)

    def _load_file_async(self, filepath: str):
        """Start background load thread for one file."""
        size_mb = Path(filepath).stat().st_size / (1024 * 1024)
        lazy    = size_mb > 150
        self.status.showMessage(f'Loading {Path(filepath).name}…')

        loader = FileLoaderThread(filepath, lazy=lazy)
        loader.progress.connect(lambda msg: self.status.showMessage(msg))
        loader.finished.connect(self._on_file_loaded)
        loader.error.connect(self._on_load_error)
        self._loaders.append(loader)
        loader.start()

    def _on_file_loaded(self, data_dict: dict):
        fp   = Path(data_dict['filepath'])
        meta = data_dict['metadata']
        tab  = SwathTab(data_dict, parent=self)
        label = f"{meta['swath_name']} [{meta['dtype_name']}]"
        idx  = self.tabs.addTab(tab, label)
        self.tabs.setCurrentIndex(idx)
        self.status.showMessage(
            f"Loaded {fp.name}  ·  "
            f"{meta['channels_count']} ch  ·  "
            f"{meta['slices_count']} slices  ·  "
            f"{meta['dtype_name']}",
            5000
        )

    def _on_load_error(self, msg: str):
        QMessageBox.critical(self, 'Load error', msg)
        self.status.showMessage('Error loading file.')

    # ------------------------------------------------------------------
    # Tab management
    # ------------------------------------------------------------------

    def _close_tab(self, index: int):
        widget = self.tabs.widget(index)
        if isinstance(widget, SwathTab):
            self.tabs.removeTab(index)
            widget.deleteLater()
        elif index != 0:   # don't close welcome tab via X (no X on it anyway)
            self.tabs.removeTab(index)

    # ------------------------------------------------------------------
    # Actions forwarded to current tab
    # ------------------------------------------------------------------

    def _export_current(self):
        w = self.tabs.currentWidget()
        if isinstance(w, SwathTab):
            w.export_image()
        else:
            QMessageBox.information(self, 'Export', 'No swath tab selected.')

    def _show_about(self):
        QMessageBox.about(
            self, 'About OGPR Radar Viewer',
            '<h3>OGPR Radar Viewer v2.0</h3>'
            '<p>GPR data visualization & processing.</p>'
            '<p><b>Antenna support:</b></p>'
            '<ul>'
            '<li>IDS Stream UP — float32</li>'
            '<li>IDS Stream DP — int16 (auto-converted to float32)</li>'
            '</ul>'
            '<p><b>OGPR versions:</b> 1.x, 2.x</p>'
            '<p>Multi-file / multi-tab interface.</p>'
            '<p><b>License:</b> MIT</p>'
        )

    def closeEvent(self, event):
        self.settings.setValue('last_directory', self.last_directory)
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setApplicationName('OGPR Radar Viewer')
    app.setOrganizationName('OGPR')
    window = OGPRViewerMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
