"""
Main GUI Application for OGPR Radar Viewer.

Provides interactive visualization and processing of GPR data.
"""

import sys
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QComboBox, QFileDialog,
    QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox, QMessageBox,
    QSplitter, QStatusBar
)
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QAction, QKeySequence

from .ogpr_parser import OGPRParser
from .signal_processing import SignalProcessor
from .visualization import RadarCanvas


class OGPRViewerMainWindow(QMainWindow):
    """
    Main window for OGPR Radar Viewer application.
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle('OGPR Radar Viewer')
        self.setGeometry(100, 100, 1400, 900)
        
        # Data
        self.parser = None
        self.data_dict = None
        self.processor = None
        self.current_channel = 0
        
        # Settings
        self.settings = QSettings('OGPR', 'RadarViewer')
        
        # UI Setup
        self.init_ui()
        self.create_menu_bar()
        self.create_status_bar()
        
        # Restore last directory
        self.last_directory = self.settings.value('last_directory', str(Path.home()))
    
    def init_ui(self):
        """Initialize user interface."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Visualization
        self.canvas = RadarCanvas(self, width=10, height=8)
        splitter.addWidget(self.canvas)
        
        # Right panel: Controls
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        # Set initial sizes
        splitter.setSizes([1000, 400])
        
        main_layout.addWidget(splitter)
    
    def create_control_panel(self) -> QWidget:
        """Create control panel with processing options."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # File info group
        info_group = self.create_info_group()
        layout.addWidget(info_group)
        
        # Navigation group
        nav_group = self.create_navigation_group()
        layout.addWidget(nav_group)
        
        # Processing group
        proc_group = self.create_processing_group()
        layout.addWidget(proc_group)
        
        # Display options group
        display_group = self.create_display_group()
        layout.addWidget(display_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton('Reset All')
        self.reset_btn.clicked.connect(self.reset_processing)
        self.reset_btn.setEnabled(False)
        button_layout.addWidget(self.reset_btn)
        
        self.export_btn = QPushButton('Export Image')
        self.export_btn.clicked.connect(self.export_image)
        self.export_btn.setEnabled(False)
        button_layout.addWidget(self.export_btn)
        
        layout.addLayout(button_layout)
        
        # Stretch at bottom
        layout.addStretch()
        
        return panel
    
    def create_info_group(self) -> QGroupBox:
        """Create file information group."""
        group = QGroupBox('File Information')
        layout = QVBoxLayout()
        
        self.info_label = QLabel('No file loaded')
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        
        group.setLayout(layout)
        return group
    
    def create_navigation_group(self) -> QGroupBox:
        """Create navigation controls."""
        group = QGroupBox('Navigation')
        layout = QVBoxLayout()
        
        # Channel selector
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(QLabel('Channel:'))
        self.channel_spin = QSpinBox()
        self.channel_spin.setMinimum(0)
        self.channel_spin.valueChanged.connect(self.on_channel_changed)
        self.channel_spin.setEnabled(False)
        channel_layout.addWidget(self.channel_spin)
        layout.addLayout(channel_layout)
        
        # Slice range (for future multi-slice view)
        slice_layout = QHBoxLayout()
        slice_layout.addWidget(QLabel('Slice Range:'))
        self.slice_start_spin = QSpinBox()
        self.slice_start_spin.setMinimum(0)
        self.slice_start_spin.setEnabled(False)
        slice_layout.addWidget(self.slice_start_spin)
        slice_layout.addWidget(QLabel('to'))
        self.slice_end_spin = QSpinBox()
        self.slice_end_spin.setMinimum(0)
        self.slice_end_spin.setEnabled(False)
        slice_layout.addWidget(self.slice_end_spin)
        layout.addLayout(slice_layout)
        
        # Update button
        self.update_view_btn = QPushButton('Update View')
        self.update_view_btn.clicked.connect(self.update_visualization)
        self.update_view_btn.setEnabled(False)
        layout.addWidget(self.update_view_btn)
        
        group.setLayout(layout)
        return group
    
    def create_processing_group(self) -> QGroupBox:
        """Create signal processing controls."""
        group = QGroupBox('Signal Processing')
        layout = QVBoxLayout()
        
        # Background removal
        self.bg_removal_check = QCheckBox('Background Removal')
        self.bg_removal_check.stateChanged.connect(self.apply_processing)
        layout.addWidget(self.bg_removal_check)
        
        bg_method_layout = QHBoxLayout()
        bg_method_layout.addWidget(QLabel('  Method:'))
        self.bg_method_combo = QComboBox()
        self.bg_method_combo.addItems(['mean', 'median'])
        self.bg_method_combo.currentTextChanged.connect(self.apply_processing)
        bg_method_layout.addWidget(self.bg_method_combo)
        layout.addLayout(bg_method_layout)
        
        # Dewow
        self.dewow_check = QCheckBox('Dewow Filter')
        self.dewow_check.stateChanged.connect(self.apply_processing)
        layout.addWidget(self.dewow_check)
        
        dewow_layout = QHBoxLayout()
        dewow_layout.addWidget(QLabel('  Window:'))
        self.dewow_spin = QSpinBox()
        self.dewow_spin.setRange(10, 200)
        self.dewow_spin.setValue(50)
        self.dewow_spin.setSuffix(' samples')
        self.dewow_spin.valueChanged.connect(self.apply_processing)
        dewow_layout.addWidget(self.dewow_spin)
        layout.addLayout(dewow_layout)
        
        # Bandpass filter
        self.bandpass_check = QCheckBox('Bandpass Filter')
        self.bandpass_check.stateChanged.connect(self.apply_processing)
        layout.addWidget(self.bandpass_check)
        
        bp_low_layout = QHBoxLayout()
        bp_low_layout.addWidget(QLabel('  Low Freq:'))
        self.bp_low_spin = QDoubleSpinBox()
        self.bp_low_spin.setRange(1.0, 2000.0)
        self.bp_low_spin.setValue(100.0)
        self.bp_low_spin.setSuffix(' MHz')
        self.bp_low_spin.valueChanged.connect(self.apply_processing)
        bp_low_layout.addWidget(self.bp_low_spin)
        layout.addLayout(bp_low_layout)
        
        bp_high_layout = QHBoxLayout()
        bp_high_layout.addWidget(QLabel('  High Freq:'))
        self.bp_high_spin = QDoubleSpinBox()
        self.bp_high_spin.setRange(1.0, 2000.0)
        self.bp_high_spin.setValue(800.0)
        self.bp_high_spin.setSuffix(' MHz')
        self.bp_high_spin.valueChanged.connect(self.apply_processing)
        bp_high_layout.addWidget(self.bp_high_spin)
        layout.addLayout(bp_high_layout)
        
        # Gain
        self.gain_check = QCheckBox('Apply Gain')
        self.gain_check.stateChanged.connect(self.apply_processing)
        layout.addWidget(self.gain_check)
        
        gain_type_layout = QHBoxLayout()
        gain_type_layout.addWidget(QLabel('  Type:'))
        self.gain_type_combo = QComboBox()
        self.gain_type_combo.addItems(['exp', 'linear', 'agc'])
        self.gain_type_combo.currentTextChanged.connect(self.apply_processing)
        gain_type_layout.addWidget(self.gain_type_combo)
        layout.addLayout(gain_type_layout)
        
        gain_factor_layout = QHBoxLayout()
        gain_factor_layout.addWidget(QLabel('  Factor:'))
        self.gain_factor_spin = QDoubleSpinBox()
        self.gain_factor_spin.setRange(0.1, 10.0)
        self.gain_factor_spin.setValue(2.0)
        self.gain_factor_spin.setSingleStep(0.1)
        self.gain_factor_spin.valueChanged.connect(self.apply_processing)
        gain_factor_layout.addWidget(self.gain_factor_spin)
        layout.addLayout(gain_factor_layout)
        
        # Hilbert transform
        self.hilbert_check = QCheckBox('Hilbert Transform (Envelope)')
        self.hilbert_check.stateChanged.connect(self.apply_processing)
        layout.addWidget(self.hilbert_check)
        
        # Time-zero correction
        self.time_zero_check = QCheckBox('Time-Zero Correction')
        self.time_zero_check.stateChanged.connect(self.apply_processing)
        layout.addWidget(self.time_zero_check)
        
        # Normalize
        self.normalize_check = QCheckBox('Normalize')
        self.normalize_check.stateChanged.connect(self.apply_processing)
        layout.addWidget(self.normalize_check)
        
        norm_method_layout = QHBoxLayout()
        norm_method_layout.addWidget(QLabel('  Method:'))
        self.norm_method_combo = QComboBox()
        self.norm_method_combo.addItems(['minmax', 'zscore', 'robust'])
        self.norm_method_combo.currentTextChanged.connect(self.apply_processing)
        norm_method_layout.addWidget(self.norm_method_combo)
        layout.addLayout(norm_method_layout)
        
        group.setLayout(layout)
        return group
    
    def create_display_group(self) -> QGroupBox:
        """Create display options."""
        group = QGroupBox('Display Options')
        layout = QVBoxLayout()
        
        # Colormap
        cmap_layout = QHBoxLayout()
        cmap_layout.addWidget(QLabel('Colormap:'))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(['gray', 'seismic', 'RdBu_r', 'viridis', 'jet', 'hot'])
        self.cmap_combo.currentTextChanged.connect(self.on_colormap_changed)
        cmap_layout.addWidget(self.cmap_combo)
        layout.addLayout(cmap_layout)
        
        group.setLayout(layout)
        return group
    
    def create_menu_bar(self):
        """Create menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        open_action = QAction('&Open OGPR File...', self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        export_action = QAction('&Export Image...', self)
        export_action.setShortcut('Ctrl+E')
        export_action.triggered.connect(self.export_image)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        about_action = QAction('&About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_status_bar(self):
        """Create status bar."""
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('Ready')
    
    def open_file(self):
        """Open OGPR file dialog."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            'Open OGPR File',
            self.last_directory,
            'OGPR Files (*.ogpr);;All Files (*)'
        )
        
        if filename:
            self.load_file(filename)
    
    def load_file(self, filepath: str):
        """Load and parse OGPR file."""
        try:
            self.statusBar.showMessage(f'Loading {Path(filepath).name}...')
            QApplication.processEvents()
            
            # Parse file
            self.parser = OGPRParser(filepath)
            
            # Check file size for lazy loading
            file_size_mb = Path(filepath).stat().st_size / (1024 * 1024)
            use_lazy = file_size_mb > 100  # Use lazy loading for files > 100 MB
            
            # Load data
            self.data_dict = self.parser.load_data(lazy=use_lazy)
            
            # Update UI
            self.update_file_info()
            self.enable_controls()
            
            # Initialize visualization
            self.current_channel = 0
            self.channel_spin.setValue(0)
            self.update_visualization()
            
            # Save directory
            self.last_directory = str(Path(filepath).parent)
            self.settings.setValue('last_directory', self.last_directory)
            
            self.statusBar.showMessage(f'Loaded {Path(filepath).name}', 3000)
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load file:\n{str(e)}')
            self.statusBar.showMessage('Error loading file')
    
    def update_file_info(self):
        """Update file information display."""
        if self.data_dict:
            meta = self.data_dict['metadata']
            info_text = (
                f"<b>File:</b> {self.parser.filepath.name}<br>"
                f"<b>Swath:</b> {meta['swath_name']}<br>"
                f"<b>Samples:</b> {meta['samples_count']}<br>"
                f"<b>Channels:</b> {meta['channels_count']}<br>"
                f"<b>Slices:</b> {meta['slices_count']}<br>"
                f"<b>Frequency:</b> {meta['frequency_mhz']} MHz<br>"
                f"<b>Sampling:</b> {meta['sampling_time_ns']} ns"
            )
            self.info_label.setText(info_text)
    
    def enable_controls(self):
        """Enable controls after file is loaded."""
        meta = self.data_dict['metadata']
        
        # Navigation
        self.channel_spin.setMaximum(meta['channels_count'] - 1)
        self.channel_spin.setEnabled(True)
        
        self.slice_start_spin.setMaximum(meta['slices_count'] - 1)
        self.slice_start_spin.setValue(0)
        self.slice_start_spin.setEnabled(True)
        
        self.slice_end_spin.setMaximum(meta['slices_count'] - 1)
        self.slice_end_spin.setValue(meta['slices_count'] - 1)
        self.slice_end_spin.setEnabled(True)
        
        self.update_view_btn.setEnabled(True)
        
        # Buttons
        self.reset_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
    
    def on_channel_changed(self, value: int):
        """Handle channel change."""
        self.current_channel = value
        self.update_visualization()
    
    def update_visualization(self):
        """Update radar visualization."""
        if self.data_dict is None:
            return
        
        try:
            # Get slice range
            slice_start = self.slice_start_spin.value()
            slice_end = self.slice_end_spin.value() + 1
            
            # Get data for current channel and slice range
            radar_volume = self.data_dict['radar_volume']
            data_2d = radar_volume[:, self.current_channel, slice_start:slice_end]
            
            # Create processor
            sampling_time = self.data_dict['metadata']['sampling_time_ns']
            self.processor = SignalProcessor(data_2d, sampling_time)
            
            # Apply processing
            self.apply_processing()
            
        except Exception as e:
            QMessageBox.warning(self, 'Warning', f'Visualization error:\n{str(e)}')
    
    def apply_processing(self):
        """Apply selected processing steps."""
        if self.processor is None:
            return
        
        try:
            # Reset to original
            self.processor.reset()
            
            # Background removal
            if self.bg_removal_check.isChecked():
                method = self.bg_method_combo.currentText()
                self.processor.remove_background(method=method)
            
            # Dewow
            if self.dewow_check.isChecked():
                window = self.dewow_spin.value()
                self.processor.dewow(window_size=window)
            
            # Bandpass filter
            if self.bandpass_check.isChecked():
                low = self.bp_low_spin.value()
                high = self.bp_high_spin.value()
                self.processor.apply_bandpass(low, high)
            
            # Gain
            if self.gain_check.isChecked():
                gain_type = self.gain_type_combo.currentText()
                factor = self.gain_factor_spin.value()
                self.processor.apply_gain(gain_type, factor)
            
            # Time-zero correction
            if self.time_zero_check.isChecked():
                self.processor.correct_time_zero()
            
            # Hilbert transform
            if self.hilbert_check.isChecked():
                self.processor.apply_hilbert()
            
            # Normalize
            if self.normalize_check.isChecked():
                method = self.norm_method_combo.currentText()
                self.processor.normalize(method=method)
            
            # Update plot
            self.plot_data()
            
        except Exception as e:
            QMessageBox.warning(self, 'Warning', f'Processing error:\n{str(e)}')
    
    def plot_data(self):
        """Plot processed data."""
        if self.processor is None:
            return
        
        data = self.processor.get_processed_data()
        time_axis = self.processor.get_time_axis()
        
        # Generate title
        meta = self.data_dict['metadata']
        title = f"Channel {self.current_channel} - {meta['swath_name']}"
        
        # Plot
        cmap = self.cmap_combo.currentText()
        self.canvas.plot_bscan(
            data,
            time_axis=time_axis,
            cmap=cmap,
            title=title
        )
    
    def on_colormap_changed(self, cmap: str):
        """Handle colormap change."""
        self.canvas.update_colormap(cmap)
    
    def reset_processing(self):
        """Reset all processing options."""
        self.bg_removal_check.setChecked(False)
        self.dewow_check.setChecked(False)
        self.bandpass_check.setChecked(False)
        self.gain_check.setChecked(False)
        self.hilbert_check.setChecked(False)
        self.time_zero_check.setChecked(False)
        self.normalize_check.setChecked(False)
        
        if self.processor:
            self.processor.reset()
            self.plot_data()
    
    def export_image(self):
        """Export current view as image."""
        if self.canvas.image is None:
            QMessageBox.warning(self, 'Warning', 'No image to export')
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            'Export Image',
            self.last_directory,
            'PNG Image (*.png);;PDF Document (*.pdf);;All Files (*)'
        )
        
        if filename:
            try:
                self.canvas.save_figure(filename)
                self.statusBar.showMessage(f'Exported to {Path(filename).name}', 3000)
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Export failed:\n{str(e)}')
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            'About OGPR Radar Viewer',
            '<h3>OGPR Radar Viewer v1.0</h3>'
            '<p>Professional GPR data visualization and processing tool.</p>'
            '<p><b>Features:</b></p>'
            '<ul>'
            '<li>OGPR format support</li>'
            '<li>Multiple signal processing filters</li>'
            '<li>Interactive visualization</li>'
            '<li>Export capabilities</li>'
            '</ul>'
            '<p><b>Developer:</b> Heryx</p>'
            '<p><b>License:</b> MIT</p>'
        )
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Save settings
        self.settings.setValue('last_directory', self.last_directory)
        event.accept()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName('OGPR Radar Viewer')
    app.setOrganizationName('OGPR')
    
    window = OGPRViewerMainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()