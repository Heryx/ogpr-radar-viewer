"""
Visualization utilities for GPR data.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from typing import Optional, Tuple


class RadarCanvas(FigureCanvasQTAgg):
    """
    Matplotlib canvas for displaying radar data in Qt.
    """
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.image = None
        self.colorbar = None
    
    def plot_bscan(self, 
                   data: np.ndarray,
                   time_axis: Optional[np.ndarray] = None,
                   trace_axis: Optional[np.ndarray] = None,
                   cmap: str = 'gray',
                   aspect: str = 'auto',
                   vmin: Optional[float] = None,
                   vmax: Optional[float] = None,
                   title: str = 'GPR B-Scan'):
        """
        Plot B-scan radargram.
        
        Args:
            data: 2D array (samples x traces)
            time_axis: Time axis values (ns)
            trace_axis: Trace/distance axis values
            cmap: Colormap name
            aspect: Aspect ratio
            vmin, vmax: Color scale limits
            title: Plot title
        """
        self.axes.clear()
        
        # Set up axes
        if time_axis is None:
            time_axis = np.arange(data.shape[0])
        if trace_axis is None:
            trace_axis = np.arange(data.shape[1])
        
        # Set color limits if not provided
        if vmin is None or vmax is None:
            # Use percentiles for robust color scaling
            vmin = np.percentile(data, 2)
            vmax = np.percentile(data, 98)
        
        # Plot
        extent = [trace_axis[0], trace_axis[-1], time_axis[-1], time_axis[0]]
        self.image = self.axes.imshow(
            data,
            cmap=cmap,
            aspect=aspect,
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            interpolation='bilinear'
        )
        
        # Labels
        self.axes.set_xlabel('Trace Number', fontsize=10)
        self.axes.set_ylabel('Time (ns)', fontsize=10)
        self.axes.set_title(title, fontsize=12, fontweight='bold')
        
        # Colorbar
        if self.colorbar is not None:
            self.colorbar.remove()
        self.colorbar = self.fig.colorbar(self.image, ax=self.axes, label='Amplitude')
        
        # Tight layout
        self.fig.tight_layout()
        self.draw()
    
    def update_colormap(self, cmap: str):
        """Update colormap of current image."""
        if self.image is not None:
            self.image.set_cmap(cmap)
            self.draw()
    
    def update_clim(self, vmin: float, vmax: float):
        """Update color limits."""
        if self.image is not None:
            self.image.set_clim(vmin, vmax)
            self.draw()
    
    def save_figure(self, filepath: str, dpi: int = 300):
        """Save figure to file."""
        self.fig.savefig(filepath, dpi=dpi, bbox_inches='tight')


def plot_bscan_standalone(data: np.ndarray,
                          time_axis: Optional[np.ndarray] = None,
                          cmap: str = 'gray',
                          title: str = 'GPR B-Scan',
                          figsize: Tuple[int, int] = (12, 6),
                          save_path: Optional[str] = None):
    """
    Create standalone B-scan plot (non-Qt).
    
    Args:
        data: 2D radar data
        time_axis: Time values
        cmap: Colormap
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if time_axis is None:
        time_axis = np.arange(data.shape[0])
    
    # Color limits
    vmin = np.percentile(data, 2)
    vmax = np.percentile(data, 98)
    
    # Plot
    im = ax.imshow(
        data,
        cmap=cmap,
        aspect='auto',
        extent=[0, data.shape[1], time_axis[-1], time_axis[0]],
        vmin=vmin,
        vmax=vmax
    )
    
    ax.set_xlabel('Trace Number', fontsize=12)
    ax.set_ylabel('Time (ns)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Amplitude')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    return fig, ax