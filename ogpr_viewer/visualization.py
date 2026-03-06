"""
Visualization utilities for GPR data.
"""

import numpy as np
import matplotlib
matplotlib.use('QtAgg')  # Qt6-compatible backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from typing import Optional, Tuple


class RadarCanvas(FigureCanvasQTAgg):
    """
    Matplotlib canvas for displaying radar data in a PyQt6 window.
    """

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig  = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        self.image    = None
        self.colorbar = None

    # ------------------------------------------------------------------
    # Main plot
    # ------------------------------------------------------------------

    def plot_bscan(
        self,
        data: np.ndarray,
        time_axis: Optional[np.ndarray] = None,
        trace_axis: Optional[np.ndarray] = None,
        cmap: str = 'gray',
        aspect: str = 'auto',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        title: str = 'GPR B-Scan',
    ):
        """
        Plot B-scan radargram.

        Args:
            data:       2-D array (samples × traces)
            time_axis:  Time values for Y axis (ns)
            trace_axis: Trace / distance values for X axis
            cmap:       Matplotlib colormap name
            aspect:     Axes aspect ratio
            vmin/vmax:  Colour scale limits (auto if None)
            title:      Plot title
        """
        # Remove old colorbar BEFORE clearing axes to avoid stale references
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except Exception:
                pass
            self.colorbar = None

        self.axes.clear()

        if time_axis is None:
            time_axis  = np.arange(data.shape[0])
        if trace_axis is None:
            trace_axis = np.arange(data.shape[1])

        if vmin is None or vmax is None:
            vmin = np.percentile(data, 2)
            vmax = np.percentile(data, 98)

        extent = [
            float(trace_axis[0]),  float(trace_axis[-1]),
            float(time_axis[-1]),  float(time_axis[0]),
        ]

        self.image = self.axes.imshow(
            data,
            cmap=cmap,
            aspect=aspect,
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            interpolation='bilinear',
        )

        self.axes.set_xlabel('Trace Number', fontsize=10)
        self.axes.set_ylabel('Time (ns)',     fontsize=10)
        self.axes.set_title(title, fontsize=11, fontweight='bold')

        # Attach colorbar directly to the *image axes* (not the whole figure)
        self.colorbar = self.fig.colorbar(
            self.image, ax=self.axes, label='Amplitude'
        )

        self.fig.tight_layout()
        self.draw()

    # ------------------------------------------------------------------
    # Live updates (no full redraw needed)
    # ------------------------------------------------------------------

    def update_colormap(self, cmap: str):
        """
        Update colormap of the current image without replotting.
        Safe even when a colorbar is present.
        """
        if self.image is not None:
            self.image.set_cmap(cmap)
            # Update colorbar mappable as well
            if self.colorbar is not None:
                self.colorbar.update_normal(self.image)
            self.draw()

    def update_clim(self, vmin: float, vmax: float):
        """Update colour limits of the current image."""
        if self.image is not None:
            self.image.set_clim(vmin, vmax)
            self.draw()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_figure(self, filepath: str, dpi: int = 300):
        """Save figure to file (PNG, PDF, …)."""
        self.fig.savefig(filepath, dpi=dpi, bbox_inches='tight')


# ---------------------------------------------------------------------------
# Standalone (non-Qt) helper
# ---------------------------------------------------------------------------

def plot_bscan_standalone(
    data: np.ndarray,
    time_axis: Optional[np.ndarray] = None,
    cmap: str = 'gray',
    title: str = 'GPR B-Scan',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
):
    """
    Create a standalone (non-Qt) B-scan figure.
    Useful for scripting / batch export.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if time_axis is None:
        time_axis = np.arange(data.shape[0])

    vmin = np.percentile(data, 2)
    vmax = np.percentile(data, 98)

    im = ax.imshow(
        data,
        cmap=cmap,
        aspect='auto',
        extent=[0, data.shape[1], float(time_axis[-1]), float(time_axis[0])],
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel('Trace Number', fontsize=12)
    ax.set_ylabel('Time (ns)',    fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Amplitude')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    return fig, ax
