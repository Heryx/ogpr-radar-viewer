"""
Visualization utilities for GPR data.

Provides:
  RadarCanvas      - single-panel canvas (legacy, kept for compatibility)
  MultiPanelCanvas - auto-grid canvas for multiple simultaneous swaths
"""

import math
import logging
import traceback
from typing import List, Optional, Tuple, Dict

import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

LOG = logging.getLogger('ogpr_viewer')


# ---------------------------------------------------------------------------
# Multi-panel canvas  (main viewer)
# ---------------------------------------------------------------------------

class MultiPanelCanvas(FigureCanvasQTAgg):
    """
    Displays 1..N B-scans side by side in an auto-computed grid.

    Grid rules:
      1 panel  -> 1x1
      2 panels -> 1x2
      3 panels -> 1x3
      4 panels -> 2x2
      5-6      -> 2x3
      7-9      -> 3x3
      etc.
    """

    def __init__(self, parent=None, dpi=100):
        self.fig = Figure(dpi=dpi)
        self.fig.patch.set_facecolor('#1e1e1e')
        super().__init__(self.fig)
        self.setParent(parent)
        self._axes: list = []
        self._images: list = []
        self._cbars: list = []

    # ------------------------------------------------------------------

    @staticmethod
    def _grid(n: int) -> Tuple[int, int]:
        """Return (rows, cols) for n panels."""
        if n <= 0:
            return 1, 1
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        return rows, cols

    # ------------------------------------------------------------------

    def render_panels(
        self,
        panels: List[Dict],
        cmap: str = 'gray',
    ):
        """
        Render a list of panels.

        Each panel dict:
          data       : 2-D ndarray  (samples x traces)
          time_axis  : 1-D ndarray  (ns)
          title      : str
        """
        LOG.debug(f'render_panels: {len(panels)} panel(s)  cmap={cmap}')

        # Clear old colorbars then old axes
        for cb in self._cbars:
            try:
                cb.remove()
            except Exception:
                pass
        self._cbars  = []
        self._axes   = []
        self._images = []
        self.fig.clf()

        if not panels:
            self.draw()
            return

        rows, cols = self._grid(len(panels))
        self.fig.set_size_inches(
            max(5, cols * 5),
            max(4, rows * 4),
            forward=True
        )

        for idx, panel in enumerate(panels):
            ax = self.fig.add_subplot(rows, cols, idx + 1)
            ax.set_facecolor('#111111')
            self._axes.append(ax)

            data      = panel['data']
            time_axis = panel.get('time_axis', np.arange(data.shape[0]))
            title     = panel.get('title', f'Panel {idx+1}')

            try:
                vmin = float(np.percentile(data, 2))
                vmax = float(np.percentile(data, 98))

                extent = [
                    0, data.shape[1],
                    float(time_axis[-1]), float(time_axis[0]),
                ]

                im = ax.imshow(
                    data,
                    cmap=cmap,
                    aspect='auto',
                    extent=extent,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation='bilinear',
                )
                self._images.append(im)

                ax.set_xlabel('Trace #',    fontsize=8,  color='#cccccc')
                ax.set_ylabel('Time (ns)',  fontsize=8,  color='#cccccc')
                ax.set_title(title,         fontsize=9,  color='#ffffff',
                             fontweight='bold')
                ax.tick_params(colors='#aaaaaa', labelsize=7)
                for spine in ax.spines.values():
                    spine.set_edgecolor('#555555')

                cb = self.fig.colorbar(im, ax=ax, label='Amplitude',
                                       fraction=0.035, pad=0.02)
                cb.ax.tick_params(labelsize=7, colors='#aaaaaa')
                cb.set_label('Amplitude', color='#aaaaaa', fontsize=7)
                self._cbars.append(cb)

            except Exception as e:
                LOG.error(f'Panel {idx} render error: {e}\n{traceback.format_exc()}')
                ax.text(0.5, 0.5, f'Render error:\n{e}',
                        ha='center', va='center',
                        color='red', transform=ax.transAxes)

        self.fig.tight_layout(pad=1.5)
        self.draw()

    # ------------------------------------------------------------------

    def clear(self):
        """Clear all panels."""
        for cb in self._cbars:
            try:
                cb.remove()
            except Exception:
                pass
        self._cbars  = []
        self._axes   = []
        self._images = []
        self.fig.clf()
        self.draw()

    def save_figure(self, filepath: str, dpi: int = 300):
        self.fig.savefig(filepath, dpi=dpi, bbox_inches='tight',
                         facecolor=self.fig.get_facecolor())


# ---------------------------------------------------------------------------
# Single-panel canvas  (kept for backward compatibility)
# ---------------------------------------------------------------------------

class RadarCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig  = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.image    = None
        self.colorbar = None

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
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except Exception:
                pass
            self.colorbar = None
        self.axes.clear()

        if time_axis  is None: time_axis  = np.arange(data.shape[0])
        if trace_axis is None: trace_axis = np.arange(data.shape[1])
        if vmin is None or vmax is None:
            vmin = np.percentile(data, 2)
            vmax = np.percentile(data, 98)

        extent = [float(trace_axis[0]), float(trace_axis[-1]),
                  float(time_axis[-1]), float(time_axis[0])]

        self.image = self.axes.imshow(
            data, cmap=cmap, aspect=aspect, extent=extent,
            vmin=vmin, vmax=vmax, interpolation='bilinear')

        self.axes.set_xlabel('Trace Number', fontsize=10)
        self.axes.set_ylabel('Time (ns)',     fontsize=10)
        self.axes.set_title(title, fontsize=11, fontweight='bold')
        self.colorbar = self.fig.colorbar(self.image, ax=self.axes, label='Amplitude')
        self.fig.tight_layout()
        self.draw()

    def update_colormap(self, cmap: str):
        if self.image is not None:
            self.image.set_cmap(cmap)
            if self.colorbar is not None:
                self.colorbar.update_normal(self.image)
            self.draw()

    def update_clim(self, vmin: float, vmax: float):
        if self.image is not None:
            self.image.set_clim(vmin, vmax)
            self.draw()

    def save_figure(self, filepath: str, dpi: int = 300):
        self.fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
