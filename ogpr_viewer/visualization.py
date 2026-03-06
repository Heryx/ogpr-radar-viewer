"""
Visualization utilities for GPR data.

MultiPanelCanvas:
  - Auto-grid layout (1x1, 1x2, 2x2, ...)
  - X axis: distance in metres  (trace_spacing_m * n_traces)
  - Y axis: switchable  time (ns) | relative depth (m) | absolute depth (m)
  - Dark theme
"""

from __future__ import annotations

import math
import logging
import traceback
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PyQt6.QtWidgets import QSizePolicy

LOG = logging.getLogger('ogpr_viewer')

YMode = Literal['time', 'depth_rel', 'depth_abs']


# ---------------------------------------------------------------------------
# Multi-panel canvas
# ---------------------------------------------------------------------------

class MultiPanelCanvas(FigureCanvasQTAgg):
    """
    Renders 1..N GPR B-scans in an auto-computed grid.

    Each panel dict expected in render_panels():
      data            : np.ndarray  shape (samples, traces)  float32
      time_axis       : np.ndarray  [ns]  length = samples
      title           : str
      trace_spacing_m : float  horizontal spacing between traces [m]
                        If absent, X axis shows trace indices.

    Shared display parameters (passed to render_panels):
      cmap            : matplotlib colormap name
      y_mode          : 'time' | 'depth_rel' | 'depth_abs'
      velocity_m_ns   : EM velocity [m/ns]  used when y_mode != 'time'
    """

    def __init__(self, parent=None, dpi=96):
        # Do NOT set a fixed figsize here - let Qt determine the size
        self.fig = Figure(facecolor='#1e1e1e', tight_layout=False)
        super().__init__(self.fig)
        self.setParent(parent)

        # Make the canvas expand to fill available space
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.updateGeometry()

        self._axes:   list = []
        self._images: list = []
        self._cbars:  list = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _grid(n: int) -> Tuple[int, int]:
        """Auto-compute (rows, cols) for n panels."""
        if n <= 0:
            return 1, 1
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        return rows, cols

    @staticmethod
    def _depth_axis(time_axis_ns: np.ndarray, velocity_m_ns: float) -> np.ndarray:
        """Convert two-way travel time to depth [m]."""
        return time_axis_ns * velocity_m_ns / 2.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_panels(
        self,
        panels:         List[Dict],
        cmap:           str    = 'gray',
        y_mode:         YMode  = 'time',
        velocity_m_ns:  float  = 0.10,
        t0_ns:          float  = 0.0,
    ):
        """
        Render all panels.

        Args:
            panels:        list of panel dicts (see class docstring)
            cmap:          matplotlib colormap
            y_mode:        'time'       -> Y axis in nanoseconds
                           'depth_rel'  -> Y axis in metres from surface
                                          (depth = (t - t0) * v / 2)
                           'depth_abs'  -> same as depth_rel (t0 always subtracted)
            velocity_m_ns: EM velocity [m/ns] for depth conversion
            t0_ns:         Time-zero offset [ns] subtracted before depth calc
        """
        LOG.debug(
            f'render_panels n={len(panels)} cmap={cmap} '
            f'y_mode={y_mode} v={velocity_m_ns} m/ns'
        )

        # --- tear down previous render ---
        for cb in self._cbars:
            try: cb.remove()
            except Exception: pass
        self._cbars  = []
        self._axes   = []
        self._images = []
        self.fig.clf()

        if not panels:
            self.draw()
            return

        rows, cols = self._grid(len(panels))

        # Use gridspec for tight control over spacing
        gs = self.fig.add_gridspec(
            rows, cols,
            left=0.07, right=0.97,
            top=0.94,  bottom=0.08,
            hspace=0.38, wspace=0.35,
        )

        for idx, panel in enumerate(panels):
            row = idx // cols
            col = idx  % cols
            ax  = self.fig.add_subplot(gs[row, col])
            ax.set_facecolor('#111111')
            self._axes.append(ax)

            data      = np.asarray(panel['data'],  dtype=np.float32)
            t_axis    = np.asarray(panel.get('time_axis', np.arange(data.shape[0])),
                                   dtype=np.float32)
            title     = panel.get('title', f'Panel {idx+1}')
            dx        = float(panel.get('trace_spacing_m', 0.0))  # 0 = unknown
            n_trc     = data.shape[1]

            # ---------- X axis ----------
            if dx > 0:
                x_min = 0.0
                x_max = dx * (n_trc - 1)
                x_label = 'Distance (m)'
            else:
                x_min = 0
                x_max = n_trc - 1
                x_label = 'Trace #'

            # ---------- Y axis ----------
            if y_mode == 'time':
                y_top    = float(t_axis[0])
                y_bottom = float(t_axis[-1])
                y_label  = 'Time (ns)'
            else:
                # depth = (t - t0) * v / 2
                d_axis   = np.maximum(t_axis - t0_ns, 0.0) * velocity_m_ns / 2.0
                y_top    = float(d_axis[0])
                y_bottom = float(d_axis[-1])
                y_label  = 'Depth (m)'

            extent = [x_min, x_max, y_bottom, y_top]

            try:
                # Robust colour limits (2nd / 98th percentile)
                vmin = float(np.percentile(data, 2))
                vmax = float(np.percentile(data, 98))
                if vmin == vmax:          # flat data guard
                    vmax = vmin + 1.0

                im = ax.imshow(
                    data,
                    cmap=cmap,
                    aspect='auto',        # fill the axes rectangle
                    extent=extent,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation='bilinear',
                    origin='upper',
                )
                self._images.append(im)

                # Axes decoration
                ax.set_xlabel(x_label, fontsize=8,  color='#cccccc', labelpad=2)
                ax.set_ylabel(y_label, fontsize=8,  color='#cccccc', labelpad=2)
                ax.set_title(title,    fontsize=8,  color='#ffffff',
                             fontweight='bold', pad=4)
                ax.tick_params(colors='#aaaaaa', labelsize=7, length=3)
                for sp in ax.spines.values():
                    sp.set_edgecolor('#555555')

                # Colorbar
                cb = self.fig.colorbar(
                    im, ax=ax, fraction=0.030, pad=0.02
                )
                cb.ax.tick_params(labelsize=6, colors='#888888')
                cb.set_label('Amplitude', color='#888888', fontsize=6)
                self._cbars.append(cb)

                LOG.debug(
                    f'Panel {idx}: shape={data.shape} '
                    f'x=[{x_min:.2f},{x_max:.2f}] '
                    f'y=[{y_top:.3f},{y_bottom:.3f}] '
                    f'vmin={vmin:.2f} vmax={vmax:.2f}'
                )

            except Exception as e:
                LOG.error(
                    f'Panel {idx} render error: {e}\n{traceback.format_exc()}'
                )
                ax.text(
                    0.5, 0.5, f'Render error:\n{e}',
                    ha='center', va='center',
                    color='red', fontsize=8,
                    transform=ax.transAxes,
                )

        self.draw()

    # ------------------------------------------------------------------

    def clear(self):
        for cb in self._cbars:
            try: cb.remove()
            except Exception: pass
        self._cbars  = []
        self._axes   = []
        self._images = []
        self.fig.clf()
        self.draw()

    def save_figure(self, filepath: str, dpi: int = 200):
        self.fig.savefig(
            filepath, dpi=dpi,
            bbox_inches='tight',
            facecolor=self.fig.get_facecolor(),
        )


# ---------------------------------------------------------------------------
# Single-panel canvas  (legacy compatibility)
# ---------------------------------------------------------------------------

class RadarCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=8, height=6, dpi=96):
        self.fig  = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.image    = None
        self.colorbar = None

    def plot_bscan(
        self,
        data:       np.ndarray,
        time_axis:  Optional[np.ndarray] = None,
        trace_axis: Optional[np.ndarray] = None,
        cmap:       str   = 'gray',
        aspect:     str   = 'auto',
        vmin:       Optional[float] = None,
        vmax:       Optional[float] = None,
        title:      str   = 'GPR B-Scan',
    ):
        if self.colorbar is not None:
            try: self.colorbar.remove()
            except Exception: pass
            self.colorbar = None
        self.axes.clear()

        if time_axis  is None: time_axis  = np.arange(data.shape[0])
        if trace_axis is None: trace_axis = np.arange(data.shape[1])
        if vmin is None or vmax is None:
            vmin = float(np.percentile(data, 2))
            vmax = float(np.percentile(data, 98))

        extent = [
            float(trace_axis[0]),  float(trace_axis[-1]),
            float(time_axis[-1]),  float(time_axis[0]),
        ]
        self.image = self.axes.imshow(
            data, cmap=cmap, aspect=aspect, extent=extent,
            vmin=vmin, vmax=vmax, interpolation='bilinear',
        )
        self.axes.set_xlabel('Trace', fontsize=10)
        self.axes.set_ylabel('Time (ns)', fontsize=10)
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

    def save_figure(self, filepath: str, dpi: int = 200):
        self.fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
