"""
Visualization utilities for GPR data.

MultiPanelCanvas:
  - Syncs matplotlib Figure size to the actual Qt widget pixel size
    (avoids blank canvas on first render)
  - Auto-grid layout for 1..N panels
  - X axis: distance in metres (trace_spacing_m * n_traces)
  - Y axis: switchable  time (ns) | depth_rel (m) | depth_abs (m)
  - Dark theme
  - Symmetric vmin/vmax centred at zero (98th percentile of |data|)
    so that positive and negative GPR reflections are equally visible.
"""

from __future__ import annotations

import logging
import math
import traceback
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use('QtAgg')

logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from PyQt6.QtCore    import QTimer
from PyQt6.QtWidgets import QSizePolicy

LOG = logging.getLogger('ogpr_viewer')

YMode = Literal['time', 'depth_rel', 'depth_abs']

# Percentile of |data| used for symmetric colour clipping.
# 98 keeps the top 2% of spikes from compressing the colour scale.
_CLIP_PCT = 98.0


# ---------------------------------------------------------------------------
# Multi-panel canvas
# ---------------------------------------------------------------------------

class MultiPanelCanvas(FigureCanvasQTAgg):
    """
    Renders 1..N GPR B-scans in an auto-computed grid.

    Panel dict keys (passed to render_panels):
        data            : np.ndarray  (samples, traces)  float32
        time_axis       : np.ndarray  [ns],  length = samples
        title           : str
        trace_spacing_m : float  [m] per trace  (0 = unknown -> show Trace #)
    """

    _MIN_W = 320
    _MIN_H = 220

    def __init__(self, parent=None, dpi: int = 96):
        self._dpi = dpi
        self.fig = Figure(
            figsize=(8, 5),
            dpi=dpi,
            facecolor='#1e1e1e',
        )
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setMinimumSize(self._MIN_W, self._MIN_H)
        self.updateGeometry()

        self._last_panels:   List[Dict] = []
        self._last_cmap:     str   = 'gray'
        self._last_y_mode:   YMode = 'time'
        self._last_vel:      float = 0.10
        self._last_t0:       float = 0.0

        self._axes:   list = []
        self._images: list = []
        self._cbars:  list = []

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._last_panels:
            QTimer.singleShot(30, self._re_render)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _grid(n: int) -> Tuple[int, int]:
        if n <= 0:
            return 1, 1
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        return rows, cols

    def _sync_fig_size(self):
        w_px = max(self.width(),  self._MIN_W)
        h_px = max(self.height(), self._MIN_H)
        self.fig.set_size_inches(
            w_px / self._dpi,
            h_px / self._dpi,
            forward=False,
        )
        LOG.debug(f'Canvas sync: {w_px}x{h_px}px  '
                  f'{w_px/self._dpi:.1f}x{h_px/self._dpi:.1f}in')

    def _re_render(self):
        if self._last_panels:
            self._render(
                self._last_panels,
                self._last_cmap,
                self._last_y_mode,
                self._last_vel,
                self._last_t0,
            )

    @staticmethod
    def _symmetric_clim(data: np.ndarray, pct: float = _CLIP_PCT) -> Tuple[float, float]:
        """
        Compute a symmetric colour range centred at zero.

        Uses the `pct`-th percentile of |data| so that the top (100-pct)%
        of amplitude spikes do not compress the colour scale.  Both positive
        and negative GPR reflections are rendered with equal contrast.

        Returns (vmin, vmax) where vmin = -vmax.
        """
        vabs = float(np.percentile(np.abs(data), pct))
        if vabs < 1e-10:
            vabs = 1.0          # avoid degenerate all-zero panels
        return -vabs, vabs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_panels(
        self,
        panels:        List[Dict],
        cmap:          str   = 'gray',
        y_mode:        YMode = 'time',
        velocity_m_ns: float = 0.10,
        t0_ns:         float = 0.0,
    ):
        """
        Render panels and cache parameters for re-render on resize.

        y_mode options:
          'time'      -> Y axis in ns
          'depth_rel' -> Y axis depth from surface [m]  d = t*v/2
          'depth_abs' -> Y axis depth with t0 removed   d = (t-t0)*v/2
        """
        self._last_panels = panels
        self._last_cmap   = cmap
        self._last_y_mode = y_mode
        self._last_vel    = velocity_m_ns
        self._last_t0     = t0_ns
        self._render(panels, cmap, y_mode, velocity_m_ns, t0_ns)

    def _render(
        self,
        panels:        List[Dict],
        cmap:          str,
        y_mode:        YMode,
        velocity_m_ns: float,
        t0_ns:         float,
    ):
        LOG.debug(
            f'_render: n={len(panels)} cmap={cmap} '
            f'y_mode={y_mode} v={velocity_m_ns} m/ns'
        )

        self._sync_fig_size()

        for cb in self._cbars:
            try: cb.remove()
            except Exception: pass
        self._cbars  = []
        self._axes   = []
        self._images = []
        self.fig.clf()

        if not panels:
            self.draw_idle()
            return

        rows, cols = self._grid(len(panels))

        self.fig.set_constrained_layout(True)
        self.fig.set_constrained_layout_pads(
            w_pad=0.04, h_pad=0.06,
            wspace=0.04, hspace=0.06,
        )

        for idx, panel in enumerate(panels):
            ax = self.fig.add_subplot(rows, cols, idx + 1)
            ax.set_facecolor('#111111')
            self._axes.append(ax)

            data   = np.asarray(panel['data'], dtype=np.float32)
            t_axis = np.asarray(
                panel.get('time_axis', np.arange(data.shape[0], dtype=np.float32)),
                dtype=np.float32,
            )
            title    = panel.get('title', f'Panel {idx+1}')
            dx       = float(panel.get('trace_spacing_m', 0.0))
            n_smp, n_trc = data.shape

            # --- X axis ---
            if dx > 0:
                x_min, x_max = 0.0, dx * (n_trc - 1)
                x_label = f'Distance  ({x_max:.2f} m)'
            else:
                x_min, x_max = 0, n_trc - 1
                x_label = f'Trace  (0 – {n_trc-1})'

            # --- Y axis ---
            if y_mode == 'time':
                y_top    = float(t_axis[0])
                y_bottom = float(t_axis[-1])
                y_label  = f'Time  (0 – {y_bottom:.1f} ns)'
            else:
                d_axis   = np.maximum(t_axis - t0_ns, 0.0) * velocity_m_ns / 2.0
                y_top    = float(d_axis[0])
                y_bottom = float(d_axis[-1])
                y_label  = f'Depth  (0 – {y_bottom:.2f} m)'

            # extent: [left, right, bottom, top]  (origin='upper')
            extent = [x_min, x_max, y_bottom, y_top]

            try:
                # Symmetric colour limits centred at zero:
                # the grey point of the colormap always represents zero amplitude.
                # Positive reflections -> white/bright, negative -> dark, zero -> mid-grey.
                vmin, vmax = self._symmetric_clim(data)

                LOG.debug(
                    f'Panel {idx}: shape={data.shape}  '
                    f'extent={[round(v,3) for v in extent]}  '
                    f'vmin={vmin:.3g}  vmax={vmax:.3g}  '
                    f'(symmetric {_CLIP_PCT}th-pct clipping)'
                )

                im = ax.imshow(
                    data,
                    cmap=cmap,
                    aspect='auto',
                    extent=extent,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation='bilinear',
                    origin='upper',
                )
                self._images.append(im)

                ax.set_xlabel(x_label, fontsize=8,  color='#bbbbbb', labelpad=2)
                ax.set_ylabel(y_label, fontsize=8,  color='#bbbbbb', labelpad=2)
                ax.set_title( title,   fontsize=8,  color='#ffffff',
                              fontweight='bold', pad=3)
                ax.tick_params(colors='#888888', labelsize=7, length=3, width=0.6)
                for sp in ax.spines.values():
                    sp.set_edgecolor('#444444')
                    sp.set_linewidth(0.6)

                cb = self.fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
                cb.ax.tick_params(labelsize=6, colors='#777777')
                cb.set_label('Amplitude', color='#777777', fontsize=6)
                self._cbars.append(cb)

            except Exception as e:
                LOG.error(f'Panel {idx} render error: {e}\n{traceback.format_exc()}')
                ax.text(
                    0.5, 0.5, f'Render error:\n{e}',
                    ha='center', va='center',
                    color='red', fontsize=8,
                    transform=ax.transAxes,
                    wrap=True,
                )

        self.draw_idle()

    # ------------------------------------------------------------------

    def clear(self):
        self._last_panels = []
        for cb in self._cbars:
            try: cb.remove()
            except Exception: pass
        self._cbars  = []
        self._axes   = []
        self._images = []
        self.fig.clf()
        self.draw_idle()

    def save_figure(self, filepath: str, dpi: int = 200):
        self._sync_fig_size()
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

        # Symmetric clipping also for the legacy canvas
        if vmin is None or vmax is None:
            vabs = float(np.percentile(np.abs(data), _CLIP_PCT))
            if vabs < 1e-10: vabs = 1.0
            vmin, vmax = -vabs, vabs

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
        self.draw_idle()

    def save_figure(self, filepath: str, dpi: int = 200):
        self.fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
