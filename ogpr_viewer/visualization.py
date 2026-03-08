"""
Visualization utilities for GPR data.

MultiPanelCanvas:
  - Syncs matplotlib Figure size to the actual Qt widget pixel size
    (avoids blank canvas on first render)
  - Auto-grid layout for 1..N panels
  - X axis: distance in metres (trace_spacing_m * n_traces)
  - Y axis: switchable  time (ns) | depth_rel (m) | depth_abs (m)
  - Dark theme
  - Configurable symmetric vmin/vmax centred at zero
    * clip_pct: percentile of |data| for color clipping (default 98.0)
    * dw_skip_frac: fraction of samples to skip (default 0.25)
    Percentile is computed on the sub-surface portion only (after skip)
    to prevent strong near-surface signals from compressing the colour scale.
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
from matplotlib.colors import SymLogNorm

from PyQt6.QtCore    import QTimer
from PyQt6.QtWidgets import QSizePolicy

LOG = logging.getLogger('ogpr_viewer')

YMode = Literal['time', 'depth_rel', 'depth_abs']


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
        x_start_m       : float  optional absolute x-origin for the panel
        x_display_width_m : float optional visible x-window width (fixed scale)
        total_length_m  : float  optional full profile length for labels
    """

    _MIN_W = 320
    _MIN_H = 220

    def __init__(
        self, 
        parent=None, 
        dpi: int = 96,
        clip_pct: float = 98.0,
        dw_skip_frac: float = 0.25,
        wiggle_overlay: bool = False,
        wiggle_scale: float = 1.0,
        wiggle_stride: int = 8,
    ):
        """
        Args:
            dpi: Display DPI
            clip_pct: Percentile for symmetric color clipping (90-99.9).
                      Lower values reveal more detail in saturated data.
            dw_skip_frac: Fraction of samples to skip when computing percentile (0.0-0.5).
                          Higher values ignore more of the direct wave zone.
        """
        self._dpi = dpi
        self._clip_pct = float(clip_pct)
        self._dw_skip_frac = float(dw_skip_frac)
        self._wiggle_overlay = bool(wiggle_overlay)
        self._wiggle_scale = float(wiggle_scale)
        self._wiggle_stride = int(max(1, wiggle_stride))
        
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
        self._last_aspect:   float = 0.5
        self._last_norm_mode: str = 'linear'
        self._last_symlog_linthresh_pct: float = 3.0

        self._axes:   list = []
        self._images: list = []
        self._cbars:  list = []
        self._zoom_limits: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
        self._panel_bounds: Dict[int, Tuple[float, float, float, float]] = {}
        self._panel_defaults: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
        self._pan_state: Optional[Dict] = None

        self.mpl_connect('scroll_event', self._on_scroll_zoom)
        self.mpl_connect('button_press_event', self._on_button_press)
        self.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.mpl_connect('button_release_event', self._on_button_release)

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._last_panels:
            QTimer.singleShot(30, self._re_render)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_clip_pct(self, pct: float):
        """Update color clipping percentile and re-render."""
        self._clip_pct = float(np.clip(pct, 50.0, 100.0))
        LOG.debug(f'Clip percentile updated: {self._clip_pct:.1f}%')
        self._re_render()

    def set_dw_skip_frac(self, frac: float):
        """Update direct-wave skip fraction and re-render."""
        self._dw_skip_frac = float(np.clip(frac, 0.0, 0.9))
        LOG.debug(f'DW skip fraction updated: {self._dw_skip_frac:.1%}')
        self._re_render()

    def get_clip_pct(self) -> float:
        return self._clip_pct

    def get_dw_skip_frac(self) -> float:
        return self._dw_skip_frac

    def set_wiggle_overlay(self, enabled: bool):
        self._wiggle_overlay = bool(enabled)
        LOG.debug(f'Wiggle overlay: {self._wiggle_overlay}')
        self._re_render()

    def set_wiggle_scale(self, scale: float):
        self._wiggle_scale = float(np.clip(scale, 0.1, 10.0))
        LOG.debug(f'Wiggle scale: {self._wiggle_scale:.2f}')
        self._re_render()

    def set_wiggle_stride(self, stride: int):
        self._wiggle_stride = int(max(1, stride))
        LOG.debug(f'Wiggle stride: {self._wiggle_stride}')
        self._re_render()

    def get_wiggle_overlay(self) -> bool:
        return self._wiggle_overlay

    def get_wiggle_scale(self) -> float:
        return self._wiggle_scale

    def get_wiggle_stride(self) -> int:
        return self._wiggle_stride

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

    @staticmethod
    def _clamp_axis_limits(
        lim: Tuple[float, float],
        lo: float,
        hi: float,
    ) -> Tuple[float, float]:
        a, b = float(lim[0]), float(lim[1])
        if not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(lo) or not np.isfinite(hi):
            return lim
        if hi <= lo:
            return lim

        rev = a > b
        l, r = (b, a) if rev else (a, b)
        full = hi - lo
        min_span = max(1e-9, full * 1e-4)
        span = max(min_span, r - l)

        if span >= full:
            l, r = lo, hi
        else:
            if l < lo:
                r += (lo - l)
                l = lo
            if r > hi:
                l -= (r - hi)
                r = hi
            l = max(lo, l)
            r = min(hi, r)
            if (r - l) < min_span:
                mid = 0.5 * (l + r)
                l = max(lo, mid - 0.5 * min_span)
                r = min(hi, mid + 0.5 * min_span)

        return (r, l) if rev else (l, r)

    def _apply_zoom_state(self, idx: int, ax):
        if idx not in self._zoom_limits:
            return
        bounds = self._panel_bounds.get(idx)
        if bounds is None:
            return
        xlim, ylim = self._zoom_limits[idx]
        xlim = self._clamp_axis_limits(xlim, bounds[0], bounds[1])
        ylim = self._clamp_axis_limits(ylim, bounds[2], bounds[3])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        self._zoom_limits[idx] = (tuple(ax.get_xlim()), tuple(ax.get_ylim()))

    @staticmethod
    def _button_name(button) -> str:
        return str(button).lower()

    def _is_left_button(self, button) -> bool:
        name = self._button_name(button)
        return name in ('1', 'mousebutton.left', 'left')

    def _is_middle_button(self, button) -> bool:
        name = self._button_name(button)
        return name in ('2', 'mousebutton.middle', 'middle')

    @staticmethod
    def _event_data_coords(event, ax) -> Tuple[Optional[float], Optional[float]]:
        x = getattr(event, 'xdata', None)
        y = getattr(event, 'ydata', None)
        if x is not None and y is not None:
            return float(x), float(y)
        if getattr(event, 'x', None) is None or getattr(event, 'y', None) is None:
            return None, None
        try:
            inv = ax.transData.inverted()
            xd, yd = inv.transform((event.x, event.y))
            return float(xd), float(yd)
        except Exception:
            return None, None

    def _on_scroll_zoom(self, event):
        ax = getattr(event, 'inaxes', None)
        if ax is None or ax not in self._axes:
            return
        idx = int(self._axes.index(ax))
        bounds = self._panel_bounds.get(idx)
        if bounds is None:
            return

        step = getattr(event, 'step', 0)
        if step == 0:
            btn = str(getattr(event, 'button', ''))
            if btn == 'up':
                step = 1
            elif btn == 'down':
                step = -1
        if step == 0:
            return

        base = 0.85
        scale = base ** float(step) if step > 0 else (1.0 / base) ** float(-step)

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        xc = float(event.xdata) if event.xdata is not None else 0.5 * (x0 + x1)
        yc = float(event.ydata) if event.ydata is not None else 0.5 * (y0 + y1)

        nx0 = xc - (xc - x0) * scale
        nx1 = xc + (x1 - xc) * scale
        ny0 = yc - (yc - y0) * scale
        ny1 = yc + (y1 - yc) * scale

        xlim = self._clamp_axis_limits((nx0, nx1), bounds[0], bounds[1])
        ylim = self._clamp_axis_limits((ny0, ny1), bounds[2], bounds[3])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        self._zoom_limits[idx] = (tuple(ax.get_xlim()), tuple(ax.get_ylim()))
        self.draw_idle()

    def _on_button_press(self, event):
        ax = getattr(event, 'inaxes', None)
        if ax is None or ax not in self._axes:
            return

        button = getattr(event, 'button', None)
        idx = int(self._axes.index(ax))

        if self._is_left_button(button):
            xd, yd = self._event_data_coords(event, ax)
            if xd is None or yd is None:
                return
            self._pan_state = {
                'idx': idx,
                'x_anchor': xd,
                'y_anchor': yd,
                'xlim0': tuple(ax.get_xlim()),
                'ylim0': tuple(ax.get_ylim()),
            }
            return

        # Middle click resets zoom for current panel.
        if not self._is_middle_button(button):
            return

        defaults = self._panel_defaults.get(idx)
        if defaults is None:
            return
        ax.set_xlim(*defaults[0])
        ax.set_ylim(*defaults[1])
        if idx in self._zoom_limits:
            del self._zoom_limits[idx]
        self.draw_idle()

    def _on_mouse_move(self, event):
        st = self._pan_state
        if not st:
            return
        idx = int(st.get('idx', -1))
        if idx < 0 or idx >= len(self._axes):
            self._pan_state = None
            return

        ax = self._axes[idx]
        bounds = self._panel_bounds.get(idx)
        if bounds is None:
            self._pan_state = None
            return

        xd, yd = self._event_data_coords(event, ax)
        if xd is None or yd is None:
            return

        x0, y0 = float(st['x_anchor']), float(st['y_anchor'])
        xlim0 = st['xlim0']
        ylim0 = st['ylim0']
        dx = xd - x0
        dy = yd - y0

        new_xlim = (xlim0[0] - dx, xlim0[1] - dx)
        new_ylim = (ylim0[0] - dy, ylim0[1] - dy)

        xlim = self._clamp_axis_limits(new_xlim, bounds[0], bounds[1])
        ylim = self._clamp_axis_limits(new_ylim, bounds[2], bounds[3])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        self._zoom_limits[idx] = (tuple(ax.get_xlim()), tuple(ax.get_ylim()))
        self.draw_idle()

    def _on_button_release(self, event):
        if self._pan_state is None:
            return
        button = getattr(event, 'button', None)
        if self._is_left_button(button):
            self._pan_state = None

    def _re_render(self):
        if self._last_panels:
            self._render(
                self._last_panels,
                self._last_cmap,
                self._last_y_mode,
                self._last_vel,
                self._last_t0,
                self._last_aspect,
                self._last_norm_mode,
                self._last_symlog_linthresh_pct,
            )

    def _symmetric_clim(
        self,
        data: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute a symmetric colour range centred at zero.

        Uses instance parameters self._clip_pct and self._dw_skip_frac.

        The percentile is computed on the *sub-surface* portion of the data:
        the first dw_skip_frac rows (direct-wave + ringing zone) are
        excluded from the calculation so that strong near-surface signals
        cannot compress the colour scale and wash out subsurface reflections.

        The direct wave and ringing are still drawn; they may appear saturated
        (full white/black) which correctly represents their much higher amplitude.

        Returns (vmin, vmax) where vmin = -vmax.
        """
        abs_all = np.abs(np.asarray(data, dtype=np.float32))
        n_rows  = abs_all.shape[0]
        skip    = max(0, int(n_rows * self._dw_skip_frac))
        d_sub   = abs_all[skip:, :] if n_rows > skip + 4 else abs_all

        vabs_sub = float(np.nanpercentile(d_sub, self._clip_pct))
        vabs_all = float(np.nanpercentile(abs_all, self._clip_pct))

        if not np.isfinite(vabs_sub):
            vabs_sub = 0.0
        if not np.isfinite(vabs_all):
            vabs_all = 0.0

        # Guard rail for Time-Zero / strong direct-wave cases:
        # if top rows are excluded, the sub-surface percentile can become too
        # small and cause aggressive clipping. Keep at least a fraction of the
        # global percentile to prevent visual over-saturation.
        vabs = max(vabs_sub, 0.40 * vabs_all)

        if vabs < 1e-10:
            vabs = vabs_all
        if vabs < 1e-10:
            vabs = 1.0
        return -vabs, vabs

    def _draw_wiggles(
        self,
        ax,
        data: np.ndarray,
        dx: float,
        y_top: float,
        y_bottom: float,
        x_offset_m: float = 0.0,
    ):
        """
        Draw a sparse wiggle overlay (seismic style) on top of imshow.
        Inspired by classic wigglesc plots, but intentionally lightweight.
        """
        n_smp, n_trc = data.shape
        if n_smp < 2 or n_trc < 2:
            return

        stride = max(1, int(self._wiggle_stride))
        idx = np.arange(0, n_trc, stride, dtype=int)
        if idx.size == 0:
            return

        if dx > 0:
            x_center = float(x_offset_m) + idx.astype(np.float32) * dx
            trace_step = dx * stride
        else:
            x_center = float(x_offset_m) + idx.astype(np.float32)
            trace_step = float(stride)

        traces = np.asarray(data[:, idx], dtype=np.float32)
        scale_ref = np.nanpercentile(np.abs(traces), 95, axis=0)
        scale_ref = np.where(np.isfinite(scale_ref) & (scale_ref > 1e-8), scale_ref, 1.0)
        traces = traces / scale_ref[np.newaxis, :]

        amp_x = 0.35 * trace_step * float(self._wiggle_scale)
        y = np.linspace(y_top, y_bottom, n_smp, dtype=np.float32)

        for j, xc in enumerate(x_center):
            x = xc + traces[:, j] * amp_x
            ax.plot(
                x, y,
                color='#f5f5f5',
                linewidth=0.45,
                alpha=0.32,
                zorder=5,
            )

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
        aspect_factor: float = 0.5,
        norm_mode:     str   = 'linear',
        symlog_linthresh_pct: float = 3.0,
    ):
        self._last_panels = panels
        self._last_cmap   = cmap
        self._last_y_mode = y_mode
        self._last_vel    = velocity_m_ns
        self._last_t0     = t0_ns
        self._last_aspect = float(aspect_factor)
        self._last_norm_mode = str(norm_mode)
        self._last_symlog_linthresh_pct = float(symlog_linthresh_pct)
        self._render(
            panels, cmap, y_mode, velocity_m_ns, t0_ns, aspect_factor,
            norm_mode, symlog_linthresh_pct,
        )

    def _render(
        self,
        panels:        List[Dict],
        cmap:          str,
        y_mode:        YMode,
        velocity_m_ns: float,
        t0_ns:         float,
        aspect_factor: float,
        norm_mode:     str,
        symlog_linthresh_pct: float,
    ):
        LOG.debug(
            f'_render: n={len(panels)} cmap={cmap} '
            f'y_mode={y_mode} v={velocity_m_ns} m/ns '
            f'norm={norm_mode} symlog_lin={symlog_linthresh_pct:.2f}%'
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
        valid_idx = set(range(len(panels)))
        self._panel_bounds = {}
        self._panel_defaults = {}
        for k in list(self._zoom_limits.keys()):
            if k not in valid_idx:
                del self._zoom_limits[k]

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
            x_start  = float(panel.get('x_start_m', 0.0))
            x_disp_w = panel.get('x_display_width_m', None)
            x_disp_w = float(x_disp_w) if x_disp_w is not None else None
            x_total  = panel.get('total_length_m', None)
            x_total  = float(x_total) if x_total is not None else None
            n_smp, n_trc = data.shape

            # --- X axis ---
            if dx > 0:
                x_span = max(0.0, dx * (n_trc - 1))
                x_min, x_max = x_start, x_start + x_span
                if x_total is None:
                    x_total = x_span
                x_label = f'Distance  ({x_total:.2f} m total)'
            else:
                x_min, x_max = x_start, x_start + (n_trc - 1)
                x_label = f'Trace  ({int(x_min)} - {int(x_max)})'

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

            extent = [x_min, x_max, y_bottom, y_top]

            try:
                dplot = np.asarray(data, dtype=np.float32)
                if not np.isfinite(dplot).all():
                    dplot = np.nan_to_num(
                        dplot, nan=0.0, posinf=0.0, neginf=0.0
                    ).astype(np.float32)

                vmin, vmax = self._symmetric_clim(dplot)

                LOG.debug(
                    f'Panel {idx}: shape={data.shape}  '
                    f'extent={[round(v,3) for v in extent]}  '
                    f'vmin={vmin:.3g}  vmax={vmax:.3g}  '
                    f'(symm {self._clip_pct:.1f}th-pct, skip={self._dw_skip_frac:.0%} rows)'
                )

                nm = str(norm_mode).lower().strip()
                if nm == 'symlog':
                    vabs = max(abs(vmin), abs(vmax))
                    lin_pct = float(np.clip(symlog_linthresh_pct, 0.1, 50.0))
                    # Robust linear threshold: keep at least median absolute
                    # amplitude so weak reflectors are not over-compressed.
                    abs_med = float(np.nanmedian(np.abs(dplot)))
                    if not np.isfinite(abs_med):
                        abs_med = 0.0
                    linth = max(vabs * (lin_pct / 100.0), abs_med, 1e-8)
                    norm = SymLogNorm(
                        linthresh=linth,
                        linscale=1.0,
                        vmin=vmin,
                        vmax=vmax,
                        base=10.0,
                    )
                    im = ax.imshow(
                        dplot,
                        cmap=cmap,
                        aspect='auto',
                        extent=extent,
                        norm=norm,
                        interpolation='bilinear',
                        origin='upper',
                    )
                else:
                    im = ax.imshow(
                        dplot,
                        cmap=cmap,
                        aspect='auto',
                        extent=extent,
                        vmin=vmin,
                        vmax=vmax,
                        interpolation='bilinear',
                        origin='upper',
                    )
                self._images.append(im)

                # Keep matplotlib in automatic aspect mode for radargrams.
                # Forcing a numeric axis aspect here distorts geometry
                # (hyperbolas/layering) when X and Y use heterogeneous units.
                ax.set_aspect('auto')

                if (dx > 0) and (x_disp_w is not None) and (x_disp_w > 0):
                    ax.set_xlim(x_min, x_min + x_disp_w)

                x0d, x1d = ax.get_xlim()
                y0d, y1d = ax.get_ylim()
                self._panel_defaults[idx] = ((x0d, x1d), (y0d, y1d))
                self._panel_bounds[idx] = (
                    min(x_min, x_max), max(x_min, x_max),
                    min(y_top, y_bottom), max(y_top, y_bottom),
                )
                self._apply_zoom_state(idx, ax)

                if self._wiggle_overlay:
                    self._draw_wiggles(
                        ax=ax,
                        data=dplot,
                        dx=dx,
                        y_top=y_top,
                        y_bottom=y_bottom,
                        x_offset_m=x_min,
                    )

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
                LOG.error(f'Panel {idx} render error: {e}\\n{traceback.format_exc()}')
                ax.text(
                    0.5, 0.5, f'Render error:\\n{e}',
                    ha='center', va='center',
                    color='red', fontsize=8,
                    transform=ax.transAxes,
                    wrap=True,
                )

        self.draw_idle()

    # ------------------------------------------------------------------

    def clear(self):
        self._last_panels = []
        self._zoom_limits = {}
        self._panel_bounds = {}
        self._panel_defaults = {}
        self._pan_state = None
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
        clip_pct:   float = 98.0,
        dw_skip_frac: float = 0.25,
    ):
        if self.colorbar is not None:
            try: self.colorbar.remove()
            except Exception: pass
            self.colorbar = None
        self.axes.clear()

        if time_axis  is None: time_axis  = np.arange(data.shape[0])
        if trace_axis is None: trace_axis = np.arange(data.shape[1])

        if vmin is None or vmax is None:
            n = data.shape[0]
            skip = max(0, int(n * dw_skip_frac))
            d_sub = data[skip:, :] if n > skip + 4 else data
            vabs  = float(np.percentile(np.abs(d_sub), clip_pct))
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
