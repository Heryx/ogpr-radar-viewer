"""
GPR Signal Processing Module

Implements GPR processing algorithms described in:
  Goodman & Piro (2013) GPR Remote Sensing in Archaeology, Chapter 3.

Pipeline (in recommended order):
  1.  Time-Zero Correction        - align first breaks, crop pre-trigger
  2.  Dewow                       - remove low-freq DC drift along each trace
  3.  Background Removal          - subtract mean/median trace (global or rolling)
  4.  Bandpass Filter             - Butterworth band-pass in frequency domain
  5.  Notch Filter                - remove a single interference frequency
  6.  Spectral Whitening          - balance amplitudes across all frequencies
  7.  Gain  (exp / linear / SEC / AGC)
      SEC = Spreading & Exponential Compensation  (physically correct for GPR)
  8.  Hilbert Transform           - instantaneous amplitude (envelope)
  9.  Normalize
  10. Kirchhoff Migration         - collapse hyperbolic diffractions
"""

from __future__ import annotations

import logging
import traceback
from typing import Literal, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal

LOG = logging.getLogger('ogpr_viewer')


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _safe_copy(data: np.ndarray) -> np.ndarray:
    return np.array(data, dtype=np.float32, copy=True)


# ---------------------------------------------------------------------------
# SignalProcessor
# ---------------------------------------------------------------------------

class SignalProcessor:
    """
    Stateful processor: each method modifies self.processed_data in place
    and returns it.  Call reset() to start over from the original data.

    Args:
        data:              2-D float array  shape=(samples, traces)
        sampling_time_ns:  Time step between samples [ns]
        trace_spacing_m:   Horizontal distance between traces [m]
                           (required for migration; defaults to 0.05 m)
    """

    def __init__(
        self,
        data: np.ndarray,
        sampling_time_ns: float = 0.117,
        trace_spacing_m:  float = 0.05,
    ):
        self.original_data    = _safe_copy(data)
        self.processed_data   = _safe_copy(data)
        self.sampling_time_ns = float(sampling_time_ns)
        self.trace_spacing_m  = float(trace_spacing_m)
        self.sampling_freq_mhz = 1000.0 / sampling_time_ns

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset(self):
        self.processed_data = _safe_copy(self.original_data)

    def get_processed_data(self) -> np.ndarray:
        return self.processed_data.copy()

    def get_time_axis(self) -> np.ndarray:
        n = self.processed_data.shape[0]
        return np.arange(n, dtype=np.float32) * self.sampling_time_ns

    def get_depth_axis(self, velocity_m_ns: float = 0.10) -> np.ndarray:
        """Two-way travel time -> depth [m]."""
        return self.get_time_axis() * velocity_m_ns / 2.0

    # ------------------------------------------------------------------
    # 1. Time-Zero Correction
    # ------------------------------------------------------------------

    def correct_time_zero(
        self,
        method:    Literal['threshold', 'max'] = 'threshold',
        threshold: float = 0.1,
    ) -> np.ndarray:
        """
        Align all traces to a common first-break (time-zero) and crop
        the pre-trigger region.

        Steps:
          1. Detect the first-break sample in each trace:
             - 'threshold': first sample exceeding threshold * max_amplitude
             - 'max':       sample of maximum absolute amplitude
          2. Shift all traces so every first-break aligns to the median
             first-break position.
          3. Crop the profile: remove the first med samples (pre-trigger
             noise / air gap before the antenna fires).  After this step
             the profile starts at t = 0 (first break = surface).
        """
        data     = self.processed_data
        n_smp    = data.shape[0]
        n_trc    = data.shape[1]
        tz       = np.zeros(n_trc, dtype=int)

        for i in range(n_trc):
            tr = np.abs(data[:, i])
            if method == 'max':
                tz[i] = int(np.argmax(tr))
            else:
                thr = threshold * float(np.max(tr))
                idx = np.where(tr > thr)[0]
                tz[i] = int(idx[0]) if len(idx) else 0

        med = int(np.median(tz))
        out = np.zeros_like(data)
        for i in range(n_trc):
            shift = med - tz[i]
            if   shift > 0: out[shift:,         i] = data[: n_smp - shift, i]
            elif shift < 0: out[: n_smp + shift, i] = data[-shift:,        i]
            else:           out[:, i] = data[:, i]

        if med > 0:
            out = out[med:, :]

        LOG.debug(
            f'Time-zero: median={med} shift_range=[{tz.min()},{tz.max()}] '
            f'cropped={med} samples  new_shape={out.shape}'
        )
        self.processed_data = out
        return out

    # ------------------------------------------------------------------
    # 2. Dewow  (low-frequency drift removal)
    # ------------------------------------------------------------------

    def dewow(self, window_size: int = 50) -> np.ndarray:
        data = self.processed_data.copy()
        w    = max(3, int(window_size))
        kernel = np.ones(w, dtype=np.float32) / w

        for i in range(data.shape[1]):
            ma = np.convolve(data[:, i], kernel, mode='same')
            data[:, i] -= ma

        LOG.debug(f'Dewow: window={w} samples')
        self.processed_data = data
        return data

    # ------------------------------------------------------------------
    # 3. Background Removal
    # ------------------------------------------------------------------

    def remove_background(
        self,
        method:        Literal['mean', 'median'] = 'mean',
        rolling:       bool  = False,
        window_traces: int   = 50,
    ) -> np.ndarray:
        data = self.processed_data.copy()
        n_trc = data.shape[1]
        half  = window_traces // 2

        if not rolling:
            if method == 'mean':
                bg = np.mean(data, axis=1, keepdims=True)
            else:
                bg = np.median(data, axis=1, keepdims=True)
            data -= bg
        else:
            out = data.copy()
            for i in range(n_trc):
                lo  = max(0,     i - half)
                hi  = min(n_trc, i + half)
                win = data[:, lo:hi]
                bg  = np.mean(win, axis=1) if method == 'mean' else np.median(win, axis=1)
                out[:, i] = data[:, i] - bg
            data = out

        LOG.debug(f'Background removal: method={method} rolling={rolling} win={window_traces}')
        self.processed_data = data
        return data

    # ------------------------------------------------------------------
    # 4. Bandpass Filter
    # ------------------------------------------------------------------

    def apply_bandpass(
        self,
        low_freq:  float = 100.0,
        high_freq: float = 800.0,
        order:     int   = 4,
    ) -> np.ndarray:
        nyq  = self.sampling_freq_mhz / 2.0
        lo   = float(np.clip(low_freq  / nyq, 1e-4, 0.9999))
        hi   = float(np.clip(high_freq / nyq, 1e-4, 0.9999))

        if lo >= hi:
            LOG.warning(f'Bandpass: low={lo:.4f} >= high={hi:.4f} — skipped')
            return self.processed_data

        try:
            sos  = sp_signal.butter(order, [lo, hi], btype='band', output='sos')
            data = self.processed_data.copy()
            for i in range(data.shape[1]):
                data[:, i] = sp_signal.sosfiltfilt(sos, data[:, i])
            LOG.debug(f'Bandpass: {low_freq}-{high_freq} MHz order={order}')
            self.processed_data = data
        except Exception as e:
            LOG.error(f'Bandpass error: {e}\n{traceback.format_exc()}')

        return self.processed_data

    # ------------------------------------------------------------------
    # 5. Notch Filter
    # ------------------------------------------------------------------

    def apply_notch(
        self,
        freq_mhz:      float = 50.0,
        bandwidth_mhz: float = 5.0,
        order:         int   = 2,
    ) -> np.ndarray:
        nyq   = self.sampling_freq_mhz / 2.0
        lo    = float(np.clip((freq_mhz - bandwidth_mhz / 2) / nyq, 1e-4, 0.9999))
        hi    = float(np.clip((freq_mhz + bandwidth_mhz / 2) / nyq, 1e-4, 0.9999))

        if lo >= hi:
            LOG.warning(f'Notch: lo={lo} >= hi={hi} — skipped')
            return self.processed_data

        try:
            sos  = sp_signal.butter(order, [lo, hi], btype='bandstop', output='sos')
            data = self.processed_data.copy()
            for i in range(data.shape[1]):
                data[:, i] = sp_signal.sosfiltfilt(sos, data[:, i])
            LOG.debug(f'Notch: {freq_mhz} ± {bandwidth_mhz/2} MHz')
            self.processed_data = data
        except Exception as e:
            LOG.error(f'Notch error: {e}\n{traceback.format_exc()}')

        return self.processed_data

    # ------------------------------------------------------------------
    # 6. Spectral Whitening
    # ------------------------------------------------------------------

    def apply_spectral_whitening(
        self,
        bp_low:  Optional[float] = None,
        bp_high: Optional[float] = None,
        eps:     float = 1e-6,
    ) -> np.ndarray:
        data   = self.processed_data.copy()
        n_smp  = data.shape[0]
        freqs  = np.fft.rfftfreq(n_smp, d=self.sampling_time_ns)
        freqs_mhz = freqs * 1000.0

        if bp_low is not None or bp_high is not None:
            lo   = bp_low  if bp_low  is not None else 0.0
            hi   = bp_high if bp_high is not None else freqs_mhz[-1]
            mask = ((freqs_mhz >= lo) & (freqs_mhz <= hi)).astype(np.float32)
        else:
            mask = np.ones(len(freqs_mhz), dtype=np.float32)

        for i in range(data.shape[1]):
            spec = np.fft.rfft(data[:, i])
            mag  = np.abs(spec)
            spec_white = spec / (mag + eps * mag.max())
            spec_white *= mask
            data[:, i] = np.fft.irfft(spec_white, n=n_smp).astype(np.float32)

        LOG.debug(f'Spectral whitening: bp=[{bp_low},{bp_high}] MHz')
        self.processed_data = data
        return data

    # ------------------------------------------------------------------
    # 7. Gain
    # ------------------------------------------------------------------

    def apply_gain(
        self,
        gain_type:   Literal['exp', 'linear', 'sec', 'agc'] = 'agc',
        factor:      float = 20.0,
        alpha:       float = 0.5,
        t_start_ns:  float = 5.0,
        window_ns:   float = 25.0,
    ) -> np.ndarray:
        """
        Apply time-varying gain.

        gain_type='exp'
            g(t) = exp(factor * t/t_max)

        gain_type='linear'
            g(t) = 1 + factor * t/t_max

        gain_type='sec'  (Goodman & Piro 2013)
            g(t) = 1 + (t^2 * exp(alpha*t) / max) * (factor - 1)
            'factor' is the maximum gain multiplier at t=t_max.
            e.g. factor=20  -> 1x at surface, 20x at bottom.
            This fixes the previous bug where factor was ignored and
            gain was hardcoded to 1e4.

        gain_type='agc'
            Normalise each sample by the local RMS within window_ns.
            - RMS is computed ONLY on samples >= start_smp (direct wave
              region is excluded to prevent it inflating the RMS).
            - A noise floor of 5% of the per-trace global RMS prevents
              near-zero zones (e.g. after background removal) from being
              amplified to saturation.
            - mode='reflect' avoids border artefacts at trace ends.

        Args:
            factor:     SEC/exp/lin: max gain multiplier at t=t_max.
                        AGC: not used.
            alpha:      SEC absorption coefficient [1/ns].
            t_start_ns: Gain starts at this time [ns].
                        For AGC: samples before t_start keep original values.
                        Default 5 ns skips the direct wave.
            window_ns:  AGC sliding window length [ns].
                        Should be 2-3x the dominant wavelength.
                        Default 25 ns suits 200-800 MHz antennas.
        """
        data  = self.processed_data.copy()
        n_smp = data.shape[0]
        t     = np.arange(n_smp, dtype=np.float64) * self.sampling_time_ns
        t_max = t[-1] if t[-1] > 0 else 1.0

        if gain_type == 'exp':
            g = np.exp(factor * t / t_max)

        elif gain_type == 'linear':
            g = 1.0 + factor * t / t_max

        elif gain_type == 'sec':
            t_shifted = np.maximum(t - t_start_ns, 0.0)
            with np.errstate(over='ignore'):
                g = (t_shifted ** 2) * np.exp(alpha * t_shifted)
            g_max = float(g.max())
            if g_max > 0:
                # 'factor' is the max gain multiplier at t=t_max.
                # g ranges from 0 at t_start to factor at t_max.
                # We add 1.0 so there is never a gain < 1 (no attenuation).
                g = 1.0 + (g / g_max) * max(float(factor) - 1.0, 0.0)
            else:
                g = np.ones_like(t)
            g[t < t_start_ns] = 1.0   # no change before t_start
            LOG.debug(
                f'SEC: factor={factor} (max_gain={factor}x at t_max)  '
                f'alpha={alpha}  t_start={t_start_ns} ns'
            )

        elif gain_type == 'agc':
            from scipy.ndimage import uniform_filter1d

            win_smp   = max(3, int(window_ns / self.sampling_time_ns))
            start_smp = max(0, int(t_start_ns / self.sampling_time_ns))

            out = data.copy()   # pre-start rows kept as-is

            if start_smp < n_smp:
                sub = data[start_smp:, :].astype(np.float64)

                # Local RMS within the sliding window.
                # mode='reflect' mirrors data at both ends:
                #   - top border: avoids direct-wave bleed into sub block
                #   - bottom border: avoids value-repetition artefact
                local_rms = np.sqrt(
                    uniform_filter1d(
                        sub ** 2,
                        size=win_smp,
                        axis=0,
                        mode='reflect',
                    )
                )

                # Noise floor: per-trace global RMS of the sub-surface block.
                # Prevents the AGC from amplifying dead zones (near-zero
                # amplitude after background removal) to saturation.
                # Floor = 5% of global RMS: regions quieter than this
                # are not boosted beyond 20x their natural level.
                global_rms = np.sqrt(
                    np.mean(sub ** 2, axis=0, keepdims=True)
                )
                noise_floor = 0.05
                rms_floor   = np.maximum(local_rms, global_rms * noise_floor)
                rms_floor   = np.where(rms_floor < 1e-12, 1e-12, rms_floor)

                out[start_smp:, :] = (sub / rms_floor).astype(np.float32)

            self.processed_data = out.astype(np.float32)
            LOG.debug(
                f'AGC: window={window_ns} ns ({win_smp} smp)  '
                f't_start={t_start_ns} ns ({start_smp} smp protected)  '
                f'noise_floor=5%  sub_block={n_smp-start_smp} smp'
            )
            return self.processed_data

        else:
            LOG.warning(f'Unknown gain_type: {gain_type}')
            return self.processed_data

        g = g.astype(np.float32).reshape(-1, 1)
        data = (data * g).astype(np.float32)
        LOG.debug(f'Gain: type={gain_type} factor={factor} alpha={alpha} t_start={t_start_ns}')
        self.processed_data = data
        return data

    # ------------------------------------------------------------------
    # 8. Hilbert Transform (envelope)
    # ------------------------------------------------------------------

    def apply_hilbert(self) -> np.ndarray:
        data = self.processed_data.copy()
        for i in range(data.shape[1]):
            data[:, i] = np.abs(sp_signal.hilbert(data[:, i]))
        LOG.debug('Hilbert transform applied')
        self.processed_data = data
        return data

    # ------------------------------------------------------------------
    # 9. Normalize
    # ------------------------------------------------------------------

    def normalize(
        self,
        method: Literal['minmax', 'zscore', 'robust'] = 'minmax',
    ) -> np.ndarray:
        data = self.processed_data.copy()

        if method == 'minmax':
            lo, hi = data.min(), data.max()
            if hi > lo:
                data = (data - lo) / (hi - lo)

        elif method == 'zscore':
            mu, sigma = data.mean(), data.std()
            if sigma > 0:
                data = (data - mu) / sigma

        elif method == 'robust':
            p5, p95 = np.percentile(data, 5), np.percentile(data, 95)
            if p95 > p5:
                data = np.clip((data - p5) / (p95 - p5), 0.0, 1.0)

        LOG.debug(f'Normalize: method={method}')
        self.processed_data = data.astype(np.float32)
        return self.processed_data

    # ------------------------------------------------------------------
    # 10. Kirchhoff Migration
    # ------------------------------------------------------------------

    def apply_migration(
        self,
        velocity_m_ns:   float = 0.10,
        aperture_traces: int   = 30,
    ) -> np.ndarray:
        data   = self.processed_data.copy()
        n_smp, n_trc = data.shape
        dt     = self.sampling_time_ns
        dx     = self.trace_spacing_m
        v      = velocity_m_ns
        out    = np.zeros_like(data)

        LOG.debug(
            f'Kirchhoff migration: v={v} m/ns dx={dx} m '
            f'aperture=±{aperture_traces} traces'
        )

        for j0 in range(n_trc):
            for i0 in range(n_smp):
                t0     = i0 * dt
                x0     = j0 * dx
                acc    = 0.0
                n_used = 0
                j_lo   = max(0,     j0 - aperture_traces)
                j_hi   = min(n_trc, j0 + aperture_traces + 1)

                for j in range(j_lo, j_hi):
                    x      = j * dx
                    offset = 2.0 * (x - x0) / v
                    t      = np.sqrt(t0 ** 2 + offset ** 2)
                    i_frac = t / dt
                    i_lo   = int(i_frac)
                    i_hi   = i_lo + 1
                    if i_hi >= n_smp:
                        continue
                    frac   = i_frac - i_lo
                    acc   += (1 - frac) * data[i_lo, j] + frac * data[i_hi, j]
                    n_used += 1

                if n_used > 0:
                    out[i0, j0] = acc / n_used

        LOG.debug('Kirchhoff migration: done')
        self.processed_data = out.astype(np.float32)
        return self.processed_data

    # ------------------------------------------------------------------
    # Power spectrum  (for GUI inspector)
    # ------------------------------------------------------------------

    def get_power_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        data  = self.processed_data
        n_smp = data.shape[0]
        specs = np.abs(np.fft.rfft(data, axis=0)) ** 2
        mean_power    = specs.mean(axis=1)
        mean_power_db = 10.0 * np.log10(mean_power + 1e-30)
        freqs_mhz     = np.fft.rfftfreq(n_smp, d=self.sampling_time_ns) * 1000.0
        return freqs_mhz.astype(np.float32), mean_power_db.astype(np.float32)
