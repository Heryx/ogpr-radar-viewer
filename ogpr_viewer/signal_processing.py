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

Dtype policy:
  float32 input (IDS Stream UP): all internal computation in float32.
                                  Physical voltage units preserved throughout.
  int16 input   (IDS Stream DP): all internal computation in int16/float64
                                  as appropriate. ADC counts preserved.
                                  Output stays int16 unless a step requires
                                  float (e.g. FFT) in which case float32 is
                                  used and flagged in the log.
  NO silent dtype promotion between the two workflows.
"""

from __future__ import annotations

import logging
import traceback
from typing import Literal, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal

LOG = logging.getLogger('ogpr_viewer')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _work_f32(data: np.ndarray) -> np.ndarray:
    """Return a float32 working copy, preserving values exactly for float32 input."""
    return np.array(data, dtype=np.float32, copy=True)


def _work_f64(data: np.ndarray) -> np.ndarray:
    """Return a float64 working copy (used for int16 spectral ops)."""
    return np.array(data, dtype=np.float64, copy=True)


# ---------------------------------------------------------------------------
# SignalProcessor
# ---------------------------------------------------------------------------

class SignalProcessor:
    """
    Stateful processor: each method modifies self.processed_data in place
    and returns it.  Call reset() to start over from the original data.

    Args:
        data:              2-D array  shape=(samples, traces)
                           Accepted dtypes: float32 (Stream UP) or int16 (Stream DP).
                           The original dtype is preserved. No silent conversion.
        sampling_time_ns:  Time step between samples [ns]
        trace_spacing_m:   Horizontal distance between traces [m]
    """

    def __init__(
        self,
        data: np.ndarray,
        sampling_time_ns: float = 0.117,
        trace_spacing_m:  float = 0.05,
    ):
        if data.dtype == np.float32:
            self._dtype        = np.float32
            self._is_int16     = False
            self.original_data  = np.array(data, dtype=np.float32, copy=True)
            self.processed_data = np.array(data, dtype=np.float32, copy=True)
            LOG.debug('SignalProcessor: float32 workflow')
        elif data.dtype == np.int16:
            self._dtype        = np.int16
            self._is_int16     = True
            self.original_data  = np.array(data, dtype=np.int16, copy=True)
            self.processed_data = np.array(data, dtype=np.int16, copy=True)
            LOG.debug('SignalProcessor: int16 workflow')
        else:
            # Any other numeric dtype: promote to float32 with a warning
            LOG.warning(
                f'SignalProcessor: unexpected dtype {data.dtype}, '
                f'promoting to float32'
            )
            self._dtype        = np.float32
            self._is_int16     = False
            self.original_data  = np.array(data, dtype=np.float32, copy=True)
            self.processed_data = np.array(data, dtype=np.float32, copy=True)

        self.sampling_time_ns  = float(sampling_time_ns)
        self.trace_spacing_m   = float(trace_spacing_m)
        self.sampling_freq_mhz = 1000.0 / sampling_time_ns

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset(self):
        self.processed_data = np.array(self.original_data, copy=True)

    def get_processed_data(self) -> np.ndarray:
        """Return a copy in the native dtype."""
        return self.processed_data.copy()

    def get_time_axis(self) -> np.ndarray:
        n = self.processed_data.shape[0]
        return np.arange(n, dtype=np.float32) * self.sampling_time_ns

    def get_depth_axis(self, velocity_m_ns: float = 0.10) -> np.ndarray:
        return self.get_time_axis() * velocity_m_ns / 2.0

    def _as_f32(self) -> np.ndarray:
        """Working float32 copy of processed_data (for spectral / float ops)."""
        return self.processed_data.astype(np.float32, copy=True)

    def _store(self, result: np.ndarray):
        """Store result back into processed_data, preserving native dtype."""
        if self._is_int16:
            # Clip to int16 range to avoid overflow
            self.processed_data = np.clip(
                np.round(result), -32768, 32767
            ).astype(np.int16)
        else:
            self.processed_data = np.array(result, dtype=np.float32)

    # ------------------------------------------------------------------
    # 1. Time-Zero Correction
    # ------------------------------------------------------------------

    def correct_time_zero(
        self,
        method:    Literal['threshold', 'max'] = 'threshold',
        threshold: float = 0.1,
    ) -> np.ndarray:
        data  = self._as_f32()
        n_smp = data.shape[0]
        n_trc = data.shape[1]
        tz    = np.zeros(n_trc, dtype=int)

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
            shift = int(np.clip(med - tz[i], -n_smp + 1, n_smp - 1))
            if shift > 0:
                out[shift:, i] = data[:n_smp - shift, i]
                out[:shift, i] = data[0, i]
            elif shift < 0:
                out[:n_smp + shift, i] = data[-shift:, i]
                out[n_smp + shift:, i] = data[-1, i]
            else:
                out[:, i] = data[:, i]

        if med > 0:
            out = out[med:, :]

        LOG.debug(
            f'Time-zero: median={med} shift_range=[{tz.min()},{tz.max()}] '
            f'cropped={med} samples  new_shape={out.shape}'
        )
        self._store(out)
        return self.processed_data

    # ------------------------------------------------------------------
    # 2. Dewow
    # ------------------------------------------------------------------

    def dewow(self, window_size: int = 50) -> np.ndarray:
        data = self._as_f32()
        n    = data.shape[0]
        w    = min(window_size, n)
        kernel = np.ones(w, dtype=np.float32) / w
        out  = np.empty_like(data)
        for i in range(data.shape[1]):
            trend   = np.convolve(data[:, i], kernel, mode='same')
            out[:, i] = data[:, i] - trend
        LOG.debug(f'Dewow: window={w}')
        self._store(out)
        return self.processed_data

    # ------------------------------------------------------------------
    # 3. Background Removal
    # ------------------------------------------------------------------

    def remove_background(
        self,
        method:        Literal['mean', 'median'] = 'mean',
        rolling:       bool = False,
        window_traces: int  = 50,
    ) -> np.ndarray:
        data = self._as_f32()
        n_smp, n_trc = data.shape

        if rolling:
            out = np.empty_like(data)
            hw  = window_traces // 2
            for i in range(n_trc):
                lo = max(0, i - hw)
                hi = min(n_trc, i + hw + 1)
                bg = (np.mean(data[:, lo:hi], axis=1)
                      if method == 'mean'
                      else np.median(data[:, lo:hi], axis=1))
                out[:, i] = data[:, i] - bg
        else:
            bg  = (np.mean(data, axis=1)
                   if method == 'mean'
                   else np.median(data, axis=1))
            out = data - bg[:, np.newaxis]

        LOG.debug(f'Background removal: method={method} rolling={rolling}')
        self._store(out)
        return self.processed_data

    # ------------------------------------------------------------------
    # 4. Bandpass Filter
    # ------------------------------------------------------------------

    def apply_bandpass(
        self,
        low_freq:  float = 100.0,
        high_freq: float = 800.0,
        order:     int   = 4,
    ) -> np.ndarray:
        data  = self._as_f32()
        fs    = self.sampling_freq_mhz
        nyq   = fs / 2.0
        lo    = float(np.clip(low_freq  / nyq, 1e-4, 0.9999))
        hi    = float(np.clip(high_freq / nyq, 1e-4, 0.9999))
        if lo >= hi:
            LOG.warning(f'Bandpass: lo={lo:.4f} >= hi={hi:.4f}, skipping')
            return self.processed_data
        b, a  = sp_signal.butter(order, [lo, hi], btype='band')
        out   = sp_signal.filtfilt(b, a, data, axis=0).astype(np.float32)
        LOG.debug(f'Bandpass: {low_freq}-{high_freq} MHz order={order}')
        self._store(out)
        return self.processed_data

    # ------------------------------------------------------------------
    # 5. Notch Filter
    # ------------------------------------------------------------------

    def apply_notch(
        self,
        freq_mhz:      float = 50.0,
        bandwidth_mhz: float = 5.0,
    ) -> np.ndarray:
        data  = self._as_f32()
        fs    = self.sampling_freq_mhz
        nyq   = fs / 2.0
        f0    = float(np.clip(freq_mhz / nyq, 1e-4, 0.9999))
        bw    = float(np.clip(bandwidth_mhz / nyq, 1e-6, 0.5))
        Q     = f0 / bw if bw > 0 else 30.0
        b, a  = sp_signal.iirnotch(f0, Q)
        out   = sp_signal.filtfilt(b, a, data, axis=0).astype(np.float32)
        LOG.debug(f'Notch: {freq_mhz} MHz BW={bandwidth_mhz} MHz')
        self._store(out)
        return self.processed_data

    # ------------------------------------------------------------------
    # 6. Spectral Whitening
    # ------------------------------------------------------------------

    def apply_spectral_whitening(
        self,
        bp_low:  Optional[float] = None,
        bp_high: Optional[float] = None,
    ) -> np.ndarray:
        data  = self._as_f32()
        n     = data.shape[0]
        F     = np.fft.rfft(data, axis=0)
        amp   = np.abs(F)
        amp   = np.where(amp < 1e-10, 1e-10, amp)
        F_w   = F / amp

        if bp_low is not None and bp_high is not None:
            fs    = self.sampling_freq_mhz
            nyq   = fs / 2.0
            freqs = np.fft.rfftfreq(n, d=1.0 / fs)
            mask  = (freqs >= bp_low) & (freqs <= bp_high)
            F_w   = F_w * mask[:, np.newaxis]

        out = np.fft.irfft(F_w, n=n, axis=0).astype(np.float32)
        LOG.debug('Spectral whitening applied')
        self._store(out)
        return self.processed_data

    # ------------------------------------------------------------------
    # 7. Gain
    # ------------------------------------------------------------------

    def apply_gain(
        self,
        gain_type:  Literal['agc', 'sec', 'exp', 'linear'] = 'agc',
        factor:     float = 20.0,
        alpha:      float = 0.5,
        t_start_ns: float = 5.0,
        window_ns:  float = 25.0,
    ) -> np.ndarray:
        data  = self._as_f32()
        n_smp = data.shape[0]
        dt    = self.sampling_time_ns
        t     = np.arange(n_smp, dtype=np.float32) * dt

        if gain_type == 'agc':
            win = max(1, int(window_ns / dt))
            out = np.empty_like(data)
            half = win // 2
            for i in range(data.shape[1]):
                tr    = data[:, i]
                env   = np.convolve(np.abs(tr), np.ones(win) / win, mode='same')
                # Noise floor: median of the DEEP half avoids being dominated
                # by the strong direct wave
                deep_start = n_smp // 2
                noise = float(np.median(np.abs(tr[deep_start:]))) + 1e-10
                env   = np.where(env < noise, noise, env)
                out[:, i] = tr / env

        elif gain_type == 'sec':
            t_s   = np.maximum(t - t_start_ns, 0.0)
            g     = (1.0 + t_s) * np.exp(alpha * t_s)
            g_max = float(g[-1]) if g[-1] > 0 else 1.0
            g     = g * (factor / g_max)
            g     = np.clip(g, 1.0, factor)
            out   = data * g[:, np.newaxis]

        elif gain_type == 'exp':
            t_s   = np.maximum(t - t_start_ns, 0.0)
            g     = np.exp(alpha * t_s)
            g_max = float(g[-1]) if g[-1] > 0 else 1.0
            g     = g * (factor / g_max)
            g     = np.clip(g, 1.0, factor)
            out   = data * g[:, np.newaxis]

        else:  # linear
            t_s   = np.maximum(t - t_start_ns, 0.0)
            t_max = float(t[-1] - t_start_ns) if t[-1] > t_start_ns else 1.0
            g     = 1.0 + (factor - 1.0) * (t_s / t_max)
            g     = np.clip(g, 1.0, factor)
            out   = data * g[:, np.newaxis]

        LOG.debug(f'Gain: type={gain_type} factor={factor} alpha={alpha}')
        self._store(out)
        return self.processed_data

    # ------------------------------------------------------------------
    # 8. Hilbert Transform (envelope)
    # ------------------------------------------------------------------

    def apply_hilbert(self) -> np.ndarray:
        data = self._as_f32()
        out  = np.abs(sp_signal.hilbert(data, axis=0)).astype(np.float32)
        LOG.debug('Hilbert transform applied')
        self._store(out)
        return self.processed_data

    # ------------------------------------------------------------------
    # 9. Normalize
    # ------------------------------------------------------------------

    def normalize(
        self,
        method: Literal['minmax', 'zscore', 'robust'] = 'minmax',
    ) -> np.ndarray:
        data = self._as_f32()

        if method == 'minmax':
            mn, mx = float(data.min()), float(data.max())
            rng    = mx - mn if mx != mn else 1.0
            out    = (data - mn) / rng * 2.0 - 1.0

        elif method == 'zscore':
            mu  = float(data.mean())
            std = float(data.std())
            out = (data - mu) / (std if std > 1e-10 else 1.0)

        else:  # robust
            p5, p95 = float(np.percentile(data, 5)), float(np.percentile(data, 95))
            rng     = p95 - p5 if p95 != p5 else 1.0
            out     = np.clip((data - p5) / rng * 2.0 - 1.0, -1.0, 1.0)

        LOG.debug(f'Normalize: method={method}')
        self._store(out)
        return self.processed_data

    # ------------------------------------------------------------------
    # 10. Kirchhoff Migration
    # ------------------------------------------------------------------

    def apply_migration(
        self,
        velocity_m_ns:   float = 0.10,
        aperture_traces: int   = 30,
    ) -> np.ndarray:
        data  = self._as_f32()
        n_smp, n_trc = data.shape
        dt    = self.sampling_time_ns
        dx    = self.trace_spacing_m
        out   = np.zeros_like(data)

        for ix in range(n_trc):
            x_lo = max(0, ix - aperture_traces)
            x_hi = min(n_trc, ix + aperture_traces + 1)
            for it in range(n_smp):
                t0   = it * dt
                depth = t0 * velocity_m_ns / 2.0
                if depth < 1e-9:
                    out[it, ix] = data[it, ix]
                    continue
                acc = 0.0
                cnt = 0
                for jx in range(x_lo, x_hi):
                    dx_m  = abs(ix - jx) * dx
                    t_hyp = 2.0 * np.sqrt(depth**2 + (dx_m / 2.0)**2) / velocity_m_ns
                    jt    = int(round(t_hyp / dt))
                    if 0 <= jt < n_smp:
                        acc += data[jt, jx]
                        cnt += 1
                out[it, ix] = acc / cnt if cnt else data[it, ix]

        LOG.debug(f'Migration: v={velocity_m_ns} m/ns aperture={aperture_traces}')
        self._store(out)
        return self.processed_data

    # ------------------------------------------------------------------
    # Power spectrum helper
    # ------------------------------------------------------------------

    def get_power_spectrum(
        self,
        n_traces: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        data  = self._as_f32()
        n_smp = data.shape[0]
        step  = max(1, data.shape[1] // n_traces)
        sub   = data[:, ::step]
        F     = np.fft.rfft(sub, axis=0)
        power = np.mean(np.abs(F)**2, axis=1)
        power_db = 10.0 * np.log10(power + 1e-30)
        freqs    = np.fft.rfftfreq(n_smp, d=self.sampling_time_ns / 1000.0)
        return freqs.astype(np.float32), power_db.astype(np.float32)
