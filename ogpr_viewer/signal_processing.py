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
             first-break position. Use edge padding instead of zero padding
             to prevent artifacts (black lines on edges).
          3. Crop the profile: remove the first med samples (pre-trigger
             noise / air gap before the antenna fires).
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
            # Clamp shift to valid range to prevent out-of-bounds
            shift = np.clip(shift, -n_smp + 1, n_smp - 1)
            
            if shift > 0: 
                # Shift down: pad top with edge value
                out[shift:, i] = data[:n_smp - shift, i]
                out[:shift, i] = data[0, i]  # Edge padding instead of zeros
            elif shift < 0: 
                # Shift up: pad bottom with edge value
                out[:n_smp + shift, i] = data[-shift:, i]
                out[n_smp + shift:, i] = data[-1, i]  # Edge padding instead of zeros
            else:           
                out[:, i] = data[:, i]

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
            # Use symmetric padding to ensure output size matches input
            # For even window: pad left more, for odd: symmetric
            pad_left = w // 2
            pad_right = w - pad_left - 1
            
            padded = np.pad(data[:, i], (pad_left, pad_right), mode='edge')
            ma = np.convolve(padded, kernel, mode='valid')
            
            # Trim to exact input size if needed
            if len(ma) != len(data[:, i]):
                # Fallback: use standard mode='same' with zero padding
                ma = np.convolve(data[:, i], kernel, mode='same')
                LOG.warning(f'Dewow: size mismatch, falling back to mode=same for trace {i}')
                
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
        sample_start:  Optional[int] = None,
        sample_end:    Optional[int] = None,
    ) -> np.ndarray:
        """
        Subtract mean or median trace to remove horizontal banding.

        WARNING: Background removal may produce blank output on int16 data
        with saturated direct wave (±32k counts). This occurs because:
        - Direct wave saturates entire dynamic range
        - Subsurface signals < 1 LSB (invisible in int16)
        - All traces are identical → bg_removal subtracts everything
        
        For int16 saturated data: use dewow + AGC only, skip bg_removal.
        For float32 v2.0 data: bg_removal should work correctly.

        Args:
            method:        'mean' or 'median'
            rolling:       If True, compute background from a sliding window
                           of traces around each trace. If False, compute
                           global background from all traces.
            window_traces: Number of traces in rolling window (ignored if
                           rolling=False).
            sample_start:  Optional. If provided, compute mean/median only
                           from samples >= sample_start. Useful to exclude
                           the direct wave zone from background calculation.
            sample_end:    Optional. If provided, compute mean/median only
                           from samples < sample_end.

        Returns:
            Processed data with background removed.

        Example:
            # Exclude first 50 samples (direct wave) from background calc
            processor.remove_background(
                method='mean',
                sample_start=50,
                sample_end=500
            )
        """
        data = self.processed_data.copy()
        n_smp = data.shape[0]
        n_trc = data.shape[1]
        half  = window_traces // 2

        # Compute std before background removal for saturation detection
        std_before = float(np.std(data))

        # Validate sample range
        if sample_start is not None:
            sample_start = max(0, int(sample_start))
        else:
            sample_start = 0

        if sample_end is not None:
            sample_end = min(n_smp, int(sample_end))
        else:
            sample_end = n_smp

        if sample_start >= sample_end:
            LOG.warning(
                f'Background removal: invalid sample range '
                f'[{sample_start}, {sample_end}) — using full range'
            )
            sample_start = 0
            sample_end = n_smp

        # Extract subset for background calculation
        data_subset = data[sample_start:sample_end, :]

        if not rolling:
            # Global background
            if method == 'mean':
                bg_subset = np.mean(data_subset, axis=1, keepdims=True)
            else:
                bg_subset = np.median(data_subset, axis=1, keepdims=True)
            
            # Create full-size background array
            bg = np.zeros((n_smp, 1), dtype=np.float32)
            bg[sample_start:sample_end, :] = bg_subset
            data -= bg
        else:
            # Rolling background
            out = data.copy()
            for i in range(n_trc):
                lo  = max(0,     i - half)
                hi  = min(n_trc, i + half)
                win_subset = data_subset[:, lo:hi]
                
                if method == 'mean':
                    bg_subset = np.mean(win_subset, axis=1)
                else:
                    bg_subset = np.median(win_subset, axis=1)
                
                # Apply background only to the computed range
                out[sample_start:sample_end, i] -= bg_subset
            data = out

        # Check for saturation-induced blanking
        std_after = float(np.std(data))
        if std_before > 0:
            ratio = std_after / std_before
            if ratio < 0.01:  # Less than 1% of original variance remains
                LOG.warning(
                    f'Background removal produced near-zero output '
                    f'(std_after/std_before = {ratio:.6f}). '
                    f'This typically indicates int16 data with saturated direct wave. '
                    f'Subsurface signal < 1 LSB is invisible in int16. '
                    f'Recommendation: DISABLE background removal, use dewow + AGC only. '
                    f'Background removal is useful only for float32 v2.0 data '
                    f'or int16 with strong subsurface reflectors (>50% of dynamic range).'
                )

        range_str = f'[{sample_start}:{sample_end}]' if (sample_start > 0 or sample_end < n_smp) else 'all'
        LOG.debug(
            f'Background removal: method={method} rolling={rolling} '
            f'win={window_traces} sample_range={range_str} '
            f'std_before={std_before:.2f} std_after={std_after:.2f} '
            f'ratio={std_after/std_before if std_before > 0 else 0:.4f}'
        )
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
            n_smp = data.shape[0]
            
            for i in range(data.shape[1]):
                # Dynamic padding: max(50, 3*order) or 10% of trace length
                pad_len = max(50, 3 * order, int(0.1 * n_smp))
                pad_len = min(pad_len, n_smp // 2)  # Cap at 50% of trace length
                
                padded = np.pad(data[:, i], pad_len, mode='edge')
                filtered = sp_signal.sosfiltfilt(sos, padded)
                data[:, i] = filtered[pad_len:-pad_len]
                
            LOG.debug(f'Bandpass: {low_freq}-{high_freq} MHz order={order} pad={pad_len}')
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
            n_smp = data.shape[0]
            
            for i in range(data.shape[1]):
                # Add edge padding like bandpass for consistency
                pad_len = max(50, 3 * order, int(0.1 * n_smp))
                pad_len = min(pad_len, n_smp // 2)
                
                padded = np.pad(data[:, i], pad_len, mode='edge')
                filtered = sp_signal.sosfiltfilt(sos, padded)
                data[:, i] = filtered[pad_len:-pad_len]
                
            LOG.debug(f'Notch: {freq_mhz} ± {bandwidth_mhz/2} MHz order={order} pad={pad_len}')
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
        eps:     float = 1e-10,
    ) -> np.ndarray:
        """
        Apply spectral whitening to balance frequency amplitudes.
        
        Args:
            bp_low:  Low frequency cutoff [MHz]
            bp_high: High frequency cutoff [MHz]
            eps:     Regularization to prevent division by zero (absolute value)
        """
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
            # Use absolute epsilon instead of relative to prevent explosive values
            spec_white = spec / (mag + eps)
            spec_white *= mask
            data[:, i] = np.fft.irfft(spec_white, n=n_smp).astype(np.float32)

        LOG.debug(f'Spectral whitening: bp=[{bp_low},{bp_high}] MHz eps={eps}')
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
        noise_floor: float = 0.01,
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

        gain_type='agc'
            Normalise each sample by the local RMS within window_ns.
            - RMS is computed ONLY on samples >= start_smp (direct wave
              region is excluded).
            - Noise floor is computed from the DEEP HALF of the sub-surface
              block (bottom 50%) to avoid ringing contamination from the
              direct wave zone (which persists 10-20 ns after removal).
            - mode='reflect' avoids border artefacts at trace ends.

        Args:
            factor:      SEC/exp/lin: max gain multiplier at t=t_max.
                         AGC: not used.
            alpha:       SEC absorption coefficient [1/ns].
            t_start_ns:  Gain starts at this time [ns].
                         For AGC: samples before t_start keep original values.
                         Default 5 ns skips the direct wave.
            window_ns:   AGC sliding window length [ns].
                         Should be 2-3x the dominant wavelength.
                         Default 25 ns suits 200-800 MHz antennas.
            noise_floor: AGC noise floor as fraction of deep global RMS.
                         Default 0.01 (1%). Lower values = more aggressive,
                         higher values = more conservative.
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
                g = 1.0 + (g / g_max) * max(float(factor) - 1.0, 0.0)
            else:
                g = np.ones_like(t)
            g[t < t_start_ns] = 1.0
            LOG.debug(
                f'SEC: factor={factor} (max_gain={factor}x at t_max)  '
                f'alpha={alpha}  t_start={t_start_ns} ns'
            )

        elif gain_type == 'agc':
            from scipy.ndimage import uniform_filter1d

            win_smp   = max(3, int(window_ns / self.sampling_time_ns))
            start_smp = max(0, int(t_start_ns / self.sampling_time_ns))

            out = data.copy()

            if start_smp < n_smp:
                sub = data[start_smp:, :].astype(np.float64)
                n_sub = sub.shape[0]

                # Local RMS sliding window
                local_rms = np.sqrt(
                    uniform_filter1d(
                        sub ** 2,
                        size=win_smp,
                        axis=0,
                        mode='reflect',
                    )
                )

                # Noise floor: compute global RMS from DEEP HALF only.
                # After background removal, the direct-wave ringing zone
                # (5-15 ns) still has 10-100x higher amplitude than true
                # subsurface reflections.  Computing global_rms from the
                # entire sub-block would yield a noise floor too high for
                # the deeper signal, making it appear flat in the final image.
                # Using only the bottom 50% ensures the noise floor is
                # calibrated to actual subsurface amplitudes, not ringing.
                mid_idx = n_sub // 2
                if n_sub > 4:
                    deep_half = sub[mid_idx:, :]
                else:
                    deep_half = sub

                global_rms = np.sqrt(
                    np.mean(deep_half ** 2, axis=0, keepdims=True)
                )
                rms_floor = np.maximum(local_rms, global_rms * noise_floor)
                rms_floor = np.where(rms_floor < 1e-12, 1e-12, rms_floor)

                out[start_smp:, :] = (sub / rms_floor).astype(np.float32)

            self.processed_data = out.astype(np.float32)
            LOG.debug(
                f'AGC: window={window_ns} ns ({win_smp} smp)  '
                f't_start={t_start_ns} ns ({start_smp} smp protected)  '
                f'noise_floor={noise_floor*100:.1f}% of deep-half global RMS  '
                f'sub_block={n_smp-start_smp} smp'
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
