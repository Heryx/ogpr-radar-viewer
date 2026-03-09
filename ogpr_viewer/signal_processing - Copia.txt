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


def _weighted_median_int(values: np.ndarray, weights: np.ndarray) -> int:
    """
    Weighted median for integer samples.

    Args:
        values: 1-D integer-like array (e.g. pick indices)
        weights: 1-D non-negative weights with same length
    """
    if values.size == 0:
        return 0

    order = np.argsort(values)
    v = np.asarray(values[order], dtype=np.int32)
    w = np.asarray(weights[order], dtype=np.float64)

    total_w = float(np.sum(w))
    if total_w <= 0.0:
        return int(np.median(v))

    cdf = np.cumsum(w)
    idx = int(np.searchsorted(cdf, 0.5 * total_w, side='left'))
    idx = min(max(idx, 0), v.size - 1)
    return int(v[idx])


def _rolling_median_1d(x: np.ndarray, window: int) -> np.ndarray:
    """Median smoothing for 1-D arrays (edge-padded)."""
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    if w <= 1 or x.size <= 2:
        return np.array(x, copy=True)

    pad = w // 2
    xp = np.pad(np.asarray(x), (pad, pad), mode='edge')
    out = np.empty_like(np.asarray(x))
    for i in range(x.size):
        out[i] = np.median(xp[i:i + w])
    return out


def _robust_mean_axis1(win: np.ndarray, clip_sigma: float = 3.5) -> np.ndarray:
    """
    Robust row-wise mean using MAD-based winsorization.

    This strongly reduces "target shadowing" from bright hyperbolas while
    keeping the behavior close to mean subtraction.
    """
    w = np.asarray(win, dtype=np.float32)
    med = np.nanmedian(w, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(w - med), axis=1, keepdims=True)
    sig = 1.4826 * mad
    sig = np.where(np.isfinite(sig) & (sig > 1e-9), sig, 1e-9)

    lo = med - float(clip_sigma) * sig
    hi = med + float(clip_sigma) * sig
    wc = np.where(np.isfinite(w), np.clip(w, lo, hi), np.nan)
    out = np.nanmean(wc, axis=1)

    # Fallback on median where mean is undefined.
    med1 = np.nanmedian(w, axis=1)
    out = np.where(np.isfinite(out), out, med1)
    out = np.where(np.isfinite(out), out, 0.0).astype(np.float32)
    return out


def _trimmed_mean_axis1(win: np.ndarray, trim_percent: float = 10.0) -> np.ndarray:
    """
    Row-wise trimmed/winsorized mean via percentile clipping.
    """
    w = np.asarray(win, dtype=np.float32)
    p = float(np.clip(trim_percent, 0.0, 45.0))
    if p <= 0.0:
        out = np.nanmean(w, axis=1)
        return np.where(np.isfinite(out), out, 0.0).astype(np.float32)

    lo = np.nanpercentile(w, p, axis=1)[:, np.newaxis]
    hi = np.nanpercentile(w, 100.0 - p, axis=1)[:, np.newaxis]
    wc = np.where(np.isfinite(w), np.clip(w, lo, hi), np.nan)
    out = np.nanmean(wc, axis=1)
    med1 = np.nanmedian(w, axis=1)
    out = np.where(np.isfinite(out), out, med1)
    out = np.where(np.isfinite(out), out, 0.0).astype(np.float32)
    return out


def _background_stat_axis1(
    win: np.ndarray,
    method: str,
    trim_percent: float = 10.0,
) -> np.ndarray:
    m = str(method).lower()
    if m == 'median':
        out = np.nanmedian(win, axis=1)
        return np.where(np.isfinite(out), out, 0.0).astype(np.float32)
    if m == 'trimmed':
        return _trimmed_mean_axis1(win, trim_percent=trim_percent)
    # default: robust mean (drop-in upgrade over plain mean)
    return _robust_mean_axis1(win, clip_sigma=3.5)


def _repair_bad_traces(
    data: np.ndarray,
    rel_std_thr: float = 0.02,
    max_bad_frac: float = 0.20,
    max_edge_frac: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Repair very low-energy / invalid traces that commonly appear at swath edges.

    Returns:
        repaired_data, bad_mask
    """
    arr = np.asarray(data, dtype=np.float32)
    n_smp, n_trc = arr.shape
    if n_trc < 3:
        return np.array(arr, copy=True), np.zeros(n_trc, dtype=bool)

    std = np.nanstd(arr, axis=0)
    finite = np.isfinite(std)
    pos = finite & (std > 0.0)
    if not np.any(pos):
        return np.array(arr, copy=True), np.ones(n_trc, dtype=bool)

    med_std = float(np.median(std[pos]))
    thr = max(1e-9, float(rel_std_thr) * med_std)
    bad = (~finite) | (std <= thr)

    bad_frac = float(np.mean(bad))
    if (not np.any(bad)) or (bad_frac > float(max_bad_frac)):
        return np.array(arr, copy=True), bad

    out = np.array(arr, copy=True)
    max_edge = max(1, int(round(float(max_edge_frac) * n_trc)))

    left = 0
    while left < n_trc and bad[left]:
        left += 1
    if 0 < left <= max_edge and left < n_trc:
        out[:, :left] = out[:, left:left + 1]

    right = 0
    while right < n_trc and bad[n_trc - 1 - right]:
        right += 1
    if 0 < right <= max_edge and right < n_trc:
        src = n_trc - right - 1
        out[:, n_trc - right:] = out[:, src:src + 1]

    bad_idx = np.where(bad)[0]
    good_idx = np.where(~bad)[0]
    if good_idx.size == 0:
        return out, bad

    # Fill isolated bad traces by linear interpolation between nearest good scans.
    for i in bad_idx:
        if (i < left and left <= max_edge) or (i >= n_trc - right and right <= max_edge):
            continue
        pos = int(np.searchsorted(good_idx, i))
        if pos <= 0:
            out[:, i] = out[:, good_idx[0]]
        elif pos >= good_idx.size:
            out[:, i] = out[:, good_idx[-1]]
        else:
            il = int(good_idx[pos - 1])
            ir = int(good_idx[pos])
            if il == ir:
                out[:, i] = out[:, il]
            else:
                w = float(i - il) / float(ir - il)
                out[:, i] = (1.0 - w) * out[:, il] + w * out[:, ir]

    return out, bad


def _stabilize_edge_transients(
    data: np.ndarray,
    max_edge_traces: Optional[int] = None,
    low_ratio: float = 0.15,
    high_ratio: float = 1.45,
) -> Tuple[np.ndarray, int]:
    """
    Replace contiguous startup/ending transients at profile edges.

    Edge traces with extremely low/high energy versus nearby stable traces
    are substituted by the first/last stable trace.
    """
    arr = np.asarray(data, dtype=np.float32)
    n_smp, n_trc = arr.shape
    if n_trc < 6:
        return np.array(arr, copy=True), 0

    out = np.array(arr, copy=True)
    std = np.nanstd(arr, axis=0)

    edge_max = (
        int(max_edge_traces)
        if max_edge_traces is not None
        else max(6, min(64, n_trc // 12))
    )
    edge_max = max(2, min(edge_max, n_trc - 1))

    def _edge_ref(start: int, stop: int) -> Tuple[float, float]:
        x = std[start:stop]
        x = x[np.isfinite(x) & (x > 0)]
        if x.size == 0:
            return 0.0, 0.0
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        sig = 1.4826 * mad
        return med, sig

    fixed = 0

    # Left edge
    left_ref, left_sig = _edge_ref(edge_max, min(n_trc, edge_max * 3))
    if left_ref > 0:
        lo_thr = min(low_ratio * left_ref, max(0.0, left_ref - 6.0 * left_sig))
        hi_thr = max(high_ratio * left_ref, left_ref + 6.0 * left_sig)
        left = 0
        while left < edge_max and left < (n_trc - 1):
            s = float(std[left]) if np.isfinite(std[left]) else np.nan
            if (not np.isfinite(s)) or (s < lo_thr) or (s > hi_thr):
                left += 1
            else:
                break
        if 0 < left < n_trc:
            out[:, :left] = out[:, left:left + 1]
            fixed += left

    # Right edge
    right_ref, right_sig = _edge_ref(max(0, n_trc - 3 * edge_max), max(0, n_trc - edge_max))
    if right_ref > 0:
        lo_thr = min(low_ratio * right_ref, max(0.0, right_ref - 6.0 * right_sig))
        hi_thr = max(high_ratio * right_ref, right_ref + 6.0 * right_sig)
        right = 0
        while right < edge_max and right < (n_trc - 1):
            idx = n_trc - 1 - right
            s = float(std[idx]) if np.isfinite(std[idx]) else np.nan
            if (not np.isfinite(s)) or (s < lo_thr) or (s > hi_thr):
                right += 1
            else:
                break
        if 0 < right < n_trc:
            src = n_trc - right - 1
            out[:, n_trc - right:] = out[:, src:src + 1]
            fixed += right

    return out, fixed


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
        method:    Literal['threshold', 'max', 'xcorr'] = 'threshold',
        threshold: float = 0.1,
        offset_mode: Literal['scan', 'line'] = 'scan',
        max_shift_smp: Optional[int] = None,
    ) -> np.ndarray:
        data  = self._as_f32()
        n_smp = data.shape[0]
        n_trc = data.shape[1]
        data, bad_traces = _repair_bad_traces(data)
        repaired_bad = int(np.sum(bad_traces))
        tz    = np.zeros(n_trc, dtype=int)
        peak  = np.zeros(n_trc, dtype=np.float32)
        valid = np.zeros(n_trc, dtype=bool)

        # Goodman/Piro-style detection: trigger on the initial rise zone
        # (early samples only), then align scan-by-scan; noisy scans fall back
        # to the line reference.
        det_end   = max(16, int(round(0.25 * n_smp)))
        noise_end = max(8, min(det_end // 4, 64))
        smooth_w  = 5
        kernel    = np.ones(smooth_w, dtype=np.float32) / smooth_w
        ref_fixed: Optional[int] = None

        def _pick_first_arrival(trace: np.ndarray) -> Tuple[int, float, bool]:
            tr = np.abs(np.asarray(trace, dtype=np.float32))
            tr_s = np.convolve(tr, kernel, mode='same')
            tr_w = tr_s[:det_end]
            mx = float(np.max(tr_w))
            if mx <= 1e-12:
                return 0, 0.0, False

            n_win = tr_w[:noise_end]
            n_med = float(np.median(n_win))
            n_mad = float(np.median(np.abs(n_win - n_med)))
            n_sig = 1.4826 * n_mad
            thr_rel = float(threshold) * mx
            thr_noise = n_med + 5.0 * n_sig
            thr = max(thr_rel, thr_noise)

            idx = np.where(tr_w >= thr)[0]
            if len(idx) == 0:
                return 0, mx, False

            # Pick initial rise (onset), not just first high-amplitude
            # sample, to better match seismic-style time-zero.
            i_amp = int(idx[0])
            d = np.diff(tr_w, prepend=tr_w[0])
            d_noise = d[:noise_end]
            d_med = float(np.median(d_noise))
            d_mad = float(np.median(np.abs(d_noise - d_med)))
            d_sig = 1.4826 * d_mad
            d_thr = d_med + 4.0 * d_sig

            lo = max(1, i_amp - 6)
            seg = d[lo:i_amp + 1]
            if seg.size > 0:
                i_rise = int(lo + np.argmax(seg))
                if float(d[i_rise]) >= d_thr:
                    return i_rise, mx, True
            return i_amp, mx, True

        if method == 'xcorr':
            ref_trace = np.nanmedian(data, axis=1).astype(np.float32)
            if not np.all(np.isfinite(ref_trace)):
                ref_trace = np.nan_to_num(ref_trace, nan=0.0).astype(np.float32)

            ref_pick, ref_peak, ref_ok = _pick_first_arrival(ref_trace)
            ref = int(ref_pick if ref_ok else 0)
            ref_fixed = ref

            peak[:] = np.max(np.abs(data[:det_end, :]), axis=0).astype(np.float32)

            lag_limit = (
                int(np.clip(max_shift_smp, 1, n_smp - 1))
                if max_shift_smp is not None
                else max(8, int(round(0.12 * n_smp)))
            )
            corr_end = min(n_smp, max(64, 2 * det_end))
            lag_limit = min(lag_limit, corr_end - 1)
            lags = sp_signal.correlation_lags(corr_end, corr_end, mode='full')
            keep = np.abs(lags) <= lag_limit

            ref0 = np.nan_to_num(ref_trace[:corr_end], nan=0.0).astype(np.float32)
            ref0 = ref0 - float(np.mean(ref0))

            for i in range(n_trc):
                tr = np.asarray(data[:, i], dtype=np.float32)
                if not np.any(np.isfinite(tr)):
                    tz[i] = ref
                    valid[i] = False
                    continue

                pick_i, mx_i, ok_i = _pick_first_arrival(tr)
                peak[i] = mx_i
                valid[i] = bool(ok_i)

                tr0 = np.nan_to_num(tr[:corr_end], nan=0.0).astype(np.float32)
                tr0 = tr0 - float(np.mean(tr0))
                corr = sp_signal.correlate(ref0, tr0, mode='full', method='fft')
                csel = corr[keep]
                lsel = lags[keep]
                if csel.size > 0:
                    lag = int(lsel[int(np.argmax(csel))])
                else:
                    lag = 0

                # shift = lag => tz_used = ref - lag in the common shift logic
                tz[i] = int(ref - lag)
                if not ok_i:
                    tz[i] = ref

        else:
            for i in range(n_trc):
                tr = np.asarray(data[:, i], dtype=np.float32)
                tr_s = np.convolve(np.abs(tr), kernel, mode='same')
                tr_w = tr_s[:det_end]
                mx = float(np.max(tr_w))
                peak[i] = mx

                if mx <= 1e-12:
                    tz[i] = 0
                    continue

                if method == 'max':
                    tz[i] = int(np.argmax(tr_w))
                    valid[i] = True
                else:
                    pick_i, _, ok_i = _pick_first_arrival(tr)
                    tz[i] = int(pick_i)
                    valid[i] = bool(ok_i)

        valid_ref = valid & (peak > 0.0) & (~bad_traces)
        if ref_fixed is not None:
            ref = int(ref_fixed)
        elif np.any(valid_ref):
            # Stronger scans contribute more to line reference (line-by-line
            # fallback for low-SNR scans).
            w = np.square(peak[valid_ref]).astype(np.float64)
            ref = _weighted_median_int(tz[valid_ref], w)
        else:
            ref = int(np.median(tz))

        tz_used = tz.copy()
        tz_used[~valid_ref] = ref
        if offset_mode == 'line':
            # Line-by-line correction: apply one common offset to the full line.
            tz_used[:] = ref
        else:
            # Scan-by-scan correction with mild trace-wise smoothing.
            tz_used = _rolling_median_1d(tz_used.astype(np.int32), window=7).astype(int)

        if max_shift_smp is None:
            if method == 'xcorr':
                auto_max_shift = max(8, int(round(0.12 * n_smp)))
            elif np.any(valid_ref):
                q5, q95 = np.percentile(tz_used[valid_ref], [5, 95])
                spread = max(0.0, float(q95 - q5))
                auto_max_shift = int(np.clip(1.5 * spread + 4.0, 8, 0.12 * n_smp))
            else:
                auto_max_shift = max(8, int(round(0.08 * n_smp)))
        else:
            auto_max_shift = int(np.clip(max_shift_smp, 1, n_smp - 1))
        max_shift = (
            auto_max_shift
        )

        raw_shifts = ref - tz_used
        # Outlier traces (wrong first-break picks) are left unshifted to
        # avoid full-height black/white columns from extreme padding.
        shifts = np.where(np.abs(raw_shifts) <= max_shift, raw_shifts, 0).astype(int)
        outlier_count = int(np.sum(np.abs(raw_shifts) > max_shift))
        out = np.empty_like(data)

        for i in range(n_trc):
            shift = int(shifts[i])
            if shift > 0:
                out[shift:, i] = data[:n_smp - shift, i]
                out[:shift, i] = data[0, i]
            elif shift < 0:
                out[:n_smp + shift, i] = data[-shift:, i]
                out[n_smp + shift:, i] = data[-1, i]
            else:
                out[:, i] = data[:, i]

        # Truncate to common valid depth interval across all scans to remove
        # padding artifacts introduced by per-trace shifts.
        # Also force the aligned reference arrival to sample 0 (time-zero).
        pad_top = int(np.max(np.maximum(shifts, 0)))
        crop_bot = int(np.max(np.maximum(-shifts, 0)))
        crop_top = max(ref, pad_top)
        if (n_smp - crop_top - crop_bot) >= 8:
            out = out[crop_top:n_smp - crop_bot, :]
        else:
            # Fallback if overlap gets too small
            crop = max(0, min(ref, max_shift))
            if crop > 0 and (out.shape[0] - crop) >= 8:
                out = out[crop:, :]
            crop_top, crop_bot = crop, 0

        LOG.debug(
            f'Time-zero: method={method} mode={offset_mode} ref={ref} '
            f'tz_range=[{tz.min()},{tz.max()}] '
            f'valid_ref={int(np.sum(valid_ref))}/{n_trc} '
            f'raw_shift_range=[{raw_shifts.min()},{raw_shifts.max()}] '
            f'applied_shift_range=[{shifts.min()},{shifts.max()}] '
            f'max_shift={max_shift} outliers={outlier_count} '
            f'bad_traces={repaired_bad} '
            f'cropped_top={crop_top} cropped_bottom={crop_bot} '
            f'new_shape={out.shape}'
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

    def apply_top_mute(self, mute_ns: float = 0.0) -> np.ndarray:
        """
        Zero initial samples (air/direct-wave zone) before downstream filters.
        """
        if mute_ns <= 0.0:
            return self.processed_data

        data = self._as_f32()
        n_smp = data.shape[0]
        mute_smp = int(max(0, round(float(mute_ns) / self.sampling_time_ns)))
        mute_smp = min(mute_smp, n_smp)
        if mute_smp <= 0:
            return self.processed_data

        data[:mute_smp, :] = 0.0
        LOG.debug(f'Top mute: mute_ns={mute_ns:.2f} -> mute_smp={mute_smp}')
        self._store(data)
        return self.processed_data

    def apply_static_correction(
        self,
        elevations_m: np.ndarray,
        velocity_m_ns: float = 0.10,
        reference: Literal['max', 'median', 'mean'] = 'max',
        two_way: bool = False,
        max_shift_smp: Optional[int] = None,
    ) -> np.ndarray:
        """
        Topographic static correction from per-trace elevations.

        Args:
            elevations_m: Elevation vector (len ~= n_traces), in meters.
            velocity_m_ns: EM velocity for time shift conversion [m/ns].
            reference: Datum for correction ('max' recommended for archaeology).
            two_way: If True use 2*dz/v; if False use dz/v.
            max_shift_smp: Optional hard clamp on absolute sample shift.
        """
        data = self._as_f32()
        n_smp, n_trc = data.shape

        elev = np.asarray(elevations_m, dtype=np.float64).ravel()
        if elev.size == 0:
            LOG.warning('Static correction: empty elevation vector, skipping')
            return self.processed_data

        if elev.size != n_trc:
            x_src = np.linspace(0.0, 1.0, elev.size, dtype=np.float64)
            x_dst = np.linspace(0.0, 1.0, n_trc, dtype=np.float64)
            good = np.isfinite(elev)
            if int(np.sum(good)) < 2:
                LOG.warning(
                    f'Static correction: invalid elevations ({int(np.sum(good))}/{elev.size}), skipping'
                )
                return self.processed_data
            elev = np.interp(x_dst, x_src[good], elev[good])

        good = np.isfinite(elev)
        if int(np.sum(good)) < 2:
            LOG.warning(
                f'Static correction: not enough finite elevations ({int(np.sum(good))}/{n_trc}), skipping'
            )
            return self.processed_data

        med = float(np.median(elev[good]))
        elev = np.where(good, elev, med)

        ref_mode = str(reference).lower()
        if ref_mode == 'mean':
            z_ref = float(np.mean(elev))
        elif ref_mode == 'median':
            z_ref = float(np.median(elev))
        else:
            z_ref = float(np.max(elev))

        dz = z_ref - elev
        dt_ns = dz / max(float(velocity_m_ns), 1e-9)
        if two_way:
            dt_ns = 2.0 * dt_ns

        shifts = np.rint(dt_ns / max(self.sampling_time_ns, 1e-9)).astype(np.int32)

        if max_shift_smp is None:
            max_shift = max(4, int(0.30 * n_smp))
        else:
            max_shift = int(np.clip(max_shift_smp, 1, n_smp - 1))
        shifts = np.clip(shifts, -max_shift, max_shift)

        out = np.empty_like(data)
        for i in range(n_trc):
            s = int(shifts[i])
            if s > 0:
                out[s:, i] = data[:n_smp - s, i]
                out[:s, i] = data[0, i]
            elif s < 0:
                out[:n_smp + s, i] = data[-s:, i]
                out[n_smp + s:, i] = data[-1, i]
            else:
                out[:, i] = data[:, i]

        LOG.debug(
            f'Static correction: ref={ref_mode} z_ref={z_ref:.3f} m '
            f'vel={velocity_m_ns:.4f} m/ns two_way={two_way} '
            f'shift_range=[{int(np.min(shifts))},{int(np.max(shifts))}] smp'
        )
        self._store(out)
        return self.processed_data

    # ------------------------------------------------------------------
    # 3. Background Removal
    # ------------------------------------------------------------------

    def remove_background(
        self,
        method:        Literal['mean', 'median', 'trimmed'] = 'mean',
        rolling:       bool = False,
        window_traces: int  = 50,
        trim_percent:  float = 10.0,
    ) -> np.ndarray:
        data = self._as_f32()
        data, bad_low = _repair_bad_traces(
            data,
            rel_std_thr=0.02,
            max_bad_frac=0.35,
            max_edge_frac=0.10,
        )
        n_smp, n_trc = data.shape
        w_trc = max(1, int(window_traces))
        if rolling and (w_trc % 2 == 0):
            w_trc += 1
        data, edge_fixed = _stabilize_edge_transients(
            data,
            max_edge_traces=max(6, min(64, w_trc)),
            low_ratio=0.15,
            high_ratio=1.45,
        )

        tr_std = np.nanstd(data, axis=0)
        finite_std = np.isfinite(tr_std)
        med_std = float(np.median(tr_std[finite_std])) if np.any(finite_std) else 0.0
        std_thr = max(1e-9, 0.02 * med_std) if med_std > 0 else 0.0
        valid_cols = finite_std & (tr_std > std_thr)

        # Exclude extreme high-energy scans from background model only:
        # these are often startup transients or very strong local targets.
        valid_model = valid_cols.copy()
        if np.any(valid_cols):
            x = tr_std[valid_cols]
            med_e = float(np.median(x))
            mad_e = float(np.median(np.abs(x - med_e)))
            sig_e = 1.4826 * mad_e
            if sig_e > 1e-9:
                hi_thr = med_e + 8.0 * sig_e
                cand = valid_cols & (tr_std <= hi_thr)
                # Keep robust clipping only if enough columns remain.
                if int(np.sum(cand)) >= max(8, int(0.25 * np.sum(valid_cols))):
                    valid_model = cand

        if rolling:
            out = np.empty_like(data)
            hw  = w_trc // 2
            for i in range(n_trc):
                if n_trc <= w_trc:
                    lo, hi = 0, n_trc
                else:
                    lo = i - hw
                    hi = i + hw + 1
                    if lo < 0:
                        lo, hi = 0, w_trc
                    elif hi > n_trc:
                        lo, hi = n_trc - w_trc, n_trc
                idx = np.arange(lo, hi, dtype=int)
                idx = idx[valid_model[idx]]
                if idx.size < 3:
                    idx = np.arange(lo, hi, dtype=int)
                win = data[:, idx]
                bg = _background_stat_axis1(
                    win,
                    method=method,
                    trim_percent=trim_percent,
                )
                bg = np.where(np.isfinite(bg), bg, 0.0)
                out[:, i] = data[:, i] - bg
        else:
            core = data[:, valid_model] if np.any(valid_model) else data
            bg = _background_stat_axis1(
                core,
                method=method,
                trim_percent=trim_percent,
            )
            bg = np.where(np.isfinite(bg), bg, 0.0)
            out = data - bg[:, np.newaxis]

        LOG.debug(
            f'Background removal: method={method} rolling={rolling} '
            f'window_traces={w_trc} trim={trim_percent:.1f}% '
            f'valid_cols={int(np.sum(valid_cols))}/{n_trc} '
            f'model_cols={int(np.sum(valid_model))}/{n_trc} '
            f'low_energy_fixed={int(np.sum(bad_low))} '
            f'edge_transient_fixed={edge_fixed}'
        )
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
        mode:      Literal['gpr', 'butter'] = 'gpr',
    ) -> np.ndarray:
        data  = self._as_f32()
        n_smp = data.shape[0]

        if high_freq <= low_freq:
            LOG.warning(
                f'Bandpass: invalid range {low_freq}-{high_freq} MHz, skipping'
            )
            return self.processed_data

        if mode == 'gpr':
            # Matlab-compatible GPR trapezoidal bandpass (ImaGIN-style):
            # [f1 f2] cosine ramp up, [f2 f3] passband, [f3 f4] ramp down.
            dt_s = self.sampling_time_ns * 1e-9
            f2 = float(low_freq) * 1e6
            f3 = float(high_freq) * 1e6
            bw = max(f3 - f2, 1.0)
            f1 = max(1.0, f2 - bw / 8.0)
            f4 = f3 + bw / 4.0

            n_fft = 2 * n_smp
            freqs = np.fft.rfftfreq(n_fft, d=dt_s)
            H = np.zeros_like(freqs, dtype=np.float32)

            m = (freqs >= f1) & (freqs < f2)
            if np.any(m):
                x = (freqs[m] - f1) / max(f2 - f1, 1.0)
                H[m] = 0.5 * (1.0 - np.cos(np.pi * x))

            m = (freqs >= f2) & (freqs <= f3)
            H[m] = 1.0

            m = (freqs > f3) & (freqs <= f4)
            if np.any(m):
                x = (freqs[m] - f3) / max(f4 - f3, 1.0)
                H[m] = 0.5 * (1.0 + np.cos(np.pi * x))

            # Keep order control meaningful in GUI:
            # order>4 sharper ramps, order<4 softer ramps.
            expo = max(0.25, float(order) / 4.0)
            H = np.power(H, expo).astype(np.float32)

            data_pad = np.vstack([data, np.zeros_like(data)])
            F = np.fft.rfft(data_pad, axis=0)
            out_pad = np.fft.irfft(F * H[:, np.newaxis], n=n_fft, axis=0)
            out = out_pad[:n_smp, :].astype(np.float32)
            LOG.debug(
                f'Bandpass(gpr): f1={f1/1e6:.1f} f2={f2/1e6:.1f} '
                f'f3={f3/1e6:.1f} f4={f4/1e6:.1f} MHz order={order}'
            )
        else:
            fs = self.sampling_freq_mhz
            nyq = fs / 2.0
            lo = float(np.clip(low_freq / nyq, 1e-4, 0.9999))
            hi = float(np.clip(high_freq / nyq, 1e-4, 0.9999))
            if lo >= hi:
                LOG.warning(f'Bandpass(butter): lo={lo:.4f} >= hi={hi:.4f}, skipping')
                return self.processed_data
            sos = sp_signal.butter(order, [lo, hi], btype='band', output='sos')
            data_pad = np.vstack([data, np.zeros_like(data)])
            out_pad = sp_signal.sosfiltfilt(sos, data_pad, axis=0)
            out = out_pad[:n_smp, :].astype(np.float32)
            LOG.debug(f'Bandpass(butter): {low_freq}-{high_freq} MHz order={order}')

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
        gain_type:  Literal['agc', 'sec', 'exp', 'linear', 'db'] = 'agc',
        factor:     float = 20.0,
        alpha:      float = 0.5,
        t_start_ns: float = 5.0,
        window_ns:  float = 25.0,
        agc_start_ns: float = 0.0,
        gain_db_points: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        data  = self._as_f32()
        n_smp = data.shape[0]
        dt    = self.sampling_time_ns
        t     = np.arange(n_smp, dtype=np.float32) * dt

        if gain_type == 'agc':
            win = max(1, int(window_ns / dt))
            gate_smp = int(max(0, round(float(agc_start_ns) / dt)))
            gate_smp = min(gate_smp, n_smp)
            out = np.empty_like(data)
            cap = max(1.0, float(factor))
            for i in range(data.shape[1]):
                tr    = data[:, i]
                env = np.convolve(np.abs(tr), np.ones(win) / win, mode='same')
                # Noise floor from deep half avoids direct-wave dominance.
                deep_start = n_smp // 2
                noise = float(np.median(np.abs(tr[deep_start:]))) + 1e-10
                env = np.where(env < noise, noise, env)

                # Normalize AGC around unit median gain and clamp amplification.
                env_ref = float(np.median(env)) + 1e-10
                g = env_ref / env
                g = np.clip(g, 1.0 / cap, cap)
                if gate_smp > 0:
                    # Do not amplify the direct-wave zone.
                    g[:gate_smp] = 1.0
                out[:, i] = tr * g

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

        elif gain_type == 'db':
            if gain_db_points is None:
                LOG.warning('Gain(db): gain_db_points is None, skipping')
                return self.processed_data
            gp = np.asarray(gain_db_points, dtype=np.float32).ravel()
            if gp.size < 2:
                LOG.warning('Gain(db): need at least 2 dB points, skipping')
                return self.processed_data
            x_src = np.linspace(0, n_smp - 1, gp.size, dtype=np.float32)
            x_dst = np.arange(n_smp, dtype=np.float32)
            g_db = np.interp(x_dst, x_src, gp).astype(np.float32)
            g = np.power(10.0, g_db / 20.0).astype(np.float32)
            out = data * g[:, np.newaxis]

        else:  # linear
            t_s   = np.maximum(t - t_start_ns, 0.0)
            t_max = float(t[-1] - t_start_ns) if t[-1] > t_start_ns else 1.0
            g     = 1.0 + (factor - 1.0) * (t_s / t_max)
            g     = np.clip(g, 1.0, factor)
            out   = data * g[:, np.newaxis]

        LOG.debug(
            f'Gain: type={gain_type} factor={factor} alpha={alpha} '
            f'agc_start_ns={agc_start_ns:.2f}'
        )
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
