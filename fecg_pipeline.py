"""
Fetal ECG pipeline - the same logic as the Colab notebook, packaged for the Streamlit app.

Steps: load CSV -> band-pass filter -> remove maternal ECG -> find candidate peaks
-> features -> (ML or rules) beat selection -> fetal heart rate.
"""
from io import BytesIO

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks, peak_widths

CHANNELS = ["Direct_1", "Abdomen_1", "Abdomen_2", "Abdomen_3", "Abdomen_4"]
ABD = CHANNELS[1:]

FEATURE_NAMES = (
    ["amp", "abs_amp"]
    + [f"{stat}_{w}ms" for w in (10, 20, 40) for stat in ("maxabs", "meanabs", "std", "ptp", "rms")]
    + ["prominence", "width", "rr_prev", "rr_next", "rr_diff", "dist_maternal", "channel"]
)


# ----------------------------------------------------------------------------- loading
def load_ecg(raw_bytes):
    """Read the dataset CSV. Needs the 4 abdominal channels; Direct_1 is optional."""
    found = []
    df = None
    for skip in ([1], None):          # the Kaggle files have a units row under the header
        try:
            df = pd.read_csv(BytesIO(raw_bytes), skiprows=skip, quotechar="'")
        except Exception:
            continue
        df.columns = [str(c).strip().strip("'\"") for c in df.columns]
        found = list(df.columns)
        if set(ABD) <= set(found):
            break
    else:
        raise ValueError(f"CSV must contain the columns {ABD} (Direct_1 is optional). "
                         f"Columns found: {found[:8]}")

    use = [c for c in CHANNELS if c in df.columns]
    out = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in use})
    out = out.interpolate(limit_direction="both")
    if out.isna().any().any():
        raise ValueError("The file contains invalid (non-numeric) ECG samples.")
    return out


# ----------------------------------------------------------------------------- filtering
def robust_scale(x):
    med = np.median(x)
    s = 1.4826 * np.median(np.abs(x - med))
    if s < 1e-8:
        s = np.std(x)
    if s < 1e-8:
        s = 1.0
    return (x - med) / s


def preprocess_signal(x, fs):
    nyq = fs / 2.0
    high = min(45.0, 0.9 * nyq)
    if high <= 8.0:
        raise ValueError("Sampling rate is too low for this pipeline (need > ~100 Hz).")
    sos = butter(3, [8.0, high], btype="bandpass", fs=fs, output="sos")
    return robust_scale(sosfiltfilt(sos, np.asarray(x, dtype=float)))


# ----------------------------------------------------------------------------- maternal ECG
def maternal_peaks(channels, fs):
    energy = sum(np.abs(channels[ch]) for ch in ABD)
    win = max(int(0.03 * fs), 1)
    env = np.convolve(energy, np.ones(win) / win, mode="same")
    peaks, _ = find_peaks(env, height=0.5 * np.percentile(env, 99.5), distance=int(0.4 * fs))
    return peaks


def cancel_maternal(x, m_peaks, fs, n_neighbors=10):
    pre, post = int(0.12 * fs), int(0.25 * fs)
    x = np.asarray(x, dtype=float)
    keep = np.array([p for p in m_peaks if p - pre >= 0 and p + post < len(x)], dtype=int)
    if len(keep) < 3:
        return x.copy()
    segs = np.stack([x[p - pre:p + post] for p in keep])
    out = x.copy()
    for i, p in enumerate(keep):
        lo, hi = max(0, i - n_neighbors // 2), min(len(keep), i + n_neighbors // 2 + 1)
        idx = [j for j in range(lo, hi) if j != i]
        tmpl = segs[idx].mean(axis=0)
        denom = np.dot(tmpl, tmpl)
        if denom < 1e-12:
            continue
        out[p - pre:p + post] = segs[i] - (np.dot(segs[i], tmpl) / denom) * tmpl
    return out


# ----------------------------------------------------------------------------- candidates
def detect_candidates(x, fs):
    a = np.abs(x)
    peaks, props = find_peaks(a, distance=int(0.22 * fs), prominence=0.8)
    widths = peak_widths(a, peaks, rel_height=0.5)[0] if len(peaks) else np.array([])
    return peaks, props["prominences"], widths


def build_features(x, m_peaks, ch_idx, fs):
    peaks, prom, widths = detect_candidates(x, fs)
    if len(peaks) == 0:
        return np.empty((0, len(FEATURE_NAMES))), np.array([], dtype=int)

    rr = np.diff(peaks) / fs
    med_rr = np.median(rr) if len(rr) else 0.4
    rr_prev, rr_next = np.r_[med_rr, rr], np.r_[rr, med_rr]
    margin = int(0.04 * fs)

    rows, kept = [], []
    for k, p in enumerate(peaks):
        if p < margin or p >= len(x) - margin:
            continue
        row = [x[p], abs(x[p])]
        for w_ms in (10, 20, 40):
            w = int(w_ms * fs / 1000)
            seg = x[p - w:p + w + 1]
            row += [np.max(np.abs(seg)), np.mean(np.abs(seg)), np.std(seg),
                    np.ptp(seg), np.sqrt(np.mean(seg ** 2))]
        dist_m = np.min(np.abs(m_peaks - p)) / fs if len(m_peaks) else 1.0
        row += [prom[k], widths[k], rr_prev[k], rr_next[k],
                abs(rr_prev[k] - rr_next[k]), min(dist_m, 1.0), ch_idx]
        rows.append(row)
        kept.append(p)
    if not rows:
        return np.empty((0, len(FEATURE_NAMES))), np.array([], dtype=int)
    return np.asarray(rows, dtype=float), np.asarray(kept, dtype=int)


def process_record(raw_bytes, fs):
    """Everything that does not depend on the chosen detection method."""
    fs = int(fs)
    df = load_ecg(raw_bytes)
    n = len(df)
    if n < 3 * fs:
        raise ValueError(f"Recording is too short ({n / fs:.1f} s). Need at least 3 s "
                         f"at {fs} Hz - check the sampling rate.")

    filtered = {ch: preprocess_signal(df[ch].to_numpy(), fs) for ch in df.columns}
    mat = maternal_peaks(filtered, fs)
    residual = {ch: robust_scale(cancel_maternal(filtered[ch], mat, fs)) for ch in ABD}

    candidates = {}
    for ci, ch in enumerate(ABD):
        X, pk = build_features(residual[ch], mat, ci, fs)
        candidates[ch] = (X, pk)

    ref = None
    if "Direct_1" in filtered:
        ref = detect_candidates(filtered["Direct_1"], fs)[0]

    return dict(fs=fs, n=n, filtered=filtered, residual=residual, maternal=mat,
                candidates=candidates, ref=ref)


# ----------------------------------------------------------------------------- beat selection
def cluster_candidates(pos, prob, fs):
    if len(pos) == 0:
        return np.array([], int), np.array([]), np.array([], int)
    gap = int(0.08 * fs)
    order = np.argsort(pos)
    pos, prob = pos[order], prob[order]
    starts = np.r_[0, np.where(np.diff(pos) > gap)[0] + 1]
    ends = np.r_[starts[1:], len(pos)]
    bp = np.array([int(np.median(pos[s:e])) for s, e in zip(starts, ends)])
    sp = np.array([prob[s:e].sum() for s, e in zip(starts, ends)])
    nm = np.array([e - s for s, e in zip(starts, ends)])
    return bp, sp, nm


def apply_refractory(beats, scores, fs):
    min_gap = int(0.30 * fs)
    keep = []
    for i in np.argsort(beats):
        if keep and beats[i] - beats[keep[-1]] < min_gap:
            if scores[i] > scores[keep[-1]]:
                keep[-1] = i
        else:
            keep.append(i)
    return beats[keep]


def select_beats(pos, prob, thr, n_samples, fs, mode="ml"):
    """mode: 'ml' (probability averaged over channels), 'rule' (>=2 channels agree), 'union' (everything)."""
    bp, sum_p, n_mem = cluster_candidates(pos, prob, fs)
    if len(bp) == 0:
        return np.array([], dtype=int)
    if mode == "ml":
        score = np.minimum(sum_p / len(ABD), 1.0)
        keep = score >= thr
    elif mode == "rule":
        score = n_mem / len(ABD)
        keep = n_mem >= 2
    else:
        score = n_mem / len(ABD)
        keep = np.ones(len(bp), bool)
    bp, score = bp[keep], score[keep]
    edge = int(0.5 * fs)
    m = (bp >= edge) & (bp < n_samples - edge)
    bp, score = bp[m], score[m]
    if len(bp) == 0:
        return np.array([], dtype=int)
    return bp if mode == "union" else apply_refractory(bp, score, fs)


def detect_beats(proc, mode="rule", thr=0.5, model=None):
    pos, prob = [], []
    for ch in ABD:
        X, pk = proc["candidates"][ch]
        if len(pk) == 0:
            continue
        pos.append(pk)
        prob.append(model.predict_proba(X)[:, 1] if (mode == "ml" and model is not None)
                    else np.zeros(len(pk)))
    if not pos:
        return np.array([], dtype=int)
    return select_beats(np.concatenate(pos), np.concatenate(prob), thr, proc["n"], proc["fs"], mode)


# ----------------------------------------------------------------------------- heart rate & metrics
def calculate_hr(beats, fs):
    beats = np.sort(np.asarray(beats, dtype=int))
    if len(beats) < 2:
        return np.array([]), np.array([])
    rr = np.diff(beats) / fs
    hr = 60.0 / rr
    t_mid = (beats[:-1] + beats[1:]) / (2 * fs)
    valid = (hr >= 80) & (hr <= 220)
    return t_mid[valid], hr[valid]


def interval_table(beats, fs):
    beats = np.sort(np.asarray(beats, dtype=int))
    if len(beats) < 2:
        return pd.DataFrame(columns=["beat_time_s", "interval_s", "fhr_bpm", "plausible"])
    rr = np.diff(beats) / fs
    hr = 60.0 / rr
    return pd.DataFrame({"beat_time_s": np.round(beats[1:] / fs, 3), "interval_s": np.round(rr, 3),
                         "fhr_bpm": np.round(hr, 1), "plausible": (hr >= 80) & (hr <= 220)})


def match_beats(pred, ref, fs, tol_s=0.05):
    tol = int(tol_s * fs)
    pred, ref = np.sort(np.asarray(pred)), np.sort(np.asarray(ref))
    used = np.zeros(len(pred), bool)
    tp = 0
    for r in ref:
        if len(pred) == 0:
            break
        j = int(np.argmin(np.abs(pred - r)))
        if abs(pred[j] - r) <= tol and not used[j]:
            used[j] = True
            tp += 1
    fn, fp = len(ref) - tp, len(pred) - tp
    sens = tp / (tp + fn) if tp + fn else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * sens * ppv / (sens + ppv) if sens + ppv else 0.0
    return dict(Sensitivity=sens, Precision=ppv, F1=f1, TP=tp, FP=fp, FN=fn)


def median_fhr_error(beats_est, beats_ref, fs):
    _, h_e = calculate_hr(beats_est, fs)
    _, h_r = calculate_hr(beats_ref, fs)
    if len(h_e) == 0 or len(h_r) == 0:
        return float("nan")
    return float(abs(np.median(h_e) - np.median(h_r)))


def trim_edges(beats, n, fs):
    beats = np.asarray(beats, dtype=int)
    edge = int(0.5 * fs)
    return beats[(beats >= edge) & (beats < n - edge)]
