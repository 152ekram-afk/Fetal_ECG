from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import fecg_pipeline as fp  # noqa: E402

st.set_page_config(page_title="Automated Fetal ECG Monitoring", page_icon="🫀", layout="wide")

MODEL_PATH = Path(__file__).parent / "fecg_model.joblib"


# ----------------------------------------------------------------------------- cached helpers
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, "No trained model found (fecg_model.joblib). Using rules-only detection."
    try:
        bundle = joblib.load(MODEL_PATH)
        if len(bundle["features"]) != len(fp.FEATURE_NAMES):
            return None, "Model was trained with a different feature set. Using rules-only detection."
        return bundle, None
    except Exception as e:  # e.g. scikit-learn version mismatch
        return None, f"Could not load the model ({e}). Using rules-only detection."


@st.cache_data(show_spinner="Filtering signal and removing maternal ECG...")
def process(raw: bytes, fs: int):
    return fp.process_record(raw, fs)


# ----------------------------------------------------------------------------- header
st.title("🫀 Automated Fetal ECG Monitoring")
st.caption("Extracts the fetal heartbeat from abdominal ECG recordings and estimates the fetal heart rate (FHR).")
st.warning("Educational prototype. It is **not** a medical device and must not be used for diagnosis "
           "or clinical decisions.")

bundle, model_msg = load_model()

# ----------------------------------------------------------------------------- sidebar
st.sidebar.header("1. Input")
upload = st.sidebar.file_uploader("ECG recording (.csv)", type=["csv"])
default_fs = int(bundle["fs"]) if bundle else 1000
fs = int(st.sidebar.number_input("Sampling rate (Hz)", min_value=100, max_value=5000,
                                 value=default_fs, step=50))
if bundle and fs != int(bundle["fs"]):
    st.sidebar.warning(f"The model was trained at {int(bundle['fs'])} Hz.")

st.sidebar.header("2. Detection method")
if model_msg:
    st.sidebar.info(model_msg)
options = ["Rules only (no ML)"]
if bundle:
    options.insert(0, "ML model (Random Forest)")
choice = st.sidebar.radio("Method", options)
mode = "ml" if choice.startswith("ML") else "rule"
thr = 0.5
if mode == "ml":
    start_thr = min(max(round(float(bundle["threshold"]) / 0.05) * 0.05, 0.05), 0.90)
    thr = st.sidebar.slider("ML confidence threshold", 0.05, 0.90, round(start_thr, 2), 0.05)

# ----------------------------------------------------------------------------- no file yet
if upload is None:
    st.info("Upload a recording in the sidebar to begin. The CSV needs the columns "
            "`Abdomen_1`, `Abdomen_2`, `Abdomen_3`, `Abdomen_4`. If it also has `Direct_1` "
            "(the direct fetal ECG), the app will compare its results against it.")
    st.stop()

try:
    proc = process(upload.getvalue(), fs)
except Exception as e:
    st.error(f"Could not process this file: {e}")
    st.stop()

fs = proc["fs"]
n = proc["n"]
dur = n / fs
model = bundle["model"] if (bundle and mode == "ml") else None
beats = fp.detect_beats(proc, mode, thr, model)
t_hr, hr = fp.calculate_hr(beats, fs)
valid_pct = 100 * len(hr) / max(len(beats) - 1, 1)

# ----------------------------------------------------------------------------- headline numbers
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Duration", f"{dur:.1f} s")
c2.metric("Fetal beats found", f"{len(beats)}")
c3.metric("Median FHR", f"{np.median(hr):.0f} BPM" if len(hr) else "n/a")
c4.metric("FHR range", f"{hr.min():.0f}-{hr.max():.0f}" if len(hr) else "n/a")
c5.metric("Clean intervals", f"{valid_pct:.0f}%")
st.caption("A typical baseline fetal heart rate is roughly 110-160 BPM. "
           "'Clean intervals' is the share of beat-to-beat intervals with a plausible heart rate; "
           "a low value means the detection is unreliable for this recording.")

tab_sig, tab_hr, tab_eval, tab_how = st.tabs(
    ["Signal and beats", "Heart rate", "Evaluation", "How it works"])

# ----------------------------------------------------------------------------- tab 1: signal
with tab_sig:
    col_a, col_b, col_c = st.columns(3)
    ch = col_a.selectbox("Abdominal channel", fp.ABD)
    clip = col_b.checkbox("Clip large artifacts for display", value=True)
    win = min(8.0, dur)
    start = 0.0
    if dur - win > 0.5:
        start = col_c.slider("Window start (s)", 0.0, float(dur - win), 0.0, 0.5)
    a, b = int(start * fs), int((start + win) * fs)
    t = np.arange(n) / fs

    def disp(x):
        return np.clip(x, -8, 8) if clip else x

    filt, res = disp(proc["filtered"][ch]), disp(proc["residual"][ch])
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)

    ax[0].plot(t[a:b], filt[a:b], lw=0.7)
    mp = proc["maternal"]
    mp = mp[(mp >= a) & (mp < b)]
    ax[0].plot(mp / fs, filt[mp], "rx", label="maternal beats")
    ax[0].set_title(f"{ch}: filtered signal (dominated by the mother's heartbeat)")
    ax[0].legend(loc="upper right")

    ax[1].plot(t[a:b], res[a:b], lw=0.7)
    ymin, ymax = ax[1].get_ylim()
    bb = beats[(beats >= a) & (beats < b)]
    ax[1].plot(bb / fs, np.full(len(bb), ymax * 0.92), "rv", label="detected fetal beats")
    if proc["ref"] is not None:
        rr_ = fp.trim_edges(proc["ref"], n, fs)
        rr_ = rr_[(rr_ >= a) & (rr_ < b)]
        ax[1].plot(rr_ / fs, np.full(len(rr_), ymin * 0.92), "g^", label="reference (Direct_1)")
    ax[1].set_title(f"{ch}: after maternal ECG removal")
    ax[1].set_xlabel("Time (s)")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ----------------------------------------------------------------------------- tab 2: heart rate
with tab_hr:
    if len(hr) < 2:
        st.info("Not enough clean beat intervals to draw a heart-rate curve for this recording.")
    else:
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.axhspan(110, 160, color="grey", alpha=0.12, label="typical baseline range (110-160)")
        ax.plot(t_hr, hr, "s-", ms=3, label="estimated FHR")
        if proc["ref"] is not None:
            rt, rh = fp.calculate_hr(fp.trim_edges(proc["ref"], n, fs), fs)
            if len(rh):
                ax.plot(rt, rh, "o-", ms=3, label="reference FHR (Direct_1)")
        ax.set_ylim(60, 240)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("BPM")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    table = fp.interval_table(beats, fs)
    st.download_button("Download beat intervals (CSV)", table.to_csv(index=False).encode(),
                       file_name="fetal_beat_intervals.csv", mime="text/csv")

# ----------------------------------------------------------------------------- tab 3: evaluation
with tab_eval:
    if proc["ref"] is None:
        st.info("This file has no `Direct_1` column, so there is nothing to compare against. "
                "Upload a recording that includes the direct fetal ECG to see accuracy metrics.")
    else:
        ref = fp.trim_edges(proc["ref"], n, fs)
        methods = {"Baseline: every candidate (no filtering)": ("union", None),
                   "Rules only (>=2 channels agree)": ("rule", None)}
        if bundle:
            methods["ML model (Random Forest)"] = ("ml", bundle["model"])
        rows = []
        for name, (m, mdl) in methods.items():
            bts = fp.detect_beats(proc, m, thr, mdl)
            met = fp.match_beats(bts, ref, fs)
            rows.append({"Method": name,
                         "Sensitivity": round(met["Sensitivity"], 3),
                         "Precision": round(met["Precision"], 3),
                         "F1": round(met["F1"], 3),
                         "Median FHR error (BPM)": round(fp.median_fhr_error(bts, ref, fs), 2),
                         "Beats": len(bts)})
        st.dataframe(pd.DataFrame(rows).set_index("Method"))
        st.caption("A detected beat counts as correct if it is within 50 ms of a Direct_1 beat. "
                   "If this recording was used to train the model, these numbers are optimistic - "
                   "the honest figures are the leave-one-recording-out results from the notebook.")

# ----------------------------------------------------------------------------- tab 4: explanation
with tab_how:
    st.markdown(
        """
**Pipeline**
1. **Band-pass filter (8-45 Hz)** removes baseline drift and high-frequency noise.
2. **Maternal ECG removal:** the mother's beats are located and a matching template is subtracted
   from each one, leaving the much smaller fetal signal.
3. **Candidate peaks** are found in each of the 4 abdominal channels.
4. **Features** describe each candidate (amplitude, width, timing between peaks, distance to the
   nearest maternal beat, ...).
5. **Machine learning:** a Random Forest, trained on recordings where the direct fetal ECG
   (Direct_1) marks the true beats, scores each candidate as "fetal beat" or "not".
6. **Consensus:** scores are averaged across the 4 channels, and two beats closer than 300 ms are
   not allowed (the weaker one is dropped).
7. **Fetal heart rate** = 60 / (time between consecutive beats).

**Limitations:** trained on only a few short recordings; results can vary a lot between recordings;
not validated clinically.
        """
    )
