# Automated Fetal ECG Monitoring

Streamlit app for the fetal ECG project: upload an abdominal ECG recording (CSV), the app removes the
mother's ECG, detects fetal beats (ML model or simple rules), and plots the fetal heart rate.

## Files
- `app.py` - the Streamlit interface
- `fecg_pipeline.py` - the signal-processing + ML pipeline (same logic as the notebook)
- `fecg_model.joblib` - trained model saved by the last cell of your notebook (put it next to `app.py`)
- `requirements.txt`

## Run on your computer
```
pip install -r requirements.txt
streamlit run app.py
```
It opens at http://localhost:8501. Upload one of the dataset CSV files.

## Put it online for free (Streamlit Community Cloud)
1. Create a GitHub repo and upload all the files above (including `fecg_model.joblib`).
2. Go to share.streamlit.io, sign in with GitHub, choose the repo and `app.py`, then Deploy.

If the model file is missing or can't be loaded, the app still works in "Rules only" mode.

*Educational prototype - not a medical device.*
