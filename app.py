
import os
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

HF_TOKEN = os.environ.get('HF_TOKEN')
HF_MODEL_REPO = 'Gaurav328/predictive-maintenance-model'
DEFAULT_ARTIFACT = 'RandomForest_best.joblib'

st.title('DTIAS Predictive Maintenance – Inference')

@st.cache_resource
def load_model(artifact_name: str):
    p = hf_hub_download(repo_id=HF_MODEL_REPO, filename=artifact_name, token=HF_TOKEN)
    return joblib.load(p)

artifact = st.text_input('Model artifact name in HF Model Hub', DEFAULT_ARTIFACT)

feature_names = [
    'engine_rpm',
    'lub_oil_pressure', 'fuel_pressure', 'coolant_pressure',
    'lub_oil_temperature', 'coolant_temperature'
]

def predict_single(m):
    cols = st.columns(3)
    defaults = {{
        'engine_rpm': 1200,
        'lub_oil_pressure': 3.5,
        'fuel_pressure': 2.0,
        'coolant_pressure': 1.0,
        'lub_oil_temperature': 85.0,
        'coolant_temperature': 80.0
    }}
    row = {{}}
    for i, f in enumerate(feature_names):
        with cols[i%3]:
            row[f] = st.number_input(f, value=float(defaults[f]))
    if st.button('Predict (single)'):
        X = pd.DataFrame([row], columns=feature_names)
        try:
            proba = m.predict_proba(X)[:,1]
            st.metric('Fault Probability', float(proba[0]))
            st.metric('Predicted Class (1=faulty)', int((proba>=0.5)[0]))
        except Exception:
            pred = m.predict(X)
            st.metric('Predicted Class (1=faulty)', int(pred[0]))

def predict_batch(m):
    st.write('Upload CSV with columns: '+', '.join(feature_names))
    up = st.file_uploader('CSV', type=['csv'])
    if up is not None:
        df = pd.read_csv(up)
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            st.error(f'Missing columns: {missing}')
        else:
            try:
                proba = m.predict_proba(df[feature_names])[:,1]
                out = df.copy(); out['prob_faulty']=proba; out['pred']=(proba>=0.5).astype(int)
            except Exception:
                pred = m.predict(df[feature_names])
                out = df.copy(); out['pred']=pred
            st.dataframe(out.head())
            st.download_button('Download predictions', out.to_csv(index=False), file_name='predictions.csv')

if artifact:
    try:
        model = load_model(artifact)
        st.success('Model loaded from HF Model Hub')
        with st.expander('Single Prediction'):
            predict_single(model)
        with st.expander('Batch Prediction'):
            predict_batch(model)
    except Exception as e:
        st.error(f'Failed to load artifact {artifact}: {e}')
else:
    st.info('Enter an artifact name and provide HF_TOKEN env variable')
