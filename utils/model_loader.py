"""
EMIPredict AI - Model Loader

Loads the trained models and preprocessing objects once per session and
caches them, so every page can call get_models() cheaply instead of
re-reading the .joblib files from disk on every rerun.
"""

import joblib
import streamlit as st
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'


@st.cache_resource
def get_models():
    
    """Load every trained model and preprocessing object once, cached for the session."""

    classifier = joblib.load(MODELS_DIR / 'final_classifier.joblib')

    regressor = joblib.load(MODELS_DIR / 'final_regressor.joblib')

    scaler = joblib.load(MODELS_DIR / 'scaler.joblib')

    label_encoder = joblib.load(MODELS_DIR / 'label_encoder.joblib')

    feature_names = joblib.load(MODELS_DIR / 'feature_names.joblib')

    return {
        
        'classifier': classifier,
        'regressor': regressor,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': feature_names
        
    }


@st.cache_data
def get_dataset_sample(n = 20000):
    
    """Load a sample of the training dataset for the data insights page - a sample keeps page load fast without needing all 400K+ rows in memory."""

    import pandas as pd

    data_path = Path(__file__).resolve().parent.parent / 'data' / 'emi_prediction_dataset.csv'

    df = pd.read_csv(data_path, low_memory = False)

    if len(df) > n:

        df = df.sample(n, random_state = 42)

    return df
