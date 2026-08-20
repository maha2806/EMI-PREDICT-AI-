"""
EMIPredict AI - Model Performance Page

Shows how the deployed models were evaluated - metric comparison across
every model that was trained, confusion matrix, actual-vs-predicted, and
feature importance. All figures come straight from the model-development
notebook's test-set evaluation.
"""

import streamlit as st
from pathlib import Path

st.set_page_config(

    page_title = "Model Performance - EMIPredict AI",

    page_icon = ":material/insights:",

    layout = "wide"

)

ASSETS_DIR = Path(__file__).resolve().parent.parent / 'assets'

st.title(":material/insights: Model Performance")

st.caption("All metrics below are computed on a held-out test set the models never saw during training or tuning.")

tab_clf, tab_reg = st.tabs([":material/verified: Eligibility Classifier", ":material/calculate: EMI Regressor"])

with tab_clf:

    st.subheader(":material/compare_arrows: Model Comparison")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", "97.0%", icon = ":material/target:")

    c2.metric("Precision (macro)", "0.865", icon = ":material/adjust:")

    c3.metric("Recall (macro)", "0.949", icon = ":material/replay:")

    c4.metric("F1-score (macro)", "0.897", icon = ":material/balance:")

    st.image(str(ASSETS_DIR / 'model_comparison_clf.png'), use_container_width = True)

    st.caption(
        
        "XGBoost was the best-performing model on the held-out test set across every metric, "
        "and is the classifier deployed in this app."
    
    )

    c1, c2 = st.columns(2)

    with c1:

        st.subheader(":material/grid_on: Confusion Matrix")

        st.image(str(ASSETS_DIR / 'cm_xgboost.png'), use_container_width = True)

    with c2:

        st.subheader(":material/ssid_chart: Feature Importance")

        st.image(str(ASSETS_DIR / 'featimp_clf_xgb.png'), use_container_width = True)

    st.caption(
        
        "The engineered ratio features (EMI-to-income, affordability ratio, disposable income) "
        "dominate alongside existing_loans - direct confirmation that the feature engineering in the "
        "notebook is what drives the model's decisions, not any single raw input."
    
    )

with tab_reg:

    st.subheader(":material/compare_arrows: Model Comparison")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("RMSE", "\u20b9625", icon = ":material/straighten:")

    c2.metric("MAE", "\u20b9229", icon = ":material/rule:")

    c3.metric("R\u00b2 Score", "0.9935", icon = ":material/analytics:")

    c4.metric("Target", "< \u20b92,000 RMSE", icon = ":material/flag:")

    st.image(str(ASSETS_DIR / 'model_comparison_reg.png'), use_container_width = True)

    st.caption(
        
        "XGBoost is the deployed regressor - it roughly halves the RMSE and MAE of the next-best "
        "model, Random Forest."
        
    )

    c1, c2 = st.columns(2)

    with c1:

        st.subheader(":material/scatter_plot: Actual vs Predicted")

        st.image(str(ASSETS_DIR / 'avp_xgboost.png'), use_container_width = True)

    with c2:

        st.subheader(":material/ssid_chart: Feature Importance")

        st.image(str(ASSETS_DIR / 'featimp_reg_xgb.png'), use_container_width = True)

st.divider()

st.subheader(":material/menu_book: Full Methodology")

st.write(

    "Data cleaning, feature engineering, hyperparameter tuning and every evaluation shown here are documented "
    "step by step in `notebooks/EMIPredict_AI_Model_Development.ipynb`, including the exact train/validation/test "
    "split and the reasoning behind each modeling decision."

)