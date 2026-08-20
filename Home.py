"""
EMIPredict AI - Home

Landing page for the Streamlit application. Gives a quick overview of the
project and the model performance headline numbers, then points the user
to the three working pages in the sidebar.
"""

import streamlit as st
from utils.model_loader import get_models

st.set_page_config(

    page_title = "EMIPredict AI",

    page_icon = ":material/account_balance:",

    layout = "wide"

)

models = get_models()

st.title(":material/account_balance: EMIPredict AI")

st.subheader("Intelligent Financial Risk Assessment Platform")

st.markdown(
    
"""
EMIPredict AI gives loan officers and applicants an instant, data-backed
first pass on an EMI application - an eligibility decision and a safe
monthly EMI estimate, both generated in real time from a model trained on
404,800 historical applications.

**Use the sidebar to get started:**
- :material/verified: **Eligibility Prediction** - find out if an applicant is Eligible, High Risk, or Not Eligible
- :material/calculate: **EMI Calculator** - estimate the maximum monthly EMI an applicant can safely afford
- :material/insights: **Model Performance** - see how the underlying models were evaluated
- :material/bar_chart: **Data Insights** - explore trends across the historical applicant dataset
"""

)

st.divider()

st.subheader(":material/dashboard: Model Snapshot")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Classifier", type(models['classifier']).__name__, icon = ":material/account_tree:")

c2.metric("Test Accuracy", "97.0%", icon = ":material/target:")

c3.metric("Regressor", type(models['regressor']).__name__, icon = ":material/trending_up:")

c4.metric("Regression R\u00b2", "0.99", icon = ":material/analytics:")

st.caption(

"Both figures are from the held-out test set in the model-development notebook "
"(notebooks/EMIPredict_AI_Model_Development.ipynb) - see the Model Performance page for the full comparison."

)

st.divider()

st.subheader(":material/route: How It Works")

c1, c2, c3 = st.columns(3)

with c1:

    st.markdown(":material/edit_note: **1. Enter applicant details**")

    st.caption("Income, expenses, credit history, and the loan being requested.")

with c2:

    st.markdown(":material/bolt: **2. Get an instant decision**")

    st.caption("Eligibility with class probabilities, plus a safe-EMI estimate.")

with c3:

    st.markdown(":material/visibility: **3. See why**")

    st.caption("Feature importance shows which factors drove the prediction.")