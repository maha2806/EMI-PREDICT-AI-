"""
EMIPredict AI - Eligibility Prediction Page

Collects one applicant's details through the shared form and runs the
trained classifier, showing the predicted eligibility class alongside its
full probability breakdown so the decision isn't just a single label.
"""

import streamlit as st
import pandas as pd
from utils.model_loader import get_models
from utils.form_inputs import render_applicant_form
from utils.preprocessing import build_feature_row, needs_scaling

st.set_page_config(

    page_title = "Eligibility Prediction - EMIPredict AI",

    page_icon = ":material/verified:",

    layout = "wide"

)

st.title(":material/verified: EMI Eligibility Prediction")

st.caption("Fill in the applicant's details below, then click Predict to get an eligibility decision.")

models = get_models()

raw = render_applicant_form()

st.divider()

predict_clicked = st.button(

    "Predict Eligibility",

    type = "primary",

    icon = ":material/query_stats:",

    use_container_width = True

)

if predict_clicked:

    feature_row = build_feature_row(raw, models['feature_names'])

    clf = models['classifier']

    X_input = models['scaler'].transform(feature_row) if needs_scaling(clf) else feature_row

    predicted_class = clf.predict(X_input)[0]

    probabilities = clf.predict_proba(X_input)[0]

    label_encoder = models['label_encoder']

    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    proba_by_class = dict(zip(label_encoder.classes_, probabilities))

    st.divider()

    result_style = {
        
        'Eligible': (':material/check_circle:', 'green', "This applicant looks eligible for the requested EMI."),

        'High_Risk': (':material/warning:', 'orange', "This applicant is borderline - recommended for manual review."),

        'Not_Eligible': (':material/cancel:', 'red', "This applicant does not currently meet the eligibility bar.")
        
    }

    icon, color, message = result_style[predicted_label]

    st.markdown(f"## {icon} :{color}[{predicted_label.replace('_', ' ')}]")

    st.write(message)

    st.subheader(":material/percent: Class Probabilities")

    proba_df = pd.DataFrame({
        
        'Eligibility': list(proba_by_class.keys()),
        
        'Probability': list(proba_by_class.values())
        
    }).sort_values('Probability', ascending = False)

    st.bar_chart(proba_df.set_index('Eligibility'))

    c1, c2, c3 = st.columns(3)

    c1.metric("Eligible", f"{proba_by_class['Eligible'] * 100:.1f}%", icon = ":material/check_circle:")

    c2.metric("High Risk", f"{proba_by_class['High_Risk'] * 100:.1f}%", icon = ":material/warning:")

    c3.metric("Not Eligible", f"{proba_by_class['Not_Eligible'] * 100:.1f}%", icon = ":material/cancel:")

    with st.expander("Why did the model decide this?", icon = ":material/lightbulb:"):

        st.write(
            
            "The decision is based on the applicant's full financial profile - the strongest signals "
            "the underlying model relies on are the engineered ratios (EMI-to-income, disposable income, "
            "affordability ratio) rather than any single raw field. See the Model Performance page for the "
            "full feature importance ranking."
            
        )

        st.dataframe(feature_row.T.rename(columns = {0: 'Value'}), use_container_width = True)

    st.info(

        "Also want to know the maximum EMI this applicant can safely afford? Head to the "
        "**EMI Calculator** page - your details are already filled in.",

        icon = ":material/calculate:"

    )