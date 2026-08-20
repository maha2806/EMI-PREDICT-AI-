"""
EMIPredict AI - EMI Calculator Page

Collects one applicant's details through the shared form and runs the
trained regressor to estimate the maximum monthly EMI they can safely
afford, then compares that figure against what they actually requested.
"""

import streamlit as st
from utils.model_loader import get_models
from utils.form_inputs import render_applicant_form
from utils.preprocessing import build_feature_row, needs_scaling

st.set_page_config(

    page_title = "EMI Calculator - EMIPredict AI",

    page_icon = ":material/calculate:",

    layout = "wide"

)

st.title(":material/calculate: Maximum Safe EMI Calculator")

st.caption("Fill in the applicant's details below to estimate the largest monthly EMI they can safely sustain.")

models = get_models()

raw = render_applicant_form()

st.divider()

calculate_clicked = st.button(

    "Calculate Safe EMI",

    type = "primary",

    icon = ":material/calculate:",

    use_container_width = True

)

if calculate_clicked:

    feature_row = build_feature_row(raw, models['feature_names'])

    reg = models['regressor']

    X_input = models['scaler'].transform(feature_row) if needs_scaling(reg) else feature_row

    predicted_emi = reg.predict(X_input)[0]

    requested_installment = raw['requested_amount'] / raw['requested_tenure']

    st.divider()

    c1, c2 = st.columns(2)

    c1.metric("Maximum Safe Monthly EMI", f"\u20b9{predicted_emi:,.0f}", icon = ":material/savings:")

    c2.metric(

        "Requested Loan's Monthly Installment",

        f"\u20b9{requested_installment:,.0f}",

        delta = f"\u20b9{requested_installment - predicted_emi:,.0f} vs safe limit",

        delta_color = "inverse",

        icon = ":material/request_quote:"

    )

    if requested_installment <= predicted_emi:

        st.success(

            f"The requested loan's installment (\u20b9{requested_installment:,.0f}/month) is within "
            f"the applicant's safe EMI capacity.",

            icon = ":material/check_circle:"

        )

    else:

        over_by = requested_installment - predicted_emi

        st.warning(
            
            f"The requested loan's installment is \u20b9{over_by:,.0f}/month above the applicant's safe capacity. "
            f"Consider a longer tenure or a smaller loan amount.",

            icon = ":material/warning:"
            
        )

        max_affordable_amount = predicted_emi * raw['requested_tenure']

        st.write(f":material/info: At the requested {raw['requested_tenure']}-month tenure, a loan amount up to "
                 f"**\u20b9{max_affordable_amount:,.0f}** would stay within the safe EMI limit.")

    st.subheader(":material/query_stats: What Drives This Estimate")

    st.write(
        
        "The estimate is built from the applicant's income, existing expenses and debt load - not just "
        "salary. Two applicants with identical salaries can have very different safe-EMI figures depending "
        "on their existing financial commitments."
        
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Monthly Salary", f"\u20b9{raw['monthly_salary']:,.0f}", icon = ":material/payments:")

    c2.metric("Disposable Income", f"\u20b9{feature_row['disposable_income'].iloc[0]:,.0f}", icon = ":material/account_balance_wallet:")

    c3.metric("Affordability Ratio", f"{feature_row['affordability_ratio'].iloc[0] * 100:.1f}%", icon = ":material/pie_chart:")

    st.info(

        "Also want to know if this applicant is Eligible, High Risk, or Not Eligible? Head to the "
        "**Eligibility Prediction** page - your details are already filled in.",

        icon = ":material/verified:"

    )