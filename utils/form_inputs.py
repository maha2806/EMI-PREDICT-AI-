"""
EMIPredict AI - Applicant Input Form

One shared form definition used by both the Eligibility and EMI Calculator
pages, so an applicant's details only need to be entered once - Streamlit's
session_state keeps the values when the user switches pages.

Every money field uses float consistently (min_value, value, and step all
float) since the values get persisted back into session_state as floats -
Streamlit's number_input rejects mixed int/float arguments.
"""

import streamlit as st

EMPLOYMENT_TYPES = ["Government", "Private", "Self-employed"]

COMPANY_TYPES = ["Large Indian", "MNC", "Mid-size", "Small", "Startup"]

HOUSE_TYPES = ["Own", "Rented", "Family"]

EDUCATION_LEVELS = ["High School", "Graduate", "Post Graduate", "Professional"]

EMI_SCENARIOS = ["Personal Loan EMI", "Vehicle EMI", "Education EMI", "Home Appliances EMI", "E-commerce Shopping EMI"]


def render_applicant_form():

    """Render the full applicant form and return the raw input dict.
    Values persist in st.session_state under the 'applicant_' prefix, so a
    value entered on one page is still there when the user visits the other."""

    ss = st.session_state

    st.subheader("Personal Details")

    c1, c2, c3 = st.columns(3)

    age = c1.number_input(

        "Age",

        min_value = 21,

        max_value = 65,

        value = int(ss.get("applicant_age", 35))

    )

    gender = c2.selectbox(

        "Gender",

        ["Male", "Female"],

        index = ["Male", "Female"].index(ss.get("applicant_gender", "Male"))

    )

    marital_status = c3.selectbox(

        "Marital Status",

        ["Single", "Married"],

        index = ["Single", "Married"].index(
            ss.get("applicant_marital_status", "Married")
        )

    )

    education = st.selectbox(

        "Education",

        EDUCATION_LEVELS,

        index = EDUCATION_LEVELS.index(ss.get("applicant_education", "Graduate"))

    )

    st.subheader("Employment & Income")

    c1, c2, c3 = st.columns(3)

    employment_type = c1.selectbox(

        "Employment Type",

        EMPLOYMENT_TYPES,

        index = EMPLOYMENT_TYPES.index(ss.get("applicant_employment_type", "Private"))

    )

    company_type = c2.selectbox(

        "Company Type",

        COMPANY_TYPES,

        index = COMPANY_TYPES.index(ss.get("applicant_company_type", "MNC"))

    )

    years_of_employment = c3.number_input(

        "Years of Employment",

        min_value = 0.0,

        max_value = 40.0,

        value = float(ss.get("applicant_years_of_employment", 5.0)),

        step = 0.5

    )

    monthly_salary = st.number_input(

        "Monthly Salary (INR)",

        min_value = 5000.0,

        value = float(ss.get("applicant_monthly_salary", 50000.0)),

        step = 1000.0

    )

    st.subheader("Housing & Family")

    c1, c2, c3 = st.columns(3)

    house_type = c1.selectbox(

        "House Type",

        HOUSE_TYPES,

        index = HOUSE_TYPES.index(ss.get("applicant_house_type", "Own"))

    )

    monthly_rent = c2.number_input(

        "Monthly Rent (INR)",

        min_value = 0.0,

        value = float(ss.get("applicant_monthly_rent", 0.0)),

        step = 500.0,

        disabled = (house_type != "Rented"),

        help = "Only applicable if House Type is Rented"

    )

    dependents = c3.number_input(

        "Number of Dependents",

        min_value = 0,

        max_value = 10,

        value = int(ss.get("applicant_dependents", 1))

    )

    st.subheader("Monthly Expenses (INR)")

    c1, c2, c3, c4 = st.columns(4)

    school_fees = c1.number_input(

        "School Fees",

        min_value = 0.0,

        value = float(ss.get("applicant_school_fees", 0.0)),

        step = 500.0

    )

    college_fees = c2.number_input(

        "College Fees",

        min_value = 0.0,

        value = float(ss.get("applicant_college_fees", 0.0)),

        step = 500.0

    )

    travel_expenses = c3.number_input(

        "Travel",

        min_value = 0.0,

        value = float(ss.get("applicant_travel_expenses", 3000.0)),

        step = 500.0

    )

    groceries_utilities = c4.number_input(

        "Groceries & Utilities",

        min_value = 0.0,

        value = float(ss.get("applicant_groceries_utilities", 8000.0)),

        step = 500.0

    )

    other_monthly_expenses = st.number_input(

        "Other Monthly Expenses",

        min_value = 0.0,

        value = float(ss.get("applicant_other_monthly_expenses", 2000.0)),

        step = 500.0

    )

    st.subheader("Credit & Savings")

    c1, c2, c3 = st.columns(3)

    credit_score = c1.slider(

        "Credit Score",

        min_value = 300,

        max_value = 850,

        value = int(ss.get("applicant_credit_score", 700))

    )

    bank_balance = c2.number_input(

        "Bank Balance (INR)",

        min_value = 0.0,

        value = float(ss.get("applicant_bank_balance", 100000.0)),

        step = 5000.0

    )

    emergency_fund = c3.number_input(

        "Emergency Fund (INR)",

        min_value = 0.0,

        value = float(ss.get("applicant_emergency_fund", 50000.0)),

        step = 5000.0

    )

    existing_loans = st.radio(

        "Existing Loans?",

        ["No", "Yes"],

        index = ["No", "Yes"].index(ss.get("applicant_existing_loans", "No")),

        horizontal = True

    )

    current_emi_amount = st.number_input(

        "Current Monthly EMI Amount (INR)",

        min_value = 0.0,

        value = float(ss.get("applicant_current_emi_amount", 0.0)),

        step = 500.0,

        disabled = (existing_loans == "No"),

        help = "Only applicable if Existing Loans is Yes"

    )

    st.subheader("Requested Loan")

    c1, c2, c3 = st.columns(3)

    emi_scenario = c1.selectbox(

        "Loan Type",

        EMI_SCENARIOS,

        index = EMI_SCENARIOS.index(ss.get("applicant_emi_scenario", "Personal Loan EMI"))

    )

    requested_amount = c2.number_input(

        "Requested Amount (INR)",

        min_value = 1000.0,

        value = float(ss.get("applicant_requested_amount", 200000.0)),

        step = 5000.0

    )

    requested_tenure = c3.number_input(

        "Requested Tenure (months)",

        min_value = 3,

        max_value = 84,

        value = int(ss.get("applicant_requested_tenure", 24))

    )

    raw = {

        "age": age,

        "gender": gender,

        "marital_status": marital_status,

        "education": education,

        "monthly_salary": float(monthly_salary),

        "employment_type": employment_type,

        "years_of_employment": years_of_employment,

        "company_type": company_type,

        "house_type": house_type,

        "monthly_rent": float(monthly_rent if house_type == "Rented" else 0),

        "dependents": dependents,

        "school_fees": float(school_fees),

        "college_fees": float(college_fees),

        "travel_expenses": float(travel_expenses),

        "groceries_utilities": float(groceries_utilities),

        "other_monthly_expenses": float(other_monthly_expenses),

        "existing_loans": existing_loans,

        "current_emi_amount": float(current_emi_amount if existing_loans == "Yes" else 0),

        "credit_score": float(credit_score),

        "bank_balance": float(bank_balance),

        "emergency_fund": float(emergency_fund),

        "emi_scenario": emi_scenario,

        "requested_amount": float(requested_amount),

        "requested_tenure": requested_tenure

    }

    # Persist every value so the other prediction page opens with the same applicant pre-filled -
    # int fields stay int, float fields stay float, matching what each widget expects on rerun

    int_fields = {"age", "dependents", "requested_tenure"}

    for key, value in raw.items():

        ss[f"applicant_{key}"] = int(value) if key in int_fields else value

    return raw