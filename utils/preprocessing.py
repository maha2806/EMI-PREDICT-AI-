"""
EMIPredict AI - Preprocessing

Turns one applicant's raw form input into the exact feature vector the
trained models expect. Every formula here matches Section 6 of the
model-development notebook - if that notebook's feature engineering
ever changes, this file needs the same change.
"""

import pandas as pd

# Same ordinal maps used when the models were trained

EDUCATION_ORDER = {'High School': 0, 'Graduate': 1, 'Post Graduate': 2, 'Professional': 3}

CREDIT_BAND_ORDER = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very_Good': 3, 'Excellent': 4}

CREDIT_BINS = [299, 579, 669, 739, 799, 850]

CREDIT_LABELS = ['Poor', 'Fair', 'Good', 'Very_Good', 'Excellent']


def build_feature_row(raw, feature_names):

    """Convert one applicant's raw form input into a single-row DataFrame,
    column-ordered to match feature_names.joblib.

    Args:
        raw: dict of raw applicant fields, using the same keys as the
        original dataset columns (age, gender, monthly_salary, ...)
        feature_names: ordered list of column names the model expects

    Returns:
        A one-row pandas DataFrame ready to pass to .predict()"""

    row = dict(raw)

    # Binary encoding

    row['gender'] = 1 if raw['gender'] == 'Male' else 0

    row['marital_status'] = 1 if raw['marital_status'] == 'Married' else 0

    row['existing_loans'] = 1 if raw['existing_loans'] == 'Yes' else 0

    # Ordinal encoding

    row['education'] = EDUCATION_ORDER[raw['education']]

    # Derived financial ratios - identical formulas to the notebook

    total_monthly_expenses = (raw['school_fees'] + raw['college_fees'] + raw['travel_expenses'] +
                              raw['groceries_utilities'] + raw['other_monthly_expenses'] + raw['monthly_rent'])

    row['total_monthly_expenses'] = total_monthly_expenses

    row['debt_to_income_ratio'] = round(raw['current_emi_amount'] / raw['monthly_salary'], 4)

    row['expense_to_income_ratio'] = round(total_monthly_expenses / raw['monthly_salary'], 4)

    disposable_income = raw['monthly_salary'] - total_monthly_expenses - raw['current_emi_amount']

    row['disposable_income'] = disposable_income

    row['affordability_ratio'] = round(disposable_income / raw['monthly_salary'], 4)

    proposed_installment = round(raw['requested_amount'] / raw['requested_tenure'], 2)

    row['proposed_installment'] = proposed_installment

    row['emi_to_income_ratio'] = round(proposed_installment / raw['monthly_salary'], 4)

    row['emergency_fund_months'] = round(raw['emergency_fund'] / total_monthly_expenses, 2) if total_monthly_expenses else 0

    row['liquidity_ratio'] = round(raw['bank_balance'] / total_monthly_expenses, 2) if total_monthly_expenses else 0

    # Credit score band, same bins as the notebook

    band = pd.cut(

        [raw['credit_score']],

        bins = CREDIT_BINS,

        labels = CREDIT_LABELS

    )[0]

    row['credit_score_band'] = CREDIT_BAND_ORDER[band]

    # One-hot encoding - reference category (dropped in training) is left implicit as all-zeros

    for col in ['Private', 'Self-employed']:

        row[f'employment_type_{col}'] = 1 if raw['employment_type'] == col else 0

    for col in ['MNC', 'Mid-size', 'Small', 'Startup']:

        row[f'company_type_{col}'] = 1 if raw['company_type'] == col else 0

    for col in ['Own', 'Rented']:

        row[f'house_type_{col}'] = 1 if raw['house_type'] == col else 0

    for col in ['Education EMI', 'Home Appliances EMI', 'Personal Loan EMI', 'Vehicle EMI']:

        row[f'emi_scenario_{col}'] = 1 if raw['emi_scenario'] == col else 0

    features_df = pd.DataFrame([row])[feature_names]

    return features_df


def needs_scaling(model):

    """Linear models need standardized input; tree-based models don't."""

    return type(model).__name__ in ('LogisticRegression', 'LinearRegression')