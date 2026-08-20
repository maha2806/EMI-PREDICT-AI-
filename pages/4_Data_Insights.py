"""
EMIPredict AI - Data Insights Page

Historical trends from the applicant dataset - the static charts are the
validated ones from the notebook's EDA, and the segment explorer below
runs live against a sample of the data so the user can slice it themselves.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from utils.model_loader import get_dataset_sample

st.set_page_config(

    page_title = "Data Insights - EMIPredict AI",

    page_icon = ":material/bar_chart:",

    layout = "wide"

)

ASSETS_DIR = Path(__file__).resolve().parent.parent / 'assets'

st.title(":material/bar_chart: Historical Data Insights")

st.caption("Trends from the 404,800-applicant dataset the models were trained on.")

df = get_dataset_sample()

c1, c2, c3 = st.columns(3)

c1.metric("Applicants in Dataset", "404,800", icon = ":material/groups:")

c2.metric("Eligible Rate", "18.4%", icon = ":material/check_circle:")

c3.metric("High Risk Rate", "4.3%", icon = ":material/warning:")

st.divider()

c1, c2 = st.columns(2)

with c1:

    st.subheader(":material/donut_small: Eligibility Distribution")

    st.image(str(ASSETS_DIR / 'chart01_eligibility_distribution.png'), use_container_width = True)

with c2:

    st.subheader(":material/monitoring: Maximum Safe EMI Distribution")

    st.image(str(ASSETS_DIR / 'chart02_max_emi_distribution.png'), use_container_width = True)

st.subheader(":material/grid_on: Feature Correlations")

st.image(str(ASSETS_DIR / 'chart12_correlation_heatmap.png'), use_container_width = True)

st.divider()

st.subheader(":material/tune: Explore a Segment")

st.caption("Filter the dataset yourself to see how eligibility and safe EMI shift across applicant segments.")

c1, c2 = st.columns(2)

employment_filter = c1.multiselect(

    "Employment Type",

    sorted(df['employment_type'].unique()),

    default = sorted(df['employment_type'].unique())

)

scenario_filter = c2.multiselect(

    "Loan Scenario",

    sorted(df['emi_scenario'].unique()),

    default = sorted(df['emi_scenario'].unique())

)

filtered = df[df['employment_type'].isin(employment_filter) & df['emi_scenario'].isin(scenario_filter)]

if len(filtered) == 0:

    st.warning("No applicants match this filter - widen your selection.", icon = ":material/filter_alt_off:")

else:

    c1, c2, c3 = st.columns(3)

    c1.metric("Applicants in Segment", f"{len(filtered):,}", icon = ":material/groups:")

    eligible_rate = (filtered['emi_eligibility'] == 'Eligible').mean() * 100

    c2.metric("Eligible Rate", f"{eligible_rate:.1f}%", icon = ":material/check_circle:")

    c3.metric("Median Max Safe EMI", f"\u20b9{filtered['max_monthly_emi'].median():,.0f}", icon = ":material/savings:")

    c1, c2 = st.columns(2)

    with c1:

        st.write(":material/pie_chart: **Eligibility mix in this segment**")

        st.bar_chart(filtered['emi_eligibility'].value_counts())

    with c2:

        st.write(":material/bar_chart: **Safe EMI by loan scenario in this segment**")

        st.bar_chart(filtered.groupby('emi_scenario')['max_monthly_emi'].median())