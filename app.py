import streamlit as st
import joblib
import pandas as pd
from datetime import date


model = joblib.load("model/salary_model.pkl")


st.title("Employee Salary Prediction")
job_mapping = {
    "AC_ACCOUNT": 0,
    "AC_MGR": 1,
    "AD_ASST": 2,
    "AD_PRES": 3,
    "AD_VP": 4,
    "FI_ACCOUNT": 5,
    "FI_MGR": 6,
    "HR_REP": 7,
    "IT_PROG": 8,
    "MK_MAN": 9,
    "MK_REP": 10,
    "PR_REP": 11,
    "PU_CLERK": 12,
    "PU_MAN": 13,
    "SH_CLERK": 14,
    "ST_CLERK": 15,
    "ST_MAN": 16
}


hire_date = st.date_input(
    "Hire Date",
    value=date(2007, 6, 21)
)

job_id = st.selectbox(
    "Job ID",
    list(job_mapping.keys())
)

commission_pct = st.number_input(
    "Commission Percentage",
    min_value=0.0,
    value=0.0
)

manager_id = st.number_input(
    "Manager ID",
    min_value=0,
    value=103
)

department_id = st.number_input(
    "Department ID",
    min_value=0,
    value=60
)


if st.button("Predict Salary"):

    hire_year = hire_date.year
    hire_month = hire_date.month
    hire_day = hire_date.day

    job_id_encoded = job_mapping[job_id]

    data = {
        "JOB_ID": [job_id_encoded],
        "COMMISSION_PCT": [commission_pct],
        "MANAGER_ID": [manager_id],
        "DEPARTMENT_ID": [department_id],
        "HIRE_YEAR": [hire_year],
        "HIRE_MONTH": [hire_month],
        "HIRE_DAY": [hire_day]
    }

    input_df = pd.DataFrame(data)

    prediction = model.predict(input_df)

    st.success(f"Predicted Salary: ₹{prediction[0]:,.2f}")
    