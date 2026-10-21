import joblib
import pandas as pd

model = joblib.load("model/salary_model.pkl")

data = {
    "HIRE_DATE": ["21-Jun-07"],
    "JOB_ID": ["IT_PROG"],
    "COMMISSION_PCT": ["-"],
    "MANAGER_ID": [103],
    "DEPARTMENT_ID": [60]
}

df = pd.DataFrame(data)

df["HIRE_DATE"] = pd.to_datetime(df["HIRE_DATE"], errors="coerce")

df["HIRE_YEAR"] = df["HIRE_DATE"].dt.year
df["HIRE_MONTH"] = df["HIRE_DATE"].dt.month
df["HIRE_DAY"] = df["HIRE_DATE"].dt.day

df = df.drop("HIRE_DATE", axis=1)

df = df.replace("-", 0)
df = df.replace(" - ", 0)

df["COMMISSION_PCT"] = pd.to_numeric(
    df["COMMISSION_PCT"],
    errors="coerce"
).fillna(0)

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
   
df["JOB_ID"] = df["JOB_ID"].map(job_mapping).fillna(0)

for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.fillna(0)

prediction = model.predict(df)

print("Predicted Salary:", prediction[0])
