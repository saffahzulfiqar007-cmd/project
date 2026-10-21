import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import joblib
import os


df = pd.read_csv("data/employees.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())

df = df.drop(columns=[
    "EMPLOYEE_ID",
    "FIRST_NAME",
    "LAST_NAME",
    "EMAIL",
    "PHONE_NUMBER"
])

print(df.columns)

df["HIRE_DATE"] = pd.to_datetime(df["HIRE_DATE"], errors="coerce")

df["HIRE_YEAR"] = df["HIRE_DATE"].dt.year
df["HIRE_MONTH"] = df["HIRE_DATE"].dt.month
df["HIRE_DAY"] = df["HIRE_DATE"].dt.day

df = df.drop("HIRE_DATE", axis=1)

df = df.replace("-", np.nan)
df = df.replace(" - ", np.nan)

df["COMMISSION_PCT"] = pd.to_numeric(
    df["COMMISSION_PCT"],
    errors="coerce"
).fillna(0)

label_encoder = LabelEncoder()
df["JOB_ID"] = label_encoder.fit_transform(df["JOB_ID"])

print("JOB_ID mapping:")
for job, number in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
    print(job, "=", number)

for column in df.columns:
    if column != "SALARY":
        df[column] = pd.to_numeric(df[column], errors="coerce")


df = df.fillna(0)

print(df.head())
print(df.dtypes)


print("Columns with text values:")
print(df.select_dtypes(include=["object"]).columns)

for column in df.select_dtypes(include=["object"]).columns:
    print(column, df[column].unique())

X = df.drop("SALARY", axis=1)
y = df["SALARY"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)



lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\n===== Linear Regression =====")
print("R2 Score:", r2_score(y_test, lr_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, lr_pred))
print("Mean Squared Error:", mean_squared_error(y_test, lr_pred))



dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

print("\n===== Decision Tree =====")
print("R2 Score:", r2_score(y_test, dt_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, dt_pred))
print("Mean Squared Error:", mean_squared_error(y_test, dt_pred))



rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\n===== Random Forest =====")
print("R2 Score:", r2_score(y_test, rf_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, rf_pred))
print("Mean Squared Error:", mean_squared_error(y_test, rf_pred))

os.makedirs("model", exist_ok=True)

joblib.dump(rf_model, "model/salary_model.pkl")

print("Best model saved successfully!")

os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(8, 5))
plt.scatter(y_test, rf_pred)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.savefig("plots/actual_vs_predicted_salary.png")
plt.show()


models = ["Linear Regression", "Decision Tree", "Random Forest"]

r2_scores = [
    r2_score(y_test, lr_pred),
    r2_score(y_test, dt_pred),
    r2_score(y_test, rf_pred)
]

plt.figure(figsize=(8, 5))
plt.bar(models, r2_scores)
plt.xlabel("Models")
plt.ylabel("R2 Score")
plt.title("Model Performance Comparison")
plt.ylim(0, 1)
plt.savefig("plots/model_comparison.png")
plt.show()


plt.figure(figsize=(8, 5))
plt.hist(df["SALARY"], bins=10)
plt.xlabel("Salary")
plt.ylabel("Number of Employees")
plt.title("Salary Distribution")
plt.savefig("plots/salary_distribution.png")
plt.show()














