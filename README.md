# Employee Salary Prediction

## Project Overview

This project uses Machine Learning to predict an employee's salary based on employee-related features.

The project compares multiple regression algorithms and selects the model that performs best on the test data.

## Objective

The main objective of this project is to build a Machine Learning model that can predict employee salaries and compare different regression algorithms to identify the best-performing model.

## Dataset

The dataset contains employee information such as:

- Employee ID
- Hire Date
- Job ID
- Salary
- Commission Percentage
- Manager ID
- Department ID

The dataset contains 50 records.

The data was divided into:

- 80% training data — 40 records
- 20% testing data — 10 records

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

## Machine Learning Models

Three regression algorithms were tested:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

## Model Evaluation

The models were evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)

### Results

| Model | R² Score | MAE | MSE |
|---|---:|---:|---:|
| Linear Regression | 0.6760 | 1539.40 | 3181540.42 |
| Decision Tree | 0.7207 | 869.20 | 2743006.40 |
| Random Forest | 0.9667 | 428.79 | 327200.82 |

## Best Model

Random Forest Regressor achieved the highest R² score of **0.9667** and the lowest MAE and MSE among the three tested models.

Therefore, Random Forest was selected as the best-performing model for this project.

## Project Structure

```text
Employee_Salary_Prediction/
│
├── data/
│   └── employees.csv
│
├── model/
│   └── salary_prediction_model.pkl
│
├── plots/
│   ├── actual_vs_predicted_salary.png
│   ├── model_comparison.png
│   └── salary_distribution.png
│
├── train_model.py
├── predict.py
└── README.md
