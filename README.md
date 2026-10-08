# Telco Customer Churn Prediction

## Project Overview

Customer churn is an important challenge for telecommunications companies, as retaining existing customers is often more cost-effective than acquiring new ones. This project uses machine learning to predict whether a customer is likely to churn based on their demographic information, account details, subscribed services, contract information, and billing characteristics.

The project covers the complete machine learning workflow, including data cleaning, exploratory data analysis, preprocessing, model development, hyperparameter tuning, and evaluation.

The final XGBoost model achieved **80.4% accuracy** and a **ROC-AUC of 0.847** on the held-out test set. The project also explores how changing the classification threshold affects the balance between identifying potential churners and generating false positives.

## Dataset

The project uses the IBM Telco Customer Churn dataset.

The dataset contains customer information including:

* Demographic information
* Account information
* Phone and internet services
* Additional services
* Contract and payment information
* Monthly and total charges
* Customer tenure
* Churn status

The dataset contains **7,043 customers**.

### Target Variable

The `Churn` variable was encoded as a binary target:

* `0` → Customer did not churn
* `1` → Customer churned

The target variable is imbalanced:

* No Churn: 5,174 customers (~73.5%)
* Churn: 1,869 customers (~26.5%)

## Machine Learning Workflow

The project follows this workflow:

1. Load the dataset
2. Perform data cleaning
3. Convert `TotalCharges` to a numeric feature
4. Handle missing numerical values through median imputation
5. Treat `SeniorCitizen` as a categorical feature
6. Remove the unique `customerID` identifier
7. Encode the target variable
8. Split the data into training and test sets
9. Build preprocessing pipelines for numerical and categorical features
10. Apply one-hot encoding to categorical variables
11. Train an XGBoost classifier
12. Tune model hyperparameters using cross-validation
13. Evaluate the final model on the test set

## Preprocessing

The preprocessing steps are implemented using a `ColumnTransformer` and `Pipeline`.

### Numerical Features

Missing numerical values are handled using median imputation:

```python
SimpleImputer(strategy="median")
```

### Categorical Features

Categorical variables are converted into numerical representations using one-hot encoding:

```python
OneHotEncoder(handle_unknown="ignore")
```

Using a pipeline ensures that preprocessing is learned from the training data and applied consistently to the test data.

## Model

The final model is an **XGBoost classifier**.

The final hyperparameters were selected through randomized hyperparameter search with 5-fold cross-validation.

### Final Parameters

```text
learning_rate = 0.05
n_estimators = 150
max_depth = 4
min_child_weight = 5
gamma = 0.2
subsample = 1.0
colsample_bytree = 1.0
reg_alpha = 0.15
reg_lambda = 30
eval_metric = logloss
```

The tuning process was stopped when further changes produced only small improvements in cross-validation performance.

## Model Performance

### Cross-Validation

The final model achieved:

**Mean 5-fold CV accuracy: 0.807**

### Test Set

| Metric          | Score |
| --------------- | ----: |
| Accuracy        | 0.804 |
| ROC-AUC         | 0.847 |
| Churn Precision |  0.66 |
| Churn Recall    |  0.53 |
| Churn F1-score  |  0.59 |

### Confusion Matrix

```text
[[935 100]
 [176 198]]
```

Where:

* **True Negatives:** 935
* **False Positives:** 100
* **False Negatives:** 176
* **True Positives:** 198

At the default probability threshold of 0.5, the model identifies approximately **53% of actual churners**.

## ROC-AUC

The model achieved a ROC-AUC score of approximately **0.847**.

ROC-AUC evaluates how well the model separates churners from non-churners across different classification thresholds rather than evaluating only the default 0.5 threshold.

A higher AUC indicates better ability to rank customers who churn above customers who do not churn.

## Classification Threshold

The default classification threshold of 0.5 is not necessarily optimal for a business problem.

For example, lowering the threshold to 0.3 increased churn recall from approximately **53% to 76%**, but also increased the number of false positives.

This demonstrates the trade-off between:

* **Precision:** How many customers predicted to churn actually churn
* **Recall:** How many actual churners the model successfully identifies

In a real business setting, the final threshold should be selected according to the relative costs of missing a potential churner versus contacting a customer who would not have churned.

The 0.3 threshold was explored as an analysis example rather than selected as the final production threshold.

## Project Structure

```text
Telco-Customer-Churn/
│
├── data/
│   └── Telco-Customer-churn.csv
│
├── src/
│   └── telco_churn.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Key Takeaways

* XGBoost achieved good classification performance on the Telco customer churn dataset.
* The complete preprocessing and modeling workflow was implemented using Scikit-learn pipelines.
* The model achieved a test accuracy of **80.4%** and a ROC-AUC of **0.847**.
* The model's churn recall at the default 0.5 threshold was **53%**.
* Lowering the classification threshold increased churn recall to **76%**, while also increasing false positives.
* The choice of classification threshold involves a trade-off between identifying more potential churners and generating more false positives.

## Future Improvements

Possible future work could include:

* Selecting a classification threshold using a validation set based on a defined business objective
* Comparing XGBoost with other classification models
* Performing more detailed feature engineering
* Investigating model interpretability using feature importance or SHAP
* Evaluating additional metrics such as precision-recall curves and cost-based metrics

## Technologies

* Python
* Pandas
* Scikit-learn
* XGBoost
* Git / GitHub
