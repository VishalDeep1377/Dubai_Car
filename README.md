# 🚀 Startup Success Prediction & Healthcare Billing Forecasting

This project combines two powerful domains — **startup success prediction** using classification algorithms and **healthcare billing trend forecasting** using time series analysis. It implements an end-to-end machine learning pipeline with real-world datasets, addressing class imbalance, visualizing insights, and applying statistical forecasting for future planning.

---

## 🧠 Objectives

- Predict startup success based on funding and operational features.
- Handle class imbalance using SMOTE.
- Train and compare ML models: Random Forest, Logistic Regression, Decision Tree.
- Evaluate model performance using classification metrics.
- Visualize data patterns and model learning behavior.
- Perform time series forecasting on healthcare billing data using ARIMA and Exponential Smoothing.

---

## 🧰 Tech Stack & Libraries

- **Language**: Python
- **ML & Stats**: `scikit-learn`, `imblearn`, `statsmodels`
- **EDA & Viz**: `pandas`, `matplotlib`, `seaborn`
- **Time Series**: `ARIMA`, `ExponentialSmoothing`
- **Modeling**: `RandomForestClassifier`, `LogisticRegression`, `DecisionTreeClassifier`
- **Preprocessing**: `StandardScaler`, `LabelEncoder`, `SMOTE`

---

## 📊 Dataset Overview

### 🔹 Startup Dataset (`car_startup.csv`)
| Feature | Description |
|---------|-------------|
| `funding_total_usd` | Total funding raised |
| `funding_rounds` | Number of rounds |
| `milestones` | Key company milestones |
| `relationships` | Investor/founder relationships |
| `age_first_funding_year`, `age_last_funding_year` | Years of funding activity |
| `status` | Binary target (success/failure) |

> Additional attributes used for extended modeling: state, categories, funding types, investor info.

---

### 🔹 Simulated Healthcare Dataset
| Feature | Description |
|---------|-------------|
| `Date of Admission` | Date of hospital admission |
| `Billing Amount` | Daily billing in USD |
| `Age`, `Gender`, `BMI` | Patient demographics |
| `Smoking`, `Alcohol`, `Exercise`, `Diet` | Lifestyle indicators |
| `Heart Disease`, `Depression`, `Diabetes` | Health conditions (binary) |

---

## 🧪 Machine Learning Pipeline

1. **Preprocessing**:
   - Label encoding for categorical features
   - Missing value imputation (median/mode)
   - Feature scaling with `StandardScaler`

2. **EDA**:
   - Histograms, boxplots, heatmaps
   - Correlation analysis
   - Distribution of success vs. failure

3. **Feature Selection**:
   - Manually curated based on domain knowledge
   - Recursive feature elimination (RFE)

4. **Class Imbalance Handling**:
   - SMOTE applied to training data to balance `status` classes

5. **Model Training**:
   - Random Forest
   - Logistic Regression
   - Decision Tree Classifier

6. **Evaluation Metrics**:
   - Accuracy, Precision, Recall, F1 Score
   - MAE, RMSE
   - Confusion Matrix
   - Learning Curves

7. **Visualization**:
   - Confusion Matrix heatmaps
   - Predicted probability distributions
   - Decision Tree logic with `graphviz`

---

## 📈 Time Series Forecasting

Used simulated healthcare billing data to generate monthly average trends and apply forecasting:

### 🔹 Monthly Aggregation
```python
monthly_billing = df.groupby('Month')['Billing Amount'].mean()
