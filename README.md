This project was developed independently as part of a machine learning portfolio. All code and analysis are original work.

# End-to-End Machine Learning Pipeline for Housing Price Prediction

## Overview

This project builds an end-to-end machine learning pipeline to predict housing prices using structured data. The workflow includes data cleaning, feature engineering, model training, and evaluation.

The objective is to demonstrate practical machine learning and data processing skills applied to real-world tabular data.

---

## Key Features

- Domain-aware handling of missing data
- Feature engineering based on data insights
- One-hot encoding of categorical variables
- Comparison of multiple regression models
- Performance evaluation using RMSE

---

## Dataset

- Ames Housing Dataset (~1460 samples, 80+ features)
- Includes structural, spatial, and quality-related attributes

---

## Methodology

### Data Cleaning
- Missing categorical values interpreted as absence and filled with `"None"`
- Numerical features representing absence filled with 0
- Location-dependent features imputed using grouped medians
- Low-information features removed

### Feature Engineering
- Total square footage (TotalSF)
- Property age and remodel age
- Total bathroom count
- Interaction feature (quality × living area)

### Modeling
- Linear Regression (baseline)
- Random Forest Regressor
- Gradient Boosting Regressor

### Evaluation
- Train/test split (80/20)
- Log transformation of target variable
- RMSE used as evaluation metric

---

## Results

| Model | RMSE (log) | Approx Error |
|------|-----------|-------------|
| Linear Regression | ~0.178 | ~19.5% |
| Random Forest | ~0.145 | ~15.7% |
| Gradient Boosting | ~0.142 | ~15.3% |

---

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- Jupyter Notebook

---

## Project Structure
data/
notebooks/
src/

---

## Future Improvements

- Cross-validation analysis
- Hyperparameter tuning for boosting models
- Model deployment as an API or web application