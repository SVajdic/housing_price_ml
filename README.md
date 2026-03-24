# End-to-End Machine Learning Pipeline for Housing Price Prediction

## Overview

This project implements an end-to-end machine learning pipeline to predict residential housing prices using structured tabular data. The workflow includes data cleaning, feature engineering, model training, and evaluation.

The objective is to demonstrate the ability to build reproducible data pipelines and apply machine learning techniques to real-world datasets.

---

## Key Features

- Domain-aware handling of missing data
- Feature engineering based on data insights
- One-hot encoding of categorical variables
- Comparison of multiple regression models
- Performance evaluation using RMSE
- Modular pipeline design (separation of concerns)

---

## Results

| Model | RMSE (log) | Approx Error |
|------|-----------|-------------|
| Linear Regression | ~0.178 | ~19.5% |
| Random Forest | ~0.145 | ~15.7% |
| Gradient Boosting | ~0.142 | ~15.3% |

Ensemble models significantly outperform linear regression, demonstrating the importance of nonlinear modeling and feature interactions in housing price prediction.

---

## Dataset

This project uses the Ames Housing dataset, a widely used dataset for regression problems involving real estate pricing.

**Note:**  
The raw dataset is not included in this repository to keep it lightweight and reproducible.

### How to obtain the data

1. Download the dataset from:  
   https://www.kaggle.com/datasets/prevek18/ames-housing-dataset

2. Place the file in:
data/raw/ames_house_prices.csv

---

## Methodology

### Data Cleaning
- Missing categorical values interpreted as absence and filled with `"None"`
- Numerical features representing absence filled with `0`
- Location-dependent features imputed using grouped medians
- Low-information features removed

### Feature Engineering
- Total square footage (`TotalSF`)
- Property age and remodel age
- Total bathroom count
- Interaction feature (`OverallQual × GrLivArea`)

### Modeling
- Linear Regression (baseline)
- Random Forest Regressor
- Gradient Boosting Regressor

### Evaluation
- Train/test split (80/20)
- Log transformation applied to target variable
- Root Mean Squared Error (RMSE) used for evaluation

---

## Project Structure

housing_price_ml/
│
├── data/
│ ├── raw/ # Raw dataset (not tracked in Git)
│
├── notebooks/
│ └── housing_analysis.ipynb # EDA and modeling workflow
│
├── src/
│ ├── data_cleaning.py
│ ├── feature_engineering.py
│ ├── modeling.py
│ └── init.py
│
├── requirements.txt
├── README.md
└── .gitignore

---

## Usage

### Install dependencies

```bash
pip install -r requirements.txt
```
### Run analysis

```bash
jupyter notebook notebooks/housing_analysis.ipynb
```

--- 

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- Jupyter Notebook

---

## Future Improvements

- Cross-validation with variance reporting
- Hyperparameter tuning for boosting models
- Model persistence (saving/loading trained models)
- Deployment as an API or batch inference pipeline

---

## Author

Stephan Vajdic 
M.S. Physics 

--- 

## License 

This project is for portfolio purposes. Attribution is required for any reuse of this work.