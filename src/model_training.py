from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.

    Parameters:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series or np.ndarray): Training target

    Returns:
        LinearRegression: Trained model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, random_state=42):
    """
    Train a Random Forest Regressor.

    Parameters:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series or np.ndarray): Training target
        random_state (int): Seed for reproducibility

    Returns:
        RandomForestRegressor: Trained model
    """
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train, random_state=42):
    """
    Train a Gradient Boosting Regressor.

    Parameters:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series or np.ndarray): Training target
        random_state (int): Seed for reproducibility

    Returns:
        GradientBoostingRegressor: Trained model
    """
    model = GradientBoostingRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def compute_rmse(y_true, y_pred):
    """
    Compute Root Mean Squared Error (RMSE).

    Parameters:
        y_true (pd.Series or np.ndarray): True target values
        y_pred (np.ndarray): Predicted target values

    Returns:
        float: RMSE
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))