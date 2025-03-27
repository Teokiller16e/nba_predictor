import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from deep_learning import TorchClassifier, LSTMModel, GRUModel, Conv1DModel
import torch
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import joblib
import optuna
# Deep Learning Models (PyTorch)
from deep_learning import LSTMModel, GRUModel, Conv1DModel, TorchClassifier


#################### Machine Learning models Hyperparameter Functions: ###################

# Hyper-parameter tuning for Random Forest:
def tune_random_forest(X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        scoring='accuracy',
        cv=tscv,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X, y)

    print("✅ Best RandomForest Parameters:", grid.best_params_)
    print("📈 Best Accuracy:", round(grid.best_score_ * 100, 2), "%")
    
    return grid.best_estimator_


# Hyper-parameter tuning for Logistic Regression:
def tune_logistic_regression(X, y):
    tscv = TimeSeriesSplit(n_splits=5)

    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear']
    }

    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid,
        scoring='accuracy',
        cv=tscv,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X, y)

    print("✅ Best LogisticRegression Parameters:", grid.best_params_)
    print("📈 Best Accuracy:", round(grid.best_score_ * 100, 2), "%")
    
    return grid.best_estimator_


# Hyper-parameter tuning for Logistic Regression:
def tune_linear_svc(X, y):
    tscv = TimeSeriesSplit(n_splits=5)

    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'max_iter': [1000, 2000]
    }

    grid = GridSearchCV(
        LinearSVC(random_state=42),
        param_grid,
        scoring='accuracy',
        cv=tscv,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X, y)

    print("✅ Best LinearSVC Parameters:", grid.best_params_)
    print("📈 Best Accuracy:", round(grid.best_score_ * 100, 2), "%")

    return grid.best_estimator_


# Hyper-parameter tuning for XGBoost:
def tune_xgboost(X, y):
    tscv = TimeSeriesSplit(n_splits=5)

    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 6],
        'eta': [0.01, 0.1],
        'subsample': [0.8, 1.0]
    }

    grid = GridSearchCV(
        XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        param_grid,
        scoring='accuracy',
        cv=tscv,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X, y)

    print("✅ Best XGBoost Parameters:", grid.best_params_)
    print("📈 Best Accuracy:", round(grid.best_score_ * 100, 2), "%")

    return grid.best_estimator_


# Hyper-parameter tuning for Decision Trees:
def tune_decision_tree(X, y):
    tscv = TimeSeriesSplit(n_splits=5)

    param_grid = {
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    }

    grid = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        scoring='accuracy',
        cv=tscv,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X, y)

    print("✅ Best DecisionTree Parameters:", grid.best_params_)
    print("📈 Best Accuracy:", round(grid.best_score_ * 100, 2), "%")

    return grid.best_estimator_


################ Deep Learning models Hyperparameter Functions: ###################

from sklearn.model_selection import TimeSeriesSplit
import torch
from sklearn.metrics import accuracy_score
import numpy as np

# Shared utility function to calculate average accuracy across folds
def evaluate_model_across_folds(model_name, model_builder, X_seq_tensor, X_conv_tensor, y_tensor, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    acc_scores = []

    for train_idx, val_idx in tscv.split(X_seq_tensor):
        if model_name == "Conv1D":
            model = model_builder()
            model.fit(X_conv_tensor[train_idx], y_tensor[train_idx])
            y_pred = model.predict(X_conv_tensor[val_idx])
        else:
            model = model_builder()
            model.fit(X_seq_tensor[train_idx], y_tensor[train_idx])
            y_pred = model.predict(X_seq_tensor[val_idx])

        acc = accuracy_score(y_tensor[val_idx].cpu().numpy(), y_pred)
        acc_scores.append(acc)

    return np.mean(acc_scores)




# Hyper-parameter tuning for LSTM:
def objective_lstm(trial, X_seq_tensor, X_conv_tensor, y_tensor, input_size):
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = trial.suggest_int("epochs", 5, 20)

    def model_builder():
        return TorchClassifier(
            model_class=LSTMModel,
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs
        )

    return evaluate_model_across_folds("LSTM", model_builder, X_seq_tensor, X_conv_tensor, y_tensor)





# Hyper-parameter tuning GRU:
def objective_gru(trial, X_seq_tensor, X_conv_tensor, y_tensor, input_size):
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = trial.suggest_int("epochs", 5, 20)

    def model_builder():
        return TorchClassifier(
            model_class=GRUModel,
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs
        )

    return evaluate_model_across_folds("GRU", model_builder, X_seq_tensor, X_conv_tensor, y_tensor)




# Hyper-parameter tuning CONV1D:
def objective_conv1d(trial, X_seq_tensor, X_conv_tensor, y_tensor, input_size):
    num_filters = trial.suggest_categorical("num_filters", [32, 64, 128])
    kernel_size = trial.suggest_categorical("kernel_size", [2, 3])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = trial.suggest_int("epochs", 5, 20)

    def model_builder():
        return TorchClassifier(
            model_class=lambda input_size: Conv1DModel(
                input_size=input_size,
                num_filters=num_filters,
                kernel_size=kernel_size,
                dropout=dropout
            ),
            input_size=input_size,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr
        )

    return evaluate_model_across_folds("Conv1D", model_builder, X_seq_tensor, X_conv_tensor, y_tensor)


