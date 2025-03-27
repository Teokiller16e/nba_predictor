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



def pre_processing(df):
    # Print the columns that contain NaN values :
    missing_cols = df.columns[df.isnull().any()]
    print("Columns with missing values:", missing_cols.tolist())

    # Now let's see the exact count of missing values per column among these:
    missing_values_count = df[missing_cols].isnull().sum()
    print("\nCount of missing values in each column:")
    print(missing_values_count)

    # 1. Define the list of columns you DO want to include in X
    features = [
        'WL', 'MIN', 'PTS', 'FGM', 'FGA', 'FG_PCT',
        'FG3M', 'FG3A',  # <--- FG3_PCT is excluded
        'FTM', 'FTA', 'FT_PCT',
        'OREB', 'DREB', 'REB',
        'AST', 'STL', 'BLK', 'TOV', 'PF'
    ]  # Notice that 'PLUS_MINUS' and 'FG3_PCT' are not here

    # 2. Drop rows that have missing values in those selected columns
    df.dropna(subset=features, inplace=True)


    # Part2: Initialize all_games and try to utilize them as the final form dataset: 

    all_games = df.copy() # previously it was df without .copy()
    # Convert GAME_DATE to datetime format
    all_games['GAME_DATE'] = pd.to_datetime(all_games['GAME_DATE'])

    # Sort by TEAM_ID and GAME_DATE to ensure proper chronological order
    #all_games = all_games.sort_values(by=['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)
    all_games = all_games.sort_values(by=['GAME_DATE']).reset_index(drop=True)

    # Track game results
    all_games['WIN'] = all_games['WL'].apply(lambda x: 1 if x == 'W' else 0)

    # Convert PTS to float
    all_games['PTS'] = all_games['PTS'].astype(float)

    # Calculate team’s average points per game
    all_games['Points_Per_Game'] = all_games.groupby('TEAM_ID')['PTS'].transform('mean')

    # ✅ Build `team_abbr_to_id` dictionary from the CSV instead of calling API
    team_abbr_to_id = dict(zip(all_games['TEAM_ABBREVIATION'], all_games['TEAM_ID']))

    def get_opponent_team_id(matchup, team_abbr_to_id, team_id):
        if '@' in matchup:
            opponent_abbr = matchup.split(' @ ')[-1]
        else:
            opponent_abbr = matchup.split(' vs. ')[-1]
        return team_abbr_to_id.get(opponent_abbr, team_id)

    # Get opponent team ID
    all_games['OPPONENT_TEAM_ID'] = all_games.apply(lambda row: get_opponent_team_id(row['MATCHUP'], team_abbr_to_id, row['TEAM_ID']), axis=1)

    # Mark home games
    all_games['HOME_GAME'] = all_games['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)

    # **FIXED: Get the result of the previous game for each team**
    all_games['LAST_GAME_RESULT'] = all_games.groupby('TEAM_ID')['WIN'].shift(1).fillna(0)

     # ✅ Assign proper NBA SEASON instead of calendar year
    def get_season_year(date):
        return date.year if date.month >= 10 else date.year - 1

    all_games['SEASON'] = all_games['GAME_DATE'].apply(get_season_year)

    return all_games

# # ✅ SAFE Rolling Feature
#     all_games['PTS_rolling3'] = all_games.groupby('TEAM_ID')['PTS'].transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())


#     # ✅ SAFE Win Streak Feature
#     def calc_streak(series):
#         streak = 0
#         result = []
#         for win in series:
#             if win == 1:
#                 streak += 1
#             else:
#                 streak = 0
#             result.append(streak)
#         return result


#     # ✅ Fix: Shift before applying streak logic
#     all_games['WIN_STREAK'] = all_games.groupby('TEAM_ID')['WIN'].transform(lambda x: calc_streak(x.shift(1).fillna(0)))

#     # Drop rows that include NaNs due to shifting
#     all_games.dropna(subset=['PTS_rolling3', 'WIN_STREAK'], inplace=True)

    


#################### FIT & EVALUATION FUNCTIONS FOR BOTH ML/DL MODELS: #################
# Fit and evaluation for machine learning models: 
def evaluate_classical_models(start_index, context_window, unique_years, all_games, feature_cols, models):
    results = []
    reports = {}


    for i in range(start_index, len(unique_years)):
        test_year = unique_years[i]
        train_years = unique_years[i - context_window : i]

        train_df = all_games[all_games['YEAR'].isin(train_years)]
        test_df  = all_games[all_games['YEAR'] == test_year]

        X_train = train_df[feature_cols]
        y_train = train_df['WIN']
        X_test  = test_df[feature_cols]
        y_test  = test_df['WIN']

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        print(f"\n=== [ML] Predicting Year {test_year} ===")
        print(f"Training on years: {train_years}\n")

        for model_name, model_fn in models.items():
            model = model_fn()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred) * 100

            print(f"{model_name:<16} Accuracy: {acc:.2f}%")

            # Save results
            results.append({
                'test_year': test_year,
                'model': model_name,
                'accuracy': acc
            })

            # Save classification report
            report = classification_report(y_test, y_pred, target_names=["Loss", "Win"], output_dict=True)
            reports[(test_year, model_name)] = report

    return results, reports


# Fit and evaluation for deep learning models:
def evaluate_deep_models(start_index, context_window, unique_years, all_games, feature_cols, models):
    results = []
    reports = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(start_index, len(unique_years)):
        test_year = unique_years[i]
        train_years = unique_years[i - context_window : i]

        train_df = all_games[all_games['YEAR'].isin(train_years)]
        test_df  = all_games[all_games['YEAR'] == test_year]

        X_train = train_df[feature_cols]
        y_train = train_df['WIN']
        X_test  = test_df[feature_cols]
        y_test  = test_df['WIN']

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # Prepare input formats
        X_train_conv = torch.as_tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1).to(device)  # [batch, channels=1, features]
        X_test_conv  = torch.as_tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1).to(device)

        X_train_seq = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
        X_test_seq  = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])


        y_train_tensor = torch.as_tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1).to(device)
        y_test_tensor  = torch.as_tensor(y_test.to_numpy(), dtype=torch.float32).unsqueeze(1).to(device)

        print(f"\n=== [DL] Predicting Year {test_year} ===")
        print(f"Training on years: {train_years}\n")

        for model_name, model_fn in models.items():
            model = model_fn()
            if model_name == "Conv1D":
                model.fit(X_train_conv, y_train_tensor)
                y_pred = model.predict(X_test_conv)
            else:  # LSTM / GRU
                model.fit(X_train_seq, y_train_tensor)
                y_pred = model.predict(X_test_seq)

            acc = accuracy_score(y_test_tensor.cpu().numpy(), y_pred)
            print(f"{model_name:<16} Accuracy: {acc * 100:.2f}%")

            results.append({
                'test_year': test_year,
                'model': model_name,
                'accuracy': acc * 100
            })

            report = classification_report(y_test_tensor.cpu().numpy(), y_pred, target_names=["Loss", "Win"], output_dict=True)
            reports.append({
                'test_year': test_year,
                'model': model_name,
                'report': report
            })

    return results, reports


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

# Hyper-parameter tuning for LSTM:
def objective_lstm(trial, X_train, y_train, X_val, y_val, input_size):
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = trial.suggest_int("epochs", 5, 20)

    model = TorchClassifier(
        model_class=LSTMModel,  # ✅ pass class directly, not lambda
        input_size=input_size,
        hidden_size=hidden_size,
        dropout=dropout,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val.cpu().numpy(), preds)
    return acc



# Hyper-parameter tuning GRU:
def objective_gru(trial, X_train, y_train, X_val, y_val, input_size):
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    dropout     = trial.suggest_float("dropout", 0.1, 0.5)
    lr          = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size  = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs      = trial.suggest_int("epochs", 5, 20)

    model = TorchClassifier(
        GRUModel,
        input_size=input_size,
        hidden_size=hidden_size,
        dropout=dropout,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val.cpu().numpy(), y_pred)
    return acc



# Hyper-parameter tuning CONV1D:
def objective_conv1d(trial, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, input_size):
    # Suggest hyperparameters
    num_filters = trial.suggest_categorical("num_filters", [32, 64, 128])
    kernel_size = trial.suggest_categorical("kernel_size", [2, 3])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = trial.suggest_int("epochs", 5, 20)

    # Conv1DModel expects input_size, num_filters, kernel_size, dropout
    model_class = lambda input_size: Conv1DModel(
        input_size=input_size,
        num_filters=num_filters,
        kernel_size=kernel_size,
        dropout=dropout
    )

    model = TorchClassifier(
        model_class=model_class,
        input_size=input_size,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr
    )

    model.fit(X_train_tensor, y_train_tensor)
    y_pred = model.predict(X_val_tensor)

    acc = accuracy_score(y_val_tensor.cpu().numpy(), y_pred)
    return acc



############################## Start of the main process ############################################

print("Using GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device name:", torch.cuda.get_device_name(0))
    print("Current device:", torch.cuda.current_device())
else:
    print("GPU not detected. Using CPU.")


df = pd.read_csv("c:/Users/teodo/Desktop/nba_predictor/data/nba_games_data.csv", index_col=0)
all_games = pre_processing(df)

# Drop rows that contain NaNs from rolling/shift calculations
# engineered_cols = [
#     'PTS_rolling3', 'AST_rolling3', 'REB_rolling3', 'FG_PCT_rolling3',
#     'WIN_STREAK', 'PTS_DIFF', 'AST_DIFF', 'REB_DIFF', 'FG_PCT_DIFF'
# ]
# all_games.dropna(subset=engineered_cols, inplace=True)

# Drop NaNs only from features currently used in evaluation
#used_engineered_cols = ['PTS_rolling3', 'WIN_STREAK', 'PTS_DIFF']
#all_games.dropna(subset=used_engineered_cols, inplace=True)


le = LabelEncoder()
all_games['TEAM_ID'] = le.fit_transform(all_games['TEAM_ID'])
all_games['OPPONENT_TEAM_ID'] = le.fit_transform(all_games['OPPONENT_TEAM_ID'])



# 1. Extract the year from GAME_DATE
all_games['YEAR'] = all_games['GAME_DATE'].dt.year

# Assuming you already have `YEAR` = all_games['GAME_DATE'].dt.year
# Filter rows to keep only games from 2015 through 2025 ( Στην ουσία θέτουμε target scope για να μην είμαστε στα κουτουρού και στοχεύουμε γιόλο.)
filtered_df = all_games[(all_games['YEAR'] >= 2014) & (all_games['YEAR'] <= 2024)]


# 3. Get the unique years in ascending order
unique_years = sorted(filtered_df['YEAR'].unique())


# Define the features we want:

# Initial features:
#feature_cols = ['TEAM_ID', 'OPPONENT_TEAM_ID', 'Points_Per_Game', 'HOME_GAME', 'LAST_GAME_RESULT']


# Updated features:
feature_cols = ['TEAM_ID', 'OPPONENT_TEAM_ID', 'Points_Per_Game', 'HOME_GAME', 'LAST_GAME_RESULT']#, 'PTS_rolling3', 'WIN_STREAK']



timesteps = 1
features = len(feature_cols)

# Baseline Classical Models
ml_models = {
    "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "DecisionTree": lambda: DecisionTreeClassifier(random_state=42),
    "LinearSVC": lambda: LinearSVC(max_iter=2000, random_state=42),
    "XGBoost": lambda: XGBClassifier(eval_metric='logloss', random_state=42),
}



# ✅ Tuned Classical Models (from GridSearchCV results)
# ml_models = {
#     "RandomForest": lambda: RandomForestClassifier(
#         n_estimators=200,
#         max_depth=10,
#         min_samples_split=5,
#         random_state=42
#     ),
#     "LogisticRegression": lambda: LogisticRegression(
#         C=10,
#         penalty='l2',
#         solver='liblinear',
#         max_iter=1000,
#         random_state=42
#     ),
#     "DecisionTree": lambda: DecisionTreeClassifier(
#         max_depth=5,
#         min_samples_split=2,
#         criterion='gini',
#         random_state=42
#     ),
#     "LinearSVC": lambda: LinearSVC(
#         C=10,
#         max_iter=1000,
#         random_state=42
#     ),
#     "XGBoost": lambda: XGBClassifier(
#         eta=0.01,
#         max_depth=3,
#         n_estimators=50,
#         subsample=0.8,
#         eval_metric='logloss',
#         random_state=42
#     )
# }



# Baseline Deep Learning Models (PyTorch)
deep_models = {
    "LSTM": lambda: TorchClassifier(LSTMModel, input_size=features),
    "GRU": lambda: TorchClassifier(GRUModel, input_size=features),
    "Conv1D": lambda: TorchClassifier(Conv1DModel, input_size=features)
}


# ✅ Tuned Deep Learning Models (from Optuna optimal hyperparameters results)
# deep_models = {
#     "LSTM": lambda: TorchClassifier(
#         model_class=LSTMModel,
#         input_size=features,
#         hidden_size=32,
#         dropout=0.4944786384563793,
#         lr=0.0004934890068286345,
#         batch_size=64,
#         epochs=12
#     ),
#     "GRU": lambda: TorchClassifier(
#         model_class=GRUModel,
#         input_size=features,
#         hidden_size=32,
#         dropout=0.4944786384563793,
#         lr=0.0004934890068286345,
#         batch_size=64,
#         epochs=12
#     ),
#     "Conv1D": lambda: TorchClassifier(
#         model_class=lambda input_size: Conv1DModel(
#             input_size=input_size,
#             num_filters=32,
#             kernel_size=3,
#             dropout=0.319223061048337
#         ),
#         input_size=features,
#         batch_size=32,
#         epochs=7,
#         lr=0.00018443769951647187
#     )
# }


# -----------------------------
# Hyperparameters for our approach
# -----------------------------
context_window = 5  # number of consecutive seasons used as training
start_index = context_window  # we can skip the first 'context_window' seasons
# because we need that many seasons to form the training set

# Run evaluations
ml_results, ml_reports = evaluate_classical_models(start_index, context_window, unique_years, filtered_df, feature_cols, ml_models)
dl_results, dl_reports = evaluate_deep_models(start_index, context_window, unique_years, filtered_df, feature_cols, deep_models)

# =============================
# Aggregated Accuracy Results
# =============================
print("\n==================== ML Aggregated Accuracy ====================")
ml_results_df = pd.DataFrame(ml_results)
print(ml_results_df.groupby('model')['accuracy'].mean().sort_values(ascending=False))

print("\n==================== DL Aggregated Accuracy ====================")
dl_results_df = pd.DataFrame(dl_results)
print(dl_results_df.groupby('model')['accuracy'].mean().sort_values(ascending=False))


# =============================
# Classification Reports (ML)
# =============================
print("\n==================== ML Classification Reports ====================")
for (year, model_name), report in ml_reports.items():
    print(f"\nYear: {year} | Model: {model_name}")
    print(f"Accuracy: {report['accuracy'] * 100:.2f}%")
    print(f"Win Precision: {report['Win']['precision']:.2f} | Recall: {report['Win']['recall']:.2f} | F1-score: {report['Win']['f1-score']:.2f}")
    print(f"Loss Precision: {report['Loss']['precision']:.2f} | Recall: {report['Loss']['recall']:.2f} | F1-score: {report['Loss']['f1-score']:.2f}")



# =============================
# Classification Reports (DL)
# =============================
print("\n==================== DL Classification Reports ====================")
for entry in dl_reports:
    year = entry['test_year']
    model_name = entry['model']
    report = entry['report']
    
    print(f"\nYear: {year} | Model: {model_name}")
    print(f"Accuracy: {report['accuracy']*100:.2f}%")
    print(f"Precision (Win): {report['Win']['precision']:.2f} | Recall: {report['Win']['recall']:.2f} | F1: {report['Win']['f1-score']:.2f}")


# =============================================
# 🔍 Step 1: Hyperparameter Tuning for ML Models
# =============================================

# =====================
# Phase 2A: DL Tuning
# =====================

print("\n==================== HYPERPARAMETER TUNING (ML Models) ====================")

X_all = filtered_df[feature_cols]
y_all = filtered_df['WIN']

print("\n🔧 Tuning Random Forest...")
best_rf = tune_random_forest(X_all, y_all)

print("\n🔧 Tuning Logistic Regression...")
best_lr = tune_logistic_regression(X_all, y_all)

print("\n🔧 Tuning Linear SVC...")
best_svc = tune_linear_svc(X_all, y_all)

print("\n🔧 Tuning XGBoost...")
best_xgb = tune_xgboost(X_all, y_all)

print("\n🔧 Tuning Decision Tree...")
best_dt = tune_decision_tree(X_all, y_all)





# =====================
# Phase 2B: DL Tuning
# =====================

print("\n==================== PHASE 2B: DL HYPERPARAMETER TUNING ====================")

# ✅ Select a window (manually, for tuning only)
tune_start = start_index  # Same as used in evaluation
train_years = unique_years[tune_start - 5:tune_start - 1]
val_year    = unique_years[tune_start - 1]

train_df = filtered_df[filtered_df['YEAR'].isin(train_years)]
val_df   = filtered_df[filtered_df['YEAR'] == val_year]

X_train = train_df[feature_cols]
y_train = train_df['WIN']
X_val   = val_df[feature_cols]
y_val   = val_df['WIN']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)

# Reshape
X_train_seq = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_val_seq   = X_val_scaled.reshape(X_val_scaled.shape[0], 1, X_val_scaled.shape[1])

# Convert to tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1).to(device)
X_val_tensor   = torch.tensor(X_val_seq, dtype=torch.float32).to(device)
y_val_tensor   = torch.tensor(y_val.to_numpy(), dtype=torch.float32).unsqueeze(1).to(device)

# FOR LSTM 
study = optuna.create_study(direction="maximize")
study.optimize(lambda trial: objective_lstm(trial, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, features), n_trials=30)

print("✅ LSTM Best hyperparameters:", study.best_params)
print("📈 LSTM Best validation accuracy:", study.best_value)


# For GRU
study_gru = optuna.create_study(direction="maximize")
study_gru.optimize(lambda trial: objective_gru(trial, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, features), n_trials=30)

print("✅ GRU Best hyperparameters:", study_gru.best_params)
print("📈 GRU Best validation accuracy:", study_gru.best_value)


# For Conv1D
study_conv = optuna.create_study(direction="maximize")
study_conv.optimize(lambda trial: objective_conv1d(trial, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, features), n_trials=30)

print("✅ Conv1D Best hyperparameters:", study_conv.best_params)
print("📈 Conv1D Best validation accuracy:", study_conv.best_value)







































































##########################################################################################################################################################################

# X = all_games[['TEAM_ID', 'OPPONENT_TEAM_ID', 'Points_Per_Game', 'HOME_GAME', 'LAST_GAME_RESULT']]

# #X = all_games[['TEAM_ID', 'OPPONENT_TEAM_ID', 'Points_Per_Game', 'HOME_GAME', 'LAST_GAME_RESULT', 'MIN', 'PTS', 'FGM', 'FGA', 'FG_PCT',
# #                'FG3M', 'FG3A', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB',	'AST', 'STL', 'BLK', 'TOV',	'PF']]
# # PLUS_MINUS : are a lot of missing values
# # FG3_PCT : are a lot of missing values
# y = all_games['WIN']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Train RandomForest Classifier model: 
# random_forest = RandomForestClassifier(n_estimators=100, random_state=42) # Initialize 100 decision trees
# logistic_regression = LogisticRegression(max_iter=1000, random_state=42) # Initialize 100 iteration of Logistic Regression
# decision_tree =  DecisionTreeClassifier(random_state=42) 
# svc = LinearSVC(random_state=42, max_iter=1000)
# xgboost = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)


# # Train all the machine learning models:
# random_forest.fit(X_train, y_train)
# logistic_regression.fit(X_train, y_train)
# decision_tree.fit(X_train, y_train)
# xgboost.fit(X_train, y_train)
# svc.fit(X_train, y_train)


# # Evaluation Process: 

# print('\n')
# # Print Evaluation Metrics for Random_Forest:
# y_pred_forest = random_forest.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred_forest) * 100
# print(f"Random Forest Accuracy: {accuracy:.2f}%")
# print(classification_report(y_test, y_pred_forest))

# print('\n')
# # Print Evaluation Metrics for Logistic Regression:
# y_pred_lr = logistic_regression.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred_lr) * 100
# print(f"Logistic Regression Accuracy: {accuracy:.2f}%")
# print(classification_report(y_test, y_pred_lr))

# print('\n')
# # Print Evaluation Metrics for Decision Tree Classifier:
# y_pred_tree = decision_tree.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred_tree) * 100
# print(f" Decision Tree Classifier Accuracy: {accuracy:.2f}%")
# print(classification_report(y_test, y_pred_tree))

# print('\n')
# # Print Evaluation Metrics for Support Vector Classifier:
# y_pred_svc = svc.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred_svc) * 100
# print(f" SVC Accuracy: {accuracy:.2f}%")
# print(classification_report(y_test, y_pred_svc))

# print('\n')
# #Print Evaluation Metrics for XGBoost:
# y_pred_boost = xgboost.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred_boost) * 100
# print(f"XGBOOST Accuracy: {accuracy:.2f}%")
# print(classification_report(y_test, y_pred_boost))


# #################################### FEATURE IMPORTANCE FOR EACH CLASSIFIER ##########################################

# print('\n')
# print('\n')

# # Feature Improtance Analysis for Random_Forest:
# feature_importances_forest = pd.DataFrame(random_forest.feature_importances_, index = X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
# print("Random Forest Feature Importances:\n", feature_importances_forest)
# print('\n')
# print('\n')

# # Feature Improtance Analysis for Logistic Regression:
# lr_coefficients = logistic_regression.coef_[0]  # For binary classification, shape = (n_features,)
# feature_names = X_train.columns
# # Convert to absolute values to indicate "importance"
# importances = np.abs(lr_coefficients)
# # Create a DataFrame for easy sorting and viewing
# feature_importances_lr = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
# print("Logistic Regression Feature Importances (by coef magnitude):")
# print(feature_importances_lr)
# print('\n')
# print('\n')


# # Feature Improtance Analysis for Decision Trees:
# feature_importances_tree = pd.DataFrame(decision_tree.feature_importances_, index = X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
# print("Decision Tree Feature Importances:\n", feature_importances_tree)
# print('\n')
# print('\n')

# # Feature Importance Analysis for Support Vector Classifier:
# coefficients = svc.coef_[0]
# importances = np.abs(coefficients)
# feature_importances_svc = pd.DataFrame({'feature': feature_names,'importance': importances}).sort_values('importance', ascending=False)
# print("LinearSVC Feature Importances (by coef magnitude):")
# print(feature_importances_svc)
# print('\n')
# print('\n')

# # Feature Improtance Analysis for XGBoost:
# feature_importances_xgboost = pd.DataFrame(xgboost.feature_importances_, index = X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
# print("XGBoost Feature Importances:\n", feature_importances_xgboost)
# print('\n')
# print('\n')




# 1. I am doing prediction not forecasting because basically the model knows all the data features and just predicts the outcome not the next game :

# 2. I need to carefully consider the splitting dataset because when splitting the dataset I won't be able to evaluate the next games but random games

# 3. I need to consider the features because the model is not supposed to know field goals, points_per_game etc. when it comes to predicting the Y Labels:



# load dataset:
#df = pd.read_csv("c:/Users/teodo/Desktop/nba_predictor/data/nba_games.csv", index_col=0)

#print(df.head())


# Create the plot
#g = sns.displot(df.won, kde=False)

# Save the figure
#g.figure.savefig("histogram.png", dpi=300, bbox_inches='tight')

