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
from utils import tune_decision_tree, tune_linear_svc, tune_logistic_regression, tune_random_forest, tune_xgboost, objective_lstm, objective_conv1d, objective_gru, evaluate_model_across_folds

#################### PRE-PROCESSING FUNCTION: #################

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
        'FG3M', 'FG3A',
        'FTM', 'FTA', 'FT_PCT',
        'OREB', 'DREB', 'REB',
        'AST', 'STL', 'BLK', 'TOV', 'PF'
    ]
    df.dropna(subset=features, inplace=True)

    all_games = df.copy()
    all_games['GAME_DATE'] = pd.to_datetime(all_games['GAME_DATE'])
    all_games = all_games.sort_values(by=['GAME_DATE']).reset_index(drop=True)
    all_games['WIN'] = all_games['WL'].apply(lambda x: 1 if x == 'W' else 0)
    all_games['PTS'] = all_games['PTS'].astype(float)
    all_games['Points_Per_Game'] = all_games.groupby('TEAM_ID')['PTS'].transform('mean')

    team_abbr_to_id = dict(zip(all_games['TEAM_ABBREVIATION'], all_games['TEAM_ID']))

    def get_opponent_team_id(matchup, team_abbr_to_id, team_id):
        if '@' in matchup:
            opponent_abbr = matchup.split(' @ ')[-1]
        else:
            opponent_abbr = matchup.split(' vs. ')[-1]
        return team_abbr_to_id.get(opponent_abbr, team_id)

    all_games['OPPONENT_TEAM_ID'] = all_games.apply(lambda row: get_opponent_team_id(row['MATCHUP'], team_abbr_to_id, row['TEAM_ID']), axis=1)
    all_games['HOME_GAME'] = all_games['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    all_games['LAST_GAME_RESULT'] = all_games.groupby('TEAM_ID')['WIN'].shift(1).fillna(0)

    # ✅ Assign proper NBA SEASON instead of calendar year
    def get_season_year(date):
        return date.year if date.month >= 10 else date.year - 1

    all_games['SEASON'] = all_games['GAME_DATE'].apply(get_season_year)

    return all_games

#################### FIT & EVALUATION FUNCTIONS FOR BOTH ML/DL MODELS: #################


# Fit and evaluation for machine learning models: 
def evaluate_classical_models_time_series(df, feature_cols, models, n_splits=5):
    results = []
    reports = {}

    df_sorted = df.sort_values(by="GAME_DATE").reset_index(drop=True)
    X = df_sorted[feature_cols]
    y = df_sorted['WIN']
    seasons = df_sorted['SEASON'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        season_test = np.unique(seasons[test_idx])

        print(f"\n=== [ML Fold {fold+1}] Predicting on seasons: {season_test.tolist()} ===")

        for model_name, model_fn in models.items():
            model = model_fn()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred) * 100

            print(f"{model_name:<16} Accuracy: {acc:.2f}%")

            results.append({
                'fold': fold + 1,
                'season': "-".join(map(str, season_test)),
                'model': model_name,
                'accuracy': acc
            })

            report = classification_report(y_test, y_pred, target_names=["Loss", "Win"], output_dict=True)
            reports[(fold + 1, model_name)] = report

    return results, reports


# Fit and evaluation for deep learning models:
def evaluate_deep_models_time_series(filtered_df, feature_cols, models, n_splits=5):
    results = []
    reports = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = filtered_df[feature_cols]
    y = filtered_df['WIN']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_seq = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    X_conv = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(device)
    X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.to_numpy(), dtype=torch.float32).unsqueeze(1).to(device)

    tscv = TimeSeriesSplit(n_splits = n_splits)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_seq)):
        print(f"\n=== [DL Fold {fold + 1}] ===")

        for model_name, model_fn in models.items():
            model = model_fn()

            if model_name == "Conv1D":
                model.fit(X_conv[train_idx], y_tensor[train_idx])
                y_pred = model.predict(X_conv[test_idx])
            else:
                model.fit(X_seq_tensor[train_idx], y_tensor[train_idx])
                y_pred = model.predict(X_seq_tensor[test_idx])

            acc = accuracy_score(y_tensor[test_idx].cpu().numpy(), y_pred)
            print(f"{model_name:<16} Accuracy: {acc * 100:.2f}%")

            results.append({
                'fold': fold + 1,
                'model': model_name,
                'accuracy': acc * 100
            })

            report = classification_report(y_tensor[test_idx].cpu().numpy(), y_pred, target_names=["Loss", "Win"], output_dict=True)
            reports.append({
                'fold': fold + 1,
                'model': model_name,
                'report': report
            })

    return results, reports



# Inside your main process:

print("Using GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device name:", torch.cuda.get_device_name(0))
    print("Current device:", torch.cuda.current_device())
else:
    print("GPU not detected. Using CPU.")


df = pd.read_csv("c:/Users/teodo/Desktop/nba_predictor/data/nba_games_data.csv", index_col=0)
all_games = pre_processing(df)

le = LabelEncoder()
all_games['TEAM_ID'] = le.fit_transform(all_games['TEAM_ID'])
all_games['OPPONENT_TEAM_ID'] = le.fit_transform(all_games['OPPONENT_TEAM_ID'])

# ✅ Filter by season, not year
target_seasons = list(range(2014, 2025))
filtered_df = all_games[all_games['SEASON'].isin(target_seasons)]
unique_seasons = sorted(filtered_df['SEASON'].unique())

feature_cols = ['TEAM_ID', 'OPPONENT_TEAM_ID', 'Points_Per_Game', 'HOME_GAME', 'LAST_GAME_RESULT']



ml_models = {
    "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "DecisionTree": lambda: DecisionTreeClassifier(random_state=42),
    "LinearSVC": lambda: LinearSVC(max_iter=2000, random_state=42),
    "XGBoost": lambda: XGBClassifier(eval_metric='logloss', random_state=42),
}

deep_models = {
    "LSTM": lambda: TorchClassifier(LSTMModel, input_size=len(feature_cols)),
    "GRU": lambda: TorchClassifier(GRUModel, input_size=len(feature_cols)),
    "Conv1D": lambda: TorchClassifier(Conv1DModel, input_size=len(feature_cols))
}

ml_results, ml_reports = evaluate_classical_models_time_series(filtered_df, feature_cols, ml_models, n_splits=5)

dl_results, dl_reports = evaluate_deep_models_time_series(filtered_df, feature_cols, deep_models,  n_splits=5)






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
for entry in dl_reports:
    fold = entry['fold']
    model_name = entry['model']
    report = entry['report']
    
    print(f"\nFold: {fold} | Model: {model_name}")
    print(f"Accuracy: {report['accuracy'] * 100:.2f}%")
    print(f"Precision (Win): {report['Win']['precision']:.2f} | Recall: {report['Win']['recall']:.2f} | F1: {report['Win']['f1-score']:.2f}")
    print(f"Precision (Loss): {report['Loss']['precision']:.2f} | Recall: {report['Loss']['recall']:.2f} | F1: {report['Loss']['f1-score']:.2f}")






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

# Use the same TimeSeriesSplit approach for tuning
X_all = filtered_df[feature_cols]
y_all = filtered_df['WIN']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# Reshape for LSTM/GRU input: [samples, timesteps=1, features]
X_seq = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

# Convert to tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_tensor_seq = torch.tensor(X_seq, dtype=torch.float32).to(device)
X_tensor_conv = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(device)
y_tensor = torch.tensor(y_all.to_numpy(), dtype=torch.float32).unsqueeze(1).to(device)

# Split once for tuning (you could loop over several folds too, but let's keep it consistent)
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X_seq):
    X_train_seq, X_val_seq = X_tensor_seq[train_idx], X_tensor_seq[val_idx]
    X_train_conv, X_val_conv = X_tensor_conv[train_idx], X_tensor_conv[val_idx]
    y_train_tensor, y_val_tensor = y_tensor[train_idx], y_tensor[val_idx]
    break  # Only use the first fold for tuning

# Get number of features
input_size = X_scaled.shape[1]


# === LSTM ===
study_lstm = optuna.create_study(direction="maximize")
study_lstm.optimize(lambda trial: objective_lstm(trial, X_tensor_seq, X_tensor_conv, y_tensor, input_size), n_trials=30)

print("✅ LSTM Best hyperparameters:", study_lstm.best_params)
print("📈 LSTM Best validation accuracy:", study_lstm.best_value)


# === GRU ===
study_gru = optuna.create_study(direction="maximize")
study_gru.optimize(lambda trial: objective_gru(trial, X_tensor_seq, X_tensor_conv, y_tensor, input_size), n_trials=30)

print("✅ GRU Best hyperparameters:", study_gru.best_params)
print("📈 GRU Best validation accuracy:", study_gru.best_value)

# === Conv1D ===
study_conv = optuna.create_study(direction="maximize")
study_conv.optimize(lambda trial: objective_conv1d(trial, X_tensor_seq, X_tensor_conv, y_tensor, input_size), n_trials=30)

print("✅ Conv1D Best hyperparameters:", study_conv.best_params)
print("📈 Conv1D Best validation accuracy:", study_conv.best_value)

