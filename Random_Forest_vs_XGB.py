import time
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
from tqdm import tqdm

moyenne_meteorologiques = False

print("Starting script...")

# ---------------------------
# Load data
# ---------------------------
if moyenne_meteorologiques:
    X = pd.read_csv("raw_data/pressions_petit_moyenne.csv", encoding="latin1")
    y = pd.read_csv("raw_data/abondances_petit_moyenne.csv", encoding="latin1")
else:
    X = pd.read_csv("raw_data/pressions_petit.csv", encoding="latin1")
    y = pd.read_csv("raw_data/abondances_petit.csv", encoding="latin1")

# Ensure y is 1D
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]

# Save species before encoding
species_col = X["species"].copy()

# One-hot encode species
X = pd.get_dummies(X, columns=["species"], prefix="species")

# Train/test split
X_train, X_test, y_train, y_test, species_train, species_test = train_test_split(
    X, y, species_col, test_size=0.2, random_state=42
)

# ---------------------------
# XGBoost with manual training loop + tqdm
# ---------------------------
feature_names = list(X_train.columns)

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
}

num_round = 300
print("Training XGBoost...")

pbar_xgb = tqdm(total=num_round, desc="Entraînement XGBoost")

booster = None
for i in range(num_round):
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=1,
        xgb_model=booster
    )
    pbar_xgb.update(1)

pbar_xgb.close()
xgb_model = booster

y_pred_xgb = xgb_model.predict(dtest)
print("XGBoost trained.")

# ---------------------------
# Random Forest model + training (allégée + timing)
# ---------------------------
print("Training RandomForest...")

start_time = time.time()

rf_model = RandomForestRegressor(
    n_estimators=150,   # moins d'arbres pour aller plus vite
    max_depth=15,      # limite la profondeur
    n_jobs=2,          # évite de surcharger le CPU
    random_state=42,
)

rf_model.fit(X_train, y_train)

elapsed = time.time() - start_time
print(f"RandomForest trained in {elapsed:.2f} seconds.")

y_pred_rf = rf_model.predict(X_test)

# ---------------------------
# Results table for XGBoost
# ---------------------------
resultats_xgb = pd.DataFrame({
    "Species": species_test.values,
    "Reel": y_test.values,
    "Pred": y_pred_xgb
})

print("\nQuelques prédictions XGBoost :")
print(resultats_xgb.head())

# ---------------------------
# Per-species metrics (XGBoost)
# ---------------------------
scores = []
for espece, subset in resultats_xgb.groupby("Species"):
    if len(subset) > 3:
        rmse = np.sqrt(mean_squared_error(subset["Reel"], subset["Pred"]))
        mae = mean_absolute_error(subset["Reel"], subset["Pred"])
        r2 = r2_score(subset["Reel"], subset["Pred"])
        scores.append({"Espèce": espece, "RMSE": rmse, "MAE": mae, "R²": r2})

scores_df = pd.DataFrame(scores).sort_values(by="RMSE")
print("\nClassement XGBoost par espèce :")
print(scores_df)

# ---------------------------
# Global scores
# ---------------------------
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\nScores globaux XGBoost :")
print(f"RMSE : {rmse_xgb:.4f}")
print(f"MAE  : {mae_xgb:.4f}")
print(f"R²   : {r2_xgb:.4f}")

print("\nScores globaux Random Forest :")
print(f"RMSE : {rmse_rf:.4f}")
print(f"MAE  : {mae_rf:.4f}")
print(f"R²   : {r2_rf:.4f}")

# ---------------------------
# Scatter plot (models vs true values)
# ---------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_xgb, alpha=0.4, label="XGBoost")
plt.scatter(y_test, y_pred_rf, alpha=0.4, label="Random Forest")
plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.title("Comparaison des modèles")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    label="Idéal"
)
plt.legend()
plt.show()

# ---------------------------
# XGBoost feature importance (with species grouped)
# ---------------------------
species_cols = [col for col in X_train.columns if col.startswith("species_")]

importance_dict = xgb_model.get_score(importance_type="gain")

feature_importance = pd.DataFrame({
    "Feature": list(importance_dict.keys()),
    "Importance": list(importance_dict.values()),
}).sort_values(by="Importance", ascending=False)

print("\nImportance des features (XGBoost) :")
print(feature_importance)

species_importance = feature_importance[
    feature_importance["Feature"].isin(species_cols)
]["Importance"].sum()

print(f"\nImportance totale de 'Espèce' : {species_importance:.4f}")

feature_importance["Feature_mod"] = feature_importance["Feature"].apply(
    lambda x: "Espèce" if x in species_cols else x
)

grouped_importance = (
    feature_importance
    .groupby("Feature_mod")["Importance"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))
plt.barh(grouped_importance.index, grouped_importance.values)
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Importance des variables (XGBoost)")
plt.show()

print("Script finished.")
