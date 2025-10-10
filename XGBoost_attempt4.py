import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


df_vol = pd.read_csv(
    "/Users/louis/Desktop/Tronc commun 2A/Projet S7/AI_and_Biodiversity/raw_data/STOC_pressions_bio_assol_meteo_pesti_140525.csv",
    sep=";", encoding="latin1", low_memory=False
)
df_norm = pd.read_csv(
    "/Users/louis/Desktop/Tronc commun 2A/Projet S7/AI_and_Biodiversity/raw_data/STOC_pressions_bio_assol_meteo_pesti_routes_lum_120625.csv",
    sep=";", encoding="latin1", low_memory=False
)


df = df_norm.copy()

months = range(1, 13)
monthly_features = [
    "Temps_mean_mensu_grid_{}",
    "Precipitation_mean_mensu_grid_{}",
    "TAMPLI_mean_mensu_grid_{}",
]
monthly_cols = [f.format(i) for f in monthly_features for i in months]

static_cols = [
    "species",
    "Area_grid_tot_Urba", "ValueSPred_Urba",
    "Area_grid_tot_For", "ValueSPred_For",
    "Area_grid_tot_CC", "ValueSPred_CC",
    "Area_grid_tot_M", "ValueSPred_M",
    "Area_grid_tot_ZH", "ValueSPred_ZH",
    "mean_concentration_air_grid_sum",
    "mean_tii_grid_sum",
    "mean_tii_scale_grid_sum",
    "mean_concentration_water_grid_sum",
    "all_pesticide_exposure_grid_sum",
    "SurfBio_cell", "nb_exp_cell",
    "Long_route", "nb_roads",
    "Lum_mean",
]

target_col = "abondance_capped"
site_col = "site"
date_col = "Year"


if not np.issubdtype(np.array(df[date_col]).dtype, np.number):
    df[date_col] = pd.to_numeric(df[date_col], errors="coerce")

num_cols = monthly_cols + [
    "Area_grid_tot_Urba","ValueSPred_Urba",
    "Area_grid_tot_For","ValueSPred_For",
    "Area_grid_tot_CC","ValueSPred_CC",
    "Area_grid_tot_M","ValueSPred_M",
    "Area_grid_tot_ZH","ValueSPred_ZH",
    "mean_concentration_air_grid_sum","mean_tii_grid_sum",
    "mean_tii_scale_grid_sum","mean_concentration_water_grid_sum",
    "all_pesticide_exposure_grid_sum",
    "SurfBio_cell","nb_exp_cell","Long_route","nb_roads","Lum_mean",
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

needed = [site_col, date_col, "species", target_col] + monthly_cols + static_cols[1:]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns in df_norm: {missing}")

df = df.sort_values([site_col, "species", date_col])
group_keys = [site_col, "species"]

# y_{t-1}
df["y_lag1"] = df.groupby(group_keys)[target_col].shift(1)

# x_{t-1} for all monthly features (build all at once to avoid fragmentation)
lagged = df.groupby(group_keys)[monthly_cols].shift(1)
lagged.columns = [f"{c}_lag1" for c in lagged.columns]
df = pd.concat([df, lagged], axis=1)

# Drop rows with missing lags or missing target
lag_cols = ["y_lag1"] + [f"{c}_lag1" for c in monthly_cols]
df_model = df.dropna(subset=lag_cols + [target_col]).copy()

# Keep meta BEFORE one-hot so we can display species & site later 
meta = df_model[[site_col, "species", date_col]].copy()

# One-hot encode species
df_model = pd.get_dummies(df_model, columns=["species"], drop_first=True)

X_cols = (
    monthly_cols
    + [f"{c}_lag1" for c in monthly_cols]
    + ["y_lag1"]
    + [c for c in df_model.columns if c.startswith("species_")]
    + static_cols[1:]
)
X = df_model[X_cols]
y = df_model[target_col]


cutoff = df_model[date_col].quantile(0.8)
is_train = df_model[date_col] <= cutoff
X_train, X_valid = X[is_train], X[~is_train]
y_train, y_valid = y[is_train], y[~is_train]


dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X_train.columns))
dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=list(X_valid.columns))

params = {
    "objective": "reg:squarederror",
    "eta": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "nthread": 8,
    "seed": 42
}
#params["objective"] = "reg:pseudohubererror" # Pseudo-Huber
params["objective"] = "reg:absoluteerror"  # MAE

booster = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=2000,
    evals=[(dtrain, "train"), (dvalid, "valid")],
    early_stopping_rounds=50,
    verbose_eval=False,
)


preds = booster.predict(dvalid)
mae = mean_absolute_error(y_valid, preds)
rmse = np.sqrt(mean_squared_error(y_valid, preds))

print("\n===== MODEL PERFORMANCE =====")
print(f"Validation MAE  : {mae:.4f}")
print(f"Validation RMSE : {rmse:.4f}")


meta_valid = meta[~is_train].copy()
preview = meta_valid.assign(
    y_true = y_valid.values,
    y_pred = preds
)
preview["abs_err"] = (preview["y_true"] - preview["y_pred"]).abs()

print("\n===== SAMPLE PREDICTIONS (site, species, year) =====")
print(preview.head(15).to_string(index=False))

print("\n===== WORST 15 ERRORS (validation) =====")
print(preview.sort_values("abs_err", ascending=False).head(15).to_string(index=False))

plt.figure(figsize=(8,5))
plt.scatter(y_valid, preds, alpha=0.5)
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--')
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.title(f"XGBoost — True vs Predicted (MAE={mae:.2f}, RMSE={rmse:.2f})")
plt.tight_layout()
plt.show()

residuals = y_valid - preds
plt.figure(figsize=(8,4))
plt.hist(residuals, bins=50, alpha=0.7)
plt.axvline(0, color='black', linestyle='--')
plt.title("Residuals distribution (y_true - y_pred)")
plt.xlabel("Prediction error")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

importance_type = "gain"
imp_dict = booster.get_score(importance_type=importance_type)
feat_names = list(X_train.columns)
imp_series = pd.Series({f: imp_dict.get(f, 0.0) for f in feat_names}).sort_values(ascending=False)

print(f"\n===== TOP 20 FEATURES ({importance_type}) =====")
print(imp_series.head(20))
