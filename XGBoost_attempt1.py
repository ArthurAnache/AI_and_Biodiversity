import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# ---------- Load data ----------
df_vol = pd.read_csv(
    "/Users/louis/Desktop/Tronc commun 2A/Projet S7/AI_and_Biodiversity/raw_data/STOC_pressions_bio_assol_meteo_pesti_140525.csv",
    sep=";", encoding="latin1", low_memory=False
)
df_norm = pd.read_csv(
    "/Users/louis/Desktop/Tronc commun 2A/Projet S7/AI_and_Biodiversity/raw_data/STOC_pressions_bio_assol_meteo_pesti_routes_lum_120625.csv",
    sep=";", encoding="latin1", low_memory=False
)

# Use df_norm as main table (routes / light features present here)
df = df_norm.copy()

# ---------- Columns ----------
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

# ---------- Basic hygiene ----------
# Ensure Year is numeric
if not np.issubdtype(np.array(df[date_col]).dtype, np.number):
    df[date_col] = pd.to_numeric(df[date_col], errors="coerce")

# Force numeric dtype for numeric feature columns (avoid mixed-types warnings)
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

# Verify required columns exist
needed = [site_col, date_col, "species", target_col] + monthly_cols + static_cols[1:]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns in df_norm: {missing}")

# ---------- Sort and build lags by (site, species) ----------
df = df.sort_values([site_col, "species", date_col])
group_keys = [site_col, "species"]

# y_{t-1}
df["y_lag1"] = df.groupby(group_keys)[target_col].shift(1)

# x_{t-1} for all monthly features
# Create all lagged columns at once to avoid fragmentation
lagged = df.groupby(group_keys)[monthly_cols].shift(1)
lagged.columns = [f"{c}_lag1" for c in lagged.columns]

# Concatenate them in one go
df = pd.concat([df, lagged], axis=1)

# Drop rows with missing lags or missing target
lag_cols = ["y_lag1"] + [f"{c}_lag1" for c in monthly_cols]
df_model = df.dropna(subset=lag_cols + [target_col]).copy()

# One-hot encode species
df_model = pd.get_dummies(df_model, columns=["species"], drop_first=True)

# ---------- Final feature matrix ----------
X_cols = (
    monthly_cols
    + [f"{c}_lag1" for c in monthly_cols]
    + ["y_lag1"]
    + [c for c in df_model.columns if c.startswith("species_")]
    + static_cols[1:]
)
X = df_model[X_cols]
y = df_model[target_col]

# ---------- Chronological split (80% earliest years -> train) ----------
cutoff = df_model[date_col].quantile(0.8)
is_train = df_model[date_col] <= cutoff
X_train, X_valid = X[is_train], X[~is_train]
y_train, y_valid = y[is_train], y[~is_train]

# ---------- Train with native XGBoost API + Early Stopping ----------
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X_train.columns))
dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=list(X_valid.columns))

params = {
    "objective": "reg:squarederror",
    "eta": 0.05,            # learning_rate
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "nthread": 8,
    "seed": 42
}

booster = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=2000,
    evals=[(dtrain, "train"), (dvalid, "valid")],
    early_stopping_rounds=50,
    verbose_eval=False,
)

# ---------- Evaluate ----------
preds = booster.predict(dvalid)
mae = mean_absolute_error(y_valid, preds)
print("Validation MAE:", mae)

# ---------- Feature importances ----------
importance_type = "gain"
imp_dict = booster.get_score(importance_type=importance_type)
feat_names = list(X_train.columns)
imp_series = pd.Series({f: imp_dict.get(f, 0.0) for f in feat_names}).sort_values(ascending=False)

print(f"\nTop 20 features ({importance_type}):\n", imp_series.head(20))

# ---------- Compare full vs reduced model (without species_) ----------
imp_no_species = imp_series[~imp_series.index.str.startswith("species_")]

# Compute total gain and cumulative contribution
total_gain = imp_no_species.sum()
threshold = 0.20 * total_gain  # keep top 80% of gain
cumulative_gain = imp_no_species.cumsum()
selected_features = cumulative_gain[cumulative_gain <= total_gain - threshold].index.tolist()

print(f"\nSelected {len(selected_features)} features covering 80% of total gain (no species_):")
print(selected_features[:15], "...")

# Reduced feature matrix
X_reduced = X[selected_features]
X_train_r, X_valid_r = X_reduced[is_train], X_reduced[~is_train]

# Retrain reduced model
dtrain_r = xgb.DMatrix(X_train_r, label=y_train, feature_names=list(X_train_r.columns))
dvalid_r = xgb.DMatrix(X_valid_r, label=y_valid, feature_names=list(X_valid_r.columns))

booster_reduced = xgb.train(
    params=params,
    dtrain=dtrain_r,
    num_boost_round=booster.best_iteration,
    evals=[(dtrain_r, "train"), (dvalid_r, "valid")],
    verbose_eval=False,
)

# Evaluate reduced model
preds_r = booster_reduced.predict(dvalid_r)
mae_reduced = mean_absolute_error(y_valid, preds_r)

print("\n--- Model comparison ---")
print(f"Full model MAE     : {mae:.5f}")
print(f"Reduced model MAE  : {mae_reduced:.5f}")
print(f"ΔMAE (reduced-full): {mae_reduced - mae:.5f}")
