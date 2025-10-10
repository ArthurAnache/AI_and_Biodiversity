import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras import layers, callbacks, optimizers, losses, metrics

# -------- Reproducibility --------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

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

# ---------- Sort and build lags by (site, species) ----------
df = df.sort_values([site_col, "species", date_col])
group_keys = [site_col, "species"]

# y_{t-1}
df["y_lag1"] = df.groupby(group_keys)[target_col].shift(1)

# x_{t-1} for all monthly features
lagged = df.groupby(group_keys)[monthly_cols].shift(1)
lagged.columns = [f"{c}_lag1" for c in lagged.columns]
df = pd.concat([df, lagged], axis=1)

# Drop rows with missing lags or missing target
lag_cols = ["y_lag1"] + [f"{c}_lag1" for c in monthly_cols]
df_model = df.dropna(subset=lag_cols + [target_col]).copy()

# ===== Keep meta BEFORE one-hot so we can display species & site later =====
meta = df_model[[site_col, "species", date_col]].copy()

# One-hot encode species (simplest to start with)
df_model = pd.get_dummies(df_model, columns=["species"], drop_first=True)

# ---------- Feature matrix ----------
X_cols = (
    monthly_cols
    + [f"{c}_lag1" for c in monthly_cols]
    + ["y_lag1"]
    + [c for c in df_model.columns if c.startswith("species_")]
    + static_cols[1:]
)
X = df_model[X_cols].copy()
y = df_model[target_col].astype(float).copy()

# ---------- Chronological split ----------
cutoff = df_model[date_col].quantile(0.8)
is_train = df_model[date_col] <= cutoff
X_train, X_valid = X[is_train], X[~is_train]
y_train, y_valid = y[is_train], y[~is_train]
meta_valid = meta[~is_train].copy()

# ---------- Scale numeric features (not one-hot) ----------
# Detect one-hot columns for species
one_hot_cols = [c for c in X_train.columns if c.startswith("species_")]
# Numeric = all features minus one-hot
numeric_cols = [c for c in X_train.columns if c not in one_hot_cols]

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_valid_scaled = X_valid.copy()

X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_valid_scaled[numeric_cols] = scaler.transform(X_valid[numeric_cols])

# Convert to float32 for tf
X_train_np = X_train_scaled.values.astype(np.float32)
X_valid_np = X_valid_scaled.values.astype(np.float32)
y_train_np = y_train.values.astype(np.float32)
y_valid_np = y_valid.values.astype(np.float32)

# ---------- Build the MLP ----------
def rmse_keras(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

input_dim = X_train_np.shape[1]

model = tf.keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.BatchNormalization(),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.25),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.25),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="linear")
])

# You can switch loss to "mae" to match your XGB objective exactly
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss=losses.Huber(delta=1.0),
    metrics=[metrics.MeanAbsoluteError(name="mae"), rmse_keras]
)

early_stop = callbacks.EarlyStopping(
    monitor="val_mae", patience=20, restore_best_weights=True
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_mae", factor=0.5, patience=8, min_lr=1e-5, verbose=1
)

history = model.fit(
    X_train_np, y_train_np,
    validation_data=(X_valid_np, y_valid_np),
    epochs=500,
    batch_size=1024,
    callbacks=[early_stop, reduce_lr],
    verbose=0
)

# ---------- Evaluate ----------
preds = model.predict(X_valid_np, verbose=0).ravel()
mae = mean_absolute_error(y_valid_np, preds)
rmse = np.sqrt(mean_squared_error(y_valid_np, preds))

print("\n===== MODEL PERFORMANCE (MLP) =====")
print(f"Validation MAE  : {mae:.4f}")
print(f"Validation RMSE : {rmse:.4f}")

# ---------- Readable preview with site + species + year ----------
preview = meta_valid.assign(
    y_true = y_valid.values,
    y_pred = preds
)
preview["abs_err"] = (preview["y_true"] - preview["y_pred"]).abs()

print("\n===== SAMPLE PREDICTIONS (site, species, year) =====")
print(preview.head(15).to_string(index=False))

print("\n===== WORST 15 ERRORS (validation) =====")
print(preview.sort_values("abs_err", ascending=False).head(15).to_string(index=False))

# ---------- Plot predictions vs true ----------
plt.figure(figsize=(8,5))
plt.scatter(y_valid_np, preds, alpha=0.5)
min_v, max_v = float(np.min(y_valid_np)), float(np.max(y_valid_np))
plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.title(f"MLP — True vs Predicted (MAE={mae:.2f}, RMSE={rmse:.2f})")
plt.tight_layout()
plt.show()

# ---------- Plot residuals ----------
residuals = y_valid_np - preds
plt.figure(figsize=(8,4))
plt.hist(residuals, bins=50, alpha=0.7)
plt.axvline(0, color='black', linestyle='--')
plt.title("Residuals distribution (y_true - y_pred) — MLP")
plt.xlabel("Prediction error")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()