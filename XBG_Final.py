# =========================
# 1. IMPORTS
# =========================
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# =========================
# 2. CHARGEMENT DES DONNÉES
# =========================
path = "raw_data/STOC_pressions_bio_assol_meteo_pesti_routes_lum_120625.csv"
df_full = pd.read_csv(path, sep=";", encoding="latin1", low_memory=False)

print(df_full.columns.tolist())

# =========================
# 3. TRI TEMPOREL (INDISPENSABLE)
# =========================
df = df_full.sort_values(["id_cell", "Year"]).copy()

# =========================
# 4. VARIABLES CLIMATIQUES ANNUELLES
# =========================
temps_cols = [f"Temps_mean_mensu_grid_{i}" for i in range(1, 13)]
prec_cols  = [f"Precipitation_mean_mensu_grid_{i}" for i in range(1, 13)]

df["Temps_mean_AllY"] = df[temps_cols].mean(axis=1, skipna=True)
df["Precipitation_mean_AllY"] = df[prec_cols].mean(axis=1, skipna=True)

# =========================
# 5. PRESSIONS EXPLICATIVES (SANS URBANISATION)
# =========================
base_pressures = [
    "ValueSPred_For",
    "ValueSPred_CC",
    "ValueSPred_M",
    "ValueSPred_ZH",
    "all_pesticide_exposure_grid_max",
    "SurfBio_cell",
    "Long_route",
    "nb_roads",
    "Lum_max",
    "Temps_mean_AllY",
    "Precipitation_mean_AllY"
]

# =========================
# 6. CRÉATION DES LAGS t-1 UNIQUEMENT
# =========================
for col in base_pressures:
    df[f"{col}_t-1"] = df.groupby("id_cell")[col].shift(1)

# =========================
# 7. DATASET FINAL
# =========================
target = "ValueSPred_Urba"

features = [f"{col}_t-1" for col in base_pressures]

df_model = df[features + [target]].copy()

# Conversion numérique
for col in df_model.columns:
    df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

# Suppression des lignes sans lag
df_model = df_model.dropna()

X = df_model[features]
y = df_model[target]

print("=" * 70)
print("XGBoost – Urbanisation(t) ~ Pressions(t-1)")
print("=" * 70)
print(f"Nombre d'observations finales : {len(df_model)}")

# =========================
# 8. SPLIT TRAIN / TEST (ALÉATOIRE, COMME AVANT)
# =========================
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 9. MODÈLE XGBOOST
# =========================
model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.5,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

model.fit(X_tr, y_tr)

# =========================
# 10. ÉVALUATION
# =========================
r2_train = r2_score(y_tr, model.predict(X_tr))
r2_test  = r2_score(y_te, model.predict(X_te))

print("\n📊 RÉSULTATS")
print("-" * 70)
print(f"R² train : {r2_train:.3f}")
print(f"R² test  : {r2_test:.3f}")
print(f"Overfitting (train - test) : {r2_train - r2_test:.3f}")
print("-" * 70)