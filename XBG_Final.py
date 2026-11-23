# =========================
# 1. IMPORTS
# =========================
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# =========================
# 2. CHARGEMENT DES DONNÉES
# =========================
path = "raw_data/STOC_pressions_bio_assol_meteo_pesti_routes_lum_120625.csv"
df_full = pd.read_csv(path, sep=";", encoding="latin1", low_memory=False)

# =========================
# 3. VARIABLES CLIMATIQUES ANNUELLES
# =========================
temps_cols = [f"Temps_mean_mensu_grid_{i}" for i in range(1, 13)]
prec_cols  = [f"Precipitation_mean_mensu_grid_{i}" for i in range(1, 13)]

df = df_full.copy()
df["Temps_mean_AllY"] = df[temps_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
df["Precipitation_mean_AllY"] = df[prec_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

# =========================
# 4. PRESSIONS EXPLICATIVES (SANS URBANISATION)
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
target = "ValueSPred_Urba"

for col in base_pressures + [target]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# 5. UNE LIGNE PAR (CELLULE, ANNÉE)
# =========================
# Le fichier contient ~30 lignes par cellule et par année (une par espèce).
# Les pressions sont identiques pour toutes ces lignes : on les regroupe,
# sinon le modèle voit des doublons à la fois en train et en test.
df_cell = (
    df.groupby(["id_cell", "Year"])[base_pressures + [target]]
      .mean()
      .reset_index()
)

# =========================
# 6. LAGS t-1 (VRAIE ANNÉE PRÉCÉDENTE)
# =========================
# On décale l'année de +1 puis on fusionne : la valeur de l'année N-1
# est associée à l'année N. Si l'année N-1 manque, la ligne disparaît.
df_prev = df_cell[["id_cell", "Year"] + base_pressures].copy()
df_prev["Year"] = df_prev["Year"] + 1
df_prev = df_prev.rename(columns={col: f"{col}_t-1" for col in base_pressures})

features = [f"{col}_t-1" for col in base_pressures]

df_model = df_cell[["id_cell", "Year", target]].merge(
    df_prev, on=["id_cell", "Year"], how="inner"
)

print("=" * 70)
print("XGBoost – Urbanisation(t) ~ Pressions(t-1)")
print("=" * 70)
print(f"Couples (cellule, année) avec année précédente dispo : {len(df_model)}")

# Valeurs manquantes par variable (pour repérer celles qui suppriment des lignes)
print("\nValeurs manquantes par variable :")
print(df_model[features + [target]].isna().sum().sort_values(ascending=False).to_string())

df_model = df_model.dropna()
print(f"\nNombre d'observations finales (après dropna) : {len(df_model)}")
print(f"Nombre de cellules : {df_model['id_cell'].nunique()}")

X = df_model[features]
y = df_model[target]
groups = df_model["id_cell"]

# =========================
# 7. SPLIT TRAIN / TEST PAR CELLULE
# =========================
# Les cellules du test ne sont jamais vues à l'entraînement.
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
idx_tr, idx_te = next(splitter.split(X, y, groups=groups))
X_tr, X_te = X.iloc[idx_tr], X.iloc[idx_te]
y_tr, y_te = y.iloc[idx_tr], y.iloc[idx_te]

# =========================
# 8. MODÈLE XGBOOST
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
# 9. ÉVALUATION
# =========================
r2_train = r2_score(y_tr, model.predict(X_tr))
r2_test  = r2_score(y_te, model.predict(X_te))

print("\n📊 RÉSULTATS")
print("-" * 70)
print(f"R² train : {r2_train:.3f}")
print(f"R² test  : {r2_test:.3f}")
print(f"Overfitting (train - test) : {r2_train - r2_test:.3f}")
print("-" * 70)
