############################################################
# XGBOOST : PREDIRE L'URBANISATION (ValueSPred_Urba)
# EN FONCTION DES AUTRES PRESSIONS + PREDICTIONS JUSQU'EN 2050
# (Baseline : features futures constantes au dernier niveau observé par id_cell)
############################################################

# =========================
# 1. IMPORTS
# =========================
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# =========================
# 2. CONFIG
# =========================
path = "/Users/r/Documents/CS/PROJET_S7/AI_and_Biodiversity/raw_data/#METTRE_LES_2_FICHIERS_ICI/STOC_pressions_bio_assol_meteo_pesti_routes_lum_120625.csv"

TARGET = "ValueSPred_Urba"

# Predictors = other pressures (remove urbanisation)
FEATURES = [
    "Temps_mean_AllY",
    "Precipitation_mean_AllY",
    "ValueSPred_For",
    "ValueSPred_CC",
    "ValueSPred_M",
    "ValueSPred_ZH",
    "all_pesticide_exposure_grid_max",
    "SurfBio_cell",
    "Long_route",
    "nb_roads",
    "Lum_max",
    # Optionnel : si tu veux capturer une tendance temporelle globale
    # "Year",
]

RANDOM_STATE = 42

# XGBoost params (à ajuster selon perf/overfit)
XGB_PARAMS = dict(
    n_estimators=800,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# =========================
# 3. HELPERS
# =========================
def compute_annual_climate_means(df: pd.DataFrame) -> pd.DataFrame:
    """Compute annual mean temperature and precipitation from monthly columns (1..12)."""
    temps_cols = [f"Temps_mean_mensu_grid_{i}" for i in range(1, 13)]
    prec_cols  = [f"Precipitation_mean_mensu_grid_{i}" for i in range(1, 13)]

    # If columns are missing, raise a clear error
    missing_t = [c for c in temps_cols if c not in df.columns]
    missing_p = [c for c in prec_cols if c not in df.columns]
    if missing_t or missing_p:
        raise ValueError(
            "Missing monthly climate columns. "
            f"Missing Temps: {missing_t[:5]}{'...' if len(missing_t)>5 else ''} | "
            f"Missing Precip: {missing_p[:5]}{'...' if len(missing_p)>5 else ''}"
        )

    df["Temps_mean_AllY"] = df[temps_cols].mean(axis=1, skipna=True)
    df["Precipitation_mean_AllY"] = df[prec_cols].mean(axis=1, skipna=True)
    return df

def make_future_constant_features(df_hist: pd.DataFrame,
                                 start_year: int,
                                 end_year: int,
                                 features: list[str],
                                 group_col: str = "id_cell",
                                 year_col: str = "Year") -> pd.DataFrame:
    """
    Build a future dataframe where each id_cell keeps its LAST observed feature values
    for all future years in [start_year, end_year].
    """
    # last observed row per id_cell
    last = (df_hist.sort_values([group_col, year_col])
                  .groupby(group_col)
                  .tail(1)[[group_col] + features]
                  .reset_index(drop=True))

    years = np.arange(start_year, end_year + 1)
    future_years = pd.DataFrame({year_col: years})

    # cross join
    last["_tmp"] = 1
    future_years["_tmp"] = 1
    df_future = last.merge(future_years, on="_tmp", how="outer").drop(columns=["_tmp"])

    return df_future

def apply_scenario_multipliers(df_future: pd.DataFrame,
                               scenario: dict[str, float]) -> pd.DataFrame:
    """
    Apply multiplicative scenario shocks to selected features (e.g. Lum_max * 1.3).
    """
    df_sc = df_future.copy()
    for feat, mult in scenario.items():
        if feat not in df_sc.columns:
            raise ValueError(f"Scenario feature '{feat}' not in df_future columns.")
        df_sc[feat] = df_sc[feat] * mult
    return df_sc

# =========================
# 4. LOAD + PREP DATA
# =========================
print("=" * 80)
print("LOAD DATA")
print("=" * 80)

df_full = pd.read_csv(path, low_memory=False)

# Compute annual climate features
df_full = compute_annual_climate_means(df_full)

# Keep needed cols
keep_cols = ["id_cell", "Year"] + FEATURES + [TARGET]
missing_keep = [c for c in keep_cols if c not in df_full.columns]
if missing_keep:
    raise ValueError(f"Missing required columns: {missing_keep}")

df = df_full[keep_cols].copy()

# Coerce numeric
for col in FEATURES + [TARGET]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows where target missing
df = df.dropna(subset=[TARGET]).reset_index(drop=True)

# Fill missing feature values with column means
df[FEATURES] = df[FEATURES].fillna(df[FEATURES].mean())

print(f"Rows used: {len(df):,}")
print(f"Years range: {int(df['Year'].min())} -> {int(df['Year'].max())}")
print(f"Unique cells: {df['id_cell'].nunique():,}")

# =========================
# 5. TRAIN / TEST SPLIT BY CELL (avoid leakage)
# =========================
X = df[FEATURES]
y = df[TARGET]
groups = df["id_cell"].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

# =========================
# 6. TRAIN MODEL
# =========================
print("\n" + "=" * 80)
print("TRAIN XGBOOST")
print("=" * 80)

model = XGBRegressor(**XGB_PARAMS)
model.fit(X_tr, y_tr)

pred_tr = model.predict(X_tr)
pred_te = model.predict(X_te)

r2_train = r2_score(y_tr, pred_tr)
r2_test  = r2_score(y_te, pred_te)
rmse_test = np.sqrt(mean_squared_error(y_te, pred_te))
y_std = y_te.std()
nrmse = rmse_test / y_std if y_std > 0 else np.nan

print(f"R2_train = {r2_train:.3f}")
print(f"R2_test  = {r2_test:.3f}")
print(f"RMSE_test = {rmse_test:.4f}")
print(f"NRMSE     = {nrmse:.4f}")
print(f"Overfit   = {r2_train - r2_test:.3f}")

# =========================
# 7. SHAP (global importance)
# =========================
print("\n" + "=" * 80)
print("SHAP EXPLANATIONS (GLOBAL)")
print("=" * 80)

sample_size = min(2000, len(X_te))
X_shap = X_te.sample(sample_size, random_state=RANDOM_STATE)

explainer = shap.Explainer(model, X_tr)
shap_values = explainer(X_shap)

shap_importance = np.abs(shap_values.values).mean(axis=0)
shap_rank = pd.Series(shap_importance, index=FEATURES).sort_values(ascending=False)

print("\nTop 10 features by mean(|SHAP|):")
print(shap_rank.head(10).to_string())

# SHAP bar plot
plt.figure(figsize=(10, 6))
shap_rank.sort_values(ascending=True).plot(kind="barh")
plt.title("SHAP mean(|SHAP|) - Urbanisation model")
plt.xlabel("mean(|SHAP|)")
plt.tight_layout()
plt.savefig("shap_importance_urbanisation.png", dpi=300, bbox_inches="tight")
plt.show()
print("✅ Saved: shap_importance_urbanisation.png")

# =========================
# 8. FUTURE PREDICTIONS TO 2050 (baseline: constant features)
# =========================
print("\n" + "=" * 80)
print("PREDICT URBANISATION TO 2050 (CONSTANT FEATURES BASELINE)")
print("=" * 80)

last_year = int(df["Year"].max())
start_year = last_year + 1
end_year = 2050

df_future = make_future_constant_features(
    df_hist=df,
    start_year=start_year,
    end_year=end_year,
    features=FEATURES,
    group_col="id_cell",
    year_col="Year"
)

df_future["ValueSPred_Urba_pred"] = model.predict(df_future[FEATURES])

out_future = "urbanisation_predictions_to_2050_constant_features.csv"
df_future.to_csv(out_future, index=False)
print(f"✅ Saved: {out_future}")
print(df_future.head().to_string(index=False))

# =========================
# 9. OPTIONAL: SCENARIO PREDICTIONS (multipliers)
# =========================
print("\n" + "=" * 80)
print("OPTIONAL: SCENARIO PREDICTIONS (MULTIPLIERS)")
print("=" * 80)

# Example scenario: more light + roads, less pesticides
scenario_1 = {
    "Lum_max": 1.30,
    "Long_route": 1.10,
    "nb_roads": 1.05,
    "all_pesticide_exposure_grid_max": 0.80
}

df_future_sc1 = apply_scenario_multipliers(df_future, scenario_1)
df_future_sc1["ValueSPred_Urba_pred"] = model.predict(df_future_sc1[FEATURES])

out_sc1 = "urbanisation_predictions_to_2050_scenario1.csv"
df_future_sc1.to_csv(out_sc1, index=False)
print(f"✅ Saved: {out_sc1}")
print(df_future_sc1.head().to_string(index=False))

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
############################################################
# END
############################################################
