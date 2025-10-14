import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import unicodedata
import os
import joblib
import shap

# --- Paramètres ---
data_path = "/Users/r/Documents/CS/PROJET_S7/AI_and_Biodiversity/raw_data/#METTRE_LES_2_FICHIERS_ICI"
file = "STOC_pressions_bio_assol_meteo_pesti_routes_lum_120625.csv"
target_column = "abondance_capped"

# --- Chargement ---
df = pd.read_csv(os.path.join(data_path, file), sep=";", encoding="latin1", low_memory=False)
print(f"✅ Données chargées ({df.shape[0]} lignes, {df.shape[1]} colonnes)")

# --- Fonction de normalisation du nom d'espèce --- 
def normalize_str(s):
    return unicodedata.normalize('NFKD', str(s)).encode('ASCII', 'ignore').decode('utf-8').lower().strip()

species_list = sorted(df['species'].dropna().unique())
print(f"Nombre d'espèces détectées : {len(species_list)}")

# --- Dossier de sauvegarde des modèles ---
model_path = "/Users/r/Documents/CS/PROJET_S7/AI_and_Biodiversity/models"
os.makedirs(model_path, exist_ok=True)

# --- Résultats stockés ---
results = []

# --- Boucle principale ---
for species_name in species_list:
    print(f"\n=== Espèce : {species_name} ===")

    # Filtrage
    df_species = df[df['species'].apply(lambda x: normalize_str(x)) == normalize_str(species_name)].copy()
    if df_species.empty:
        continue

    # Tri par cellule et année
    df_species = df_species.sort_values(by=["id_cell", "Year"])

    # Création de la colonne abondance décalée (t-1)
    df_species["abondance_capped_t-1"] = df_species.groupby("id_cell")[target_column].shift(1)

    # Colonnes à exclure (identifiants + abondances)
    drop_cols = [c for c in df.columns if 'abondance' in c.lower()] + ['species', 'id_cell', 'Year']

    # Colonnes pressions (tout sauf les drop_cols et la target)
    pressure_cols = [c for c in df_species.columns if c not in drop_cols + [target_column]]

    # Créer les versions t-1 pour toutes les pressions
    for col in pressure_cols:
        df_species[f"{col}_t-1"] = df_species.groupby("id_cell")[col].shift(1)

    # Supprimer les lignes sans t-1 dispo
    df_species = df_species.dropna(subset=["abondance_capped_t-1"])
    if df_species.empty:
        continue

    # Définition de X et y
    y = df_species[target_column].fillna(df_species[target_column].median())
    X = df_species.drop(columns=drop_cols, errors='ignore')

    # Moyenne annuelle de température (exemple : si tu veux garder un agrégat)
    temps_cols = [c for c in X.columns if c.startswith("Temps_mean_mensu_grid_")]
    if temps_cols:
        X['Temps_mean_annual'] = X[temps_cols].mean(axis=1)
        X = X.drop(columns=temps_cols)

    # Conversion numérique et nettoyage
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median())
    X = pd.get_dummies(X, drop_first=True)

    if X.empty or X.shape[0] < 30:
        print(f"⚠️ Trop peu de données ({len(X)} lignes) → skip.")
        continue

    # Split train/test (pas de shuffle car temporel)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # Modèle XGBoost
    xgb_model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train) 
        
    # SHAP
    try:
        explainer = shap.Explainer(xgb_model, X_train)
        shap_values = explainer(X_test)

        print(f"→ Importance globale des variables pour {species_name}")
        shap.summary_plot(shap_values, X_test, plot_type="bar")
        shap.summary_plot(shap_values, X_test)
    except Exception as e:
        print(f"⚠️ Impossible de calculer SHAP pour {species_name} : {e}")

    # Évaluation
    y_pred = xgb_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Sauvegarde modèle + résultats
    joblib.dump(xgb_model, os.path.join(model_path, f"xgb_{species_name.replace(' ', '_')}.pkl"))
    results.append({'species': species_name, 'n_samples': len(df_species), 'RMSE': rmse, 'R2': r2})

    print(f"→ RMSE = {rmse:.3f}, R² = {r2:.3f} (n={len(df_species)})")

# --- Résumé global ---
results_df = pd.DataFrame(results).sort_values(by="RMSE", ascending=True)
print("\n=== Résumé des performances ===")
print(results_df.head(10))

# --- Sauvegarde CSV ---
results_df.to_csv(os.path.join(model_path, "xgb_species_results.csv"), index=False)

# --- Visualisation ---
plt.figure(figsize=(12, 6))
plt.barh(results_df['species'], results_df['RMSE'], color='skyblue', edgecolor='k')
plt.xlabel("RMSE")
plt.ylabel("Espèce")
plt.title("Erreur quadratique moyenne (RMSE) par espèce")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
