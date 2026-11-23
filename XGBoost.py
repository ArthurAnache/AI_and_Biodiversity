import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, balanced_accuracy_score, roc_auc_score, make_scorer, confusion_matrix, mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

import matplotlib.pyplot as plt

moyenne_meteorologiques = False

# Chargement des données
if moyenne_meteorologiques:
    X = pd.read_csv('raw_data/pressions_petit_moyenne.csv', encoding='latin1')
    y = pd.read_csv('raw_data/abondances_petit_moyenne.csv', encoding='latin1')
else:
    X = pd.read_csv('raw_data/pressions_petit.csv', encoding='latin1')
    y = pd.read_csv('raw_data/abondances_petit.csv', encoding='latin1')

# On garde une colonne avec le nom d'espèce
species_col = X['species'].copy()

# Encodage one-hot de la colonne species
X = pd.get_dummies(X, columns=['species'], prefix='species')

# Affichage des premières lignes du DataFrame
#print(df.head())

# On ne garde qu'une espèce d'oiseaux
#df = df[df['species'] == 'APUAPU']

#On enlève les colonnes qui ne servent à rien
#colonnes_a_garder = [f'Temps_mean_mensu_grid_{i}' for i in range(1, 13)] + [f'Precipitation_mean_mensu_grid_{i}' for i in range(1, 13)] + [f'Temps_max_mensu_grid_{i}' for i in range(1, 13)] + [f'Temps_min_mensu_grid_{i}' for i in range(1, 13)] + ['Year', 'abondance_capped', 'ValueSPred_Urba', 'ValueSPred_For', 'ValueSPred_CC', 'ValueSPred_M', 'ValueSPred_ZH', 'mean_concentration_air_grid_max', 'mean_tii_grid_max', 'mean_tii_scale_grid_max', 'mean_concentration_water_grid_max', 'all_pesticide_exposure_grid_max', 'SurfBio_cell', 'nb_exp_cell', 'Long_route', 'nb_roads', 'Lum_mean']
#df = df[colonnes_a_garder]

# Remplacement des NaN et inf par 0
#df = df.replace([np.inf, -np.inf], np.nan)
#df = df.fillna(0)

#print(df.head())

#pd.set_option('display.max_rows', None)
#print(df.dtypes)

# On divise les données en 2 ensembles : les données que l'on utilisent pour la prédiction (X) et la variable que l'on veut prédire (y)
#X = df.drop(columns=['abondance_capped'])
#y = df['abondance_capped']

# On divise les données en un ensemble d'entraînement et un ensemble de test
X_train, X_test, y_train, y_test, species_train, species_test = train_test_split(
    X, y, species_col, test_size=0.2, random_state=42
)

# On crée le modèle XGBoost
clf_xgb = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')

# Entraînement du modèle
clf_xgb.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = clf_xgb.predict(X_test)

# Affichage des prédictions
'''for oiseau_true, true, pred in zip(X_test.to_dict('records'), y_test, y_pred):
    print(f"Pour l'oiseau {oiseau_true['Year']}, Vrai: {true}, Prédit: {pred}")

# Évaluation du modèle
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Valeurs réelles (abondance_capped)")
plt.ylabel("Valeurs prédites")
plt.title("Prédiction XGBoost : Réel vs Prédit")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # diagonale
plt.show()

print(y_test)
print(y_pred)
'''

# S'assurer que y_test est une Series
if isinstance(y_test, pd.DataFrame):
    y_test = y_test.squeeze()
'''
plt.figure(figsize=(8, 6))
plt.hist(y_test - y_pred, bins=30)
plt.xlabel("Erreur de prédiction (Vrai - Prédit)")
plt.ylabel("Nombre d'observations")
plt.title("Distribution des erreurs de prédiction")
plt.show()
'''
# Création d'une table avec les valeurs réelles et prédites
resultats = pd.DataFrame({
    'Species': species_test.values,
    'Reel': y_test.values.squeeze(),
    'Pred': y_pred
})

pd.set_option('display.max_rows', None)
print(resultats.head(20))
'''
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE :", rmse)
print("MAE :", np.mean(abs(y_test - y_pred)))


r2 = r2_score(y_test, y_pred)
print("R² :", r2)
'''
# =========================
# 🔹 Évaluation par espèce
# =========================
scores = []
for espece, subset in resultats.groupby('Species'):
    if len(subset) > 3:  # éviter les espèces avec trop peu de points
        rmse = np.sqrt(mean_squared_error(subset['Reel'], subset['Pred']))
        mae = mean_absolute_error(subset['Reel'], subset['Pred'])
        r2 = r2_score(subset['Reel'], subset['Pred'])
        scores.append({'Espèce': espece, 'RMSE': rmse, 'MAE': mae, 'R²': r2})

scores_df = pd.DataFrame(scores).sort_values(by='RMSE', ascending=True)

print("\n Classement des espèces (de la meilleure à la moins bonne RMSE) :")
print(scores_df)

# =========================
# 🔹 Évaluation globale
# =========================
rmse_global = np.sqrt(mean_squared_error(y_test, y_pred))
mae_global = mean_absolute_error(y_test, y_pred)
r2_global = r2_score(y_test, y_pred)

print("\n Résultats globaux :")
print(f"RMSE : {rmse_global:.4f}")
print(f"MAE  : {mae_global:.4f}")
print(f"R²   : {r2_global:.4f}")

# =========================
# 🔹 Optionnel : graphique
# =========================
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.title("Prédiction XGBoost : Réel vs Prédit (toutes espèces)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

species_cols = [col for col in X_train.columns if col.startswith('species_')]

# Récupérer l'importance des features
importance = clf_xgb.feature_importances_

# Mettre en DataFrame pour plus de clarté
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

print(feature_importance.head(30))
# Importance globale de l'espèce
esp_importance = feature_importance[feature_importance['Feature'].isin(species_cols)]['Importance'].sum()

print(f"Importance globale de l'espèce : {esp_importance:.4f}")

# Marquer les espèces comme une seule catégorie
feature_importance['Feature_mod'] = feature_importance['Feature'].apply(
    lambda x: 'Espèce' if x in species_cols else x
)

# Grouper pour sommer l’importance
feature_importance_grouped = feature_importance.groupby('Feature_mod')['Importance'].sum().sort_values(ascending=False)

# Barplot
plt.figure(figsize=(10,6))
plt.barh(feature_importance_grouped.index, feature_importance_grouped.values)
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Importance des variables (espèce regroupée)")
plt.show()


