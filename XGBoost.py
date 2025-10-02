import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error,balanced_accuracy_score, roc_auc_score, make_scorer, confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt

# Chargement des données
df = pd.read_csv('raw_data/donnees_test.csv')

# Affichage des premières lignes du DataFrame
#print(df.head())

# On ne garde qu'une espèce d'oiseaux
#df = df[df['species'] == 'APUAPU']

#On enlève les colonnes qui ne servent à rien
colonnes_a_garder = [f'Temps_mean_mensu_grid_{i}' for i in range(1, 13)] + [f'Precipitation_mean_mensu_grid_{i}' for i in range(1, 13)] + [f'Temps_max_mensu_grid_{i}' for i in range(1, 13)] + [f'Temps_min_mensu_grid_{i}' for i in range(1, 13)] + ['Year', 'abondance_capped', 'ValueSPred_Urba', 'ValueSPred_For', 'ValueSPred_CC', 'ValueSPred_M', 'ValueSPred_ZH', 'mean_concentration_air_grid_max', 'mean_tii_grid_max', 'mean_tii_scale_grid_max', 'mean_concentration_water_grid_max', 'all_pesticide_exposure_grid_max', 'SurfBio_cell', 'nb_exp_cell', 'Long_route', 'nb_roads', 'Lum_mean']
df = df[colonnes_a_garder]

#print(df.head())

#pd.set_option('display.max_rows', None)
#print(df.dtypes)

# On divise les données en 2 ensembles : les données que l'on utilisent pour la prédiction (X) et la variable que l'on veut prédire (y)
X = df.drop(columns=['abondance_capped'])
y = df['abondance_capped']

# On divise les données en un ensemble d'entraînement et un ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# On crée le modèle XGBoost
clf_xgb = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')

# Entraînement du modèle
clf_xgb.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = clf_xgb.predict(X_test)

# Affichage des prédictions
for oiseau_true, true, pred in zip(X_test.to_dict('records'), y_test, y_pred):
    print(f"Pour l'oiseau {oiseau_true['Year']}, Vrai: {true}, Prédit: {pred}")
'''
print(y_test)
print(y_pred)
'''

