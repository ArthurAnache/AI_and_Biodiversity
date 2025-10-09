import pandas as pd
import numpy as np

# Chargement des données
df = pd.read_csv('raw_data/STOC_pressions_bio_assol_meteo_pesti_routes_lum_120625.csv', encoding='latin1')

# On ne garde que les colonnes qui nous intéressent
colonnes_a_garder = [f'Temps_mean_mensu_grid_{i}' for i in range(1, 13)] + [f'Precipitation_mean_mensu_grid_{i}' for i in range(1, 13)] + ['Year', 'abondance_capped', 'ValueSPred_Urba', 'ValueSPred_For', 'ValueSPred_CC', 'ValueSPred_M', 'ValueSPred_ZH', 'mean_concentration_air_grid_sum', 'mean_tii_grid_sum', 'mean_tii_scale_grid_sum', 'mean_concentration_water_grid_sum', 'all_pesticide_exposure_grid_sum', 'SurfBio_cell', 'nb_exp_cell', 'Long_route', 'nb_roads', 'Lum_mean', 'site', 'species']
df = df[colonnes_a_garder]

# Remplacement des NaN et inf par 0
df = df.replace([np.inf, -np.inf], np.nan)
#df = df.fillna(0)

# Création d'une copie du DataFrame avec l'année incrémentée de 1
df_year_plus_1 = df.copy()
df_year_plus_1['Year'] = df_year_plus_1['Year'] + 1

# Renommer les colonnes en ajoutant _year_plus_1
df_year_plus_1 = df_year_plus_1.rename(columns={col: f"{col}_year_plus_1" for col in df_year_plus_1.columns})

# Fusionner les deux DataFrames sur l'année (Year = Year_year_plus_1), le site et l'espèce
df_merged = pd.merge(
    df,
    df_year_plus_1,
    left_on=['Year', 'site', 'species'],
    right_on=['Year_year_plus_1', 'site_year_plus_1', 'species_year_plus_1'],
    how='inner'
)

pd.set_option('display.max_columns', None)
#print(df_merged.head)

# Supprimer les colonnes inutiles après la fusion
df_merged = df_merged.drop(columns=['site', 'site_year_plus_1', 'species_year_plus_1', 'Year', 'Year_year_plus_1'])
#print(df_merged.head)

# Séparer les données en deux DataFrames
abondance = df_merged[['abondance_capped']]
pression = df_merged.drop(columns=['abondance_capped'])

# Exporter les deux DataFrames dans le dossier raw_data
abondance.to_csv('raw_data/abondances_petit.csv', index=False)
pression.to_csv('raw_data/pressions_petit.csv', index=False)
