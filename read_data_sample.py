import pandas as pd

# Remplace par le chemin de ton fichier
csv_file = "raw_data/STOC_pressions_bio_assol_meteo_pesti_routes_lum_120625.csv"

# Lire seulement les 10000 premières lignes
df_sample = pd.read_csv(csv_file, nrows=10000, encoding="latin1")


# Sauvegarder l'échantillon dans un petit CSV
df_sample.to_csv("sample_data/sample.csv", index=False)


print("\n✅ Échantillon sauvegardé sous 'sample_data/sample.csv'")