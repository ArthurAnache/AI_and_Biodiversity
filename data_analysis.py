import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TABLEAU_COLORS

df = pd.read_csv("raw_data/STOC_pressions_bio_assol_meteo_pesti_routes_lum_120625.csv", encoding='iso-8859-1')
# S'assurer que la colonne d'abondance est numérique
df['abondance_capped'] = pd.to_numeric(df['abondance_capped'], errors='coerce')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')




def read_sample():
    # Lire le fichier CSV

    # Sélectionner les colonnes qui nous intéressent
    columns_of_interest = ['Year', 'abondance_capped', 'species', 'french_name']
    selected_data = df[columns_of_interest]

    # Afficher les données
    print("Données pour les colonnes year, abondance_capped, species et french_name :")
    print(selected_data.head())

    #valeur max de year
    print(max(selected_data['Year']))



def plot_species_all_sites(species_name, top_n_sites=10):
    """
    Superpose les courbes d'évolution pour tous les sites (ou top N sites)
    """
    # Filtrer pour l'espèce et les années 2001-2023
    species_data = df[(df['species'] == species_name) & 
                    (df['Year'] >= 2001) & 
                    (df['Year'] <= 2023)].copy()
    
    if len(species_data) == 0:
        print(f"Aucune donnée trouvée pour l'espèce {species_name}")
        return
    
    # Prendre les top N sites avec le plus d'observations
    site_counts = species_data['site'].value_counts()
    top_sites = site_counts.head(top_n_sites).index
    
    # Créer le graphique
    plt.figure(figsize=(15, 8))
    
    # Palette de couleurs stable
    base_colors = list(TABLEAU_COLORS.values())
    if len(base_colors) < len(top_sites):
        # Étendre la palette si besoin en la répétant
        repeats = int(np.ceil(len(top_sites) / len(base_colors)))
        colors = (base_colors * repeats)[:len(top_sites)]
    else:
        colors = base_colors[:len(top_sites)]
    
    for i, site in enumerate(top_sites):
        site_data = species_data[species_data['site'] == site]
        yearly_abundance = site_data.groupby('Year')['abondance_capped'].sum().reset_index()
        
        plt.plot(yearly_abundance['Year'], yearly_abundance['abondance_capped'], 
                marker='o', markersize=4, linewidth=1.5, 
                label=f'{site} ({len(site_data)} obs)', color=colors[i])
    
    plt.xlabel('Année')
    plt.ylabel('Abondance')
    plt.title(f'Évolution de l\'espèce {species_name} - Comparaison des {top_n_sites} sites principaux (2001-2023)')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(2001, 2024, 2))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # Statistiques globales
    print(f"\nStatistiques globales pour {species_name} :")
    print(f"Nombre total de sites : {species_data['site'].nunique()}")
    print(f"Sites affichés (top {top_n_sites}) : {list(top_sites)}")
    
    # Moyenne d'abondance par site
    site_stats = species_data.groupby('site')['abondance_capped'].agg(['mean', 'max', 'count']).round(2)
    site_stats.columns = ['abondance_moyenne', 'abondance_max', 'nb_observations']
    print("\nTop 5 sites par abondance moyenne :")
    print(site_stats.sort_values('abondance_moyenne', ascending=False).head())


def get_top_abondance_record():
    """
    Retourne la ligne avec la plus grande valeur d'"abondance_capped" sur l'ensemble des données.
    """
    s = pd.to_numeric(df['abondance_capped'], errors='coerce')
    if s.notna().sum() == 0:
        raise ValueError("Aucune valeur numérique valide trouvée pour 'abondance_capped'.")
    idx = s.idxmax()
    return df.loc[idx]


def plot_species_with_max_abondance(top_n_sites: int = 10):
    """
    Trouve l'espèce ayant le relevé maximal d'abondance_capped et trace:
      - son évolution sur le site où ce maximum a été observé (avec annotation du max)
      - (optionnel) la comparaison multi-sites pour cette espèce
    """
    top_row = get_top_abondance_record()
    species = top_row['species']
    site = top_row['site']
    year = int(top_row['Year']) if not pd.isna(top_row['Year']) else None
    value = float(top_row['abondance_capped']) if not pd.isna(top_row['abondance_capped']) else None
    french = top_row['french_name'] if 'french_name' in top_row else species

    print("\n=== Relevé maximal d'abondance_capped ===")
    print(f"Espèce: {species} ({french})")
    print(f"Site: {site}")
    print(f"Année: {year}")
    print(f"Valeur: {value}")

    # Afficher quelques sites disponibles
    _ = show_available_sites(species)

    # Superposer les principaux sites pour cette espèce
    plot_species_all_sites(species, top_n_sites=top_n_sites)


def compute_sites_per_species() -> pd.DataFrame:
        """
        Pour chaque espèce, calcule le nombre de sites distincts avec des données.
        Retourne un DataFrame trié par nombre de sites décroissant.
        """
        result = (
                df.groupby('species')
                    .agg(n_sites=('site', 'nunique'), french_name=('french_name', 'first'))
                    .reset_index()
                    .sort_values('n_sites', ascending=False)
        )
        return result


def compute_years_per_site_species() -> pd.DataFrame:
        """
        Pour chaque couple (site, espèce), calcule le nombre d'années distinctes avec des mesures.
        Retourne un DataFrame trié par n_years décroissant.
        """
        result = (
                df.groupby(['site', 'species'])
                    .agg(n_years=('Year', 'nunique'), french_name=('french_name', 'first'))
                    .reset_index()
                    .sort_values(['site', 'n_years'], ascending=[True, False])
        )
        return result


def export_counts(proc_dir: str = 'proc_data') -> None:
        """
        Exporte deux fichiers CSV dans proc_data/ :
            - nb_sites_par_espece.csv
            - nb_annees_par_site_espece.csv
        """
        os.makedirs(proc_dir, exist_ok=True)

        sites_per_species = compute_sites_per_species()
        years_per_site_species = compute_years_per_site_species()

        sites_csv = os.path.join(proc_dir, 'nb_sites_par_espece.csv')
        years_csv = os.path.join(proc_dir, 'nb_annees_par_site_espece.csv')

        sites_per_species.to_csv(sites_csv, index=False, encoding='utf-8-sig')
        years_per_site_species.to_csv(years_csv, index=False, encoding='utf-8-sig')

        print("\nAperçu - nb_sites_par_espece.csv:")
        print(sites_per_species.head(10))
        print(f"Fichier écrit: {sites_csv}")

        print("\nAperçu - nb_annees_par_site_espece.csv:")
        print(years_per_site_species.head(10))
        print(f"Fichier écrit: {years_csv}")


# Afficher les sites disponibles pour une espèce
def show_available_sites(species_name):
    species_data = df[df['species'] == species_name]
    sites = species_data['site'].value_counts()
    print(f"Sites disponibles pour {species_name} :")
    print(sites.head(10))
    return sites



if __name__ == "__main__":
    # Exemple d'utilisation : espèce avec le plus grand relevé d'abondance_capped
    plot_species_with_max_abondance(top_n_sites=10)
    # Comptages demandés
    export_counts(proc_dir='proc_data')
    
    
#Filtrer 