import os
import hashlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TABLEAU_COLORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'raw_data', 'STOC_pressions_bio_assol_meteo_pesti_routes_lum_120625.csv')

df = pd.read_csv(RAW_DATA_PATH, encoding='iso-8859-1')
# S'assurer que la colonne d'abondance est numérique
df['abondance_capped'] = pd.to_numeric(df['abondance_capped'], errors='coerce')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')


def _normalize_species_codes(species: list[str]) -> list[str]:
    """Corrige quelques codes d'espèces fréquents (typos) sans casser l'exécution."""
    species = [s.strip() for s in species if isinstance(s, str) and s.strip()]
    if len(species) == 0:
        return species

    available = set(df['species'].dropna().astype(str).unique()) if 'species' in df.columns else set()
    normalized: list[str] = []
    for sp in species:
        # Typo courant vu dans la discussion
        if sp == 'ANTRI' and 'ANTTRI' in available:
            normalized.append('ANTTRI')
            continue
        normalized.append(sp)

    missing = [sp for sp in normalized if available and sp not in available]
    if len(missing) > 0:
        print(f"Attention: codes espèce inconnus dans le CSV: {missing}")
    return normalized


def select_sites_with_recent_coverage(
    species: list[str],
    end_year: int = 2023,
    n_recent_years: int = 20,
) -> list[int | str]:
    """Retourne les sites qui ont des données pour TOUTES les espèces et TOUTES les
    `n_recent_years` dernières années (fenêtre [end_year-n_recent_years+1, end_year]).
    """
    species = _normalize_species_codes(species)
    if len(species) == 0:
        return []

    start_year = end_year - n_recent_years + 1
    window = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)].copy()
    window = window.dropna(subset=['Year', 'site', 'species', 'abondance_capped'])
    window = window[window['species'].isin(species)]

    if len(window) == 0:
        return []

    # Nombre d'années distinctes par (site, espèce) dans la fenêtre récente
    counts = (
        window.groupby(['site', 'species'], as_index=False)['Year']
        .nunique()
        .rename(columns={'Year': 'n_years'})
    )

    pivot = counts.pivot(index='site', columns='species', values='n_years')

    # Il faut avoir exactement n_recent_years pour chaque espèce.
    # IMPORTANT: on calcule sur pivot.dropna() (sites qui ont au moins une valeur pour chaque espèce),
    # donc on doit retourner directement l'index de cette série.
    pivot_complete = pivot.dropna()
    eligible = pivot_complete.eq(n_recent_years).all(axis=1)
    return eligible[eligible].index.tolist()


def plot_total_abundance_by_year_for_species(
    species: list[str] | None = None,
    n_species: int = 3,
    start_year: int = 2001,
    end_year: int = 2023,
    fill_missing_years_with: float | None = 0.0,
    save_path: str = os.path.join(BASE_DIR, 'figures', 'abondances_totales_especes.png'),
    show: bool = True,
) -> pd.DataFrame:
    """Trace l'abondance totale annuelle (somme de abondance_capped) pour 2-3 espèces.

    - Si `species` est None, sélectionne automatiquement les `n_species` espèces avec
      la plus grande abondance totale sur la période.
    - `fill_missing_years_with=None` laisse des trous (NaN) quand une année manque.
      `0.0` remplit à 0 (utile pour un graphe descriptif).

    Retourne le DataFrame pivoté (index=Year, colonnes=species) utilisé pour le plot.
    """
    if 'species' not in df.columns or 'Year' not in df.columns or 'abondance_capped' not in df.columns:
        raise ValueError("Colonnes requises manquantes: Year, species, abondance_capped")

    period_df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)].copy()
    period_df = period_df.dropna(subset=['Year', 'species', 'abondance_capped'])

    if species is None:
        totals = (
            period_df.groupby('species', as_index=False)['abondance_capped']
            .sum()
            .sort_values('abondance_capped', ascending=False)
        )
        species = totals['species'].head(max(1, n_species)).tolist()
    else:
        # Nettoyage basique: enlever None / ''
        species = [s for s in species if isinstance(s, str) and s.strip()]
        if len(species) == 0:
            raise ValueError("La liste 'species' est vide.")

    grouped = (
        period_df[period_df['species'].isin(species)]
        .groupby(['Year', 'species'], as_index=False)['abondance_capped']
        .sum()
    )

    pivot = (
        grouped.pivot(index='Year', columns='species', values='abondance_capped')
        .sort_index()
        .reindex(range(start_year, end_year + 1))
    )
    if fill_missing_years_with is not None:
        pivot = pivot.fillna(fill_missing_years_with)

    # Labels plus lisibles si french_name est dispo
    label_by_species: dict[str, str] = {s: s for s in species}
    if 'french_name' in df.columns:
        names = (
            period_df[period_df['species'].isin(species)]
            .groupby('species', as_index=False)['french_name']
            .agg(lambda x: x.dropna().iloc[0] if len(x.dropna()) else None)
        )
        for _, row in names.iterrows():
            if isinstance(row.get('french_name'), str) and row['french_name'].strip():
                label_by_species[str(row['species'])] = f"{row['french_name']} ({row['species']})"

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    plt.figure(figsize=(12, 6))
    base_colors = list(TABLEAU_COLORS.values())
    colors = (base_colors * int(np.ceil(len(species) / max(1, len(base_colors)))))[:len(species)]

    for i, sp in enumerate(species):
        if sp not in pivot.columns:
            continue
        plt.plot(
            pivot.index,
            pivot[sp],
            linewidth=2,
            marker='o',
            markersize=3,
            label=label_by_species.get(sp, sp),
            color=colors[i],
        )

    plt.xlabel('Année')
    plt.ylabel("Abondance totale (somme de 'abondance_capped')")
    plt.title(f"Abondances totales annuelles (tous sites) – {len(species)} espèces ({start_year}-{end_year})")
    plt.grid(True, alpha=0.3)
    if end_year - start_year <= 30:
        plt.xticks(range(start_year, end_year + 1, 2))
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()

    print(f"Figure enregistrée: {save_path}")
    print(f"Espèces tracées: {species}")
    return pivot


def plot_mean_abundance_by_year_for_species(
    species: list[str] | None = None,
    n_species: int = 3,
    start_year: int = 2001,
    end_year: int = 2023,
    fill_missing_years_with: float | None = 0.0,
    save_path: str = os.path.join(BASE_DIR, 'figures', 'abondances_moyennes_especes.png'),
    show: bool = True,
) -> pd.DataFrame:
    """Trace l'abondance moyenne annuelle sur les sites pour 2-3 espèces.

    Interprétation: pour une année donnée et une espèce donnée, on prend la moyenne de
    `abondance_capped` sur tous les sites (chaque ligne = un site, car (Year, site, species) est unique).

    - Si `species` est None, sélectionne automatiquement les `n_species` espèces avec
      la plus grande abondance totale sur la période.
    - `fill_missing_years_with=None` laisse des trous (NaN) quand une année manque.

    Retourne le DataFrame pivoté (index=Year, colonnes=species) utilisé pour le plot.
    """
    if 'species' not in df.columns or 'Year' not in df.columns or 'abondance_capped' not in df.columns:
        raise ValueError("Colonnes requises manquantes: Year, species, abondance_capped")

    period_df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)].copy()
    period_df = period_df.dropna(subset=['Year', 'species', 'abondance_capped'])

    if species is None:
        totals = (
            period_df.groupby('species', as_index=False)['abondance_capped']
            .sum()
            .sort_values('abondance_capped', ascending=False)
        )
        species = totals['species'].head(max(1, n_species)).tolist()
    else:
        species = [s for s in species if isinstance(s, str) and s.strip()]
        if len(species) == 0:
            raise ValueError("La liste 'species' est vide.")

    grouped = (
        period_df[period_df['species'].isin(species)]
        .groupby(['Year', 'species'], as_index=False)['abondance_capped']
        .mean()
    )

    pivot = (
        grouped.pivot(index='Year', columns='species', values='abondance_capped')
        .sort_index()
        .reindex(range(start_year, end_year + 1))
    )
    if fill_missing_years_with is not None:
        pivot = pivot.fillna(fill_missing_years_with)

    label_by_species: dict[str, str] = {s: s for s in species}
    if 'french_name' in df.columns:
        names = (
            period_df[period_df['species'].isin(species)]
            .groupby('species', as_index=False)['french_name']
            .agg(lambda x: x.dropna().iloc[0] if len(x.dropna()) else None)
        )
        for _, row in names.iterrows():
            if isinstance(row.get('french_name'), str) and row['french_name'].strip():
                label_by_species[str(row['species'])] = f"{row['french_name']} ({row['species']})"

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    plt.figure(figsize=(12, 6))
    base_colors = list(TABLEAU_COLORS.values())
    colors = (base_colors * int(np.ceil(len(species) / max(1, len(base_colors)))))[:len(species)]

    for i, sp in enumerate(species):
        if sp not in pivot.columns:
            continue
        plt.plot(
            pivot.index,
            pivot[sp],
            linewidth=2,
            marker='o',
            markersize=3,
            label=label_by_species.get(sp, sp),
            color=colors[i],
        )

    plt.xlabel('Année')
    plt.ylabel("Abondance moyenne sur les sites (moyenne de 'abondance_capped')")
    plt.title(f"Abondances moyennes annuelles (moyenne sur sites) – {len(species)} espèces ({start_year}-{end_year})")
    plt.grid(True, alpha=0.3)
    if end_year - start_year <= 30:
        plt.xticks(range(start_year, end_year + 1, 2))
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()

    print(f"Figure enregistrée: {save_path}")
    print(f"Espèces tracées: {species}")
    return pivot


def plot_sites_species_total_abundance_by_year(
    sites: list[int | str],
    species: list[str],
    start_year: int = 2001,
    end_year: int = 2023,
    fill_missing_years_with: float | None = 0.0,
    impute_missing_with_site_mean: bool = False,
    save_path: str | None = None,
    show: bool = True,
) -> pd.DataFrame:
    """Trace la somme annuelle de l'abondance (abondance_capped) pour des espèces données sur une liste de sites.

    Comme (Year, site, species) est unique dans ce dataset, la somme par année/espèce sur une liste de sites
    correspond bien à la somme des valeurs site-par-site (sans double-comptage par mois).

    Retourne un DataFrame pivoté (index=Year, colonnes=species) utilisé pour le plot.
    """
    sites = [s for s in sites if s is not None]
    if len(sites) == 0:
        raise ValueError("La liste 'sites' est vide.")

    species = [s for s in species if isinstance(s, str) and s.strip()]
    if len(species) == 0:
        raise ValueError("La liste 'species' est vide.")

    if save_path is None:
        # Sur Windows, un nom de fichier trop long déclenche OSError: [Errno 22].
        # On utilise donc un slug compact: nombre de sites + hash stable.
        sites_str = ','.join(str(s) for s in sites)
        sites_hash = hashlib.sha1(sites_str.encode('utf-8')).hexdigest()[:10]
        sites_slug = f"{len(sites)}sites_{sites_hash}"
        save_path = os.path.join(
            BASE_DIR,
            'figures',
            f"abondances_sites_{sites_slug}_{'_'.join(species)}.png",
        )

    # Filtre sites/année/espèces
    sites_df = df[(df['site'].isin(sites)) & (df['Year'] >= start_year) & (df['Year'] <= end_year)].copy()
    sites_df = sites_df.dropna(subset=['Year', 'site', 'species', 'abondance_capped'])
    sites_df = sites_df[sites_df['species'].isin(species)]

    if len(sites_df) == 0:
        raise ValueError(f"Aucune donnée trouvée pour sites={sites} et espèces={species}")

    # Agrégation par (Year, site, species) (robuste même si doublons)
    obs_site_year = (
        sites_df.groupby(['Year', 'site', 'species'], as_index=False)['abondance_capped']
        .sum()
    )

    if impute_missing_with_site_mean:
        # Remplit les (Year, site, species) manquants avec la moyenne du site pour cette espèce
        site_means = (
            obs_site_year.groupby(['site', 'species'], as_index=False)['abondance_capped']
            .mean()
            .rename(columns={'abondance_capped': 'site_species_mean'})
        )

        full_index = pd.MultiIndex.from_product(
            [range(start_year, end_year + 1), sites, species],
            names=['Year', 'site', 'species'],
        )
        full = full_index.to_frame(index=False)
        full = full.merge(obs_site_year, on=['Year', 'site', 'species'], how='left')
        full = full.merge(site_means, on=['site', 'species'], how='left')
        full['abondance_capped'] = full['abondance_capped'].fillna(full['site_species_mean'])
        full = full.drop(columns=['site_species_mean'])

        if fill_missing_years_with is not None:
            # Si un site n'a jamais de données pour une espèce, la moyenne reste NaN
            full['abondance_capped'] = full['abondance_capped'].fillna(fill_missing_years_with)

        grouped = full.groupby(['Year', 'species'], as_index=False)['abondance_capped'].sum()
    else:
        grouped = obs_site_year.groupby(['Year', 'species'], as_index=False)['abondance_capped'].sum()

    pivot = (
        grouped.pivot(index='Year', columns='species', values='abondance_capped')
        .sort_index()
        .reindex(range(start_year, end_year + 1))
    )
    if fill_missing_years_with is not None:
        pivot = pivot.fillna(fill_missing_years_with)

    label_by_species: dict[str, str] = {s: s for s in species}
    if 'french_name' in df.columns:
        names = (
            df[(df['site'].isin(sites)) & (df['species'].isin(species))]
            .groupby('species', as_index=False)['french_name']
            .agg(lambda x: x.dropna().iloc[0] if len(x.dropna()) else None)
        )
        for _, row in names.iterrows():
            if isinstance(row.get('french_name'), str) and row['french_name'].strip():
                label_by_species[str(row['species'])] = f"{row['french_name']} ({row['species']})"

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    plt.figure(figsize=(12, 6))
    base_colors = list(TABLEAU_COLORS.values())
    colors = (base_colors * int(np.ceil(len(species) / max(1, len(base_colors)))))[:len(species)]

    for i, sp in enumerate(species):
        if sp not in pivot.columns:
            continue
        plt.plot(
            pivot.index,
            pivot[sp],
            linewidth=2,
            marker='o',
            markersize=3,
            label=label_by_species.get(sp, sp),
            color=colors[i],
        )

    plt.xlabel('Année')
    plt.ylabel("Abondance totale (somme de 'abondance_capped')")
    plt.title(f"Abondances totales annuelles – {len(sites)} sites ({start_year}-{end_year})")
    plt.grid(True, alpha=0.3)
    if end_year - start_year <= 30:
        plt.xticks(range(start_year, end_year + 1, 2))
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()

    print(f"Figure enregistrée: {save_path}")
    print(f"Sites: {sites} | Espèces tracées: {species} | Imputation moyenne site: {impute_missing_with_site_mean}")
    return pivot


def plot_total_abundance_all_sites_with_extrapolation(
    species: list[str],
    start_year: int = 2001,
    end_year: int = 2023,
    fit_start_year: int = 2004,
    extrapolate_years: list[int] | None = None,
    fill_missing_years_with: float | None = 0.0,
    save_path: str | None = None,
    show: bool = True,
) -> pd.DataFrame:
    """Somme annuelle de `abondance_capped` sur tous les sites disponibles.

    Objectif: illustrer une tendance globale même si des années anciennes sont peu couvertes.
    - Pour les années >= fit_start_year: somme sur tous les sites ayant des données cette année.
    - Pour `extrapolate_years` (par défaut 2001-2003): extrapole via une régression linéaire
      ajustée sur les années >= fit_start_year (par espèce), puis remplit ces années.
    """
    if extrapolate_years is None:
        extrapolate_years = [2001, 2002, 2003]

    species = _normalize_species_codes(species)
    if len(species) == 0:
        raise ValueError("La liste 'species' est vide.")

    if save_path is None:
        save_path = os.path.join(
            BASE_DIR,
            'figures',
            f"abondances_totales_tous_sites_extrap_{fit_start_year}_{'_'.join(species)}.png",
        )

    period_df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)].copy()
    period_df = period_df.dropna(subset=['Year', 'species', 'abondance_capped'])
    period_df = period_df[period_df['species'].isin(species)]

    if len(period_df) == 0:
        raise ValueError(f"Aucune donnée trouvée pour espèces={species} sur {start_year}-{end_year}")

    grouped = (
        period_df.groupby(['Year', 'species'], as_index=False)['abondance_capped']
        .sum()
    )

    pivot = (
        grouped.pivot(index='Year', columns='species', values='abondance_capped')
        .sort_index()
        .reindex(range(start_year, end_year + 1))
    )

    # Extrapolation des années anciennes
    for sp in species:
        if sp not in pivot.columns:
            continue

        # Série observée à partir de fit_start_year
        obs = pivot.loc[fit_start_year:end_year, sp].dropna()
        if len(obs) < 2:
            # Pas assez de points: fallback constant = premier point dispo
            if len(obs) == 0:
                continue
            a = 0.0
            b = float(obs.iloc[0])
        else:
            x = obs.index.to_numpy(dtype=float)
            y = obs.to_numpy(dtype=float)
            a, b = np.polyfit(x, y, 1)

        for y in extrapolate_years:
            if y < start_year or y > end_year:
                continue
            # On ne remplace que si la valeur est manquante
            if pd.isna(pivot.loc[y, sp]):
                pred = a * float(y) + b
                pivot.loc[y, sp] = max(0.0, float(pred))

    if fill_missing_years_with is not None:
        pivot = pivot.fillna(fill_missing_years_with)

    label_by_species: dict[str, str] = {s: s for s in species}
    if 'french_name' in df.columns:
        names = (
            df[df['species'].isin(species)]
            .groupby('species', as_index=False)['french_name']
            .agg(lambda x: x.dropna().iloc[0] if len(x.dropna()) else None)
        )
        for _, row in names.iterrows():
            if isinstance(row.get('french_name'), str) and row['french_name'].strip():
                label_by_species[str(row['species'])] = f"{row['french_name']} ({row['species']})"

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    plt.figure(figsize=(12, 6))
    base_colors = list(TABLEAU_COLORS.values())
    colors = (base_colors * int(np.ceil(len(species) / max(1, len(base_colors)))))[:len(species)]

    for i, sp in enumerate(species):
        if sp not in pivot.columns:
            continue
        plt.plot(
            pivot.index,
            pivot[sp],
            linewidth=2,
            marker='o',
            markersize=3,
            label=label_by_species.get(sp, sp),
            color=colors[i],
        )

    plt.axvline(fit_start_year, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    plt.xlabel('Année')
    plt.ylabel("Abondance totale (somme de 'abondance_capped')")
    plt.title(
        f"Abondances totales annuelles (tous sites) – extrapolation {min(extrapolate_years)}-{max(extrapolate_years)} | fit >= {fit_start_year}"
    )
    plt.grid(True, alpha=0.3)
    if end_year - start_year <= 30:
        plt.xticks(range(start_year, end_year + 1, 2))
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()

    print(f"Figure enregistrée: {save_path}")
    print(f"Espèces tracées: {species}")
    print(f"Extrapolation: années {extrapolate_years} (fit à partir de {fit_start_year})")
    return pivot




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
    if not os.path.isabs(proc_dir):
        proc_dir = os.path.join(BASE_DIR, proc_dir)
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
    # Comptages demandés
    export_counts(proc_dir='proc_data')

    # Plot rapport: utiliser tous les sites qui ont AU MOINS les 20 dernières années de données
    # pour toutes les espèces demandées (donc, en pratique: présents sur chaque année de la fenêtre).
    species_codes = ['ANTTRI', 'APUAPU', 'PASDOM', 'COLPAL']
    eligible_sites = select_sites_with_recent_coverage(species=species_codes, end_year=2023, n_recent_years=20)
    print(f"Sites éligibles (couverture 20 ans, {species_codes}): {len(eligible_sites)}")

    # Si une année manque sur un site (hors fenêtre récente), on remplit par la moyenne de ce site (pour l'espèce).
    plot_sites_species_total_abundance_by_year(
        sites=eligible_sites,
        species=species_codes,
        start_year=2001,
        end_year=2023,
        impute_missing_with_site_mean=True,
        fill_missing_years_with=0.0,
        show=True,
    )
    
    
#Filtrer 