
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import argparse

# Import utilities from the existing model
# We assume species_evolution_model.py is in the same directory
from species_evolution_model import (
    load_cluster_mapping, 
    prepare_xy, 
    time_based_split, 
    build_pipeline
)

def prepare_data_generic(df_raw, filter_species_list=None):
    """
    Standard data preparation steps copied/adapted from species_evolution_model
    """
    colonnes_a_garder = [f'Temps_mean_mensu_grid_{i}' for i in range(1, 13)] + \
                        [f'Precipitation_mean_mensu_grid_{i}' for i in range(1, 13)] + \
                        ['Year', 'abondance_capped', 'ValueSPred_Urba', 'ValueSPred_For', 
                         'ValueSPred_CC', 'ValueSPred_M', 'ValueSPred_ZH', 
                         'mean_concentration_air_grid_sum', 'mean_tii_grid_sum', 
                         'mean_tii_scale_grid_sum', 'mean_concentration_water_grid_sum', 
                         'all_pesticide_exposure_grid_sum', 'SurfBio_cell', 'nb_exp_cell', 
                         'Long_route', 'nb_roads', 'Lum_mean', 'site', 'species']
    
    existing_cols = [c for c in colonnes_a_garder if c in df_raw.columns]
    df = df_raw[existing_cols].replace([np.inf, -np.inf], np.nan)
    
    if filter_species_list is not None:
        df = df[df['species'].isin(filter_species_list)]

    # Merge N / N+1 (lag-1 features)
    df_year_plus_1 = df.copy()
    df_year_plus_1['Year'] = df_year_plus_1['Year'] + 1
    df_year_plus_1 = df_year_plus_1.rename(columns={col: f"{col}_year_plus_1" for col in df_year_plus_1.columns})

    df_merged = pd.merge(
        df,
        df_year_plus_1,
        left_on=['Year', 'site', 'species'],
        right_on=['Year_year_plus_1', 'site_year_plus_1', 'species_year_plus_1'],
        how='inner'
    )
    df_merged = df_merged.rename(columns={'Year': 'annee'})
    
    if 'abondance_capped' not in df_merged.columns:
        return None, None

    y = df_merged['abondance_capped']
    feature_cols = [c for c in df_merged.columns if c.endswith('_year_plus_1')]
    cols_to_keep = feature_cols + ['annee', 'species'] # Keep species for later grouping
    
    X = df_merged[cols_to_keep].copy()
    X, y = prepare_xy(X, y)
    
    return X, y

def train_and_eval(X, y):
    if len(X) < 50:
        return None
    
    y_log = np.log1p(y)
    
    # Split
    # Important: We need X_val to keep the 'species' column to calculate per-species R2
    # time_based_split keeps all columns in X, just splits rows.
    X_train, y_train, X_test, y_test, X_val, y_val = time_based_split(X, y_log, train_frac=0.6, test_frac=0.2)
    
    # Pipeline
    model = build_pipeline(X_train)
    
    # Fit
    prep = model.named_steps['prep']
    xgb_model = model.named_steps['model']
    
    X_train_proc = prep.fit_transform(X_train)
    X_val_proc = prep.transform(X_val)
    
    xgb_model.fit(X_train_proc, y_train, eval_set=[(X_val_proc, y_val)], verbose=False)
    
    # Predict on Validation set
    y_pred_log = xgb_model.predict(X_val_proc)

    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_val)
    
    # Store results in a DataFrame for easy grouping
    df_res = pd.DataFrame({
        'annee': X_val['annee'].values,
        'species': X_val['species'].values,
        'y_true': y_true,
        'y_pred': y_pred
    })
    
    return df_res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-data', default='raw_data/STOC_pressions_bio_assol_meteo_pesti_routes_lum_120625.csv')
    parser.add_argument('--clusters-file', default='figures/species_clusters.csv')
    args = parser.parse_args()

    print("Loading Raw Data...")
    df_raw = pd.read_csv(args.raw_data, encoding='latin1')
    
    print("Loading Clusters...")
    cluster_series = load_cluster_mapping(args.clusters_file)
    unique_clusters = sorted(cluster_series.unique())
    
    results = []

    # --- 1. Train Global Model ---
    print("\nTraining Global Model (XGBoost)...")
    X_global, y_global = prepare_data_generic(df_raw)
    df_res_global = train_and_eval(X_global, y_global)
    
    if df_res_global is not None:
        # Calculate R2 per species for the Global Model
        for species_name, grp in df_res_global.groupby('species'):
            if len(grp) > 10: # Only calculate if enough samples
                r2 = r2_score(grp['y_true'], grp['y_pred'])
                results.append({
                    'species': species_name,
                    'type': 'Global (XGB)',
                    'r2': r2,
                    'cluster': cluster_series.get(species_name, -1)
                })

    # --- 2. Train Cluster Models ---
    print("\nTraining Cluster Models (XGBoost)...")
    for cluster_id in unique_clusters:
        species_in_cluster = cluster_series[cluster_series == cluster_id].index.tolist()
        print(f"  > Cluster {cluster_id} ({len(species_in_cluster)} species)")
        
        X_cluster, y_cluster = prepare_data_generic(df_raw, filter_species_list=species_in_cluster)
        
        if X_cluster is None or len(X_cluster) < 50:
            print("    Not enough data.")
            continue
            
        df_res_cluster = train_and_eval(X_cluster, y_cluster)
        
        if df_res_cluster is not None:
            # Calculate R2 per species for this Cluster Model
            for species_name, grp in df_res_cluster.groupby('species'):
                if len(grp) > 10:
                    r2 = r2_score(grp['y_true'], grp['y_pred'])
                    # Store
                    results.append({
                        'species': species_name,
                        'type': 'Cluster (XGB)',
                        'r2': r2,
                        'cluster': cluster_id
                    })

    # --- 3. Analysis & Plotting ---
    df_final = pd.DataFrame(results)
    if df_final.empty:
        print("No results to plot.")
        return

    # Pivot
    df_pivot = df_final.pivot_table(index=['species', 'cluster'], columns='type', values='r2').reset_index()
    
    # Filter only species that have Cluster and Global XGB
    df_pivot = df_pivot.dropna(subset=['Global (XGB)', 'Cluster (XGB)'])
    
    # Sort
    df_pivot = df_pivot.sort_values(by=['cluster', 'species'])

    # Plot Bar Chart
    print(f"\nPlotting comparison for {len(df_pivot)} species...")
    
    _, ax = plt.subplots(figsize=(15, 8))
    
    x = np.arange(len(df_pivot))
    width = 0.35 # thinner bars for 2 types
    
    # Define colors and plotting order
    columns_map = {
        'Global (XGB)': {'color': 'gray', 'label': 'Global XGB', 'offset': -width/2},
        'Cluster (XGB)': {'color': 'teal', 'label': 'Cluster XGB', 'offset': width/2},
    }
    
    # Check which columns actually exist in pivot
    existing_cols = [c for c in columns_map.keys() if c in df_pivot.columns]
    
    for col in existing_cols:
        props = columns_map[col]
        ax.bar(x + props['offset'], df_pivot[col], width, label=props['label'], color=props['color'], alpha=0.8)
    
    ax.set_ylabel('R² Score (Validation)')
    ax.set_title('Comparison: Global XGB vs Cluster XGB')
    ax.set_xticks(x)
    ax.set_xticklabels(df_pivot['species'], rotation=90, fontsize=8)
    ax.legend()
    
    # Lines
    prev_c = df_pivot.iloc[0]['cluster']
    for i in range(1, len(df_pivot)):
        curr_c = df_pivot.iloc[i]['cluster']
        if curr_c != prev_c:
            ax.axvline(i - 0.5, color='black', linestyle='--', alpha=0.3)
            prev_c = curr_c

    plt.tight_layout()
    plt.savefig('figures/model_comparison_bars.png')
    print("Plot saved to figures/model_comparison_bars.png")

    # --- Histogram Comparison ---
    print("Generating Histogram...")
    plt.figure(figsize=(10, 6))
    
    # Use same colors/labels
    for col in existing_cols:
        props = columns_map[col]
        plt.hist(df_pivot[col], bins=20, alpha=0.5, label=props['label'], color=props['color'], edgecolor='black')
    
    plt.xlabel('R² Score (Validation)')
    plt.ylabel('Number of Species')
    plt.title('Distribution of Model Performance (R² Scores)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig('figures/model_comparison_hist.png')
    print("Histogram saved to figures/model_comparison_hist.png")
    
    # --- Critique Text ---
    # Update critique
    with open('figures/comparison_critique.txt', 'w', encoding='utf-8') as f:
        f.write("Critique de la comparaison\n")
        f.write("==========================\n\n")
        f.write(f"Nombre d'espèces : {len(df_pivot)}\n")
        
        for col in existing_cols:
            f.write(f"Moyenne {col} : {df_pivot[col].mean():.4f}\n")
        
        f.write("\nTop Gains (Cluster vs Global XGB):\n")
        df_pivot['Diff_XGB'] = df_pivot['Cluster (XGB)'] - df_pivot['Global (XGB)']
        f.write(df_pivot.sort_values('Diff_XGB', ascending=False).head(5)[['species', 'Cluster (XGB)', 'Global (XGB)', 'Diff_XGB']].to_string())

if __name__ == '__main__':
    main()
