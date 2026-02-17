import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def geometric_mean_abundance(values: np.ndarray) -> float:
    """Geometric mean adapted to (possibly zero) abundances.

    Uses log1p/expm1 so that zero abondance is handled correctement:
    gm = exp(mean(log(1 + x))) - 1
    """
    arr = np.asarray(values, dtype=float)
    arr = np.maximum(arr, 0.0)
    return float(np.expm1(np.mean(np.log1p(arr))))

def prepare_data_from_raw(raw_path: str, n_lags: int = 1):
    """
    Reads raw data, processes it similar to formatage_donnees.py but KEEPS the Year column.
    Returns X (features) and y (target).
    """
    if n_lags not in (1, 2):
        raise ValueError("n_lags must be 1 or 2")
    print(f"Reading raw data from {raw_path}...")
    df = pd.read_csv(raw_path, encoding='latin1')

    # Columns to keep (same as formatage_donnees.py)
    colonnes_a_garder = [f'Temps_mean_mensu_grid_{i}' for i in range(1, 13)] + \
                        [f'Precipitation_mean_mensu_grid_{i}' for i in range(1, 13)] + \
                        ['Year', 'abondance_capped', 'ValueSPred_Urba', 'ValueSPred_For', 
                         'ValueSPred_CC', 'ValueSPred_M', 'ValueSPred_ZH', 
                         'mean_concentration_air_grid_sum', 'mean_tii_grid_sum', 
                         'mean_tii_scale_grid_sum', 'mean_concentration_water_grid_sum', 
                         'all_pesticide_exposure_grid_sum', 'SurfBio_cell', 'nb_exp_cell', 
                         'Long_route', 'nb_roads', 'Lum_mean', 'site', 'species']
    
    # Filter columns if they exist
    existing_cols = [c for c in colonnes_a_garder if c in df.columns]
    df = df[existing_cols]

    # Replace infs
    df = df.replace([np.inf, -np.inf], np.nan)

    def _shift_and_rename(base: pd.DataFrame, lag: int) -> pd.DataFrame:
        shifted = base.copy()
        shifted['Year'] = shifted['Year'] + lag
        return shifted.rename(columns={col: f"{col}_year_plus_{lag}" for col in shifted.columns})

    # We want to predict Y(t) using X(t-1) (and optionally X(t-2)).
    df_year_plus_1 = _shift_and_rename(df, lag=1)

    print("Merging data (Year N with Year N-1 features)...")
    df_merged = pd.merge(
        df,
        df_year_plus_1,
        left_on=['Year', 'site', 'species'],
        right_on=['Year_year_plus_1', 'site_year_plus_1', 'species_year_plus_1'],
        how='inner',
    )

    if n_lags >= 2:
        df_year_plus_2 = _shift_and_rename(df, lag=2)
        print("Merging data (Year N with Year N-2 features)...")
        df_merged = pd.merge(
            df_merged,
            df_year_plus_2,
            left_on=['Year', 'site', 'species'],
            right_on=['Year_year_plus_2', 'site_year_plus_2', 'species_year_plus_2'],
            how='inner',
        )

    df_merged = df_merged.rename(columns={'Year': 'annee'})
    
    # Target is abundance in year N (Left side)
    if 'abondance_capped' in df_merged.columns:
        y = df_merged['abondance_capped']
    else:
        raise ValueError("Target column 'abondance_capped' not found.")

    # Features: Keep only Year N-1 features (Right side) and 'annee'
    # We also keep 'abondance_capped_year_plus_1' as an autoregressive feature
    
    # Identify lagged feature columns (exclude merge keys)
    feature_cols: list[str] = []
    for lag in range(1, n_lags + 1):
        lag_cols = [c for c in df_merged.columns if c.endswith(f'_year_plus_{lag}')]
        drop_keys = {
            f'Year_year_plus_{lag}',
            f'site_year_plus_{lag}',
            f'species_year_plus_{lag}',
        }
        feature_cols.extend([c for c in lag_cols if c not in drop_keys])
    
    # Also keep 'annee' for splitting and 'species'/'site' if needed (but usually dropped for X)
    # We keep 'species' to filter later if needed, but prepare_xy usually handles X.
    # Wait, prepare_data_from_raw returns X, y.
    # If we keep 'species' in X, we must handle it in pipeline (OHE) or drop it.
    # The current pipeline handles categorical features.
    
    cols_to_keep = feature_cols + ['annee']
    if 'species' in df_merged.columns:
        cols_to_keep.append('species')
        
    X = df_merged[cols_to_keep].copy()
    
    return X, y

def prepare_xy(X: pd.DataFrame, y: pd.Series):
    """
    - Drops rows with NaN target (y).
    - Replaces +/-inf in X with NaN (to be imputed).
    """
    before = len(y)
    mask = ~y.isna()
    dropped = int(before - mask.sum())
    if dropped > 0:
        print(f"Dropping {dropped} rows due to NaN target values.")
    X_clean = X.loc[mask].copy()
    y_clean = y.loc[mask].copy()
    # Sanitize infinities
    X_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    return X_clean, y_clean

def time_based_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_frac: float = 0.8,
    test_frac: float = 0.2,
    val_last_n_years: int = 3,
    test_last_n_years: int = 0,
):
    """Time-based split where the **last ``val_last_n_years`` years** are held out.

    - Toutes les années **sauf les ``val_last_n_years`` dernières** servent au *train* et à la *validation interne*.
      On découpe chronologiquement ces années en Train et Validation-interne via ``train_frac``.
    - Les ``val_last_n_years`` dernières années forment un **jeu de test final** totalement hors entraînement.

    La fonction retourne (X_train, y_train, X_val_int, y_val_int, X_test_final, y_test_final) où
    "val_int" est utilisé pour l'early stopping et "test_final" correspond aux 3 dernières années.
    """
    # Combine X and y to sort together
    data = X.copy()
    data['target_y'] = y

    # Sort by year
    data = data.sort_values('annee')

    years_sorted = np.sort(data['annee'].unique())
    if len(years_sorted) <= val_last_n_years + 1:
        # Trop peu d'années pour un vrai holdout : on garde uniquement la dernière en test
        holdout_years = years_sorted[-1:]
    else:
        holdout_years = years_sorted[-val_last_n_years:]

    is_holdout = data['annee'].isin(holdout_years)

    early_data = data.loc[~is_holdout]
    test_data = data.loc[is_holdout]

    # Fractional split inside early years for Train / Validation-interne
    n_early = len(early_data)
    train_end = int(n_early * train_frac)

    train_data = early_data.iloc[:train_end]
    val_int_data = early_data.iloc[train_end:]

    print("Time-based split (sorted by year):")
    print(f"  Train: {len(train_data)} samples ({train_data['annee'].min()} - {train_data['annee'].max()})")
    print(f"  Val (interne):  {len(val_int_data)} samples ({val_int_data['annee'].min()} - {val_int_data['annee'].max()})")
    print(f"  Test final (dernieres annees):   {len(test_data)} samples ({test_data['annee'].min()} - {test_data['annee'].max()})")

    X_train = train_data.drop(columns=['target_y'])
    y_train = train_data['target_y']

    X_val_int = val_int_data.drop(columns=['target_y'])
    y_val_int = val_int_data['target_y']

    X_test_final = test_data.drop(columns=['target_y'])
    y_test_final = test_data['target_y']

    return X_train, y_train, X_val_int, y_val_int, X_test_final, y_test_final

def build_pipeline(X: pd.DataFrame, random_state=42):
    """
    Builds a preprocessing + XGBoost regression pipeline.
    """
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Drop columns that are entirely NaN
    numeric_features = [c for c in numeric_features if X[c].notna().any()]
    categorical_features = [c for c in categorical_features if X[c].notna().any()]

    transformers = []
    if numeric_features:
        # XGBoost handles NaNs, but scaling might help convergence slightly or interpretation
        # Actually XGBoost doesn't need scaling. But let's keep SimpleImputer if we want to be safe,
        # though XGBoost handles missing values. Let's just pass through or simple impute.
        # We'll keep the structure but remove scaling to keep it raw for XGBoost (it works well).
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            # ('scaler', StandardScaler()), # Not strictly necessary for trees
        ])
        transformers.append(('num', num_pipe, numeric_features))
    if categorical_features:
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ])
        transformers.append(('cat', cat_pipe, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )

    # XGBRegressor
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=1.0,
        objective="reg:squarederror",
    )

    pipe = Pipeline(steps=[
        ('prep', preprocessor),
        ('model', model),
    ])
    return pipe

def evaluate_model(model, X, y, set_name="Test"):
    # Predict on log scale if model was trained on log scale?
    # We will handle log transform outside or inside.
    # Let's assume the model predicts the transformed target.
    
    y_pred_trans = model.predict(X)
    
    # Inverse transform (expm1)
    y_pred = np.expm1(y_pred_trans)
    y_true = np.expm1(y) # Assuming y passed here is also log-transformed
    
    # Clip negative predictions just in case
    y_pred = np.maximum(y_pred, 0)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"[{set_name}] RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return y_pred

def plot_evolution(X_val, y_val_log, y_pred_val, species_name=None, save_path=None):
    """
    Plots the true vs predicted values over time (years) for the validation set.
    If multiple species are present, plots the aggregate mean and individual species trends.
    """
    df_res = X_val.copy()
    df_res['True'] = np.expm1(y_val_log)
    df_res['Predicted'] = y_pred_val
    
    plt.figure(figsize=(12, 7))
    
    # Check if we have multiple species
    if 'species' in df_res.columns and df_res['species'].nunique() > 1:
        species_list = df_res['species'].unique()
        print(f"Plotting evolution for {len(species_list)} species...")
        
        # Plot individual species (faint)
        # Use a colormap
        colors = plt.cm.tab20(np.linspace(0, 1, len(species_list)))
        
        for i, spec in enumerate(species_list):
            spec_data = df_res[df_res['species'] == spec]
            yearly_spec = spec_data.groupby('annee')[['True', 'Predicted']].mean()
            
            # Plot only if we have enough points
            if len(yearly_spec) > 1:
                plt.plot(yearly_spec.index, yearly_spec['True'], color=colors[i], alpha=0.3, linewidth=1)
                plt.plot(yearly_spec.index, yearly_spec['Predicted'], color=colors[i], alpha=0.3, linestyle='--', linewidth=1)
        
        # Add a dummy legend entry for individual species
        plt.plot([], [], color='gray', alpha=0.5, label='Individual Species (True)')
        plt.plot([], [], color='gray', alpha=0.5, linestyle='--', label='Individual Species (Pred)')

    # Group by year to see the global trend (Cluster Average)
    # Use geometric mean of abundances (oiseaux se multiplient)
    yearly_stats = df_res.groupby('annee').agg({
        'True': lambda v: geometric_mean_abundance(v.values),
        'Predicted': lambda v: geometric_mean_abundance(v.values),
    })
    
    plt.plot(yearly_stats.index, yearly_stats['True'], marker='o', color='black', linewidth=2, label='Cluster Geom. Mean (True)')
    plt.plot(yearly_stats.index, yearly_stats['Predicted'], marker='x', color='red', linestyle='--', linewidth=2, label='Cluster Geom. Mean (Pred)')
    
    # Add vertical line to separate Train/Test/Val if possible?
    # We don't know the split years here easily without passing them.
    # But we can just plot.
    
    title = "Species Evolution (Full Timeline)"
    if species_name:
        title += f" - {species_name}"
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Mean Abundance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def load_cluster_mapping(csv_path: str):
    """
    Loads the species-to-cluster mapping from CSV.
    Returns a dictionary {cluster_id: [species_list]} or a DataFrame.
    """
    print(f"Loading cluster mapping from {csv_path}...")
    # Assuming the first column is the species name (index)
    df = pd.read_csv(csv_path, index_col=0)
    if 'cluster' not in df.columns:
        raise ValueError("Cluster CSV must contain a 'cluster' column.")
    return df['cluster']

def eval_manual(model, X, y, label="Set"):
    """
    Evaluates the model on a given set (X, y).
    Returns predictions (original scale).
    """
    y_pred_log = model.predict(X)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y)
    
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"[{label}] R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return y_pred

def main():
    parser = argparse.ArgumentParser(description='Train a model to predict species evolution (time-split).')
    parser.add_argument('--raw-data', default='raw_data/STOC_pressions_bio_assol_meteo_pesti_routes_lum_120625.csv', help='Path to raw CSV')
    parser.add_argument('--species', type=str, default=None, help='Filter for a specific species (optional)')
    parser.add_argument('--eval-species', type=str, default=None, help='Evaluate model specifically on this species within the cluster')
    parser.add_argument('--cluster', type=int, default=None, help='Filter for all species in a specific cluster (optional)')
    parser.add_argument('--all-clusters', action='store_true', help='Run for all clusters sequentially')
    parser.add_argument('--clusters-file', default='figures/species_clusters.csv', help='Path to species clusters CSV')
    
    args = parser.parse_args()
    
    # Load cluster map first if needed
    cluster_map = None
    clusters_to_run = []
    
    if args.all_clusters:
        cluster_map = load_cluster_mapping(args.clusters_file)
        clusters_to_run = sorted(cluster_map.unique())
        print(f"Running for all clusters: {clusters_to_run}")
    elif args.cluster is not None:
        cluster_map = load_cluster_mapping(args.clusters_file)
        clusters_to_run = [args.cluster]
    else:
        # Run for all data or specific species
        clusters_to_run = [None]

    # Use the raw data loader ONCE if possible? 
    # No, prepare_data_from_raw filters columns and merges.
    # But it loads the huge CSV every time.
    # Let's load the raw CSV once here and pass it?
    # prepare_data_from_raw takes a path.
    # We can refactor prepare_data_from_raw to take a dataframe.
    
    print(f"Loading raw data from {args.raw_data}...")
    df_raw = pd.read_csv(args.raw_data, encoding='latin1')
    
    for cluster_id in clusters_to_run:
        print(f"\n{'='*40}")
        if cluster_id is not None:
            print(f"Processing Cluster {cluster_id}")
        else:
            print("Processing Full Dataset / Single Species")
        print(f"{'='*40}")
        
        # Prepare data for this cluster
        # We need to replicate logic of prepare_data_from_raw but using df_raw
        
        # 1. Filter columns
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
        
        # Filter by species/cluster BEFORE merge to save memory/time
        selected_species_name = "All Species"
        if args.species:
            df = df[df['species'] == args.species]
            selected_species_name = args.species
        elif cluster_id is not None:
            species_in_cluster = cluster_map[cluster_map == cluster_id].index.tolist()
            df = df[df['species'].isin(species_in_cluster)]
            selected_species_name = f"Cluster {cluster_id}"
            print(f"Species in Cluster {cluster_id}: {', '.join(species_in_cluster)}")
        
        # 2. Merge (Year N with Year N-1 features)
        df_year_plus_1 = df.copy()
        df_year_plus_1['Year'] = df_year_plus_1['Year'] + 1
        df_year_plus_1 = df_year_plus_1.rename(columns={col: f"{col}_year_plus_1" for col in df_year_plus_1.columns})

        # print("Merging data...")
        df_merged = pd.merge(
            df,
            df_year_plus_1,
            left_on=['Year', 'site', 'species'],
            right_on=['Year_year_plus_1', 'site_year_plus_1', 'species_year_plus_1'],
            how='inner'
        )

        df_merged = df_merged.rename(columns={'Year': 'annee'})
        
        if 'abondance_capped' not in df_merged.columns:
            print("Target not found, skipping.")
            continue
             
        y = df_merged['abondance_capped']
        
        feature_cols = [c for c in df_merged.columns if c.endswith('_year_plus_1')]
        cols_to_keep = feature_cols + ['annee']
        if 'species' in df_merged.columns:
            cols_to_keep.append('species')
            
        X = df_merged[cols_to_keep].copy()
        
        # 3. Prepare XY
        X, y = prepare_xy(X, y)
        
        if len(X) < 50:
            print("Not enough data for this cluster/species.")
            continue

        # Log-transform target
        y_log = np.log1p(y)
        
        # Time-based split
        X_train, y_train, X_test, y_test, X_val, y_val = time_based_split(X, y_log, train_frac=0.6, test_frac=0.2)
        
        model = build_pipeline(X_train)
        
        print("Training XGBoost model...")
        prep = model.named_steps['prep']
        X_train_proc = prep.fit_transform(X_train)
        X_test_proc = prep.transform(X_test)
        X_val_proc = prep.transform(X_val)
        
        xgb_model = model.named_steps['model']
        xgb_model.fit(
            X_train_proc, y_train,
            eval_set=[(X_test_proc, y_test)],
            verbose=False
        )
        
        print("Evaluating on Validation Set...")
        _ = eval_manual(xgb_model, X_val_proc, y_val, "Validation")

        # Specific evaluation on a requested species (if present in this cluster/split)
        if args.eval_species:
            print(f"--- Specific Evaluation for {args.eval_species} ---")
            if 'species' in X_val.columns:
                # Use boolean mask on the original dataframe to filter the processed array
                mask_spec = (X_val['species'] == args.eval_species).values
                if mask_spec.sum() > 0:
                    X_val_spec_proc = X_val_proc[mask_spec]
                    y_val_spec = y_val[mask_spec]
                    _ = eval_manual(xgb_model, X_val_spec_proc, y_val_spec, f"Validation ({args.eval_species})")
                else:
                    print(f"Warning: Species '{args.eval_species}' not found in the Validation set.")
            else:
                print("Warning: 'species' column not found in validation data for filtering.")

        # Compute predictions for ALL years (Train + Test + Val) and aggregate by year
        X_all_proc = prep.transform(X)
        y_pred_all = eval_manual(xgb_model, X_all_proc, y_log, "Full Set")

        df_all_plot = X.copy()
        df_all_plot['True'] = np.expm1(y_log)
        df_all_plot['Predicted'] = y_pred_all
        pred_yearly = df_all_plot.groupby('annee').agg({
            'True': lambda v: geometric_mean_abundance(v.values),
            'Predicted': lambda v: geometric_mean_abundance(v.values),
        })

        # Years used for validation (to highlight the non-trained part of the curve)
        val_years = sorted(X_val['annee'].unique())

if __name__ == '__main__':
    main()
