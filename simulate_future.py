import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from species_evolution_model import (
    prepare_data_from_raw,
    prepare_xy,
    build_pipeline,
    load_cluster_mapping
)


SCENARIO_RULES_TEXT = """Scenario (appliqué chaque année sur les variables de pression courantes, pas sur les lags):
- Pesticides: colonnes contenant 'all_pesticide_exposure' -> * 0.15 (baisse drastique)
- Bio: colonnes contenant 'SurfBio_cell' -> * 4.0 (explosion du bio)
- Lumière: colonnes contenant 'Lum_' -> * 0.4 (sobriété nocturne)
- Urbain: colonnes contenant 'Area_grid_tot_Urba' ou 'ValueSPred_Urba' -> * 0.95 (renaturation)
- Forêt: colonnes contenant 'Area_grid_tot_For' ou 'ValueSPred_For' -> * 1.15 (puits de carbone)
- Cultures: colonnes contenant 'Area_grid_tot_CC' ou 'ValueSPred_CC' -> * 0.85 (moins d'élevage/cultures)
- Température: colonnes contenant 'Temps_' ou 'Temps_mean' -> + 1.2 (°C additionnels)
"""

def apply_scenario_rules(features_df, year_step=1):
    """
    Applies the transformation rules to generate the features for the NEXT year.
    features_df: DataFrame containing features for Year N.
    Returns: DataFrame containing features for Year N+1.
    """
    next_features = features_df.copy()
    
    # 0. Update Year (if present)
    if 'annee' in next_features.columns:
        next_features['annee'] = next_features['annee'] + year_step
    
    # Identify columns by suffix/prefix in the context of the training data
    # Note: In prepare_data_from_raw, we select columns like "Temps..._year_plus_1"
    # because X(t-1) predicts Y(t). 
    # BUT, wait. 
    # In species_evolution_model, X contains columns ending in "_year_plus_1".
    # This naming is confusing. 
    # The 'prepare_data_from_raw' does:
    #   df (Left, Year=T) merge df_year_plus_1 (Right, Year=T <-- Year=T-1)
    #   So the columns in X are named `*_year_plus_1`.
    #   These columns actually represent the state at T-1.
    #   We use them to predict Abundance at T.
    
    # So if we want to predict 2024, we need input features representing 2023.
    # If we have 2023 data, we can predict 2024.
    # Then to predict 2025, we need features representing 2024.
    # We generate features_2024 from features_2023 using the rules.
    
    cols = [
        c for c in next_features.columns
        if not (str(c).endswith('_lag1') or str(c).endswith('_lag2'))
        and 'abondance' not in str(c).lower()
    ]
    
    # 1. Pesticides (baisse drastique)
    # Pattern: all_pesticide_exposure...
    pesti_cols = [c for c in cols if 'all_pesticide_exposure' in str(c)]
    for c in pesti_cols:
        next_features[c] = next_features[c] * 0.15
        
    # 2. Agriculture Bio (explosion)
    # Pattern: SurfBio_cell
    bio_cols = [c for c in cols if 'SurfBio_cell' in c]
    for c in bio_cols:
        next_features[c] = next_features[c] * 4.0
        
    # 3. Pollution Lumineuse (sobriété nocturne)
    # Prefix: Lum_
    lum_cols = [c for c in cols if 'Lum_' in c]
    for c in lum_cols:
        next_features[c] = next_features[c] * 0.4
        
    # 4. Urbanisation (renaturation)
    # Patterns: Area_grid_tot_Urba / ValueSPred_Urba
    urba_cols = [c for c in cols if ('Area_grid_tot_Urba' in c) or ('ValueSPred_Urba' in c)]
    for c in urba_cols:
        next_features[c] = next_features[c] * 0.95
        
    # 5. Forêt (puits de carbone)
    # Patterns: Area_grid_tot_For / ValueSPred_For
    forest_cols = [c for c in cols if ('Area_grid_tot_For' in c) or ('ValueSPred_For' in c)]
    for c in forest_cols:
        next_features[c] = next_features[c] * 1.15

    # 6. Cultures (moins d'élevage / cultures)
    # Patterns: Area_grid_tot_CC / ValueSPred_CC
    cc_cols = [c for c in cols if ('Area_grid_tot_CC' in c) or ('ValueSPred_CC' in c)]
    for c in cc_cols:
        next_features[c] = next_features[c] * 0.85

    # 7. Températures : +1.2
    # Prefix: Temps_ / Temps_mean
    temp_cols = [c for c in cols if ('Temps_' in c) or ('Temps_mean' in c)]
    for c in temp_cols:
        next_features[c] = next_features[c] + 1.2

    return next_features

def geometric_mean(series, *, present_only: bool = False):
    arr = np.asarray(series, dtype=float)
    arr = np.maximum(arr, 0.0)
    if present_only:
        arr = arr[arr > 0]
    if arr.size == 0:
        return float('nan')
    return float(np.expm1(np.mean(np.log1p(arr))))


def _pressure_columns_for_deltas(df: pd.DataFrame, *, use_proc: bool) -> list[str]:
    cols: list[str] = []
    if use_proc:
        for c in df.columns:
            s = str(c)
            if s == 'annee':
                continue
            if s.endswith('_lag1') or s.endswith('_lag2'):
                continue
            if 'abondance' in s.lower():
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
    else:
        # In year_plus schema, the scenario is applied to *_year_plus_1 columns
        for c in df.columns:
            s = str(c)
            if not s.endswith('_year_plus_1'):
                continue
            if 'abondance' in s.lower():
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
    return cols


def _compute_pressure_deltas(prev_state: pd.DataFrame, next_state: pd.DataFrame, *, use_proc: bool) -> dict:
    cols = _pressure_columns_for_deltas(next_state, use_proc=use_proc)
    if not cols:
        out = {
            'pressure_n_cols': 0,
            'pressure_delta_abs_mean': np.nan,
            'pressure_delta_abs_median': np.nan,
            'pressure_delta_rel_mean': np.nan,
        }
        for i in range(1, 6):
            out[f'pressure_top{i}_name'] = ''
            out[f'pressure_top{i}_abs_mean'] = np.nan
            out[f'pressure_top{i}_rel_mean'] = np.nan
        return out

    prev = prev_state.reindex(columns=cols)
    nxt = next_state.reindex(columns=cols)
    prev_num = prev.apply(pd.to_numeric, errors='coerce')
    nxt_num = nxt.apply(pd.to_numeric, errors='coerce')

    delta_df = nxt_num - prev_num
    abs_df = delta_df.abs()

    denom_df = prev_num.abs() + 1e-9
    rel_df = abs_df / denom_df

    abs_delta = abs_df.to_numpy(dtype=float)
    rel = rel_df.to_numpy(dtype=float)

    abs_mean_per_col = abs_df.mean(axis=0, skipna=True)
    rel_mean_per_col = rel_df.mean(axis=0, skipna=True)
    top = abs_mean_per_col.sort_values(ascending=False).head(5)

    out = {
        'pressure_n_cols': int(len(cols)),
        'pressure_delta_abs_mean': float(np.nanmean(abs_delta)),
        'pressure_delta_abs_median': float(np.nanmedian(abs_delta)),
        'pressure_delta_rel_mean': float(np.nanmean(rel)),
    }

    for i in range(1, 6):
        if i <= len(top):
            col_name = str(top.index[i - 1])
            out[f'pressure_top{i}_name'] = col_name
            out[f'pressure_top{i}_abs_mean'] = float(top.iloc[i - 1])
            out[f'pressure_top{i}_rel_mean'] = float(rel_mean_per_col.get(top.index[i - 1], np.nan))
        else:
            out[f'pressure_top{i}_name'] = ''
            out[f'pressure_top{i}_abs_mean'] = np.nan
            out[f'pressure_top{i}_rel_mean'] = np.nan

    return out

def main():
    parser = argparse.ArgumentParser(description="Simulate species evolution 2023-2033")
    parser.add_argument('--mode', choices=['cluster', 'species', 'all'], required=True, help="Scope of simulation")
    parser.add_argument('--target', help="Cluster ID (int) or Species Name (str)")
    parser.add_argument('--n-lags', type=int, choices=[1, 2], default=1, help="Use lagged inputs from t-1 only (1) or t-1 and t-2 (2)")
    parser.add_argument('--no-scenario', action='store_true', help="Do not apply scenario rules (useful to test pure autoregressive drift)")
    ab_group = parser.add_mutually_exclusive_group()
    ab_group.add_argument('--include-abundance-features', dest='use_abundance_features', action='store_true', help="Include abundance (and abundance lags) as model inputs (autoregressive)")
    ab_group.add_argument('--exclude-abundance-features', dest='use_abundance_features', action='store_false', help="Exclude abundance (and abundance lags) from model inputs (default)")
    parser.set_defaults(use_abundance_features=False)
    parser.add_argument('--clip-train-range', action='store_true', help="Clip simulated numeric features to the [min,max] seen in training (reduces out-of-domain drift)")
    parser.add_argument('--debug', action='store_true', help="Write per-year diagnostics CSV next to outputs")
    parser.add_argument('--gm-present-only', action='store_true', help="Compute geometric means only on values > 0 (otherwise 0 is treated as an observation)")
    parser.add_argument('--raw-data', default='raw_data/STOC_pressions_bio_assol_meteo_pesti_routes_lum_120625.csv')
    parser.add_argument('--proc-data', default=None, help="Optional processed CSV with explicit *_lag1/_lag2 columns (overrides --raw-data loader)")
    parser.add_argument('--clusters-file', default='figures/species_clusters.csv')
    
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    figures_dir = base_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print("Loading data...")
    # This prepares X (state at T-1) and y (Abundance at T) with 'annee' = T
    # The last year in 'annee' is 2023 (since we have 2002-2023 data).
    # This means X contains features from 2022.
    # WAIT. 
    # If the raw data goes up to 2023.
    # We want to start predicting FROM 2023 (to get 2024).
    # Do we have features for 2023 in the raw file?
    # prepare_data_from_raw relies on Year+1 merge.
    # If raw file has row 2023. It can be Left side (Y=2023).
    # It needs Right side (Year_year_plus_1 = 2023 -> Original Year 2022).
    # So we get X(2022) -> Y(2023).
    # To predict Y(2024), we need X(2023).
    # Does the raw file contain a row 2023 with features? Yes.
    # But prepare_data_from_raw won't output a row for 2024 because there is no Left side (Y=2024) to merge with!
    
    # We need to manually extract the MOST RECENT features (2023) from the RAW dataframe directly,
    # before the merge that drops the future.
    
    df_raw = pd.read_csv(args.raw_data, encoding='latin1')
    
    # Load clusters
    cluster_series = load_cluster_mapping(args.clusters_file)
    
    # Filter Scope
    species_list = []
    if args.mode == 'all':
        species_list = df_raw['species'].unique().tolist()
    elif args.mode == 'cluster':
        cluster_id = int(args.target)
        species_list = cluster_series[cluster_series == cluster_id].index.tolist()
    elif args.mode == 'species':
        species_list = [args.target]

    print(f"Scope: {args.mode} ({len(species_list)} species)")
    if len(species_list) == 0:
        print("No species found.")
        return

    # 2. Train Model
    print("Training model...")

    def drop_abundance_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
        # Default behavior: exclude abundance from features
        if args.use_abundance_features:
            return df
        drop_cols = [c for c in df.columns if 'abondance' in str(c).lower()]
        return df.drop(columns=drop_cols, errors='ignore')

    use_proc = args.proc_data is not None
    if use_proc:
        proc_path = (base_dir / args.proc_data).resolve() if not Path(args.proc_data).is_absolute() else Path(args.proc_data)
        df_proc = pd.read_csv(proc_path, encoding='utf-8', low_memory=False)
        if 'Year' not in df_proc.columns or 'species' not in df_proc.columns:
            raise ValueError("Processed CSV must contain 'Year' and 'species' columns")
        if 'abondance_capped' not in df_proc.columns:
            raise ValueError("Processed CSV must contain 'abondance_capped' (target)")

        df_proc = df_proc.rename(columns={'Year': 'annee'})

        # Filter scope
        df_proc = df_proc[df_proc['species'].isin(species_list)].copy()

        y_train = df_proc['abondance_capped'].copy()
        X_train = df_proc.drop(columns=['abondance_capped']).copy()
        X_train = drop_abundance_feature_columns(X_train)

        X_train, y_train = prepare_xy(X_train, y_train)
        y_train_log = np.log1p(y_train)

        pipeline = build_pipeline(X_train)
        pipeline.fit(X_train, y_train_log)

        print(f"Model trained from processed CSV: {proc_path}")
    else:
        # Use raw -> year_plus_* dataset
        X_full, y_full = prepare_data_from_raw(args.raw_data, n_lags=args.n_lags)

        mask = X_full['species'].isin(species_list)
        X_train = X_full[mask].copy()
        y_train = y_full[mask].copy()

        X_train = drop_abundance_feature_columns(X_train)

        X_train, y_train = prepare_xy(X_train, y_train)
        y_train_log = np.log1p(y_train)

        pipeline = build_pipeline(X_train)
        pipeline.fit(X_train, y_train_log)

        print("Model trained.")

    train_numeric = X_train.select_dtypes(include=[np.number]).copy()
    train_min = train_numeric.min(numeric_only=True)
    train_max = train_numeric.max(numeric_only=True)

    def clip_to_train_range(df: pd.DataFrame) -> pd.DataFrame:
        if not args.clip_train_range:
            return df
        out = df.copy()
        num_cols = out.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            if c in train_min.index and c in train_max.index:
                out[c] = out[c].clip(train_min[c], train_max[c])
        return out

    # 3. Prepare Initial State for Simulation
    if use_proc:
        max_year = int(df_proc['annee'].max())
        print(f"Last available year in processed data: {max_year}")

        # State is last observed year (t=max_year). We'll apply scenario to get t+1 pressures.
        req_cols = X_train.columns
        state = df_proc[df_proc['annee'] == max_year].copy()
        state = state[req_cols].copy()

        # Historical series for plot
        yearly_hist = df_proc.groupby('annee')['abondance_capped'].apply(lambda s: geometric_mean(s, present_only=args.gm_present_only))
    else:
        max_year = df_raw['Year'].max()
        print(f"Last available year in raw data: {max_year}")

        req_cols = X_train.columns

        if args.n_lags == 1:
            latest_data = df_raw[df_raw['Year'] == max_year].copy()
            latest_data = latest_data[latest_data['species'].isin(species_list)]

            rename_dict = {col: f"{col}_year_plus_1" for col in latest_data.columns if col not in ['Year', 'site', 'species']}
            latest_data = latest_data.rename(columns=rename_dict)
            latest_data['annee'] = max_year + 1

            state = latest_data[req_cols].copy()
        else:
            latest_lag1 = df_raw[df_raw['Year'] == max_year].copy()
            latest_lag1 = latest_lag1[latest_lag1['species'].isin(species_list)]
            latest_lag2 = df_raw[df_raw['Year'] == (max_year - 1)].copy()
            latest_lag2 = latest_lag2[latest_lag2['species'].isin(species_list)]

            rename_lag1 = {col: f"{col}_year_plus_1" for col in latest_lag1.columns if col not in ['Year', 'site', 'species']}
            rename_lag2 = {col: f"{col}_year_plus_2" for col in latest_lag2.columns if col not in ['Year', 'site', 'species']}
            latest_lag1 = latest_lag1.rename(columns=rename_lag1).drop(columns=['Year'])
            latest_lag2 = latest_lag2.rename(columns=rename_lag2).drop(columns=['Year'])

            merged = pd.merge(latest_lag1, latest_lag2, on=['site', 'species'], how='inner')
            merged['annee'] = max_year + 1
            state = merged[req_cols].copy()

        hist_stats = X_full[X_full['species'].isin(species_list)].copy()
        hist_stats['abundance'] = y_full[hist_stats.index]
        yearly_hist = hist_stats.groupby('annee')['abundance'].apply(lambda s: geometric_mean(s, present_only=args.gm_present_only))

    
    # 4. Simulation Loop
    future_years = range(max_year + 1, 2034) # 2024 to 2033
    
    # Add 2023 point to predictions list for continuity in plot (using real data)
    # Actually, plot will join them.
    
    print("Simulating future scenarios...")
    
    simulated_means = []
    
    # Loop
    # We handle the 'state' as a DataFrame (current_features)
    # encompassing ALL sites and ALL species in the scope.
    
    diagnostics_rows = []
    
    for _year in future_years:
        prev_state = state.copy()

        # In proc-data mode, state currently represents year (_year-1). We first build year _year pressures.
        if use_proc:
            next_state = state.copy()
            if 'annee' in next_state.columns:
                next_state['annee'] = next_state['annee'] + 1

            # Shift lag columns
            lag1_cols = [c for c in next_state.columns if str(c).endswith('_lag1')]
            for c1 in lag1_cols:
                c2 = c1[:-len('_lag1')] + '_lag2'
                if c2 in next_state.columns:
                    next_state[c2] = next_state[c1]

            # Update lag1 from current (non-lag) columns
            base_numeric = [
                c for c in next_state.columns
                if pd.api.types.is_numeric_dtype(next_state[c])
                and c != 'annee'
                and 'abondance' not in str(c).lower()
                and not (str(c).endswith('_lag1') or str(c).endswith('_lag2'))
            ]
            for base in base_numeric:
                c1 = f"{base}_lag1"
                if c1 in next_state.columns:
                    next_state[c1] = state[base]

            # Apply scenario to current-year pressures (non-lag)
            if not args.no_scenario:
                before_scenario = next_state.copy()
                next_state = apply_scenario_rules(next_state, year_step=0)
            else:
                before_scenario = next_state.copy()

            state_for_pred = next_state
            preds_log = pipeline.predict(state_for_pred)
            preds = np.expm1(preds_log)
            preds = np.maximum(preds, 0)

            # If abundance is part of the state/features, keep it updated; otherwise, ignore it
            if args.use_abundance_features and ('abondance_capped' in next_state.columns):
                next_state['abondance_capped'] = preds

            state = clip_to_train_range(next_state)
        else:
            preds_log = pipeline.predict(state)
            preds = np.expm1(preds_log)
            preds = np.maximum(preds, 0)

        if args.debug:
            s = pd.Series(preds)
            diag = {
                'year_predicted': int(_year),
                'n_rows': int(len(state)),
                'pred_min': float(np.min(preds)) if len(preds) else np.nan,
                'pred_p10': float(np.percentile(preds, 10)) if len(preds) else np.nan,
                'pred_median': float(np.median(preds)) if len(preds) else np.nan,
                'pred_p90': float(np.percentile(preds, 90)) if len(preds) else np.nan,
                'pred_max': float(np.max(preds)) if len(preds) else np.nan,
                'pred_zero_frac': float((s <= 0).mean()) if len(preds) else np.nan,
            }

            if use_proc:
                diag.update(_compute_pressure_deltas(before_scenario, state, use_proc=True))
            else:
                # In raw mode, state update happens later; we fill this after next_state is computed
                diag.update({
                    'pressure_n_cols': 0,
                    'pressure_delta_abs_mean': np.nan,
                    'pressure_delta_abs_median': np.nan,
                    'pressure_delta_rel_mean': np.nan,
                })
            key_cols = [
                'abondance_capped_year_plus_1',
                'abondance_capped_year_plus_2',
                'all_pesticide_exposure_grid_sum_year_plus_1',
                'mean_concentration_water_grid_sum_year_plus_1',
                'mean_tii_grid_sum_year_plus_1',
                'SurfBio_cell_year_plus_1',
                'ValueSPred_Urba_year_plus_1',
                'ValueSPred_For_year_plus_1',
                'Lum_mean_year_plus_1',
            ]
            for c in key_cols:
                if c in state.columns and pd.api.types.is_numeric_dtype(state[c]):
                    diag[f'{c}__mean'] = float(state[c].mean())
            diagnostics_rows.append(diag)
        
        # Calculate mean abundance for this year
        mean_ab = geometric_mean(preds, present_only=args.gm_present_only)
        simulated_means.append(mean_ab)
        
        # Prepare state for Next Year
        # Apply rules to 'state'
        # Note: 'state' currently represents features of Year T-1 (used to predict T).
        # We want features of Year T (to predict T+1).
        # So we apply the evolution rules to the features.
        
        if (not use_proc) and args.n_lags == 1:
            next_state = state.copy()
            if 'annee' in next_state.columns:
                next_state['annee'] = next_state['annee'] + 1

            if not args.no_scenario:
                next_state = apply_scenario_rules(next_state, year_step=0)

            if args.use_abundance_features and ('abondance_capped_year_plus_1' in next_state.columns):
                next_state['abondance_capped_year_plus_1'] = preds

            state = clip_to_train_range(next_state)

            if args.debug and diagnostics_rows:
                diagnostics_rows[-1].update(_compute_pressure_deltas(prev_state, state, use_proc=False))
        elif not use_proc:
            next_state = state.copy()
            if 'annee' in next_state.columns:
                next_state['annee'] = next_state['annee'] + 1

            lag1_cols = [c for c in state.columns if c.endswith('_year_plus_1')]
            for c1 in lag1_cols:
                c2 = c1[:-len('_year_plus_1')] + '_year_plus_2'
                if c2 in next_state.columns:
                    next_state[c2] = state[c1]

            if not args.no_scenario:
                lag1_updated = apply_scenario_rules(state[lag1_cols].copy(), year_step=0)
                for c1 in lag1_cols:
                    next_state[c1] = lag1_updated[c1]

            if args.use_abundance_features and ('abondance_capped_year_plus_1' in next_state.columns):
                next_state['abondance_capped_year_plus_1'] = preds

            state = clip_to_train_range(next_state)

            if args.debug and diagnostics_rows:
                diagnostics_rows[-1].update(_compute_pressure_deltas(prev_state, state, use_proc=False))
        
    # 5. Plotting
    plt.figure(figsize=(12, 7))
    
    # Plot History
    plt.plot(yearly_hist.index, yearly_hist.values, marker='o', color='black', label='Historical Data (2002-2023)')
    
    # Plot Future
    plt.plot(future_years, simulated_means, marker='^', color='red', linestyle='--', label='Prediction (Scenario 2023-2033)')
    
    plt.title(f"Abundance Prediction 2023-2033\nScope: {args.mode} {args.target if args.target else ''}")
    plt.xlabel("Year")
    plt.ylabel("Geometric Mean Abundance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    out_path = figures_dir / f"prediction_{args.mode}_{args.target if args.target else 'global'}.png"
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

    scenario_path = out_path.with_name(out_path.stem + '_scenario.txt')
    with open(scenario_path, 'w', encoding='utf-8') as f:
        f.write(SCENARIO_RULES_TEXT)
    print(f"Saved scenario summary to {scenario_path}")
    
    # Save numerical results
    txt_path = out_path.with_suffix('.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("Year,Abundance\n")
        f.write(f"2023,{yearly_hist.iloc[-1]:.4f}\n")
        for y, val in zip(future_years, simulated_means):
            f.write(f"{y},{val:.4f}\n")

    if args.debug and diagnostics_rows:
        diag_path = out_path.with_name(out_path.stem + '_debug.csv')
        pd.DataFrame(diagnostics_rows).to_csv(diag_path, index=False, encoding='utf-8')
        print(f"Saved diagnostics to {diag_path}")

if __name__ == "__main__":
    main()
