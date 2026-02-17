import argparse

import numpy as np
import pandas as pd

from species_evolution_model import (
    prepare_xy,
    time_based_split,
    build_pipeline,
    geometric_mean_abundance,
    load_cluster_mapping,
    eval_manual,
)


def main():
    """Train a model to predict species evolution without using previous-year abundance as a feature."""

    parser = argparse.ArgumentParser(
        description="Train a model to predict species evolution (no lagged abundance feature)."
    )
    parser.add_argument(
        "--raw-data",
        default="raw_data/STOC_pressions_bio_assol_meteo_pesti_routes_lum_120625.csv",
        help="Path to raw CSV",
    )
    parser.add_argument(
        "--species",
        type=str,
        default=None,
        help="Filter for a specific species (optional)",
    )
    parser.add_argument(
        "--cluster",
        type=int,
        default=None,
        help="Filter for all species in a specific cluster (optional)",
    )
    parser.add_argument(
        "--all-clusters",
        action="store_true",
        help="Run for all clusters sequentially",
    )
    parser.add_argument(
        "--clusters-file",
        default="figures/species_clusters.csv",
        help="Path to species clusters CSV",
    )

    args = parser.parse_args()

    # Load cluster map first if needed
    cluster_map = None
    clusters_to_run = []

    if args.all_clusters:
        cluster_map = load_cluster_mapping(args.clusters_file)
        clusters_to_run = sorted(cluster_map.unique())
        print(f"Running for all clusters (no lag feature): {clusters_to_run}")
    elif args.cluster is not None:
        cluster_map = load_cluster_mapping(args.clusters_file)
        clusters_to_run = [args.cluster]
    else:
        # Run for all data or specific species
        clusters_to_run = [None]

    print(f"Loading raw data from {args.raw_data}...")
    df_raw = pd.read_csv(args.raw_data, encoding="latin1")

    for cluster_id in clusters_to_run:
        print("\n" + "=" * 40)
        if cluster_id is not None:
            print(f"Processing Cluster {cluster_id} (no lag feature)")
        else:
            print("Processing Full Dataset / Single Species (no lag feature)")
        print("=" * 40)

        # 1. Filter columns (same as in species_evolution_model.main)
        colonnes_a_garder = [
            f"Temps_mean_mensu_grid_{i}" for i in range(1, 13)
        ] + [
            f"Precipitation_mean_mensu_grid_{i}" for i in range(1, 13)
        ] + [
            "Year",
            "abondance_capped",
            "ValueSPred_Urba",
            "ValueSPred_For",
            "ValueSPred_CC",
            "ValueSPred_M",
            "ValueSPred_ZH",
            "mean_concentration_air_grid_sum",
            "mean_tii_grid_sum",
            "mean_tii_scale_grid_sum",
            "mean_concentration_water_grid_sum",
            "all_pesticide_exposure_grid_sum",
            "SurfBio_cell",
            "nb_exp_cell",
            "Long_route",
            "nb_roads",
            "Lum_mean",
            "site",
            "species",
        ]

        existing_cols = [c for c in colonnes_a_garder if c in df_raw.columns]
        df = df_raw[existing_cols].replace([np.inf, -np.inf], np.nan)

        # Filter by species/cluster BEFORE merge to save memory/time
        if args.species:
            df = df[df["species"] == args.species]
        elif cluster_id is not None:
            species_in_cluster = cluster_map[cluster_map == cluster_id].index.tolist()
            df = df[df["species"].isin(species_in_cluster)]
            print(f"Species in Cluster {cluster_id}: {', '.join(species_in_cluster)}")

        # 2. Merge (Year N with Year N-1 features)
        df_year_plus_1 = df.copy()
        df_year_plus_1["Year"] = df_year_plus_1["Year"] + 1
        df_year_plus_1 = df_year_plus_1.rename(
            columns={col: f"{col}_year_plus_1" for col in df_year_plus_1.columns}
        )

        df_merged = pd.merge(
            df,
            df_year_plus_1,
            left_on=["Year", "site", "species"],
            right_on=["Year_year_plus_1", "site_year_plus_1", "species_year_plus_1"],
            how="inner",
        )

        df_merged = df_merged.rename(columns={"Year": "annee"})

        if "abondance_capped" not in df_merged.columns:
            print("Target not found, skipping.")
            continue

        y = df_merged["abondance_capped"]

        feature_cols = [c for c in df_merged.columns if c.endswith("_year_plus_1")]
        cols_to_keep = feature_cols + ["annee"]
        if "species" in df_merged.columns:
            cols_to_keep.append("species")

        X = df_merged[cols_to_keep].copy()

        # *** KEY CHANGE ***
        # Remove previous-year abundance from features
        if "abondance_capped_year_plus_1" in X.columns:
            X = X.drop(columns=["abondance_capped_year_plus_1"])

        # 3. Prepare XY
        X, y = prepare_xy(X, y)

        if len(X) < 50:
            print("Not enough data for this cluster/species.")
            continue

        # Log-transform target
        y_log = np.log1p(y)

        # Random split by Year (80% Train, 20% Val)
        all_years = X["annee"].unique()
        n_years = len(all_years)
        n_val = int(np.ceil(n_years * 0.2))  # Ensure at least some years in val
        
        # Consistent random shuffle
        rng = np.random.RandomState(42)
        val_years_rand = rng.choice(all_years, size=n_val, replace=False)
        
        print(f"Random split enabled. Validation years: {sorted(val_years_rand)}")
        
        mask_val = X["annee"].isin(val_years_rand)
        
        X_train = X[~mask_val].copy()
        y_train = y_log[~mask_val].copy()
        
        X_val_int = X[mask_val].copy()
        y_val_int = y_log[mask_val].copy()
        
        # For this setup, we treat the 'Validation' set as the 'Test' set 
        # to satisfy the request (80/20 split).
        X_test_final = X_val_int
        y_test_final = y_val_int

        model = build_pipeline(X_train)

        print("Training XGBoost model (no lag feature)...")
        prep = model.named_steps["prep"]
        X_train_proc = prep.fit_transform(X_train)
        X_val_int_proc = prep.transform(X_val_int)
        X_test_final_proc = prep.transform(X_test_final)

        xgb_model = model.named_steps["model"]
        xgb_model.fit(
            X_train_proc,
            y_train,
            eval_set=[(X_val_int_proc, y_val_int)],
            verbose=False,
        )

        print("Evaluating on internal Validation Set...")
        _ = eval_manual(xgb_model, X_val_int_proc, y_val_int, "Validation (interne)")

        print("Evaluating on final Test Set (last years)...")
        _ = eval_manual(xgb_model, X_test_final_proc, y_test_final, "Test final (dernieres annees)")

        # Compute predictions for ALL years (Train + Val + Test) and aggregate by year
        X_all_proc = prep.transform(X)
        y_pred_all = eval_manual(xgb_model, X_all_proc, y_log, "Full Set")

        df_all_plot = X.copy()
        df_all_plot["True"] = np.expm1(y_log)
        df_all_plot["Predicted"] = y_pred_all
        pred_yearly_full = df_all_plot.groupby("annee").agg(
            {
                "True": lambda v: geometric_mean_abundance(v.values),
                "Predicted": lambda v: geometric_mean_abundance(v.values),
            }
        )

        # Ne conserver pour l'affichage que les 3 dernières années (jeu de test final)
        test_years = sorted(X_test_final["annee"].unique())
        pred_yearly = pred_yearly_full.loc[pred_yearly_full.index.isin(test_years)]
        val_years = test_years


if __name__ == "__main__":
    main()
