import argparse

import numpy as np
import pandas as pd

from species_evolution_model import (
    prepare_xy,
    time_based_split,
    build_pipeline,
    load_cluster_mapping,
    eval_manual,
    geometric_mean_abundance,
)


def _detect_target_column(df: pd.DataFrame) -> str:
    for candidate in [
        "abondance_capped",
        "abondance",
        "abondance_mean",
        "abondance_sum",
    ]:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "No target column found. Expected one of: abondance_capped, abondance, abondance_mean, abondance_sum"
    )


def _detect_merge_keys(df: pd.DataFrame) -> list[str]:
    # We always need Year and species to align (t) with (t-1)
    keys = []
    for k in ["Year", "species"]:
        if k in df.columns:
            keys.append(k)
    if "Year" not in keys or "species" not in keys:
        raise ValueError("Input CSV must contain at least 'Year' and 'species' columns.")

    # Add optional keys when present to avoid mixing rows (monthly data, spatial ids)
    for k in ["Month", "site", "id_cell"]:
        if k in df.columns:
            keys.append(k)

    return keys


def prepare_xy_auto(df_raw: pd.DataFrame, target_col: str | None = None):
    """Builds a lag-1 dataset from ANY STOC-style CSV.

    It aligns target at year t with features at year t-1 by shifting Year in the feature table.

    If Month exists, it is included in the merge keys so that January aligns with January (t-1).
    """
    df = df_raw.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    if target_col is None:
        target_col = _detect_target_column(df)

    merge_keys = _detect_merge_keys(df)

    # Build lagged feature table (t-1 values) re-indexed to year t
    lagged = df.copy()
    lagged["Year"] = lagged["Year"] + 1

    rename_map = {}
    for c in lagged.columns:
        if c in merge_keys:
            continue
        rename_map[c] = f"{c}_lag1"
    lagged = lagged.rename(columns=rename_map)

    merged = pd.merge(df, lagged, on=merge_keys, how="inner")

    # Canonical year name used by time_based_split
    merged = merged.rename(columns={"Year": "annee"})

    # y(t)
    y = merged[target_col]

    # Features = all lagged columns + identifiers (species/site) for grouping
    feature_cols = [c for c in merged.columns if c.endswith("_lag1")]

    # Keep identifiers in X so the pipeline can use them if desired
    # (species/site/id_cell are in merged because they are merge keys)
    extra_cols = [c for c in ["annee", "species", "site", "id_cell", "Month"] if c in merged.columns]

    X = merged[feature_cols + extra_cols].copy()
    X, y = prepare_xy(X, y)

    return X, y, target_col, merge_keys


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train the same XGBoost evolution model on a CSV with auto-detected columns. "
            "Works with the heavy STOC_pressions_bio_assol_meteo_pesti_140525.csv schema."
        )
    )
    parser.add_argument(
        "--raw-data",
        default="raw_data/STOC_pressions_bio_assol_meteo_pesti_140525.csv",
        help="Path to raw CSV",
    )
    parser.add_argument(
        "--target-col",
        default=None,
        help="Target column name (default: auto-detect)",
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
        "--clusters-file",
        default="figures/species_clusters.csv",
        help="Path to species clusters CSV",
    )

    args = parser.parse_args()

    print(f"Loading raw data from {args.raw_data}...")
    df_raw = pd.read_csv(args.raw_data, encoding="latin1")

    cluster_map = None
    selected_species_name = "All Species"

    if args.cluster is not None:
        cluster_map = load_cluster_mapping(args.clusters_file)
        species_in_cluster = cluster_map[cluster_map == args.cluster].index.tolist()
        df_raw = df_raw[df_raw["species"].isin(species_in_cluster)]
        selected_species_name = f"Cluster {args.cluster}"
        print(f"Species in Cluster {args.cluster}: {', '.join(species_in_cluster)}")
    elif args.species is not None:
        df_raw = df_raw[df_raw["species"] == args.species]
        selected_species_name = args.species

    print("Preparing lag-1 dataset with auto-columns...")
    X, y, target_col, merge_keys = prepare_xy_auto(df_raw, target_col=args.target_col)
    print(f"Target: {target_col}")
    print(f"Merge keys: {merge_keys}")
    print(f"Prepared dataset: X={X.shape}, y={y.shape}")

    if len(X) < 100:
        raise ValueError("Not enough rows after lag merge/filter to train a model.")

    # Train on log1p scale (same approach as species_evolution_model.py)
    y_log = np.log1p(y)

    # Same time split behavior
    X_train, y_train, X_val_int, y_val_int, X_test_final, y_test_final = time_based_split(
        X,
        y_log,
        train_frac=0.8,
        test_frac=0.2,
        val_last_n_years=3,
    )

    model = build_pipeline(X_train)

    print("Training XGBoost model...")
    prep = model.named_steps["prep"]
    X_train_proc = prep.fit_transform(X_train)
    X_val_proc = prep.transform(X_val_int)
    X_test_proc = prep.transform(X_test_final)

    xgb_model = model.named_steps["model"]
    xgb_model.fit(
        X_train_proc,
        y_train,
        eval_set=[(X_val_proc, y_val_int)],
        verbose=False,
    )

    print("Evaluating...")
    _ = eval_manual(xgb_model, X_val_proc, y_val_int, "Val (interne)")
    _ = eval_manual(xgb_model, X_test_proc, y_test_final, "Test final (3 last years)")

    # Optional: show basic yearly trend for the filtered scope
    # (works only if annee is present)
    try:
        X_all_proc = prep.transform(X)
        y_pred_all = np.expm1(xgb_model.predict(X_all_proc))
        y_true_all = np.expm1(y_log)
        df_plot = X[["annee"]].copy()
        df_plot["True"] = y_true_all
        df_plot["Pred"] = y_pred_all
        yearly = df_plot.groupby("annee").agg(
            {
                "True": ("True", lambda v: geometric_mean_abundance(v.values)),
                "Pred": ("Pred", lambda v: geometric_mean_abundance(v.values)),
            }
        )
        print("\nYearly geometric means (first/last 3 years):")
        print(pd.concat([yearly.head(3), yearly.tail(3)]))
    except Exception:
        pass

    print(f"Done. Scope: {selected_species_name}")


if __name__ == "__main__":
    main()
