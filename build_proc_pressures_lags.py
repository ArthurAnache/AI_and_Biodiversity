import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _annual_mean_from_monthly_wide(df: pd.DataFrame, base: str) -> pd.Series:
    cols = [f"{base}_{i}" for i in range(1, 13) if f"{base}_{i}" in df.columns]
    if not cols:
        return pd.Series([np.nan] * len(df), index=df.index)
    return df[cols].mean(axis=1, skipna=True)


def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Build a processed table from STOC_pressions_bio_assol_meteo_pesti_routes_lum_120625.csv "
            "with annual-mean monthly weather variables and explicit lag1/lag2 columns."
        )
    )
    p.add_argument(
        "--input",
        default=r"raw_data\STOC_pressions_bio_assol_meteo_pesti_routes_lum_120625.csv",
        help="Input raw CSV (relative to project root).",
    )
    p.add_argument(
        "--output",
        default=r"proc_data\data_proc_pressures_lag2.csv",
        help="Output CSV path (relative to project root).",
    )
    p.add_argument(
        "--encoding",
        default="latin1",
        help="CSV encoding (default latin1 for STOC exports).",
    )
    args = p.parse_args()

    base_dir = Path(__file__).resolve().parent
    input_path = (base_dir / args.input).resolve()
    output_path = (base_dir / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, encoding=args.encoding, low_memory=False)

    # --- 1) Annual means for monthly variables ---
    df["Temps_mean_mensu_grid_mean"] = _annual_mean_from_monthly_wide(df, "Temps_mean_mensu_grid")
    df["Precipitation_mean_mensu_grid_mean"] = _annual_mean_from_monthly_wide(df, "Precipitation_mean_mensu_grid")
    df["TAMPLI_mean_mensu_grid_mean"] = _annual_mean_from_monthly_wide(df, "TAMPLI_mean_mensu_grid")

    # --- 2) Select requested columns (no aggregation) ---
    # Keep Year for lags; you can drop it later if you want.
    required_like = {
        "Year": ["Year"],
        "species": ["species"],
        "abondance_capped": ["abondance_capped"],
        "Temps_mean_mensu_grid_mean": ["Temps_mean_mensu_grid_mean"],
        "Precipitation_mean_mensu_grid_mean": ["Precipitation_mean_mensu_grid_mean"],
        "TAMPLI_mean_mensu_grid_mean": ["TAMPLI_mean_mensu_grid_mean"],
        "Area_grid_tot_Urba": ["Area_grid_tot_Urba"],
        "ValueSPred_Urba": ["ValueSPred_Urba", "ValuesPred_Urba"],
        "Area_grid_tot_For": ["Area_grid_tot_For"],
        "ValueSPred_For": ["ValueSPred_For", "ValuesPred_For"],
        "Area_grid_tot_CC": ["Area_grid_tot_CC"],
        "ValueSPred_CC": ["ValueSPred_CC", "ValuesPred_CC"],
        "Area_grid_tot_M": ["Area_grid_tot_M"],
        "ValueSPred_M": ["ValueSPred_M", "ValuesPred_M"],
        "Area_grid_tot_ZH": ["Area_grid_tot_ZH"],
        "ValueSPred_ZH": ["ValueSPred_ZH", "ValuesPred_ZH"],
        "mean_concentration_air_grid_sum": ["mean_concentration_air_grid_sum"],
        "mean_tii_grid_sum": ["mean_tii_grid_sum"],
        "mean_tii_scale_grid_sum": ["mean_tii_scale_grid_sum"],
        "mean_concentration_water_grid_sum": ["mean_concentration_water_grid_sum"],
        "all_pesticide_exposure_grid_sum": ["all_pesticide_exposure_grid_sum"],
        "SurfBio_cell": ["SurfBio_cell"],
        "nb_exp_cell": ["nb_exp_cell"],
        "Long_route": ["Long_route"],
        "nb_roads": ["nb_roads"],
        "Lum_mean": ["Lum_mean"],
    }

    rename_map: dict[str, str] = {}
    keep_cols: list[str] = []
    missing: list[str] = []

    for out_name, candidates in required_like.items():
        found = _pick_first_existing(df, candidates)
        if found is None:
            missing.append(out_name)
            continue
        keep_cols.append(found)
        if found != out_name:
            rename_map[found] = out_name

    if missing:
        print("[build_proc_pressures_lags] Warning: missing columns:")
        for m in missing:
            print(f"  - {m}")

    proc = df[keep_cols].copy().rename(columns=rename_map)

    # --- 3) Build lag1/lag2 for all numeric pressures (and the annual means), grouped by species and site if available ---
    # You asked to ignore sites; but for clean lags we need to avoid mixing sites.
    # So: compute lags within (species, site) when possible, then drop site if you don't want it.
    group_keys = ["species"]
    if "site" in df.columns:
        proc["site"] = df["site"].values
        group_keys = ["species", "site"]

    proc = proc.sort_values(group_keys + ["Year"]).reset_index(drop=True)

    numeric_cols = [c for c in proc.columns if c not in ("species", "site") and pd.api.types.is_numeric_dtype(proc[c])]

    g = proc.groupby(group_keys, sort=False)
    for lag in (1, 2):
        for c in numeric_cols:
            proc[f"{c}_lag{lag}"] = g[c].shift(lag)

    # Drop site column from final output (since you said "oublions les sites")
    if "site" in proc.columns:
        proc = proc.drop(columns=["site"])

    proc.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[build_proc_pressures_lags] Wrote {output_path} ({len(proc):,} rows)")


if __name__ == "__main__":
    main()
