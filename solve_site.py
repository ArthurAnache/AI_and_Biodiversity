from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _monthly_cols(prefix: str) -> List[str]:
    """Return [f"{prefix}_{1}", ..., f"{prefix}_{12}"]"""
    return [f"{prefix}_{i}" for i in range(1, 13)]


def _safe_row_mean(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """Row-wise mean ignoring NaNs; returns NaN if all NaN."""
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series([np.nan] * len(df), index=df.index)
    return df[existing].mean(axis=1, skipna=True)


def _update_species_name_map(
    chunk: pd.DataFrame,
    species_to_names: Dict[str, Tuple[str | None, str | None]],
) -> None:
    # species -> (scientific_name, french_name)
    if "species" not in chunk.columns:
        return
    sci = chunk["scientific_name"] if "scientific_name" in chunk.columns else None
    fr = chunk["french_name"] if "french_name" in chunk.columns else None

    for sp, sub in chunk.groupby("species", sort=False):
        if sp in species_to_names and all(v is not None for v in species_to_names[sp]):
            continue

        sci_name = None
        fr_name = None
        if sci is not None:
            s = sub["scientific_name"].dropna()
            if len(s):
                sci_name = str(s.iloc[0])
        if fr is not None:
            s = sub["french_name"].dropna()
            if len(s):
                fr_name = str(s.iloc[0])

        old = species_to_names.get(sp, (None, None))
        species_to_names[sp] = (old[0] or sci_name, old[1] or fr_name)


def build_processed_no_site(
    input_csv: Path,
    output_csv: Path,
    abundance_col: str = "abondance_capped",
    chunksize: int = 250_000,
    encoding: str = "latin1",
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    keys = ["Year", "species"]

    # Accumulators indexed by (Year, species)
    sum_logs = None  # pd.Series
    cnt_ab = None  # pd.Series

    sum_num = None  # pd.DataFrame
    cnt_num = None  # pd.DataFrame

    species_to_names: Dict[str, Tuple[str | None, str | None]] = {}

    # Columns we know we want to turn into annual means per row first
    meteo_specs = {
        "Temps_mean_mensu_grid": "Temps_mean_mensu_grid_mean",
        "Temps_max_mensu_grid": "Temps_max_mensu_grid_mean",
        "Temps_min_mensu_grid": "Temps_min_mensu_grid_mean",
        "TAMPLI_mean_mensu_grid": "TAMPLI_mean_mensu_grid_mean",
        "Precipitation_mean_mensu_grid": "Precipitation_mean_mensu_grid_mean",
    }

    # Read in chunks (heavy file)
    reader = pd.read_csv(
        input_csv,
        encoding=encoding,
        low_memory=False,
        chunksize=chunksize,
    )

    total_rows = 0
    for i, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)
        print(f"[solve_site] Chunk {i}: {len(chunk):,} rows (total read: {total_rows:,})")

        # Basic schema checks
        missing_keys = [k for k in keys if k not in chunk.columns]
        if missing_keys:
            raise ValueError(f"Missing required columns {missing_keys} in input CSV.")

        if abundance_col not in chunk.columns:
            raise ValueError(f"Abundance column '{abundance_col}' not found in input CSV.")

        # Keep names mapping (species -> names)
        _update_species_name_map(chunk, species_to_names)

        # Meteorology can come in two formats:
        # - wide: columns like Temps_mean_mensu_grid_1..12
        # - long: columns like Temps_mean_mensu_grid with a Month column
        has_long_month = "Month" in chunk.columns and any(base in chunk.columns for base in meteo_specs.keys())
        has_wide_months = any(any(c in chunk.columns for c in _monthly_cols(base)) for base in meteo_specs.keys())

        if has_wide_months:
            # Build annual meteo means per row (collapse monthly 1..12 into one value per row)
            for base, outcol in meteo_specs.items():
                chunk[outcol] = _safe_row_mean(chunk, _monthly_cols(base))

            # Drop monthly columns (reduce memory + avoid averaging them again later)
            monthly_to_drop = []
            for base in meteo_specs.keys():
                monthly_to_drop.extend([c for c in _monthly_cols(base) if c in chunk.columns])
            if monthly_to_drop:
                chunk = chunk.drop(columns=monthly_to_drop)
        elif has_long_month:
            # Each row is already a month; we will average over months in the groupby.
            # Rename meteo columns to the same "*_mean" convention as the wide format.
            rename_map = {base: outcol for base, outcol in meteo_specs.items() if base in chunk.columns}
            if rename_map:
                chunk = chunk.rename(columns=rename_map)

        # Decide numeric feature columns to aggregate with means
        # Exclude keys and obvious non-features
        exclude_cols = {
            "site",
            "id_cell",  # spatial id; after collapsing sites, not meaningful
            "Month",
            "passage",
            "scientific_name",
            "french_name",
            "abondance",
            "abondance_max_capped",
            "abondance_mean_capped",
            "abondance_sum_capped",
            abundance_col,  # aggregated separately as geometric mean
        }

        # Numeric columns left will be averaged (statu quo if not changed later)
        numeric_cols = [
            c
            for c in chunk.columns
            if c not in exclude_cols
            and c not in keys
            and pd.api.types.is_numeric_dtype(chunk[c])
        ]

        gb = chunk.groupby(keys, sort=False)

        # --- abundance geometric mean components: sum(log1p(x)) and count ---
        ab = chunk[abundance_col].astype("float64", errors="ignore")
        log1p_ab = np.log1p(ab.clip(lower=0))  # abundance should not be negative
        chunk["_log1p_ab"] = log1p_ab

        sum_log_chunk = gb["_log1p_ab"].sum(min_count=1)
        cnt_ab_chunk = gb[abundance_col].count()

        chunk = chunk.drop(columns=["_log1p_ab"])

        # --- numeric features: sum and count to compute mean later ---
        if numeric_cols:
            sum_num_chunk = gb[numeric_cols].sum(min_count=1)
            cnt_num_chunk = gb[numeric_cols].count()
        else:
            sum_num_chunk = None
            cnt_num_chunk = None

        # Accumulate
        if sum_logs is None:
            sum_logs = sum_log_chunk
            cnt_ab = cnt_ab_chunk
        else:
            sum_logs = sum_logs.add(sum_log_chunk, fill_value=0)
            cnt_ab = cnt_ab.add(cnt_ab_chunk, fill_value=0)

        if numeric_cols:
            if sum_num is None:
                sum_num = sum_num_chunk
                cnt_num = cnt_num_chunk
            else:
                # Align columns / index automatically
                sum_num = sum_num.add(sum_num_chunk, fill_value=0)
                cnt_num = cnt_num.add(cnt_num_chunk, fill_value=0)

    if sum_logs is None or cnt_ab is None:
        raise RuntimeError("No data processed (empty file?)")

    # Finalize abundance geometric mean:
    # GM = exp( sum(log1p(x))/n ) - 1
    mean_log = sum_logs / cnt_ab.replace(0, np.nan)
    ab_geo = np.expm1(mean_log).rename(f"{abundance_col}_geo_mean")

    # Finalize numeric means
    if sum_num is not None and cnt_num is not None:
        num_means = sum_num.divide(cnt_num.replace(0, np.nan))
    else:
        num_means = pd.DataFrame(index=ab_geo.index)

    out = pd.concat([ab_geo, num_means], axis=1).reset_index()

    # Add names back (optional)
    if species_to_names:
        sci = []
        fr = []
        for sp in out["species"]:
            s, f = species_to_names.get(sp, (None, None))
            sci.append(s)
            fr.append(f)
        out.insert(out.columns.get_loc("species") + 1, "scientific_name", sci)
        out.insert(out.columns.get_loc("scientific_name") + 1, "french_name", fr)

    # Save
    out.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[solve_site] Wrote: {output_csv} ({len(out):,} rows)")


def build_lagged_dataset(
    input_csv: Path,
    output_csv: Path,
    target_col: str = "abondance_capped_geo_mean",
    n_lags: int = 2,
    include_current_pressures: bool = False,
) -> None:
    """Create a supervised dataset with lagged features per species.

    Input is expected to have 1 row per (Year, species), e.g. proc_data/stoc_no_site_annual.csv.

    Output contains:
    - Target at time t: target_col
    - Lagged features: <col>_lag1, <col>_lag2 (for all numeric cols, including target)
    - Keys: Year, species (+ names if present)

    By default, current-year pressures are NOT included to avoid leaking future info when you
    later simulate (you typically only know/define pressures via scenario).
    """
    if n_lags not in (1, 2):
        raise ValueError("n_lags must be 1 or 2")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, encoding="utf-8")
    required = ["Year", "species", target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in processed file: {missing}")

    # Sort for shifting
    df = df.sort_values(["species", "Year"]).reset_index(drop=True)

    name_cols = [c for c in ["scientific_name", "french_name"] if c in df.columns]

    # Features we can lag: all numeric columns (including target) except Year
    numeric_cols = [
        c
        for c in df.columns
        if c not in ("Year",)
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    keep_cols = ["Year", "species"] + name_cols
    if include_current_pressures:
        # keep current numeric cols (except target), as additional contemporaneous features
        keep_cols += [c for c in numeric_cols if c != target_col]

    out = df[keep_cols + [target_col]].copy()

    g = df.groupby("species", sort=False)
    for lag in range(1, n_lags + 1):
        for c in numeric_cols:
            out[f"{c}_lag{lag}"] = g[c].shift(lag)

    # Remove rows without full history
    needed = [f"{target_col}_lag{lag}" for lag in range(1, n_lags + 1)]
    out = out.dropna(subset=needed).reset_index(drop=True)

    out.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[solve_site] Wrote lagged dataset: {output_csv} ({len(out):,} rows)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collapse site dimension, average monthly pressures, and build lags.")
    p.add_argument(
        "--input",
        type=str,
        default=r"raw_data\STOC_pressions_bio_assol_meteo_pesti_routes_lum_120625.csv",
        help="Input raw CSV path (relative to AI_and_Biodiversity/).",
    )
    p.add_argument(
        "--output",
        type=str,
        default=r"proc_data\stoc_no_site_annual.csv",
        help="Output processed CSV path (relative to AI_and_Biodiversity/).",
    )
    p.add_argument(
        "--build-no-site-lags",
        action="store_true",
        help=(
            "One-shot: build the no-site annual table from raw input and then build a lagged dataset "
            "(lag1/lag2) from it. Writes outputs into proc_data/."
        ),
    )
    p.add_argument(
        "--make-lags",
        action="store_true",
        help="Instead of aggregating raw->annual, build a lagged dataset from an existing annual CSV.",
    )
    p.add_argument(
        "--lags-output",
        type=str,
        default=r"proc_data\stoc_no_site_annual.csv",
        help="Output lagged CSV path (relative to AI_and_Biodiversity/).",
    )
    p.add_argument(
        "--target-col",
        type=str,
        default="abondance_capped_geo_mean",
        help="Target column in the annual CSV (default is produced by this script).")
    p.add_argument(
        "--n-lags",
        type=int,
        choices=[1, 2],
        default=2,
        help="Number of lags to generate (1 or 2).",
    )
    p.add_argument(
        "--include-current-pressures",
        action="store_true",
        help="Also keep current-year numeric pressures as features (in addition to lags).",
    )
    p.add_argument(
        "--abundance-col",
        type=str,
        default="abondance_capped",
        help="Which abundance column to aggregate as geometric mean.",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="CSV chunksize for heavy files.",
    )
    p.add_argument(
        "--encoding",
        type=str,
        default="latin1",
        help="CSV encoding (often latin1 for these STOC exports).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(__file__).resolve().parent

    input_csv = (base / args.input).resolve()
    output_csv = (base / args.output).resolve()

    # Safety: avoid conflicting modes
    if args.build_no_site_lags and args.make_lags:
        raise ValueError("Use either --build-no-site-lags or --make-lags, not both.")

    if args.build_no_site_lags:
        # 1) Raw -> annual no-site
        build_processed_no_site(
            input_csv=input_csv,
            output_csv=output_csv,
            abundance_col=args.abundance_col,
            chunksize=args.chunksize,
            encoding=args.encoding,
        )

        # 2) Annual -> lagged (explicit _lag1/_lag2 columns)
        lags_output = (base / args.lags_output).resolve()
        build_lagged_dataset(
            input_csv=output_csv,
            output_csv=lags_output,
            target_col=args.target_col,
            n_lags=args.n_lags,
            include_current_pressures=args.include_current_pressures,
        )
        return

    if args.make_lags:
        lags_output = (base / args.lags_output).resolve()
        build_lagged_dataset(
            input_csv=input_csv,
            output_csv=lags_output,
            target_col=args.target_col,
            n_lags=args.n_lags,
            include_current_pressures=args.include_current_pressures,
        )
    else:
        build_processed_no_site(
            input_csv=input_csv,
            output_csv=output_csv,
            abundance_col=args.abundance_col,
            chunksize=args.chunksize,
            encoding=args.encoding,
        )


if __name__ == "__main__":
    main()