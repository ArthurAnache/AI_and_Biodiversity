# standard_data_returns_species_site.py
# Explanations in French; code comments in English.

import os
import numpy as np
import pandas as pd

INPUT_DIR = "raw_data"
OUTPUT_DIR = "raw_data_transformed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Columns to keep for returns (your list) ---
colonnes_a_garder = (
    [f"Temps_mean_mensu_grid_{i}" for i in range(1, 13)]
    + [f"Precipitation_mean_mensu_grid_{i}" for i in range(1, 13)]
    + [f"Temps_max_mensu_grid_{i}" for i in range(1, 13)]
    + [f"Temps_min_mensu_grid_{i}" for i in range(1, 13)]
    + [
        "Year", "abondance_capped", "ValueSPred_Urba", "ValueSPred_For",
        "ValueSPred_CC", "ValueSPred_M", "ValueSPred_ZH",
        "mean_concentration_air_grid_max", "mean_tii_grid_max",
        "mean_tii_scale_grid_max", "mean_concentration_water_grid_max",
        "all_pesticide_exposure_grid_max", "SurfBio_cell", "nb_exp_cell",
        "Long_route", "nb_roads", "Lum_mean",
    ]
)

# --- Fixed identifiers for grouping ---
ID_COLS = ["species", "site"]

RETURNS_SUFFIX = "_rendements"

# --- Fast-save options ---
USE_GZIP = False          # True -> write .csv.gz; False -> plain .csv
CHUNKSIZE = 200_000      # rows per chunk when writing
FLOAT_FORMAT = "%.6g"    # output precision for floats

def detect_sep(path: str) -> str:
    """Detect separator among ',', ';', or '\\t' from the first non-empty line."""
    with open(path, "rb") as f:
        head = f.read(4096).decode("utf-8", errors="ignore")
    line = next((ln for ln in head.splitlines() if ln.strip()), "")
    counts = {",": line.count(","), ";": line.count(";"), "\t": line.count("\t")}
    if counts and max(counts.values()) > 0:
        return max(counts, key=counts.get)
    return ","  # sensible default

def read_csv_robust(path: str) -> tuple[pd.DataFrame, str, str]:
    """Read CSV with detected separator; try multiple encodings; disable low-memory dtype guessing."""
    sep = detect_sep(path)
    for enc in ("utf-8", "latin1", "ISO-8859-1"):
        try:
            df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
            return df, sep, enc
        except UnicodeDecodeError:
            continue
    # last attempt with python engine
    df = pd.read_csv(path, sep=sep, engine="python", low_memory=False)
    return df, sep, "unknown"

def to_numeric_series(s: pd.Series) -> pd.Series:
    """Clean common text-number formats (spaces, comma decimal, thousands) -> float."""
    if s.dtype.kind in "biufc":
        return s
    ss = s.astype(str)
    ss = ss.replace({
        "": np.nan, "nan": np.nan, "NaN": np.nan, "NA": np.nan, "N/A": np.nan, "-": np.nan
    })
    ss = ss.str.replace(r"\s+", "", regex=True)   # remove spaces (incl. nbsp)
    ss = ss.str.replace("'", "", regex=False)     # thousands apostrophes
    ss = ss.str.replace(",", ".", regex=False)    # decimal comma -> dot
    return pd.to_numeric(ss, errors="coerce")

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(".csv"):
        continue

    in_path = os.path.join(INPUT_DIR, filename)
    df, sep, enc = read_csv_robust(in_path)

    # Year column: prefer 'Year' if present; else assume first column is the year
    year_col = "Year" if "Year" in df.columns else df.columns[0]
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")

    # Ensure both ID columns exist; if not, skip file
    missing_ids = [c for c in ID_COLS if c not in df.columns]
    if missing_ids:
        print(f"[WARN] {filename}: missing ID columns {missing_ids} -> skipping.")
        continue

    # Keep only requested value columns present (excluding the year itself)
    value_cols = [c for c in colonnes_a_garder if c in df.columns and c != year_col]

    # Build working frame: IDs + Year + selected values
    work_cols = ID_COLS + [year_col] + value_cols
    w = df[work_cols].copy()

    # Clean numeric value columns
    for col in value_cols:
        w[col] = to_numeric_series(w[col])

    # Sort within each (species, site) by year
    w = w.sort_values(ID_COLS + [year_col])

    # Group by (species, site) and compute YoY returns on each selected value column
    rets = w.groupby(ID_COLS, dropna=False)[value_cols].pct_change(fill_method=None)
    rets = rets.replace([np.inf, -np.inf], np.nan)
    rets.columns = [f"{c}{RETURNS_SUFFIX}" for c in rets.columns]

    # Output = IDs + Year + returns; drop first year of each group (all NaN returns)
    out = pd.concat([w[ID_COLS + [year_col]], rets], axis=1)
    out = out[~rets.isna().all(axis=1)].reset_index(drop=True)

    # Smaller output: cast IDs to category (faster/lighter to_write)
    for c in ID_COLS:
        out[c] = out[c].astype("category")

    # Save
    out_name = f"{os.path.splitext(filename)[0]}_transformed.csv"
    if USE_GZIP:
        out_name += ".gz"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    out.to_csv(
        out_path,
        sep=sep,                 # preserve original separator
        index=False,
        float_format=FLOAT_FORMAT,
        chunksize=CHUNKSIZE,
        compression="infer" if USE_GZIP else None,
    )

    print(f"{filename} -> {out_name} | enc={enc} | values={len(value_cols)} | sep='{sep}'")

print("Terminé : rendements année→année par (species, site) écrits dans 'raw_data_transformed/'.")
