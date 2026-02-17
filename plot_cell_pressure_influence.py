import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import Ridge


def read_csv_robust(path: Path, *, encoding: str | None, encoding_errors: str) -> pd.DataFrame:
    """Read CSV with a small encoding fallback.

    Some raw exports contain Windows-1252 characters (e.g. é) that fail under utf-8.
    """
    if encoding is not None:
        return pd.read_csv(path, low_memory=False, encoding=encoding, encoding_errors=encoding_errors)

    # Try common encodings seen in this project.
    candidates = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err: Exception | None = None
    for enc in candidates:
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc, encoding_errors=encoding_errors)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise last_err if last_err is not None else UnicodeDecodeError("utf-8", b"", 0, 1, "Unknown decoding error")


def detect_id_col(df: pd.DataFrame) -> str:
    for c in ["id_cell", "cell_id", "idcell"]:
        if c in df.columns:
            return c
    # fallback: anything containing id_cell
    candidates = [c for c in df.columns if "id_cell" in str(c).lower()]
    if candidates:
        return candidates[0]
    raise ValueError("Could not find an id_cell column. Pass --id-col explicitly.")


def default_pressure_columns(df: pd.DataFrame) -> list[str]:
    """Heuristic selection of pressure columns (non-meteo, non-abundance).

    Goal: keep interpretable anthropogenic/landcover/pollution pressures.
    """
    exclude_prefixes = (
        "Temps_",
        "Precipitation_",
        "TAMPLI_",
        "Month",
    )

    cols: list[str] = []
    for c in df.columns:
        s = str(c)
        sl = s.lower()
        if any(s.startswith(p) for p in exclude_prefixes):
            continue
        if "abondance" in sl:
            continue
        if s in {"Year", "annee", "site", "species", "scientific_name", "french_name", "passage"}:
            continue

        if (
            s.startswith("Area_grid_tot_")
            or s.startswith("ValueSPred_")
            or "mean_concentration" in sl
            or "tii" in sl
            or "pesticide" in sl
            or "SurfBio_cell" in s
            or s in {"nb_exp_cell", "Long_route", "nb_roads"}
            or s.startswith("Lum_")
            or s == "Lum_mean"
        ):
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(s)

    # keep stable ordering
    cols = sorted(set(cols), key=cols.index)
    return cols


def fit_ridge_per_cell(
    df: pd.DataFrame,
    id_col: str,
    target_col: str,
    feature_cols: list[str],
    min_samples: int,
    alpha: float,
    log_target: bool,
    max_cells: int | None,
) -> pd.DataFrame:
    col_to_idx = {c: i for i, c in enumerate(feature_cols)}

    # Count rows per cell and keep biggest first for stability
    counts = df[id_col].value_counts(dropna=False)
    cell_ids = counts.index.tolist()
    if max_cells is not None:
        cell_ids = cell_ids[:max_cells]

    rows = []
    for i, cell_id in enumerate(cell_ids, start=1):
        sub = df[df[id_col] == cell_id]
        if len(sub) < min_samples:
            continue

        X = sub[feature_cols]
        y = sub[target_col]

        mask = ~y.isna()
        X = X.loc[mask]
        y = y.loc[mask]
        if len(y) < min_samples:
            continue

        y_arr = y.to_numpy(dtype=float)
        if log_target:
            y_arr = np.log1p(np.maximum(y_arr, 0.0))

        # Some cells can have entire columns missing; fit on available features,
        # then map coefficients back onto the full feature list.
        non_all_missing = [c for c in feature_cols if not X[c].isna().all()]
        if not non_all_missing:
            continue

        X_fit = X[non_all_missing]

        pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", MaxAbsScaler()),
                ("model", Ridge(alpha=alpha)),
            ]
        )
        pipe.fit(X_fit, y_arr)

        coefs_fit = pipe.named_steps["model"].coef_
        full = np.full(len(feature_cols), np.nan, dtype=float)
        for j, col in enumerate(non_all_missing):
            full[col_to_idx[col]] = float(coefs_fit[j])

        rows.append([cell_id, len(y_arr), *full.tolist()])

        if i % 250 == 0:
            print(f"Processed {i}/{len(cell_ids)} cells...")

    coef_df = pd.DataFrame(rows, columns=[id_col, "n_samples", *feature_cols])
    return coef_df


def plot_heatmap(coef_df: pd.DataFrame, id_col: str, feature_cols: list[str], out_path: Path, sort_by: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if coef_df.empty:
        raise ValueError("No cells had enough samples to fit a model. Lower --min-samples or check your data.")

    if sort_by == "id":
        coef_df = coef_df.sort_values(by=id_col)
    elif sort_by == "n":
        coef_df = coef_df.sort_values(by="n_samples", ascending=False)

    X = coef_df[feature_cols].to_numpy(dtype=float).T  # features x cells

    # robust symmetric scaling
    vmax = float(np.nanpercentile(np.abs(X), 98)) if np.isfinite(np.nanmax(np.abs(X))) else 1.0
    if vmax <= 0:
        vmax = 1.0

    plt.figure(figsize=(max(12, 0.03 * X.shape[1]), max(6, 0.25 * X.shape[0])))
    plt.imshow(X, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    plt.colorbar(label="Ridge coefficient (red=positive, blue=negative)")

    plt.yticks(range(len(feature_cols)), feature_cols)

    # x ticks: too many -> thin ticks
    cell_labels = coef_df[id_col].astype(str).tolist()
    if len(cell_labels) <= 60:
        plt.xticks(range(len(cell_labels)), cell_labels, rotation=90, fontsize=7)
    else:
        plt.xticks([])
        plt.xlabel(f"{id_col} (n={len(cell_labels)} cells; labels hidden)")

    plt.title("Pressure influence by cell (Ridge per id_cell)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved heatmap to: {out_path}")


def plot_pressure_scatter(
    df: pd.DataFrame,
    coef_df: pd.DataFrame,
    id_col: str,
    feature_cols: list[str],
    out_path: Path,
    sort_by: str,
    top_features: int | None,
    mode: str,
    show_legend: bool,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if coef_df.empty:
        raise ValueError("No cells had enough samples to fit a model. Lower --min-samples or check your data.")

    if sort_by == "id":
        coef_df = coef_df.sort_values(by=id_col)
    elif sort_by == "n":
        coef_df = coef_df.sort_values(by="n_samples", ascending=False)

    cell_ids = coef_df[id_col].tolist()

    # Average pressure level per cell (y axis), sign of coefficient decides color.
    means = df.groupby(id_col, dropna=False)[feature_cols].mean(numeric_only=True)
    means = means.reindex(cell_ids)

    # Pick a manageable number of features for facet plots.
    plot_cols = feature_cols
    if top_features is not None and top_features > 0 and len(feature_cols) > top_features:
        # Rank by average absolute coefficient magnitude (more interpretable than variance alone).
        scores = coef_df[feature_cols].abs().mean(axis=0, skipna=True)
        plot_cols = scores.sort_values(ascending=False).head(top_features).index.tolist()

    # X axis: numeric if possible, otherwise category codes.
    try:
        x = pd.to_numeric(pd.Series(cell_ids), errors="raise").to_numpy()
        x_labels = None
    except Exception:
        x = np.arange(len(cell_ids))
        x_labels = [str(v) for v in cell_ids]

    if mode == "facets":
        n = len(plot_cols)
        nrows = n
        fig_h = max(6, 1.6 * nrows)
        fig_w = max(12, 0.03 * len(cell_ids))
        fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(fig_w, fig_h), sharex=True)
        if nrows == 1:
            axes = [axes]

        for ax, col in zip(axes, plot_cols):
            y = means[col].to_numpy(dtype=float)
            coef = coef_df[col].to_numpy(dtype=float)
            colors = np.where(np.isnan(coef), "0.7", np.where(coef >= 0, "red", "blue"))
            ax.scatter(x, y, s=10, c=colors, alpha=0.7, linewidths=0)
            ax.set_ylabel(col)
            ax.grid(True, axis="y", alpha=0.25)

        if x_labels is not None and len(x_labels) <= 60:
            axes[-1].set_xticks(x)
            axes[-1].set_xticklabels(x_labels, rotation=90, fontsize=7)
        else:
            axes[-1].set_xlabel(f"{id_col} (n={len(cell_ids)} cells)")

        fig.suptitle("Pressure level by cell; point color = influence sign (Ridge coef)")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        print(f"Saved pressure scatter facets to: {out_path}")
        return

    if mode != "single":
        raise ValueError("mode must be 'single' or 'facets'")

    # Single-axes plot: overlay all selected pressures on the same graph.
    fig_w = max(12, 0.03 * len(cell_ids))
    fig_h = 7
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for col in plot_cols:
        y = means[col].to_numpy(dtype=float)
        coef = coef_df[col].to_numpy(dtype=float)
        colors = np.where(np.isnan(coef), "0.7", np.where(coef >= 0, "red", "blue"))
        ax.scatter(x, y, s=8, c=colors, alpha=0.25, linewidths=0, label=col)

    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylabel("Pressure value (mean per cell)")

    if x_labels is not None and len(x_labels) <= 60:
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=90, fontsize=7)
    else:
        ax.set_xlabel(f"{id_col} (n={len(cell_ids)} cells)")

    ax.set_title("id_cell vs pressures (overlay); point color = influence sign")
    if show_legend and len(plot_cols) <= 15:
        ax.legend(loc="upper right", fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved single pressure overlay plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fit a Ridge model per id_cell to estimate the sign/magnitude of each pressure's influence, "
            "then plot a heatmap (x=id_cell, y=pressure, color=coef sign)."
        )
    )
    parser.add_argument("--data", default="sample_data/sample_stoc_data.csv", help="Input CSV with id_cell, pressures and target")
    parser.add_argument(
        "--encoding",
        default=None,
        help="CSV encoding (default: auto-try utf-8/utf-8-sig/cp1252/latin1)",
    )
    parser.add_argument(
        "--encoding-errors",
        default="strict",
        help="How to handle encoding errors (strict/replace/ignore). Default: strict",
    )
    parser.add_argument("--id-col", default=None, help="Column name for spatial cell id (default: auto-detect id_cell)")
    parser.add_argument("--target-col", default="abondance_capped", help="Target column to predict")
    parser.add_argument("--min-samples", type=int, default=30, help="Minimum rows per id_cell to fit a model")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization strength")
    parser.add_argument("--log-target", action="store_true", help="Fit on log1p(abundance) instead of raw")
    parser.add_argument("--max-cells", type=int, default=300, help="Max number of id_cells to plot (top by row count). Use 0 for all")
    parser.add_argument(
        "--features",
        default=None,
        help="Comma-separated list of pressure columns to use (default: auto-select pressures)",
    )
    parser.add_argument(
        "--include-year-plus-1",
        action="store_true",
        help="Include *_year_plus_1 versions of selected pressures when available",
    )
    parser.add_argument(
        "--plot",
        choices=["heatmap", "scatter", "both"],
        default="both",
        help="Which plot(s) to generate",
    )
    parser.add_argument("--sort-by", choices=["id", "n"], default="n", help="Sort cells by id or by sample count")
    parser.add_argument("--out", default="figures/cell_pressure_influence.png", help="Output heatmap path")
    parser.add_argument(
        "--out-scatter",
        default="figures/cell_pressure_by_id.png",
        help="Output scatter facets path (y=pressure mean per cell; color=coef sign)",
    )
    parser.add_argument(
        "--scatter-mode",
        choices=["single", "facets"],
        default="single",
        help="Scatter visualization style (single axis overlay vs multiple subplots)",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Disable legend on the single-axis scatter overlay",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=12,
        help="Max number of pressures to show in scatter facets (ranked by abs coef). 0 for all",
    )
    parser.add_argument("--out-csv", default="figures/cell_pressure_influence_coeffs.csv", help="Output coefficients table")

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    data_path = (base_dir / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)

    df = read_csv_robust(data_path, encoding=args.encoding, encoding_errors=args.encoding_errors)

    id_col = args.id_col or detect_id_col(df)
    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in data")

    if args.features:
        feature_cols = [c.strip() for c in args.features.split(",") if c.strip()]
    else:
        feature_cols = default_pressure_columns(df)

    if args.include_year_plus_1:
        extra = []
        for c in feature_cols:
            c1 = f"{c}_year_plus_1"
            if c1 in df.columns:
                extra.append(c1)
        feature_cols = feature_cols + extra

    # keep only existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        raise ValueError("No feature columns selected/found. Use --features or check your dataset columns.")

    max_cells = None if args.max_cells == 0 else int(args.max_cells)

    print(f"Data: {data_path}")
    print(f"id_col: {id_col}")
    print(f"target_col: {args.target_col} (log={args.log_target})")
    print(f"Selected {len(feature_cols)} pressure columns")

    coef_df = fit_ridge_per_cell(
        df=df,
        id_col=id_col,
        target_col=args.target_col,
        feature_cols=feature_cols,
        min_samples=args.min_samples,
        alpha=args.alpha,
        log_target=args.log_target,
        max_cells=max_cells,
    )

    out_csv = (base_dir / args.out_csv).resolve() if not Path(args.out_csv).is_absolute() else Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    coef_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved coefficients to: {out_csv}")

    out_img = (base_dir / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)

    if args.plot in {"heatmap", "both"}:
        plot_heatmap(coef_df, id_col=id_col, feature_cols=feature_cols, out_path=out_img, sort_by=args.sort_by)

    if args.plot in {"scatter", "both"}:
        out_scatter = (base_dir / args.out_scatter).resolve() if not Path(args.out_scatter).is_absolute() else Path(args.out_scatter)
        top_features = None if args.top_features == 0 else int(args.top_features)
        plot_pressure_scatter(
            df=df,
            coef_df=coef_df,
            id_col=id_col,
            feature_cols=feature_cols,
            out_path=out_scatter,
            sort_by=args.sort_by,
            top_features=top_features,
            mode=args.scatter_mode,
            show_legend=(not args.no_legend),
        )


if __name__ == "__main__":
    main()
