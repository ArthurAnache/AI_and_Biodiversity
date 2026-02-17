import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler


def read_csv_robust(path: Path, *, encoding: str | None, encoding_errors: str) -> pd.DataFrame:
    if encoding is not None:
        return pd.read_csv(path, low_memory=False, encoding=encoding, encoding_errors=encoding_errors)

    candidates = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err: Exception | None = None
    for enc in candidates:
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc, encoding_errors=encoding_errors)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise last_err if last_err is not None else UnicodeDecodeError("utf-8", b"", 0, 1, "Unknown decoding error")


def _minmax_0_1(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)


def plot_tsne_clusters(
    X_cluster: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    random_state: int = 42,
    perplexity: float = 30.0,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = int(X_cluster.shape[0])
    if n < 4:
        print("Not enough points for t-SNE visualization.")
        return

    px = float(perplexity)
    px_max = max(2.0, (n - 1) / 3.0)
    if px >= n:
        px = min(30.0, px_max)
    if px < 2.0:
        px = 2.0

    tsne = TSNE(
        n_components=2,
        perplexity=px,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
    )
    Z = tsne.fit_transform(X_cluster)

    plt.figure(figsize=(10, 7))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=18, alpha=0.85, linewidths=0)
    plt.title(f"t-SNE clusters (perplexity={px:.1f})")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    cb = plt.colorbar(sc)
    cb.set_label("cluster")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"t-SNE plot saved to '{out_path}'")


def find_optimal_clusters(X: np.ndarray, *, max_k: int, out_prefix: str):
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    inertias: list[float] = []
    silhouettes: list[float] = []
    dbis: list[float] = []

    K = range(2, max_k + 1)
    print(f"Searching for optimal k (2 to {max_k})...")

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(float(kmeans.inertia_))
        silhouettes.append(float(silhouette_score(X, kmeans.labels_)))
        dbis.append(float(davies_bouldin_score(X, kmeans.labels_)))
        print(f"k={k}: Inertia={inertias[-1]:.2f}, Silhouette={silhouettes[-1]:.4f}, DBI={dbis[-1]:.4f}")

    # Elbow + silhouette
    _, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia (Elbow)", color="tab:red")
    ax1.plot(list(K), inertias, marker="o", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Silhouette Score", color="tab:blue")
    ax2.plot(list(K), silhouettes, marker="s", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    plt.title("Optimal k Analysis: Elbow Method & Silhouette Score")
    plt.grid(True, alpha=0.3)
    out1 = out_dir / f"{out_prefix}_optimal_k_analysis.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=200)
    print(f"Optimal k plot saved to {out1}")

    # DBI (lower is better)
    plt.figure(figsize=(10, 6))
    plt.plot(list(K), dbis, marker="^", color="tab:green")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Davies–Bouldin Index (lower is better)")
    plt.title("Optimal k Analysis: Davies–Bouldin Index")
    plt.grid(True, alpha=0.3)
    out2 = out_dir / f"{out_prefix}_optimal_k_dbi.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=200)
    print(f"DBI plot saved to {out2}")

    # Synthesis normalized (0-1)
    sil_n = _minmax_0_1(silhouettes)
    dbi_n = 1.0 - _minmax_0_1(dbis)
    inertia_n = 1.0 - _minmax_0_1(inertias)

    plt.figure(figsize=(10, 6))
    plt.plot(list(K), inertia_n, marker="o", label="Elbow (inertia) — inverted+norm")
    plt.plot(list(K), sil_n, marker="s", label="Silhouette — norm")
    plt.plot(list(K), dbi_n, marker="^", label="DBI — inverted+norm")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Normalized score (0–1, higher = better)")
    plt.title("Optimal k synthesis: Elbow + Silhouette + DBI")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out3 = out_dir / f"{out_prefix}_optimal_k_synthesis.png"
    plt.tight_layout()
    plt.savefig(out3, dpi=200)
    print(f"Synthesis plot saved to {out3}")

    df_metrics = pd.DataFrame({"k": list(K), "inertia": inertias, "silhouette": silhouettes, "dbi": dbis})
    out_csv = out_dir / f"{out_prefix}_optimal_k_metrics.csv"
    df_metrics.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Metrics saved to {out_csv}")


def pick_feature_columns(df: pd.DataFrame, *, species_col: str, target_col: str, exclude_meteo: bool) -> list[str]:
    # Start from numeric columns, then drop metadata and abundance-like columns.
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()

    drop_exact = {
        "Year",
        "annee",
        "id_cell",
        "site",
        "passage",
        target_col,
    }

    cols: list[str] = []
    for c in numeric:
        if c in drop_exact:
            continue
        lc = c.lower()
        if "abondance" in lc:
            continue
        if c == species_col:
            continue

        if exclude_meteo:
            # For this RPG file, meteo columns look like Mean_temp/Mean_preci/Tot_preci/Max_temp/Mean_Ampli
            if lc.startswith("mean_temp") or lc.startswith("mean_preci") or lc.startswith("tot_preci") or lc.startswith("max_temp") or lc.startswith("mean_ampli"):
                continue

        cols.append(c)

    return cols


def compute_species_coefficients(
    df: pd.DataFrame,
    *,
    species_col: str,
    target_col: str,
    feature_cols: list[str],
    min_samples: int,
    alpha: float,
    log_target: bool,
) -> pd.DataFrame:
    if species_col not in df.columns:
        raise ValueError(f"Species column '{species_col}' not found")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    species_list = df[species_col].dropna().unique().tolist()
    print(f"Computing coefficients for {len(species_list)} species using Ridge Regression...")
    print(f"Using {len(feature_cols)} features")

    results: dict[str, dict[str, float]] = {}

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MaxAbsScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )

    for i, spec in enumerate(species_list, start=1):
        sub = df[df[species_col] == spec]
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

        pipe.fit(X, y_arr)
        coefs = pipe.named_steps["model"].coef_

        results[str(spec)] = {feature_cols[j]: float(coefs[j]) for j in range(len(feature_cols))}

        if i % 50 == 0:
            print(f"  processed {i}/{len(species_list)} species...")

    coef_df = pd.DataFrame.from_dict(results, orient="index")
    return coef_df


def plot_cluster_heatmap(cluster_means: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 8))
    plt.imshow(cluster_means.T, aspect="auto", cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Mean influence")
    plt.xticks(range(len(cluster_means)), cluster_means.index)
    plt.yticks(range(len(cluster_means.columns)), cluster_means.columns)
    plt.xlabel("Cluster")
    plt.ylabel("Feature")
    plt.title("Feature influence heatmap by cluster")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Heatmap saved to '{out_path}'")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Cluster species by their pressure-response profile: fit a Ridge per species, "
            "use coefficients as embeddings, then StandardScaler + KMeans."
        )
    )
    parser.add_argument(
        "--data",
        default="raw_data/STOC_RPG_pressions_bio_assol_meteo_pesti_routes_lum_buff_haie_ZeromeanAdded2500_260925.csv",
        help="Input CSV (raw) containing species + target + pressure/meteo columns",
    )
    parser.add_argument("--species-col", default="species")
    parser.add_argument("--target-col", default="abondance_capped")
    parser.add_argument("--min-samples", type=int, default=200, help="Min rows per species to fit Ridge")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--log-target", action="store_true", help="Fit on log1p(target)")
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--find-k", action="store_true")
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--exclude-meteo", action="store_true", help="Exclude aggregated meteo columns")
    parser.add_argument("--max-rows", type=int, default=0, help="Read only first N rows (0 = all)")
    parser.add_argument("--encoding", default=None)
    parser.add_argument("--encoding-errors", default="strict")
    parser.add_argument("--out-prefix", default="rpg_raw")

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    data_path = (base_dir / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)

    print(f"Loading: {data_path}")
    if args.max_rows and args.max_rows > 0:
        # Use robust header+encoding for nrows too.
        # pandas supports the same encoding args.
        df = read_csv_robust(data_path, encoding=args.encoding, encoding_errors=args.encoding_errors)
        df = df.head(int(args.max_rows))
        print(f"Loaded first {len(df)} rows")
    else:
        df = read_csv_robust(data_path, encoding=args.encoding, encoding_errors=args.encoding_errors)
        print(f"Loaded {len(df)} rows")

    # Replace inf
    df = df.replace([np.inf, -np.inf], np.nan)

    feature_cols = pick_feature_columns(
        df,
        species_col=args.species_col,
        target_col=args.target_col,
        exclude_meteo=bool(args.exclude_meteo),
    )
    if not feature_cols:
        raise ValueError("No numeric feature columns selected. Check --exclude-meteo or your input schema.")

    influence_df = compute_species_coefficients(
        df,
        species_col=args.species_col,
        target_col=args.target_col,
        feature_cols=feature_cols,
        min_samples=int(args.min_samples),
        alpha=float(args.alpha),
        log_target=bool(args.log_target),
    )

    # Drop any species with missing coefficients (should be rare, but safe)
    influence_df = influence_df.dropna(axis=0, how="any")
    if influence_df.empty:
        raise ValueError("No species coefficients computed. Lower --min-samples or check your target column.")

    scaler_cluster = StandardScaler()
    X_cluster = scaler_cluster.fit_transform(influence_df.to_numpy(dtype=float))

    out_dir = base_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.find_k:
        find_optimal_clusters(X_cluster, max_k=int(args.max_k), out_prefix=args.out_prefix)
        return

    print(f"Clustering species into {args.n_clusters} groups...")
    kmeans = KMeans(n_clusters=int(args.n_clusters), random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_cluster)

    sil = float(silhouette_score(X_cluster, labels))
    dbi = float(davies_bouldin_score(X_cluster, labels))
    print("\nClustering quality (standardized coefficient space):")
    print(f"  Silhouette: {sil:.4f} (higher is better)")
    print(f"  DBI:        {dbi:.4f} (lower is better)\n")

    try:
        plot_tsne_clusters(X_cluster, labels, out_path=out_dir / f"{args.out_prefix}_cluster_tsne.png")
    except (ValueError, RuntimeError, TypeError) as e:
        print(f"t-SNE visualization failed: {e}")

    influence_df = influence_df.copy()
    influence_df["cluster"] = labels

    # Print groups summary + top features
    for c in range(int(args.n_clusters)):
        members = influence_df[influence_df["cluster"] == c].index.tolist()
        print(f"\n=== Cluster {c} ({len(members)} species) ===")
        print(", ".join(members[:200]) + (" ..." if len(members) > 200 else ""))

        mean_infl = influence_df[influence_df["cluster"] == c].drop(columns=["cluster"]).mean()
        top_features = mean_infl.abs().sort_values(ascending=False).head(8)
        print("Top features:", ", ".join(top_features.index.tolist()))

    # Save
    out_csv = out_dir / f"{args.out_prefix}_species_clusters.csv"
    influence_df.to_csv(out_csv, encoding="utf-8")
    print(f"Clusters saved to '{out_csv}'")

    cluster_means = influence_df.groupby("cluster").mean(numeric_only=True)
    plot_cluster_heatmap(cluster_means, out_path=out_dir / f"{args.out_prefix}_species_clusters_heatmap.png")


if __name__ == "__main__":
    main()
