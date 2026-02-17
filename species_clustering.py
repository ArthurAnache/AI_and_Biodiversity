import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


# NEW
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def _format_feature_effect(feature: str, mean_coef: float) -> str:
    """Human-readable signed effect for cluster summaries.

    mean_coef > 0: tends to increase predicted abundance
    mean_coef < 0: tends to decrease predicted abundance
    """
    if not np.isfinite(mean_coef):
        return f"{feature} (n/a)"
    direction = "+" if mean_coef >= 0 else "-"
    return f"{feature} ({direction}{mean_coef:.3g})"


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
    out_path: str = 'figures/cluster_tsne.png',
    random_state: int = 42,
    perplexity: float = 30.0,
):
    """2D t-SNE visualization of clusters.

    Intended as a visual sanity-check: distinct color clouds => separation.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    n = int(X_cluster.shape[0])
    if n < 4:
        print("Not enough points for t-SNE visualization.")
        return

    # Perplexity must be < n_samples; keep a safe default.
    px = float(perplexity)
    px_max = max(2.0, (n - 1) / 3.0)
    if px >= n:
        px = min(30.0, px_max)
    if px < 2.0:
        px = 2.0

    tsne = TSNE(
        n_components=2,
        perplexity=px,
        learning_rate='auto',
        init='pca',
        random_state=random_state,
    )
    Z = tsne.fit_transform(X_cluster)

    plt.figure(figsize=(10, 7))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=labels, cmap='tab10', s=18, alpha=0.85, linewidths=0)
    plt.title(f"t-SNE clusters (perplexity={px:.1f})")
    plt.xlabel('t-SNE dim 1')
    plt.ylabel('t-SNE dim 2')
    cb = plt.colorbar(sc)
    cb.set_label('cluster')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"t-SNE plot saved to '{out_path}'")


def plot_pca_clusters(
    X_cluster: np.ndarray,
    labels: np.ndarray,
    out_path: str = 'figures/cluster_pca.png',
    random_state: int = 42,
    n_components: int = 2,
):
    """PCA visualization of clusters.

    Note: you cannot force PC1+PC2 to explain more variance; it is a property of the data.
    If PC1+PC2 is low, use 3D (PC1-3) or report cumulative variance.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    n = int(X_cluster.shape[0])
    if n < 2:
        print("Not enough points for PCA visualization.")
        return

    n_components = int(n_components)
    if n_components < 2:
        raise ValueError("n_components must be >= 2")

    pca = PCA(n_components=n_components, random_state=random_state)
    Z = pca.fit_transform(X_cluster)
    evr = pca.explained_variance_ratio_
    cum = float(np.sum(evr[: min(len(evr), n_components)]))

    if n_components == 2:
        evr_text = " + ".join([f"{v*100:.1f}%" for v in evr[:2]])
        plt.figure(figsize=(10, 7))
        sc = plt.scatter(Z[:, 0], Z[:, 1], c=labels, cmap='tab10', s=20, alpha=0.85, linewidths=0)
        plt.title(f"PCA clusters (PC1+PC2={cum*100:.1f}%; {evr_text})")
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        cb = plt.colorbar(sc)
        cb.set_label('cluster')
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        print(f"PCA plot saved to '{out_path}'")
        return

    if n_components == 3:
        # 3D PCA plot to capture more variance.
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=labels, cmap='tab10', s=20, alpha=0.85, linewidths=0)
        ax.set_title(f"PCA 3D clusters (PC1+PC2+PC3={cum*100:.1f}%)")
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        cb = fig.colorbar(sc)
        cb.set_label('cluster')
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        print(f"PCA 3D plot saved to '{out_path}'")
        return

    print(f"PCA visualization skipped: n_components={n_components} (supported: 2 or 3).")


def plot_pca_explained_variance(
    X_cluster: np.ndarray,
    out_path: str = 'figures/pca_explained_variance.png',
    random_state: int = 42,
    max_components: int = 25,
):
    """Scree + cumulative explained variance plot.

    Helps justify why PC1+PC2 can be ~55%: the variance may be spread across many dimensions.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    n_features = int(X_cluster.shape[1])
    k = int(max(2, min(max_components, n_features)))

    pca = PCA(n_components=k, random_state=random_state)
    pca.fit(X_cluster)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    plt.figure(figsize=(10, 6))
    xs = np.arange(1, len(evr) + 1)
    plt.bar(xs, evr, alpha=0.6, label='Explained variance ratio')
    plt.plot(xs, cum, marker='o', color='tab:red', label='Cumulative')
    plt.xlabel('Principal component')
    plt.ylabel('Variance explained')
    plt.title('PCA explained variance (scree + cumulative)')
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"PCA explained variance plot saved to '{out_path}'")


def load_data(pressions_path: str, abundances_path: str):
    X = pd.read_csv(pressions_path)
    y_df = pd.read_csv(abundances_path)

    if 'abondance_capped' not in y_df.columns:
        raise ValueError("The abundances CSV must contain a column named 'abondance_capped'.")

    y = y_df['abondance_capped']

    if len(X) != len(y):
        raise ValueError(f"Row count mismatch: X={len(X)}, y={len(y)}")

    return X, y

def prepare_xy(X: pd.DataFrame, y: pd.Series):
    mask = ~y.isna()
    X_clean = X.loc[mask].copy()
    y_clean = y.loc[mask].copy()
    X_clean.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Remove 'abondance_capped_year_plus_1' if present
    if 'abondance_capped_year_plus_1' in X_clean.columns:
        print("Dropping 'abondance_capped_year_plus_1' from features.")
        X_clean.drop(columns=['abondance_capped_year_plus_1'], inplace=True)

    return X_clean, y_clean

def compute_species_coefficients(X: pd.DataFrame, y: pd.Series, species_col: str = 'species') -> pd.DataFrame:
    """
    Trains a Ridge regression for each species individually and extracts the coefficients
    for the numeric features (pressures).
    Also returns mean/std abundance to help clustering by population dynamics.
    """
    if species_col not in X.columns:
        raise ValueError(f"Species column '{species_col}' not found in X.")

    species_list = X[species_col].unique()
    print(f"Computing coefficients for {len(species_list)} species using Ridge Regression...")
    
    # Identify numeric features (pressures)
    # Filter to keep only RELEVANT structural/pressure features
    # Excluding:
    # - Monthly weather (too noisy/numerous)
    # - Time-shifted duplicates (_year_plus_1)
    
    all_numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    
    numeric_cols = []
    for col in all_numeric:
        # Exclude monthly meteo
        if col.startswith('Temps_') or col.startswith('Precipitation_'):
            continue
        # Exclude abundance metrics if they slipped in
        if 'abondance' in col and 'capped' in col:
            # We want to exclude the target 'abondance_capped' but maybe not other derived metrics?
            # Safest is to exclude explicit target-like names
            continue
            
        if 'all_pesticide_exposure_grid_sum' in col:
            continue
            
        numeric_cols.append(col)
        
    print(f"Selected {len(numeric_cols)} relevant features for clustering (including lagged): {numeric_cols}")
    
    results = {}
    
    for spec in species_list:
        # Filter data for this species
        mask = X[species_col] == spec
        X_spec = X.loc[mask, numeric_cols].copy()
        y_spec = y.loc[mask]
        
        if len(X_spec) < 10:
            print(f"Skipping {spec}: too few samples ({len(X_spec)})")
            continue
            
        # Pipeline for this species
        # We only use numeric features for the 'influence' profile
        # User requested scaling by max to handle magnitude differences
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MaxAbsScaler()),
            ('model', Ridge(alpha=1.0))
        ])
        
        pipe.fit(X_spec, y_spec)
        
        # Extract coefficients
        model = pipe.named_steps['model']
        coefs = model.coef_
        
        # Store stats
        spec_stats = {}
        # Abundance stats removed to focus only on pressure response profile
        
        for i, col in enumerate(numeric_cols):
            spec_stats[col] = coefs[i]
            
        results[spec] = spec_stats

    # Create DataFrame: Index=Species, Columns=Features + Stats
    coef_df = pd.DataFrame.from_dict(results, orient='index')
    return coef_df

def plot_cluster_heatmap(cluster_means):
    import os
    os.makedirs('figures', exist_ok=True)
    plt.figure(figsize=(14, 8))
    # Normalize columns for better visualization of relative importance
    # or just plot raw values. Let's plot raw values first.
    plt.imshow(cluster_means.T, aspect='auto', cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Mean Influence')
    plt.xticks(range(len(cluster_means)), cluster_means.index)
    plt.yticks(range(len(cluster_means.columns)), cluster_means.columns)
    plt.xlabel('Cluster')
    plt.ylabel('Feature')
    plt.title('Feature Influence Heatmap by Cluster')
    plt.tight_layout()
    plt.savefig('figures/species_clusters_heatmap.png')
    print("Heatmap saved to 'figures/species_clusters_heatmap.png'")
    # plt.show()

def find_optimal_clusters(X, max_k=10):
    inertias = []
    silhouettes = []
    dbis = []
    K = range(2, max_k + 1)
    
    print(f"Searching for optimal k (2 to {max_k})...")
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, kmeans.labels_))
        dbis.append(davies_bouldin_score(X, kmeans.labels_))
        print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouettes[-1]:.4f}, DBI={dbis[-1]:.4f}")
        
    # Plotting
    _, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia (Elbow)', color=color)
    ax1.plot(K, inertias, marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(K, silhouettes, marker='s', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Optimal k Analysis: Elbow Method & Silhouette Score')
    plt.grid(True, alpha=0.3)
    
    save_path = 'figures/optimal_k_analysis.png'
    plt.savefig(save_path)
    print(f"Optimal k analysis plot saved to {save_path}")

    # DBI plot (lower is better)
    plt.figure(figsize=(10, 6))
    plt.plot(list(K), dbis, marker='^', color='tab:green')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Davies–Bouldin Index (lower is better)')
    plt.title('Optimal k Analysis: Davies–Bouldin Index')
    plt.grid(True, alpha=0.3)
    save_path = 'figures/optimal_k_dbi.png'
    plt.savefig(save_path)
    print(f"DBI analysis plot saved to {save_path}")

    # Synthèse (superposition normalisée 0-1)
    # - Silhouette: plus haut = mieux
    # - DBI: plus bas = mieux => on inverse avant normalisation
    # - Inertia: plus bas = mieux => on inverse avant normalisation
    sil_n = _minmax_0_1(silhouettes)
    dbi_n = 1.0 - _minmax_0_1(dbis)
    inertia_n = 1.0 - _minmax_0_1(inertias)

    plt.figure(figsize=(10, 6))
    plt.plot(list(K), inertia_n, marker='o', label='Elbow (inertie) — inversé+norm')
    plt.plot(list(K), sil_n, marker='s', label='Silhouette — norm')
    plt.plot(list(K), dbi_n, marker='^', label='DBI — inversé+norm')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Score normalisé (0–1, plus haut = mieux)')
    plt.title('Synthèse optimal k: Elbow + Silhouette + DBI')
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_path = 'figures/optimal_k_synthesis.png'
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Synthesis plot saved to {save_path}")

    # Tableau récap (utile pour analyser plus finement)
    df_metrics = pd.DataFrame({
        'k': list(K),
        'inertia': inertias,
        'silhouette': silhouettes,
        'dbi': dbis,
    })
    df_metrics.to_csv('figures/optimal_k_metrics.csv', index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pressions', default='proc_data/pressions_petit.csv')
    parser.add_argument('--abundances', default='proc_data/abondances_petit.csv')
    parser.add_argument('--n-clusters', type=int, default=3)
    parser.add_argument('--find-k', action='store_true', help="Search for optimal k using Elbow and Silhouette methods")
    parser.add_argument(
        '--viz',
        choices=['tsne', 'pca', 'both', 'none'],
        default='both',
        help="Cluster visualization method (default: both)",
    )
    parser.add_argument(
        '--pca-components',
        type=int,
        default=2,
        help="Number of PCA components for the PCA visualization (2=2D, 3=3D)",
    )
    parser.add_argument(
        '--pca-max-components',
        type=int,
        default=25,
        help="Max number of components to show in the PCA explained-variance plot",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    pressions_path = Path(args.pressions)
    abundances_path = Path(args.abundances)
    if not pressions_path.is_absolute():
        pressions_path = (base_dir / pressions_path).resolve()
    if not abundances_path.is_absolute():
        abundances_path = (base_dir / abundances_path).resolve()

    print("Loading data...")
    X, y = load_data(str(pressions_path), str(abundances_path))
    X, y = prepare_xy(X, y)

    # Compute coefficients per species
    influence_df = compute_species_coefficients(X, y)
    
    # We want to cluster based on:
    # 1. The profile of response (coefficients) - MAGNITUDE MATTERS for shared model
    # (Abundance removed as requested)
    
    # Preprocessing for clustering:
    # - Standardize ALL features (coefficients)
    # This aligns the "scales" of different pressures so they contribute equally to distance
    scaler_cluster = StandardScaler()
    X_cluster = scaler_cluster.fit_transform(influence_df)
    
    # We use the standardized features for clustering
    # (Previously we normalized by vector norm, which destroyed magnitude information)
    
    if args.find_k:
        find_optimal_clusters(X_cluster)
        return

    print(f"Clustering species into {args.n_clusters} groups...")
    kmeans = KMeans(n_clusters=args.n_clusters, init='k-means++', random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_cluster)

    sil = silhouette_score(X_cluster, labels)
    dbi = davies_bouldin_score(X_cluster, labels)
    print("\nClustering quality (on standardized coefficient space):")
    print(f"  Silhouette: {sil:.4f} (higher is better)")
    print(f"  DBI:        {dbi:.4f} (lower is better)\n")

    # Visual validation (PCA / t-SNE)
    if args.viz in {'pca', 'both'}:
        try:
            plot_pca_explained_variance(X_cluster, max_components=int(args.pca_max_components))
            plot_pca_clusters(X_cluster, labels, n_components=int(args.pca_components))
        except (ValueError, RuntimeError, TypeError) as e:
            print(f"PCA visualization failed: {e}")

    if args.viz in {'tsne', 'both'}:
        try:
            plot_tsne_clusters(X_cluster, labels)
        except (ValueError, RuntimeError, TypeError) as e:
            print(f"t-SNE visualization failed: {e}")
    
    influence_df['cluster'] = labels
    
    # Print groups
    for c in range(args.n_clusters):
        species = influence_df[influence_df['cluster'] == c].index.tolist()
        print(f"\n=== Cluster {c} ({len(species)} species) ===")
        print(", ".join(species))
        
        # Identify top influential features for this cluster
        mean_infl = influence_df[influence_df['cluster'] == c].drop(columns=['cluster']).mean()
        top_features = mean_infl.abs().sort_values(ascending=False).head(5)
        top_with_sign = [_format_feature_effect(f, float(mean_infl[f])) for f in top_features.index.tolist()]
        print("Top features (sign indicates +/− effect on abundance):", ", ".join(top_with_sign))

    # Plot heatmap of cluster centers
    cluster_means = influence_df.groupby('cluster').mean()
    
    # Save clusters to CSV
    influence_df.to_csv('figures/species_clusters.csv')
    print("Clusters saved to 'figures/species_clusters.csv'")
    
    plot_cluster_heatmap(cluster_means)

if __name__ == '__main__':
    main()


