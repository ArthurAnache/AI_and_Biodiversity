import argparse
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import PartialDependenceDisplay


def load_data(pressions_path: str, abundances_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = pd.read_csv(pressions_path)
    y_df = pd.read_csv(abundances_path)
    if 'abondance_capped' not in y_df.columns:
        raise ValueError("The abundances CSV must contain a column named 'abondance_capped'.")
    y = y_df['abondance_capped']
    if len(X) != len(y):
        raise ValueError(
            f"Row count mismatch between features ({len(X)}) and target ({len(y)})."
        )
    return X, y


def prepare_xy(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    mask = ~y.isna()
    if int(len(y) - mask.sum()) > 0:
        print(f"Dropping {int(len(y) - mask.sum())} rows due to NaN target values.")
    X_clean = X.loc[mask].copy()
    y_clean = y.loc[mask].copy()
    X_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    return X_clean, y_clean


def build_pipeline(X: pd.DataFrame, hidden_layers: Tuple[int, ...], random_state: int = 42,
                   early_stopping: bool = True, max_iter: int = 200) -> Pipeline:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_features = [c for c in numeric_features if X[c].notna().any()]
    categorical_features = [c for c in categorical_features if X[c].notna().any()]

    transformers = []
    if not numeric_features and not categorical_features:
        X_numeric = X.apply(pd.to_numeric, errors='coerce')
        numeric_features = X_numeric.columns.tolist()

    if numeric_features:
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        transformers.append(('num', num_pipe, numeric_features))
    if categorical_features:
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ])
        transformers.append(('cat', cat_pipe, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    print(f"Features -> numeric: {len(numeric_features)}, categorical: {len(categorical_features)}")

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        random_state=random_state,
        early_stopping=early_stopping,
        max_iter=max_iter,
        n_iter_no_change=10,
        validation_fraction=0.1,
        verbose=False
    )

    return Pipeline(steps=[('prep', preprocessor), ('mlp', mlp)])


essentials_desc = """
Trains the same MLP pipeline on a subset filtered by one species.
No files are written; prints metrics and shows training loss + permutation-importance histogram.
"""


def train_and_evaluate_subset(X: pd.DataFrame, y: pd.Series, species: str, hidden_layers: Tuple[int, ...],
                              test_size: float = 0.2, random_state: int = 42, max_iter: int = 1000):
    if 'species' not in X.columns:
        raise ValueError("Column 'species' not found in features; required for species filtering.")
    mask = (X['species'] == species)
    if mask.sum() == 0:
        raise ValueError(f"No rows found for species '{species}'.")
    X_sub = X.loc[mask].copy()
    y_sub = y.loc[mask].copy()

    Xp, yp = prepare_xy(X_sub, y_sub)
    X_train, X_test, y_train, y_test = train_test_split(
        Xp, yp, test_size=test_size, random_state=random_state
    )

    print(f"Subset '{species}' sizes -> train: {len(X_train)}, test: {len(X_test)}")
    model = build_pipeline(Xp, hidden_layers=hidden_layers, random_state=random_state, max_iter=max_iter)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        'rmse': float(np.sqrt(np.mean((y_test - y_pred) ** 2))),
        'r2': float(r2_score(y_test, y_pred)),
        'mae': float(mean_absolute_error(y_test, y_pred)),
    }
    return model, metrics, {'X_test': X_test, 'y_test': y_test, 'y_pred_test': y_pred}


def plot_training_loss(model: Pipeline) -> None:
    mlp = model.named_steps.get('mlp')
    loss_curve = getattr(mlp, 'loss_curve_', None)
    if loss_curve is None or len(loss_curve) == 0:
        print('[Info] No loss_curve_ available to plot.')
        return
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(loss_curve, color='purple')
    plt.xlabel('Iteration')
    plt.ylabel('Training loss')
    plt.title('MLP Training Loss (subset)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_top_feature_importance_rmse(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, top_n: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_test, dtype=float)
    baseline_pred = model.predict(X_test)
    baseline_rmse = float(np.sqrt(np.mean((y_true - baseline_pred) ** 2)))

    results = []
    for col in X_test.columns:
        X_perm = X_test.copy()
        X_perm[col] = np.random.permutation(X_perm[col].values)
        perm_pred = model.predict(X_perm)
        perm_rmse = float(np.sqrt(np.mean((y_true - perm_pred) ** 2)))
        results.append({
            'feature': col,
            'baseline_rmse': baseline_rmse,
            'permuted_rmse': perm_rmse,
            'rmse_increase': perm_rmse - baseline_rmse
        })

    res_df = pd.DataFrame(results).sort_values('rmse_increase', ascending=False)
    top_df = res_df.head(top_n)

    plt.figure(figsize=(8, max(4, top_n * 0.4)))
    plt.barh(top_df['feature'][::-1], top_df['permuted_rmse'][::-1], color='steelblue')
    plt.axvline(baseline_rmse, color='red', linestyle='--', linewidth=1, label=f'Baseline RMSE {baseline_rmse:.3f}')
    plt.xlabel('RMSE (after permutation)')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} variables by RMSE impact (subset)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return top_df


def accuracy_within_abs_tolerance(y_true: np.ndarray, y_pred: np.ndarray, abs_tol: float = 1.0) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_pred - y_true) <= float(abs_tol)))


def plot_partial_dependence(model: Pipeline, X: pd.DataFrame, features: list, species_name: str):
    """
    Plots Partial Dependence for the specified features.
    This shows the marginal effect of each feature on the predicted abundance.
    """
    print(f"Generating Partial Dependence Plots for top {len(features)} features...")
    
    # Convert integer columns to float to avoid sklearn future warning
    X_eval = X.copy()
    for col in X_eval.select_dtypes(include=['int', 'int32', 'int64']).columns:
        X_eval[col] = X_eval[col].astype(float)
    
    # Laisser scikit-learn gérer la grille d'axes automatiquement
    disp = PartialDependenceDisplay.from_estimator(
        model,
        X_eval,
        features,
        kind="average",
        n_jobs=-1,
        grid_resolution=30,
    )

    plt.suptitle(
        f"Partial Dependence of Abundance for {species_name}\n(Marginal Effect of Top Pressures)",
        fontsize=16,
        y=1.02,
    )
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train MLP on a specific species subset.', epilog=essentials_desc)
    parser.add_argument('--pressions', default='proc_data/pressions_petit.csv', help='Path to pressions CSV')
    parser.add_argument('--abundances', default='proc_data/abondances_petit.csv', help='Path to abundances CSV')
    parser.add_argument('--species', required=True, help='Species name to filter on')
    parser.add_argument('--hidden', default='256,128,64', help='Hidden layer sizes, comma-separated')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test size fraction for evaluation')
    parser.add_argument('--max-iter', type=int, default=1000, help='Max epochs for MLPRegressor (early stopping on)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--acc-abs-tol', type=float, default=1.0, help='Absolute tolerance for accuracy computation')

    args = parser.parse_args()
    hidden_tuple = tuple(int(x) for x in args.hidden.split(',') if x.strip())

    print('Loading data...')
    X, y = load_data(args.pressions, args.abundances)
    print(f"Data shapes -> X: {X.shape}, y: {y.shape}")

    print(f"Training model on species: {args.species}")
    model, metrics, splits = train_and_evaluate_subset(
        X, y, species=args.species, hidden_layers=hidden_tuple,
        test_size=args.test_size, random_state=args.seed, max_iter=args.max_iter
    )

    # Compute normalized RMSEs on subset test
    y_true_test = np.asarray(splits['y_test'], dtype=float)
    y_pred_test = np.asarray(splits['y_pred_test'], dtype=float)
    rmse_val = float(np.sqrt(np.mean((y_true_test - y_pred_test) ** 2)))
    mean_y = float(np.mean(y_true_test))
    std_y = float(np.std(y_true_test))
    nrmse_mean = float(rmse_val / (abs(mean_y) if abs(mean_y) > 1e-12 else 1e-12))
    nrmse_std = float(rmse_val / (std_y if std_y > 1e-12 else 1e-12))
    print(f"Validation metrics (subset): RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}")
    print(f"Normalized RMSE (subset) -> by mean: {nrmse_mean:.4f}, by std: {nrmse_std:.4f}")

    # Training loss curve (no other plots like pred-vs-true or residuals)
    plot_training_loss(model)

    # Accuracy within absolute tolerance
    acc = accuracy_within_abs_tolerance(splits['y_test'], splits['y_pred_test'], abs_tol=args.acc_abs_tol)
    print(f"Accuracy@±{args.acc_abs_tol:g} (subset): {acc:.4f}")

    # Permutation importance histogram
    print("Computing permutation importance (RMSE impact) on subset...")
    top_imp = plot_top_feature_importance_rmse(model, splits['X_test'], splits['y_test'], top_n=10)
    print("Top 10 features by RMSE increase (subset):")
    print(top_imp[['feature', 'rmse_increase']].to_string(index=False))

    # Plot Partial Dependence for the top 6 features
    # This answers "HOW" the features affect the abundance (direction/shape)
    top_features_list = top_imp['feature'].head(6).tolist()
    plot_partial_dependence(model, splits['X_test'], top_features_list, args.species)


if __name__ == '__main__':
    main()
