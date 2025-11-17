import argparse
import os
from typing import Tuple
import json

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


def load_data(pressions_path: str, abundances_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads features (pressions) and target (abondance_capped) from CSV files.
    Assumes abundances file has a single column named 'abondance_capped',
    aligned row-wise with pressions.
    """
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
    """
    - Drops rows with NaN target (y).
    - Replaces +/-inf in X with NaN (to be imputed).
    """
    before = len(y)
    mask = ~y.isna()
    dropped = int(before - mask.sum())
    if dropped > 0:
        print(f"Dropping {dropped} rows due to NaN target values.")
    X_clean = X.loc[mask].copy()
    y_clean = y.loc[mask].copy()
    # Sanitize infinities
    X_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    return X_clean, y_clean


def build_pipeline(X: pd.DataFrame, hidden_layers: Tuple[int, ...], random_state: int = 42,
                   early_stopping: bool = True, max_iter: int = 200) -> Pipeline:
    """
    Builds a preprocessing + MLP regression pipeline.
    - Selects numeric columns and standardizes them.
    - One-hot encodes categorical columns (e.g., species) with handle_unknown='ignore'.
    - Uses an MLPRegressor with ReLU activations and Adam optimizer.
    """
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Drop columns that are entirely NaN to avoid imputer errors
    numeric_features = [c for c in numeric_features if X[c].notna().any()]
    categorical_features = [c for c in categorical_features if X[c].notna().any()]

    transformers = []
    if len(numeric_features) == 0 and len(categorical_features) == 0:
        # Fallback: try to coerce all to numeric if detection failed
        X_numeric = X.apply(pd.to_numeric, errors='coerce')
        numeric_features = X_numeric.columns.tolist()

    if numeric_features:
        # Impute with median then scale
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        transformers.append(('num', num_pipe, numeric_features))
    if categorical_features:
        # Impute missing categories then one-hot encode to dense
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ])
        transformers.append(('cat', cat_pipe, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )

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

    pipe = Pipeline(steps=[
        ('prep', preprocessor),
        ('mlp', mlp),
    ])
    return pipe


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, hidden_layers: Tuple[int, ...],
                       test_size: float = 0.2, random_state: int = 42):
    """
    Splits data into train/test, fits the pipeline, and returns model and metrics.
    """
    # Prepare data (drop NaN targets, sanitize X)
    Xp, yp = prepare_xy(X, y)

    idx = Xp.index.to_numpy()
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        Xp, yp, idx, test_size=test_size, random_state=random_state
    )

    # Report split sizes to confirm 80/20 (or chosen proportion)
    n_total = len(Xp)
    n_train = len(X_train)
    n_test = len(X_test)
    print(f"Split sizes -> train: {n_train} ({n_train / n_total:.1%}), test: {n_test} ({n_test / n_total:.1%})")

    model = build_pipeline(Xp, hidden_layers=hidden_layers, random_state=random_state)
    model.fit(X_train, y_train)

    # Diagnostics: report one-hot encoded dimensions and category counts
    try:
        prep = model.named_steps.get('prep')
        if isinstance(prep, ColumnTransformer):
            for name, trans, cols in prep.transformers_:
                if name == 'cat' and isinstance(trans, Pipeline):
                    ohe = trans.named_steps.get('ohe')
                    if hasattr(ohe, 'categories_'):
                        cat_counts = [len(c) for c in ohe.categories_]
                        total_ohe = sum(cat_counts)
                        print(f"OneHot details -> categorical columns: {len(cols)}, total OHE features: {total_ohe}, per-column categories: {cat_counts}")
                        break
    except (AttributeError, KeyError, TypeError):
        pass

    y_pred = model.predict(X_test)
    metrics = {
        'r2': float(r2_score(y_test, y_pred)),
        'mae': float(mean_absolute_error(y_test, y_pred)),
    }
    return model, metrics, {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'idx_train': idx_train,
        'idx_test': idx_test,
        'y_pred_test': y_pred,
    }


def predict_and_save(model: Pipeline, X: pd.DataFrame, output_csv: str, round_int: bool = True) -> None:
    """
    Runs prediction for all rows in X and writes a single-column CSV 'abondance_capped'.
    If round_int is True, rounds to nearest non-negative integer (clip at 0).
    """
    preds = model.predict(X)
    if round_int:
        preds = np.maximum(np.rint(preds), 0).astype(int)

    out_df = pd.DataFrame({'abondance_capped': preds})
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding='utf-8-sig')


def save_predictions_with_index(indices: np.ndarray, preds: np.ndarray, output_csv: str, round_int: bool = True) -> None:
    """
    Save predictions with original row indices for alignment.
    Columns: row_index, abondance_capped
    """
    if round_int:
        preds = np.maximum(np.rint(preds), 0).astype(int)
    out_df = pd.DataFrame({'row_index': indices, 'abondance_capped': preds})
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding='utf-8-sig')


def compute_species_pressions_influence(model: Pipeline, species_col: str = 'species', eps_scale: float = 0.5) -> pd.DataFrame:
    """
    Approximate partial derivatives dY/dX for numeric pressions per species using finite differences
    on a baseline constructed from the pipeline's imputers (median/mode).

    Returns a DataFrame of shape (n_species, n_numeric_pressions), where entries are
    (y(x + eps_j*e_j) - y(x)) / eps_j, with species fixed per row and other features at baseline.
    """
    prep = model.named_steps.get('prep')
    if not isinstance(prep, ColumnTransformer):
        raise ValueError('Unexpected pipeline structure: missing ColumnTransformer step named "prep"')

    # Identify transformers
    num_feats: list[str] = []
    cat_feats: list[str] = []
    num_imputer = None
    num_scaler = None
    cat_imputer = None
    ohe = None

    for name, trans, cols in prep.transformers_:
        if name == 'num' and isinstance(trans, Pipeline):
            num_feats = list(cols)
            num_imputer = trans.named_steps.get('imputer')
            num_scaler = trans.named_steps.get('scaler')
        elif name == 'cat' and isinstance(trans, Pipeline):
            cat_feats = list(cols)
            cat_imputer = trans.named_steps.get('imputer')
            ohe = trans.named_steps.get('ohe')

    if ohe is None or cat_imputer is None:
        raise ValueError('No categorical transformer with OneHotEncoder found; cannot compute per-species influence.')

    if species_col not in cat_feats:
        raise ValueError(f'Categorical column "{species_col}" not found among categorical features: {cat_feats}')

    # Index of species within categorical features
    species_idx = cat_feats.index(species_col)
    species_categories = list(ohe.categories_[species_idx])

    # Baseline values from imputers
    base_numeric = {}
    if num_imputer is not None and num_feats:
        for c, v in zip(num_feats, num_imputer.statistics_):
            base_numeric[c] = v

    base_categorical = {}
    for c, v in zip(cat_feats, cat_imputer.statistics_):
        base_categorical[c] = v

    # Epsilons per numeric feature from StandardScaler scale_ (std in raw units pre-scaling)
    epsilons = {}
    if num_scaler is not None and num_feats:
        for c, s in zip(num_feats, num_scaler.scale_):
            if s is None or not np.isfinite(s) or s == 0:
                epsilons[c] = 1.0
            else:
                epsilons[c] = float(eps_scale * s)

    # Construct baseline row helper
    def make_row_for_species(spec: str) -> pd.DataFrame:
        row = {**base_numeric, **base_categorical}
        row[species_col] = spec
        # Ensure all expected columns exist; if some numeric/cat were empty, fill with NaN
        for c in num_feats:
            row.setdefault(c, np.nan)
        for c in cat_feats:
            row.setdefault(c, base_categorical.get(c, None))
        return pd.DataFrame([row])

    # Compute influence matrix
    data = []
    for spec in species_categories:
        x0 = make_row_for_species(spec)
        y0 = float(model.predict(x0)[0])
        row_vals = []
        for c in num_feats:
            eps = epsilons.get(c, 1.0)
            x1 = x0.copy()
            x1.loc[:, c] = x1[c].astype(float).fillna(base_numeric.get(c, 0.0)) + eps
            y1 = float(model.predict(x1)[0])
            grad = (y1 - y0) / eps if eps != 0 else 0.0
            row_vals.append(grad)
        data.append(row_vals)

    infl_df = pd.DataFrame(data, index=species_categories, columns=num_feats)
    return infl_df


def plot_influence_heatmap(infl_df: pd.DataFrame, out_png: str | None = None, title: str = 'Influence (dY/dX) by species x pressions') -> None:
    if infl_df.empty:
        print('Influence DataFrame is empty; skipping heatmap.')
        return
    vmax = float(np.nanmax(np.abs(infl_df.values)))
    plt.figure(figsize=(max(8, infl_df.shape[1] * 0.4), max(6, infl_df.shape[0] * 0.3)))
    plt.imshow(infl_df.values, aspect='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax)
    plt.colorbar(label='Approx. dY/dX')
    plt.xticks(ticks=np.arange(infl_df.shape[1]), labels=infl_df.columns, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(infl_df.shape[0]), labels=infl_df.index)
    plt.title(title)
    plt.tight_layout()
    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=150)
        print(f'Heatmap saved to: {out_png}')
    else:
        plt.show()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute RMSE, R2 and MAE between arrays.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"rmse": rmse, "r2": r2, "mae": mae}


def evaluate_predictions_file(y_true_csv: str, y_pred_csv: str) -> dict:
    """
    Read ground-truth and predictions from CSVs and compute metrics.
    Both CSVs must contain a single column named 'abondance_capped'.
    """
    true_df = pd.read_csv(y_true_csv)
    pred_df = pd.read_csv(y_pred_csv)

    if 'abondance_capped' not in true_df.columns:
        raise ValueError("Ground-truth CSV must contain column 'abondance_capped'.")
    if 'abondance_capped' not in pred_df.columns:
        raise ValueError("Prediction CSV must contain column 'abondance_capped'.")

    y_true = true_df['abondance_capped'].values
    y_pred = pred_df['abondance_capped'].values

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)} vs y_pred={len(y_pred)}")

    return compute_metrics(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser(description='Train a DNN to predict abundances from pressions.')
    parser.add_argument('--pressions', default='proc_data/pressions_petit.csv', help='Path to pressions CSV')
    parser.add_argument('--abundances', default='proc_data/abondances_petit.csv', help='Path to abundances CSV')
    parser.add_argument('--out', default='proc_data/abondances_petit_pred.csv', help='Output CSV path for predictions')
    parser.add_argument('--hidden', default='256,128,64', help='Hidden layer sizes, comma-separated')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test size fraction for evaluation (e.g., 0.2 for 80/20)')
    parser.add_argument('--max-iter', type=int, default=200, help='Max epochs for MLPRegressor')
    parser.add_argument('--no-round', action='store_true', help='Do not round predictions to integers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--metrics-out', default='', help='Optional path to write metrics JSON for predictions vs ground truth')
    parser.add_argument('--save-train-preds', action='store_true', help='Also save predictions for the training split')

    args = parser.parse_args()

    hidden_tuple = tuple(int(x) for x in args.hidden.split(',') if x.strip())

    print('Loading data...')
    X, y = load_data(args.pressions, args.abundances)
    print(f"Data shapes -> X: {X.shape}, y: {y.shape}")

    print('Training model...')
    model, metrics, splits = train_and_evaluate(
        X, y, hidden_layers=hidden_tuple, test_size=args.test_size, random_state=args.seed
    )
    print(f"Validation metrics: R2={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}")

    # Compute and save influence matrix (species x pressions)
    try:
        infl_df = compute_species_pressions_influence(model, species_col='species', eps_scale=0.5)
        os.makedirs('proc_data', exist_ok=True)
        infl_csv = os.path.join('proc_data', 'species_pressions_influence.csv')
        infl_df.to_csv(infl_csv, encoding='utf-8-sig')
        print(f'Influence matrix written to: {infl_csv}')
        # Plot heatmap
        out_png = os.path.join('figures', 'species_pressions_influence_heatmap.png')
        plot_influence_heatmap(infl_df, out_png=out_png)
    except (ValueError, KeyError, AttributeError) as e:
        print(f'[Warn] Could not compute influence heatmap: {e}')

    # Avoid accidental overwrite of ground-truth file unless explicitly requested
    if os.path.abspath(args.out) == os.path.abspath(args.abundances):
        print('[Warning] Output path equals ground-truth file. Changing to abondances_petit_pred_test.csv to avoid overwrite.')
        out_test = os.path.join(os.path.dirname(args.out), 'abondances_petit_pred_test.csv')
    else:
        out_test = args.out

    # Save predictions for test split with original row indices
    print(f"Writing test predictions to: {out_test}")
    save_predictions_with_index(splits['idx_test'], splits['y_pred_test'], out_test, round_int=(not args.no_round))

    # Optionally also save train predictions
    if args.save_train_preds:
        y_pred_train = model.predict(splits['X_train'])
        out_train = os.path.join(os.path.dirname(out_test), 'abondances_petit_pred_train.csv')
        print(f"Writing train predictions to: {out_train}")
        save_predictions_with_index(splits['idx_train'], y_pred_train, out_train, round_int=(not args.no_round))

    # Optionally write metrics JSON (from validation metrics)
    if args.metrics_out:
        with open(args.metrics_out, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Validation metrics written to: {args.metrics_out}")

    print('Done.')


if __name__ == '__main__':
    main()

