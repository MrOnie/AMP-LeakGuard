"""
evaluate.py
===========
Rigorous cross-validation training and benchmarking suite comparing:
- Split Regime: Random CV vs Homology-Controlled CV (30%, 40%, 70%)
- Negative Sampling Regime: Naive UniProt vs Matched UniProt vs Inactives
- Estimators: Logistic Regression, Random Forest, LightGBM
- Metrics: AUPRC (primary), AUROC, MCC, F1, Balanced Accuracy, Brier Score
"""

from typing import Any, Dict, List, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Computes comprehensive classification metrics for imbalanced/balanced evaluation."""
    y_pred = (y_prob >= threshold).astype(int)

    # Avoid zero division or edge cases
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = 0.5

    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except Exception:
        auprc = 0.0

    return {
        "auprc": auprc,
        "auroc": auroc,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


class ModelEvaluator:
    """
    Executes cross-validation across models and splits, logging detailed per-fold results.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def get_models(self) -> Dict[str, Any]:
        """Returns baseline and gradient-boosting estimators."""
        return {
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=self.random_seed,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                random_state=self.random_seed,
                n_jobs=-1,
            ),
            "lightgbm": lgb.LGBMClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=15,
                min_child_samples=2,
                random_state=self.random_seed,
                verbosity=-1,
            ),
        }

    def evaluate_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        folds: List[Tuple[np.ndarray, np.ndarray]],
        model_name: str,
        scale_features: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs cross-validation across the given fold indices.
        Returns aggregate mean, std, and per-fold metrics.
        """
        fold_metrics: List[Dict[str, float]] = []
        oof_predictions = np.zeros(len(y), dtype=float)

        models = self.get_models()
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            if scale_features:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

            model = models[model_name]
            model.fit(X_train, y_train)

            y_prob_val = model.predict_proba(X_val)[:, 1]
            oof_predictions[val_idx] = y_prob_val

            metrics = compute_binary_metrics(y_val, y_prob_val)
            fold_metrics.append(metrics)

        # Compute summary statistics
        df_metrics = pd.DataFrame(fold_metrics)
        summary = {
            "model": model_name,
            "mean_auprc": float(df_metrics["auprc"].mean()),
            "std_auprc": float(df_metrics["auprc"].std()),
            "mean_auroc": float(df_metrics["auroc"].mean()),
            "std_auroc": float(df_metrics["auroc"].std()),
            "mean_mcc": float(df_metrics["mcc"].mean()),
            "std_mcc": float(df_metrics["mcc"].std()),
            "mean_f1": float(df_metrics["f1"].mean()),
            "std_f1": float(df_metrics["f1"].std()),
            "mean_brier": float(df_metrics["brier_score"].mean()),
            "per_fold": fold_metrics,
            "oof_predictions": oof_predictions,
        }

        return summary

    @staticmethod
    def compute_paired_significance(
        res_random: Dict[str, Any],
        res_homology: Dict[str, Any],
        metric_name: str = "auprc",
    ) -> Dict[str, float]:
        """
        Computes formal paired statistical hypothesis tests (Wilcoxon signed-rank and Paired t-test)
        comparing cross-validation fold scores between random and homology splits.
        """
        vals_r = [f[metric_name] for f in res_random["per_fold"]]
        vals_h = [f[metric_name] for f in res_homology["per_fold"]]

        # Paired t-test
        t_stat, p_val_ttest = ttest_rel(vals_r, vals_h)

        # Wilcoxon signed-rank test
        try:
            w_stat, p_val_wilcoxon = wilcoxon(vals_r, vals_h, zero_method="pratt")
        except Exception:
            w_stat, p_val_wilcoxon = 0.0, 1.0

        diffs = np.array(vals_r) - np.array(vals_h)
        mean_diff = float(np.mean(diffs))
        ci_lower = float(np.percentile(diffs, 2.5))
        ci_upper = float(np.percentile(diffs, 97.5))

        return {
            "metric": metric_name,
            "mean_difference": mean_diff,
            "p_value_ttest": float(p_val_ttest),
            "p_value_wilcoxon": float(p_val_wilcoxon),
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
        }
