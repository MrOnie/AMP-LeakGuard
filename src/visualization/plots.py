"""
plots.py
========
Publication-ready figure generation for AMP-LeakGuard (300 DPI, Vector/TIFF/PNG):
1. Figure 1: Performance degradation gap (Delta AUPRC and Delta MCC) comparing
   Random 5-Fold vs. Homology-Aware Cross-Validation across Negative Sampling Schemes.
2. Figure 2: SHAP Beeswarm summary plots contrasting feature attribution in
   Naive (Scheme A) vs. Physicochemically-Matched (Scheme B) regimes.
3. Figure 3: Receiver Operating Characteristic (ROC) and Precision-Recall (PRC) curves.
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import precision_recall_curve, roc_curve


# Publication style configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

FIGURES_DIR = Path("reports/figures")


def plot_performance_gap(
    summary_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Figure 1: Grouped bar chart comparing Random vs. Homology Cross-Validation
    performance across models and negative sampling schemes.
    """
    if output_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIGURES_DIR / "fig1_performance_gap.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    # Plot A: AUPRC
    plot_data_auprc = pd.melt(
        summary_df,
        id_vars=["Negative Scheme", "Model"],
        value_vars=["Random AUPRC", "Homology AUPRC"],
        var_name="Validation Split",
        value_name="AUPRC",
    )
    plot_data_auprc["Validation Split"] = plot_data_auprc["Validation Split"].str.replace(" AUPRC", "")

    sns.barplot(
        data=plot_data_auprc,
        x="Negative Scheme",
        y="AUPRC",
        hue="Validation Split",
        palette=["#2b5c8f", "#d95f02"],
        ax=axes[0],
    )
    axes[0].set_title("A) Area Under Precision-Recall Curve (AUPRC)")
    axes[0].set_ylabel("AUPRC (Higher is Better)")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend(title="Cross-Validation Protocol", frameon=True)

    # Plot B: MCC
    plot_data_mcc = pd.melt(
        summary_df,
        id_vars=["Negative Scheme", "Model"],
        value_vars=["Random MCC", "Homology MCC"],
        var_name="Validation Split",
        value_name="MCC",
    )
    plot_data_mcc["Validation Split"] = plot_data_mcc["Validation Split"].str.replace(" MCC", "")

    sns.barplot(
        data=plot_data_mcc,
        x="Negative Scheme",
        y="MCC",
        hue="Validation Split",
        palette=["#2b5c8f", "#d95f02"],
        ax=axes[1],
    )
    axes[1].set_title("B) Matthews Correlation Coefficient (MCC)")
    axes[1].set_ylabel("MCC")
    axes[1].set_ylim(-0.05, 1.0)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].legend(title="Cross-Validation Protocol", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved Figure 1 to {output_path}")
    return output_path


def plot_roc_prc_comparison(
    y_true: np.ndarray,
    y_prob_random: np.ndarray,
    y_prob_homology: np.ndarray,
    scheme_name: str,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Figure 2: Side-by-side ROC and Precision-Recall Curves demonstrating the
    overoptimism gap caused by homology leakage.
    """
    if output_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        sanitized = scheme_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        output_path = FIGURES_DIR / f"fig2_roc_prc_{sanitized}.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC Curves
    fpr_r, tpr_r, _ = roc_curve(y_true, y_prob_random)
    fpr_h, tpr_h, _ = roc_curve(y_true, y_prob_homology)

    axes[0].plot(fpr_r, tpr_r, label="Random Split (Leakage)", color="#2b5c8f", lw=2.5)
    axes[0].plot(fpr_h, tpr_h, label="Homology-Aware (De-biased)", color="#d95f02", lw=2.5, linestyle="--")
    axes[0].plot([0, 1], [0, 1], color="grey", linestyle=":")
    axes[0].set_title("A) Receiver Operating Characteristic (ROC)")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right", frameon=True)
    axes[0].grid(alpha=0.3)

    # Precision-Recall Curves
    prec_r, rec_r, _ = precision_recall_curve(y_true, y_prob_random)
    prec_h, rec_h, _ = precision_recall_curve(y_true, y_prob_homology)

    axes[1].plot(rec_r, prec_r, label="Random Split (Leakage)", color="#2b5c8f", lw=2.5)
    axes[1].plot(rec_h, prec_h, label="Homology-Aware (De-biased)", color="#d95f02", lw=2.5, linestyle="--")
    axes[1].set_title("B) Precision-Recall Curve (PRC)")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left", frameon=True)
    axes[1].grid(alpha=0.3)

    plt.suptitle(f"Evaluation Under {scheme_name}", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved ROC/PRC comparison to {output_path}")
    return output_path


def generate_shap_beeswarm(
    model,
    X_sample: pd.DataFrame,
    title: str,
    output_path: Path,
    max_display: int = 12,
) -> Path:
    """
    Figure 3: SHAP Beeswarm plot explaining biological feature importance
    and exposing decision artifacts.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    plt.figure(figsize=(9, 6))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title(title, fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved SHAP plot to {output_path}")
    return output_path
