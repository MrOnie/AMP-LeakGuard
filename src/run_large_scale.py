"""
run_large_scale.py
==================
Large-scale benchmark execution of AMP-LeakGuard using >1,000 real curated AMPs
from Swiss-Prot and controlled negative sampling schemes.

Generates all publication-grade artifacts:
- Figure 1: Performance Degradation Gap (Delta AUPRC, Delta MCC)
- Figure 2: ROC and Precision-Recall Curves (Random vs Homology Split)
- Figure 3: SHAP Feature Attribution Beeswarm Plots (Exposing Trivial vs Genuine Motifs)
- Summary Data Tables (CSV and Markdown)
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.data.curation import PeptideCurator
from src.data.negative_sampling import NegativeSampler
from src.features.physicochemical import PhysicochemicalExtractor
from src.models.evaluate import ModelEvaluator
from src.splitting.homology_split import BenchmarkSplitter
from src.visualization.plots import (
    generate_shap_beeswarm,
    plot_performance_gap,
    plot_roc_prc_comparison,
)


def run_full_benchmark():
    print("=" * 75)
    print("AMP-LeakGuard: LARGE-SCALE BENCHMARK & DE-BIASING AUDIT")
    print("=" * 75)

    # 1. Load and Curate Positive AMPs
    raw_amps_path = Path("data/raw/uniprot_reviewed_amps.tsv")
    if not raw_amps_path.exists():
        raise FileNotFoundError(f"Missing {raw_amps_path}. Run fetch_databases.py first.")

    raw_amp_df = pd.read_csv(raw_amps_path, sep="\t")
    raw_amp_df["label"] = 1
    curator = PeptideCurator(min_length=10, max_length=50)
    pos_df = curator.curate_dataframe(
        raw_amp_df,
        seq_col="Sequence",
        id_col="Entry",
        label_col="label",
    )
    print(f"[*] Curated high-confidence positive AMPs (10-50 aa): {len(pos_df)} records.")

    # 2. Load Background Proteins for Negatives
    bg_path = Path("data/raw/uniprot_background_proteins.tsv")
    bg_df = pd.read_csv(bg_path, sep="\t")
    bg_proteins = bg_df["Sequence"].dropna().tolist()
    print(f"[*] Loaded {len(bg_proteins)} background proteins for non-AMP sampling.")

    # 3. Generate Scheme A and Scheme B Negatives (1:1 ratio)
    sampler = NegativeSampler(random_seed=42)
    print("[*] Generating Negatives Scheme A (Naive UniProt random slices)...")
    neg_scheme_a = sampler.generate_naive_negatives(
        source_proteins=bg_proteins,
        target_count=len(pos_df),
        target_lengths=pos_df["length"].tolist(),
    )

    print("[*] Generating Negatives Scheme B (Physicochemically-matched slices)...")
    neg_scheme_b = sampler.generate_matched_negatives(
        source_proteins=bg_proteins,
        positive_df=pos_df,
    )

    print(f"[+] Scheme A count: {len(neg_scheme_a)}, Scheme B count: {len(neg_scheme_b)}")

    # 4. Feature Extraction & Modeling
    evaluator = ModelEvaluator(random_seed=42)
    splitter = BenchmarkSplitter(random_seed=42)
    results = []

    schemes = [
        ("Scheme A (Naive)", neg_scheme_a),
        ("Scheme B (Matched)", neg_scheme_b),
    ]

    for scheme_name, neg_df in schemes:
        print(f"\n{'='*30} Processing {scheme_name} {'='*30}")
        dataset = pd.concat([pos_df, neg_df], ignore_index=True)
        print(f"[*] Total dataset size: {len(dataset)} peptides (50% AMP / 50% Non-AMP)")

        # Feature Extraction
        print("[*] Extracting biophysical physicochemical features...")
        X_df = PhysicochemicalExtractor.extract_dataframe(dataset, seq_col="sequence")
        X = X_df.values
        y = dataset["label"].values

        # Cross-validation splits
        print("[*] Generating Random 5-Fold CV...")
        random_folds = splitter.get_random_cv_folds(dataset, n_splits=5)

        print("[*] Generating Homology-Aware 5-Fold CV (@ 40% sequence identity)...")
        homology_folds, cluster_ids = splitter.get_homology_cv_folds(
            dataset, n_splits=5, identity_threshold=0.40, seq_col="sequence", label_col="label"
        )
        n_clusters = len(np.unique(cluster_ids))
        print(f"[+] Partitioned into {n_clusters} non-redundant homology clusters via StratifiedGroupKFold.")

        # Verify no cross-fold homology leakage
        fold0_train, fold0_val = homology_folds[0]
        mean_cross_id = splitter.verify_no_cross_fold_leakage(
            dataset.iloc[fold0_train]["sequence"].tolist(),
            dataset.iloc[fold0_val]["sequence"].tolist(),
            sample_size=50,
        )
        print(f"[+] Verified mean cross-fold global alignment identity: {mean_cross_id:.3f}")

        # Evaluate Models
        for model_name in ["random_forest", "lightgbm"]:
            print(f"\n--- Model: {model_name} ---")
            res_rand = evaluator.evaluate_cv(X, y, random_folds, model_name=model_name)
            res_homo = evaluator.evaluate_cv(X, y, homology_folds, model_name=model_name)

            delta_auprc = res_rand["mean_auprc"] - res_homo["mean_auprc"]
            delta_mcc = res_rand["mean_mcc"] - res_homo["mean_mcc"]

            # Statistical significance of degradation
            sig_auprc = evaluator.compute_paired_significance(res_rand, res_homo, metric_name="auprc")
            sig_mcc = evaluator.compute_paired_significance(res_rand, res_homo, metric_name="mcc")

            results.append({
                "Negative Scheme": scheme_name,
                "Model": model_name,
                "Random AUPRC": round(res_rand["mean_auprc"], 4),
                "Homology AUPRC": round(res_homo["mean_auprc"], 4),
                "Delta AUPRC (Leakage)": round(delta_auprc, 4),
                "p-val AUPRC (Paired)": f"{sig_auprc['p_value_ttest']:.4f}",
                "Random AUROC": round(res_rand["mean_auroc"], 4),
                "Homology AUROC": round(res_homo["mean_auroc"], 4),
                "Delta AUROC": round(res_rand["mean_auroc"] - res_homo["mean_auroc"], 4),
                "Random MCC": round(res_rand["mean_mcc"], 4),
                "Homology MCC": round(res_homo["mean_mcc"], 4),
                "Delta MCC": round(delta_mcc, 4),
                "p-val MCC (Paired)": f"{sig_mcc['p_value_ttest']:.4f}",
            })

            # Plot ROC / PRC for LightGBM
            if model_name == "lightgbm":
                plot_roc_prc_comparison(
                    y_true=y,
                    y_prob_random=res_rand["oof_predictions"],
                    y_prob_homology=res_homo["oof_predictions"],
                    scheme_name=scheme_name,
                )

        # Generate SHAP Analysis for LightGBM on this scheme
        print(f"[*] Computing SHAP explanations for {scheme_name}...")
        clf = lgb.LGBMClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42, verbosity=-1)
        clf.fit(X, y)
        sanitized = scheme_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        shap_out = Path(f"reports/figures/fig3_shap_{sanitized}.png")
        generate_shap_beeswarm(
            model=clf,
            X_sample=X_df.sample(n=min(500, len(X_df)), random_state=42),
            title=f"SHAP Feature Attribution: {scheme_name}",
            output_path=shap_out,
        )

    # 5. Summarize Results & Save
    summary_df = pd.DataFrame(results)
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "large_scale_benchmark_audit.csv", index=False)
    summary_df.to_markdown(out_dir / "large_scale_benchmark_audit.md", index=False)

    print("\n" + "=" * 75)
    print("FINAL LARGE-SCALE BENCHMARK RESULTS")
    print("=" * 75)
    print(summary_df.to_string(index=False))

    # Generate Figure 1
    plot_performance_gap(summary_df)
    print("\n[+] All figures and data tables successfully generated in reports/figures/ and data/processed/")


if __name__ == "__main__":
    run_full_benchmark()
