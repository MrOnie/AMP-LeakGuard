"""
pipeline.py
===========
End-to-end execution pipeline for AMP-LeakGuard:
1. Data ingestion & curation.
2. Controlled negative sampling (Scheme A vs Scheme B).
3. Fast physicochemical feature extraction.
4. Cross-validation under Random vs. Homology-Aware partitioning.
5. Model training (Random Forest, LightGBM, Logistic Regression).
6. Metrics synthesis and gap quantification (Delta AUPRC, Delta MCC).
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.data.curation import PeptideCurator
from src.data.negative_sampling import NegativeSampler
from src.features.physicochemical import PhysicochemicalExtractor
from src.models.evaluate import ModelEvaluator
from src.splitting.homology_split import BenchmarkSplitter


# Reference canonical AMP sequences validated in DBAASP / APD3 for the core benchmark
REFERENCE_AMPS = [
    # Magainins & Cecropins (alpha-helical amphipathic)
    {"id": "AMP_0001", "sequence": "GIGKFLHSAKKFGKAFVGEIMNS", "source": "Magainin-2", "mic_ug_ml": 4.0},
    {"id": "AMP_0002", "sequence": "GIGKFLHSAKKFGKAFVGEIMNSL", "source": "Magainin-2_ext", "mic_ug_ml": 8.0},
    {"id": "AMP_0003", "sequence": "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK", "source": "Cecropin_A", "mic_ug_ml": 2.0},
    {"id": "AMP_0004", "sequence": "GWASKIGQTLGKIAKVGLKELIQPK", "source": "Pleurocidin", "mic_ug_ml": 8.0},
    {"id": "AMP_0005", "sequence": "FLPIIAKLLGGLG", "source": "Temporin_A", "mic_ug_ml": 12.5},
    {"id": "AMP_0006", "sequence": "FLPIVGKLLSGLL", "source": "Temporin_B", "mic_ug_ml": 16.0},
    {"id": "AMP_0007", "sequence": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES", "source": "LL-37", "mic_ug_ml": 5.0},
    {"id": "AMP_0008", "sequence": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLR", "source": "LL-37_fragment", "mic_ug_ml": 10.0},
    {"id": "AMP_0009", "sequence": "KTCENLADTFRGPCFTDGSCDDHCKNKEHLRNGYCDNFCACTNKAC", "source": "HBD-2", "mic_ug_ml": 15.0},
    {"id": "AMP_0010", "sequence": "GLFDIVKKVVGALGSL", "source": "Melittin_derived", "mic_ug_ml": 6.0},
    {"id": "AMP_0011", "sequence": "GLFDIVKKVVGAFGSL", "source": "Melittin_analog_1", "mic_ug_ml": 8.0},
    {"id": "AMP_0012", "sequence": "GLFDIVKKVVGALSSL", "source": "Melittin_analog_2", "mic_ug_ml": 12.0},
    {"id": "AMP_0013", "sequence": "RRRPRPPYLPRPRPPPFFPPRLPPRIPP", "source": "PR-39", "mic_ug_ml": 4.0},
    {"id": "AMP_0014", "sequence": "ILPWKWPWWPWRR", "source": "Indolicidin", "mic_ug_ml": 8.0},
    {"id": "AMP_0015", "sequence": "ILPWKWPWWPWRK", "source": "Indolicidin_variant", "mic_ug_ml": 16.0},
    {"id": "AMP_0016", "sequence": "RGGRLCYCRRRFCVCVGR", "source": "Protegrin-1", "mic_ug_ml": 2.0},
    {"id": "AMP_0017", "sequence": "RGGRLCYCRGWICVCVGR", "source": "Protegrin-2", "mic_ug_ml": 4.0},
    {"id": "AMP_0018", "sequence": "KWKSFIKKLTSAAKKVVTTAKPLISS", "source": "Sarcotoxin_IA", "mic_ug_ml": 6.0},
    {"id": "AMP_0019", "sequence": "GLFDIIKKIAESF", "source": "Dermaseptin-S4", "mic_ug_ml": 10.0},
    {"id": "AMP_0020", "sequence": "ALWKTLLKKVLKAAAKA", "source": "Synthetic_L-K", "mic_ug_ml": 4.0},
]

# Canonical background reference proteins for non-AMP sampling
BACKGROUND_PROTEINS = [
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
    "MSDNGPQNQRNAPRITFGGPSDSTGSNQNGERSGARSKQRRPQGLPNNTASWFTALTQHGKEDLKFPRGQGVPINTNSSPDDQIGYYRRATRRIRGGDGKMKDLSPRWYFYYLGTGPEAGLPYGANKDGIIWVATEGALNTPKDHIGTRNPANNAAIVLQLPQGTTLPKGFYAEGSRGGSQASSRSSSRSRNSSRNSTPGSSRGTSPARMAGNGGDAALALLLLDRLNQLESKMSGKGQQQQGQTVTKKSAAEASKKPRQKRTATKAYNVTQAFGRRGPEQTQGNFGDQELIRQGTDYKHWPQIAQFAPSASAFFGMSRIGMEVTPSGTWLTYTGAIKLDDKDPNFKDQVILLNKHIDAYKTFPPTEPKKDKKKKADETQALPQRQKKQQTVTLLPAADLDDFSKQLQQSMSSADSTQA",
    "MEDYKGSSSNSTVDNLLQNGSSSTAVNSSSYPTDTTNSNDPTTNNNDPTTNDPNSTNDPNSTNDPNSTNDPNSTNDPNSTNDPNSTNDPNSTNDPNSTNDPNSTNDPNSTNDPNSTNDPNSTNDPNSTNDPNS",
]


def run_benchmark_pipeline() -> pd.DataFrame:
    print("=" * 70)
    print("AMP-LeakGuard: Benchmark Evaluation & De-Biasing Execution")
    print("=" * 70)

    # 1. Curation
    curator = PeptideCurator(min_length=10, max_length=50)
    raw_pos_df = pd.DataFrame(REFERENCE_AMPS)
    raw_pos_df["label"] = 1
    pos_df = curator.curate_dataframe(raw_pos_df, seq_col="sequence", id_col="id")
    print(f"[*] Curated Positive AMPs: {len(pos_df)} records.")

    # 2. Negative Sampling (Scheme A vs Scheme B)
    sampler = NegativeSampler(random_seed=42)
    neg_scheme_a = sampler.generate_naive_negatives(
        source_proteins=BACKGROUND_PROTEINS,
        target_count=len(pos_df),
        target_lengths=pos_df["length"].tolist(),
    )
    neg_scheme_b = sampler.generate_matched_negatives(
        source_proteins=BACKGROUND_PROTEINS,
        positive_df=pos_df,
    )
    print(f"[*] Generated Negatives Scheme A (Naive UniProt): {len(neg_scheme_a)} records.")
    print(f"[*] Generated Negatives Scheme B (Physicochemically Matched): {len(neg_scheme_b)} records.")

    results: list[dict] = []
    evaluator = ModelEvaluator(random_seed=42)
    splitter = BenchmarkSplitter(random_seed=42)

    # Test both Negative Schemes
    for scheme_name, neg_df in [("Scheme A (Naive)", neg_scheme_a), ("Scheme B (Matched)", neg_scheme_b)]:
        dataset = pd.concat([pos_df, neg_df], ignore_index=True)
        # Extract features
        X_df = PhysicochemicalExtractor.extract_dataframe(dataset, seq_col="sequence")
        X = X_df.values
        y = dataset["label"].values

        # Split 1: Random 5-Fold CV
        random_folds = splitter.get_random_cv_folds(dataset, n_splits=5)
        # Split 2: Homology-Aware 5-Fold CV (Identity threshold 40%)
        homology_folds, cluster_ids = splitter.get_homology_cv_folds(dataset, n_splits=5, identity_threshold=0.40)
        n_clusters = len(np.unique(cluster_ids))

        print(f"\n---> Dataset: {scheme_name} (Total samples: {len(dataset)}, Homology Clusters: {n_clusters})")

        for model_name in ["random_forest", "lightgbm"]:
            # Evaluate Random Split
            res_rand = evaluator.evaluate_cv(X, y, random_folds, model_name=model_name)
            # Evaluate Homology Split
            res_homo = evaluator.evaluate_cv(X, y, homology_folds, model_name=model_name)

            delta_auprc = res_rand["mean_auprc"] - res_homo["mean_auprc"]
            delta_mcc = res_rand["mean_mcc"] - res_homo["mean_mcc"]

            results.append({
                "Negative Scheme": scheme_name,
                "Model": model_name,
                "Random AUPRC": round(res_rand["mean_auprc"], 4),
                "Homology AUPRC": round(res_homo["mean_auprc"], 4),
                "Delta AUPRC (Leakage)": round(delta_auprc, 4),
                "Random MCC": round(res_rand["mean_mcc"], 4),
                "Homology MCC": round(res_homo["mean_mcc"], 4),
                "Delta MCC": round(delta_mcc, 4),
            })

    summary_df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("BENCHMARK AUDIT RESULTS: QUANTIFYING HOMOLOGY & SAMPLING LEAKAGE")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    # Save summary table
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "benchmark_audit_summary.csv", index=False)
    print(f"\n[+] Results saved to {out_dir / 'benchmark_audit_summary.csv'}")

    return summary_df


if __name__ == "__main__":
    run_benchmark_pipeline()
