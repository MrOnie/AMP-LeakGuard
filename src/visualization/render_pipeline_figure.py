"""
render_pipeline_figure.py
=========================
Renders a publication-grade, high-resolution (300 DPI) schematic of the
AMP-LeakGuard benchmark workflow (Figure 1), ensuring pristine compilation in LaTeX.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set high DPI and aesthetic styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "figure.dpi": 300,
})

def draw_pipeline():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Colors
    c_blue_bg = "#EBF5FB"
    c_blue_border = "#2980B9"
    c_teal_bg = "#E8F8F5"
    c_teal_border = "#16A085"
    c_red_bg = "#FDEDEC"
    c_red_border = "#E74C3C"
    c_gold_bg = "#FEF9E7"
    c_gold_border = "#F39C12"
    c_purple_bg = "#F4ECF7"
    c_purple_border = "#8E44AD"
    c_dark_bg = "#EAECEE"
    c_dark_border = "#34495E"
    c_green_bg = "#E8F6F3"
    c_green_border = "#27AE60"

    def draw_box(x, y, w, h, title, text, bg_col, border_col, title_col=None):
        if title_col is None:
            title_col = border_col
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1,rounding_size=0.15",
            facecolor=bg_col,
            edgecolor=border_col,
            linewidth=1.8,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2, y + h - 0.35, title,
            ha="center", va="top",
            fontsize=10.5, fontweight="bold",
            color=title_col, zorder=3,
        )
        ax.text(
            x + 0.25, y + h - 0.75, text,
            ha="left", va="top",
            fontsize=9.0, color="#2C3E50",
            linespacing=1.35, zorder=3,
        )

    def draw_arrow(x1, y1, x2, y2, label=None):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#5D6D7E",
                lw=1.8,
                mutation_scale=15,
            ),
            zorder=1,
        )
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, label,
                    ha="center", va="bottom", fontsize=8.5, color="#5D6D7E", fontweight="bold")

    # Section 1: Data Acquisition & Curation (y: 7.6 to 9.8)
    draw_box(0.5, 7.8, 3.8, 1.8,
             "1. UniProtKB / Swiss-Prot (AMPs)",
             "• Keyword: KW-0929 (Antimicrobial)\n• Manually reviewed Swiss-Prot\n• Raw candidates: N = 2,652",
             c_blue_bg, c_blue_border)

    draw_box(4.8, 7.8, 4.4, 1.8,
             "2. Quality Control & Curation",
             "• Length constraints: 10 <= L <= 50 aa\n• Canonical 20 amino acids only\n• Deduplication & ambiguity removal\n• Yield: N = 1,026 Curated AMPs",
             c_teal_bg, c_teal_border)

    draw_box(9.7, 7.8, 3.8, 1.8,
             "3. Eukaryotic Background",
             "• Human Swiss-Prot reviewed\n• Non-AMP & non-toxic sequences\n• Background pool: N = 500 proteins",
             c_blue_bg, c_blue_border)

    draw_arrow(4.3, 8.7, 4.8, 8.7)

    # Section 2: Dual Negative Sampling Regimes (y: 5.2 to 7.0)
    draw_box(7.2, 5.2, 3.2, 1.9,
             "Scheme A: Naive Random",
             "• Random background slices\n• Length-matched distribution\n• Negative pool: N = 1,026\n• Total Dataset: N = 2,052",
             c_red_bg, c_red_border)

    draw_box(10.5, 5.2, 3.2, 1.9,
             "Scheme B: Matched Bio-physics",
             "• Paired length |ΔL| = 0\n• Net charge matched |ΔZ| <= 1.0\n• Neutralizes charge shortcut\n• Negative pool: N = 1,026",
             c_gold_bg, c_gold_border)

    draw_arrow(11.6, 7.8, 8.8, 7.1)
    draw_arrow(11.6, 7.8, 12.1, 7.1)

    # Section 3: Feature Extraction (y: 5.2 to 7.0, Left)
    draw_box(0.5, 5.2, 6.2, 1.9,
             "4. Biophysical Feature Extraction (NumPy Vectorized)",
             "• Net Charge (pH 7.0 via Henderson-Hasselbalch, pKa scales)\n• Eisenberg & Kyte-Doolittle Hydrophobicity (<H>)\n• Amphipathic Hydrophobic Moment (μα @ 100°, μβ @ 180°)\n• Aliphatic Index (Ikai 1980) & Boman Index (Boman 2003)\n• 20-dim Amino Acid Composition (AAC-20) & Grouped Ratios",
             c_purple_bg, c_purple_border)

    draw_arrow(7.0, 7.8, 3.6, 7.1)
    draw_arrow(7.2, 6.1, 6.7, 6.1)
    draw_arrow(10.5, 6.1, 6.7, 6.1)

    # Section 4: Splitting Protocols (y: 2.7 to 4.4)
    draw_box(0.5, 2.7, 6.2, 1.9,
             "Protocol R: Random 5-Fold Split",
             "• Standard StratifiedKFold cross-validation\n• Sequence homology unconstrained between folds\n• Reproduces standard overoptimistic literature setup\n• High susceptibility to memorization",
             c_dark_bg, c_dark_border)

    draw_box(7.2, 2.7, 6.5, 1.9,
             "Protocol H: Homology-Aware 5-Fold Split",
             "• Single-linkage clustering at 40% sequence identity\n• StratifiedGroupKFold prevents cross-split cluster leakage\n• Verified Mean Cross-Fold Global Identity: 16.1% - 18.0%\n• Rigorous de-biased real-world generalization test",
             c_dark_bg, c_dark_border)

    draw_arrow(3.6, 5.2, 3.6, 4.6)
    draw_arrow(3.6, 5.2, 10.4, 4.6)

    # Section 5: Machine Learning & Auditing (y: 0.3 to 2.0)
    draw_box(0.5, 0.4, 3.8, 1.8,
             "5. Model Benchmarking",
             "• Random Forest Classifier\n• LightGBM Gradient Boosting\n• 5-Fold cross-validation\n• Stratified class balance 50:50",
             c_green_bg, c_green_border)

    draw_box(4.8, 0.4, 4.2, 1.8,
             "6. Statistical Leakage Audit",
             "• Evaluation: AUPRC, AUROC, MCC\n• Paired Student's t-test & Wilcoxon\n• Quantified ΔAUPRC & ΔMCC drops\n• Bootstrap 95% Confidence Intervals",
             c_green_bg, c_green_border)

    draw_box(9.5, 0.4, 4.2, 1.8,
             "7. Explainable AI (TreeExplainer SHAP)",
             "• Scheme A: Net charge electrostatic shortcut\n• Scheme B: Lytic determinants emerge\n  - Arginine guanidinium bidentate binding\n  - Cysteine disulfide bridges (defensins)\n  - Amphipathic hydrophobic moment (μα)",
             c_green_bg, c_green_border)

    draw_arrow(3.6, 2.7, 2.4, 2.2)
    draw_arrow(10.4, 2.7, 2.4, 2.2)
    draw_arrow(4.3, 1.3, 4.8, 1.3)
    draw_arrow(9.0, 1.3, 9.5, 1.3)

    plt.tight_layout()
    
    # Save both in manuscript/figures and reports/figures
    out1 = Path("AMP-LeakGuard/manuscript/figures/fig1_pipeline.png")
    out2 = Path("AMP-LeakGuard/reports/figures/fig1_pipeline.png")
    out1.parent.mkdir(parents=True, exist_ok=True)
    out2.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out1, dpi=300, bbox_inches="tight")
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Successfully rendered Figure 1 to {out1} and {out2}")

if __name__ == "__main__":
    draw_pipeline()
