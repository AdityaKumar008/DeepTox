"""
DeepTox — Feature Importance Analysis
======================================
This script analyzes which chemical properties (Morgan fingerprint bits)
contribute most to toxicity prediction in the trained Random Forest model.

Run from the project root directory:
    python feature_importance.py

Outputs:
    - feature_importance_summary.csv   → Top 20 important features per assay
    - feature_importance_overall.png   → Bar chart of overall top features
    - feature_importance_per_assay.png → Per-assay top features heatmap
    - feature_importance_report.txt    → Plain text summary for submission
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH   = 'tox21_model.joblib'
DATA_PATH    = 'Tox21.csv'
OUTPUT_DIR   = 'feature_importance_outputs'
TOP_N        = 20   # top N features to display

# Assay target names (same order as model was trained)
ASSAY_TARGETS = [
    'Nuclear Receptor-Androgen',
    'Nuclear Receptor-Estrogen',
    'Nuclear Receptor-AhR',
    'Nuclear Receptor-Aromatase',
    'Nuclear Receptor-ER-LBD',
    'Nuclear Receptor-PPAR-gamma',
    'Stress Response-ARE',
    'Stress Response-ATAD5',
    'Stress Response-HSE',
    'Stress Response-MMP',
    'Stress Response-p53',
    'Stress Response-Mitochondrial',
]

# Short display names for charts
ASSAY_SHORT = [
    'NR-AR', 'NR-ER', 'NR-AhR', 'NR-Arom',
    'NR-ER-LBD', 'NR-PPAR', 'SR-ARE', 'SR-ATAD5',
    'SR-HSE', 'SR-MMP', 'SR-p53', 'SR-Mito',
]


def setup_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR}/")


def load_model():
    print(f"⏳ Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded. Type: {type(model).__name__}")
    return model


def extract_importances(model):
    """Extract feature importances from each estimator in the MultiOutput model."""
    importances_per_assay = []

    estimators = model.estimators_
    print(f"✅ Found {len(estimators)} assay estimators.")

    for i, est in enumerate(estimators):
        if hasattr(est, 'feature_importances_'):
            imp = est.feature_importances_
        else:
            # Some estimators may be pipelines
            imp = np.zeros(2048)
        importances_per_assay.append(imp)

    return np.array(importances_per_assay)  # shape: (n_assays, 2048)


def compute_overall_importance(importances_matrix):
    """Average importance across all assay targets."""
    return np.mean(importances_matrix, axis=0)


def plot_overall_importance(overall_imp, top_n=TOP_N):
    """Bar chart of top N most important fingerprint bits overall."""
    top_indices = np.argsort(overall_imp)[::-1][:top_n]
    top_values  = overall_imp[top_indices]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#0d1b2e')
    ax.set_facecolor('#0d1b2e')

    colors = ['#00cc6a' if v < 0.002 else '#ffb300' if v < 0.004 else '#ff4d6d'
              for v in top_values]

    bars = ax.bar(range(top_n), top_values, color=colors, edgecolor='none', width=0.7)

    ax.set_xticks(range(top_n))
    ax.set_xticklabels([f'Bit {i}' for i in top_indices], rotation=45, ha='right',
                       fontsize=9, color='#c8dff5')
    ax.set_ylabel('Mean Feature Importance', color='#c8dff5', fontsize=11)
    ax.set_title(f'Top {top_n} Most Important Molecular Features\n(Averaged Across All 12 Tox21 Assay Targets)',
                 color='#eaf4ff', fontsize=13, fontweight='bold', pad=15)
    ax.tick_params(axis='y', colors='#c8dff5')
    ax.spines['bottom'].set_color('#1a3050')
    ax.spines['left'].set_color('#1a3050')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, color='#1a3050', linewidth=0.5)
    ax.set_axisbelow(True)

    # legend
    low   = mpatches.Patch(color='#00cc6a', label='Low importance')
    mid   = mpatches.Patch(color='#ffb300', label='Medium importance')
    high  = mpatches.Patch(color='#ff4d6d', label='High importance')
    ax.legend(handles=[low, mid, high], facecolor='#0d1b2e', edgecolor='#1a3050',
              labelcolor='#c8dff5', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'feature_importance_overall.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1b2e')
    plt.close()
    print(f"✅ Saved: {out_path}")
    return top_indices


def plot_per_assay_heatmap(importances_matrix, top_indices, top_n=TOP_N):
    """Heatmap showing top features importance per assay target."""
    # Use only top N global features for the heatmap
    data = importances_matrix[:, top_indices]  # (n_assays, top_n)

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor('#0d1b2e')
    ax.set_facecolor('#0d1b2e')

    im = ax.imshow(data, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    ax.set_xticks(range(top_n))
    ax.set_xticklabels([f'Bit {i}' for i in top_indices], rotation=45, ha='right',
                       fontsize=8, color='#c8dff5')
    ax.set_yticks(range(len(ASSAY_SHORT)))
    ax.set_yticklabels(ASSAY_SHORT, fontsize=10, color='#c8dff5')
    ax.set_title(f'Feature Importance Heatmap — Top {top_n} Features × 12 Assay Targets',
                 color='#eaf4ff', fontsize=13, fontweight='bold', pad=15)
    ax.tick_params(axis='both', colors='#c8dff5')

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Feature Importance', color='#c8dff5', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='#c8dff5')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#c8dff5')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'feature_importance_per_assay.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1b2e')
    plt.close()
    print(f"✅ Saved: {out_path}")


def save_csv_summary(importances_matrix, overall_imp):
    """Save top features per assay as a CSV."""
    rows = []

    # Overall top features
    top_overall = np.argsort(overall_imp)[::-1][:TOP_N]
    for rank, bit_idx in enumerate(top_overall):
        rows.append({
            'Rank':       rank + 1,
            'Assay':      'OVERALL (Average)',
            'Bit_Index':  int(bit_idx),
            'Importance': round(float(overall_imp[bit_idx]), 6),
        })

    # Per-assay top features
    n_assays = min(len(ASSAY_TARGETS), importances_matrix.shape[0])
    for i in range(n_assays):
        imp = importances_matrix[i]
        top_bits = np.argsort(imp)[::-1][:10]
        for rank, bit_idx in enumerate(top_bits):
            rows.append({
                'Rank':       rank + 1,
                'Assay':      ASSAY_TARGETS[i] if i < len(ASSAY_TARGETS) else f'Assay_{i}',
                'Bit_Index':  int(bit_idx),
                'Importance': round(float(imp[bit_idx]), 6),
            })

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, 'feature_importance_summary.csv')
    df.to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path}")
    return df


def save_text_report(importances_matrix, overall_imp):
    """Write a plain-text summary report for hackathon submission."""
    top_overall = np.argsort(overall_imp)[::-1][:TOP_N]
    n_assays    = min(len(ASSAY_TARGETS), importances_matrix.shape[0])

    lines = []
    lines.append("=" * 70)
    lines.append("DEEPTOX — FEATURE IMPORTANCE ANALYSIS REPORT")
    lines.append("Hybrid AI Toxicity Prediction System")
    lines.append("Dataset: Tox21 (NIH) | Model: Multi-output Random Forest")
    lines.append("=" * 70)
    lines.append("")
    lines.append("OVERVIEW")
    lines.append("-" * 40)
    lines.append("This report identifies which molecular features (Morgan fingerprint")
    lines.append("bits) contribute most to toxicity prediction across 12 Tox21 assay")
    lines.append("targets. Each bit represents the presence or absence of a specific")
    lines.append("chemical substructure in the molecule.")
    lines.append("")
    lines.append(f"Total features analyzed : 2048 Morgan fingerprint bits (radius=2)")
    lines.append(f"Assay targets evaluated : {n_assays}")
    lines.append(f"Top features reported   : {TOP_N}")
    lines.append("")

    lines.append("TOP 20 MOST IMPORTANT FEATURES (OVERALL AVERAGE)")
    lines.append("-" * 40)
    lines.append(f"{'Rank':<6} {'Bit Index':<12} {'Importance':<12} {'Significance'}")
    lines.append("-" * 50)
    for rank, bit_idx in enumerate(top_overall):
        val = overall_imp[bit_idx]
        sig = 'HIGH' if val > 0.004 else 'MEDIUM' if val > 0.002 else 'LOW'
        lines.append(f"{rank+1:<6} {bit_idx:<12} {val:<12.6f} {sig}")

    lines.append("")
    lines.append("TOP 5 FEATURES PER ASSAY TARGET")
    lines.append("-" * 40)
    for i in range(n_assays):
        name = ASSAY_TARGETS[i] if i < len(ASSAY_TARGETS) else f'Assay_{i}'
        imp  = importances_matrix[i]
        top5 = np.argsort(imp)[::-1][:5]
        lines.append(f"\n{name}")
        for rank, bit_idx in enumerate(top5):
            lines.append(f"  {rank+1}. Bit {bit_idx:<6}  importance = {imp[bit_idx]:.6f}")

    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 40)
    lines.append("Morgan fingerprint bits encode the presence of specific atomic")
    lines.append("neighborhoods (substructures) within a molecule. A high feature")
    lines.append("importance score means the presence or absence of that particular")
    lines.append("chemical substructure is a strong predictor of toxicity for the")
    lines.append("corresponding biological assay target.")
    lines.append("")
    lines.append("Higher bit index importance does NOT mean a compound is toxic —")
    lines.append("it means that substructure is highly discriminative between")
    lines.append("toxic and non-toxic compounds in the training data.")
    lines.append("")
    lines.append("MODEL DETAILS")
    lines.append("-" * 40)
    lines.append("Algorithm     : Multi-output Random Forest Classifier")
    lines.append("Estimators    : 100 trees per assay target")
    lines.append("Input features: 2048-bit Morgan fingerprint (radius=2)")
    lines.append("Training data : Tox21 dataset (NIH) — 12,707 compounds")
    lines.append("Output        : Toxicity probability per assay target (0-100%)")
    lines.append("")
    lines.append("=" * 70)
    lines.append("Generated by DeepTox Feature Importance Analysis Script")
    lines.append("GitHub: https://github.com/AdityaKumar008/DeepTox")
    lines.append("=" * 70)

    out_path = os.path.join(OUTPUT_DIR, 'feature_importance_report.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"✅ Saved: {out_path}")


def print_quick_summary(overall_imp, importances_matrix):
    """Print a quick summary to terminal."""
    top5 = np.argsort(overall_imp)[::-1][:5]
    print("\n" + "="*60)
    print("QUICK SUMMARY — TOP 5 FEATURES (OVERALL)")
    print("="*60)
    for rank, bit_idx in enumerate(top5):
        val = overall_imp[bit_idx]
        bar = '█' * int(val * 10000)
        print(f"  {rank+1}. Bit {bit_idx:<6}  {val:.6f}  {bar}")
    print("="*60)
    print(f"\nAll outputs saved to: ./{OUTPUT_DIR}/")
    print("Files generated:")
    print("  📊 feature_importance_overall.png   — bar chart")
    print("  🔥 feature_importance_per_assay.png — heatmap")
    print("  📄 feature_importance_summary.csv   — full data table")
    print("  📝 feature_importance_report.txt    — text report")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🧪 DeepTox — Feature Importance Analysis")
    print("=" * 50)

    setup_output_dir()
    model = load_model()

    print("⏳ Extracting feature importances from Random Forest estimators...")
    importances_matrix = extract_importances(model)
    print(f"✅ Importances matrix shape: {importances_matrix.shape}")

    overall_imp = compute_overall_importance(importances_matrix)
    print(f"✅ Overall importance computed. Max value: {overall_imp.max():.6f}")

    print("\n⏳ Generating visualizations...")
    top_indices = plot_overall_importance(overall_imp)
    plot_per_assay_heatmap(importances_matrix, top_indices)

    print("\n⏳ Saving data files...")
    save_csv_summary(importances_matrix, overall_imp)
    save_text_report(importances_matrix, overall_imp)

    print_quick_summary(overall_imp, importances_matrix)
    print("\n✅ Feature importance analysis complete!\n")
