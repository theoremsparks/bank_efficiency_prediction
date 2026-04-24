import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import RAW_DATA_PATH, get_analysis_output_dir


def run_correlation_heatmap():
    data = pd.read_csv(RAW_DATA_PATH)
    data.columns = data.columns.str.strip()

    vars_11 = [
        "Assets",
        "Employee Expense",
        "Equity",
        "Deposits",
        "Borrowings",
        "Performing Loans",
        "Investment",
        "Net Income",
        "Net-interest Income",
        "Non-interest Income",
        "NPAs"
    ]

    # Compute correlation matrix
    corr = data[vars_11].corr(method="pearson")

    # Mask upper triangle, keep lower triangle + diagonal
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    out_dir = get_analysis_output_dir("descriptive")
    output_file = out_dir / "correlation_heatmap.png"

    # Plot
    plt.figure(figsize=(12, 10))
    plt.title('Correlation Heatmap of Bank Variables', fontsize=18)
    sns.set_style("white")


    ax = sns.heatmap(corr, mask=mask, annot=True, fmt=".3f", cmap="coolwarm", square=True, linewidths=0.8, linecolor="white",
        cbar_kws={"shrink": 0.8}, annot_kws={"size": 16})

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14, width=2, length=6)
    cbar.set_label('Pearson Correlation Coefficient', fontsize=16, labelpad=16)  
    cbar.outline.set_linewidth(1.5)  # Colorbar border thickness


    # Axis formatting
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=16)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved correlation heatmap to: {output_file}")