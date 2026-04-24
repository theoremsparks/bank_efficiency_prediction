import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import DEA_PROCESSED_PATH, get_analysis_output_dir


def add_median_labels(ax, data, year_col, value_col):
    """
    Add median value labels to each yearly boxplot.
    This preserves the behaviour of the original script.
    """
    medians = data.groupby(year_col)[value_col].median()

    for i, (year, median_val) in enumerate(medians.items()):
        ax.text(
            i,
            median_val,
            f"{median_val:.3f}",
            ha="center",
            va="center",
            fontsize=14,
            color="black",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.7,
                boxstyle="round,pad=0.2"
            )
        )


def run_efficiency_boxplots():
    # ============================
    # Load DEA results
    # ============================
    df = pd.read_csv(DEA_PROCESSED_PATH)
    df.columns = df.columns.str.strip()

    required_cols = [
        "Year",
        "Stage1_Efficiency",
        "Stage2_Efficiency",
        "Overall_Efficiency",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing required DEA columns for boxplots: {missing_cols}"
        )

    # ============================
    # Same preprocessing as original
    # ============================
    df = df.dropna(
        subset=[
            "Year",
            "Stage1_Efficiency",
            "Stage2_Efficiency",
            "Overall_Efficiency"
        ]
    ).copy()

    # Ensure Year is sorted properly
    df["Year"] = df["Year"].astype(int)
    df = df.sort_values("Year")

    # Convert Year to string for categorical plotting
    df["Year"] = df["Year"].astype(str)

    # ============================
    # Output folder
    # ============================
    out_dir = get_analysis_output_dir("boxplots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ============================
    # Same style as original
    # ============================
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)

    # Same colour palette as original
    palette = sns.color_palette("Set2", n_colors=df["Year"].nunique())

    # ==========================================================
    # Plot definitions
    # Same labels and filenames as original
    # ==========================================================
    plots = [
        {
            "column": "Stage1_Efficiency",
            "ylabel": "Intermediation Efficiency",
            "filename": "boxplot_stage1_efficiency_by_year.png"
        },
        {
            "column": "Stage2_Efficiency",
            "ylabel": "Operating Efficiency",
            "filename": "boxplot_stage2_efficiency_by_year.png"
        },
        {
            "column": "Overall_Efficiency",
            "ylabel": "Overall Efficiency",
            "filename": "boxplot_overall_efficiency_by_year.png"
        }
    ]

    for plot in plots:
        col = plot["column"]
        ylabel = plot["ylabel"]
        filename = plot["filename"]

        plt.figure(figsize=(11, 6.5))

        ax = sns.boxplot(
            data=df,
            x="Year",
            y=col,
            hue="Year",
            palette=palette,
            width=0.6,
            linewidth=1.2,
            legend=False
        )

        plt.xlabel("Year", fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.xticks(rotation=45, fontsize=16)
        plt.yticks(fontsize=16)

        # Add median labels
        add_median_labels(
            ax=ax,
            data=df,
            year_col="Year",
            value_col=col
        )

        plt.tight_layout()

        plot_file = out_dir / filename

        plt.savefig(
            plot_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved: {plot_file}")


if __name__ == "__main__":
    run_efficiency_boxplots()