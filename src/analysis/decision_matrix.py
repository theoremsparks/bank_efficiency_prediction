import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import DEA_PROCESSED_PATH, get_analysis_output_dir


# ============================================================
# Function to assign quadrant class
# Same logic as original code
# ============================================================
def assign_class(e1, e2, mean_e1, mean_e2):
    if e1 >= mean_e1 and e2 >= mean_e2:
        return 1, "Network Leaders"
    elif e1 < mean_e1 and e2 >= mean_e2:
        return 2, "Transformation Specialists"
    elif e1 < mean_e1 and e2 < mean_e2:
        return 3, "Network Laggards"
    else:
        return 4, "Funding-Rich Underperformers"


# ============================================================
# Main decision matrix function
# ============================================================
def run_decision_matrix():
    # ============================
    # Load DEA results
    # ============================
    df = pd.read_csv(DEA_PROCESSED_PATH)
    df.columns = df.columns.str.strip()

    required_cols = ["Year", "Stage1_Efficiency", "Stage2_Efficiency"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing required DEA columns for decision matrix: {missing_cols}"
        )

    # Same as original: keep valid rows only
    df = df.dropna(
        subset=["Year", "Stage1_Efficiency", "Stage2_Efficiency"]
    ).copy()

    # Same as original: force year to integer
    df["Year"] = df["Year"].astype(int)

    # ============================
    # Output folders
    # ============================
    output_dir = get_analysis_output_dir("decision_matrix")
    quadrant_csv_dir = output_dir / "quadrant_csvs"

    output_dir.mkdir(parents=True, exist_ok=True)
    quadrant_csv_dir.mkdir(parents=True, exist_ok=True)

    # Same visual style as original
    sns.set_theme(style="white")

    # ============================
    # Visual mapping by quadrant
    # Same colors and marker shapes as original
    # ============================
    quadrant_styles = {
        1: {
            "name": "Network Leaders",
            "color": "#1b9e77",
            "marker": "o"
        },
        2: {
            "name": "Transformation Specialists",
            "color": "#7570b3",
            "marker": "^"
        },
        3: {
            "name": "Network Laggards",
            "color": "#d95f02",
            "marker": "s"
        },
        4: {
            "name": "Funding-Rich Underperformers",
            "color": "#e7298a",
            "marker": "D"
        }
    }

    class_order = [
        "Network Leaders",
        "Transformation Specialists",
        "Network Laggards",
        "Funding-Rich Underperformers"
    ]

    all_yearly_data = []
    all_class_counts = []

    # ============================
    # Create one matrix per year
    # ============================
    years = sorted(df["Year"].unique())

    for yr in years:
        sub = df[df["Year"] == yr].copy()

        mean_e1 = sub["Stage1_Efficiency"].mean()
        mean_e2 = sub["Stage2_Efficiency"].mean()

        # Add yearly means to match your production dataset structure
        sub["Stage1_Mean"] = mean_e1
        sub["Stage2_Mean"] = mean_e2

        # Assign classes
        sub[["Class", "Class_Name"]] = sub.apply(
            lambda row: pd.Series(
                assign_class(
                    row["Stage1_Efficiency"],
                    row["Stage2_Efficiency"],
                    mean_e1,
                    mean_e2
                )
            ),
            axis=1
        )

        # Keep compatibility with your original naming
        sub["Quadrant_Name"] = sub["Class_Name"]

        all_yearly_data.append(sub)

        # ============================
        # Save whole classified yearly data
        # Same as original code
        # ============================
        yearly_file = output_dir / f"decision_matrix_classes_{yr}.csv"
        sub.to_csv(yearly_file, index=False)
        print(f"Saved yearly classified data: {yearly_file}")

        # ============================
        # Save each quadrant separately
        # Same as original code
        # ============================
        for class_id, style in quadrant_styles.items():
            quad_df = sub[sub["Class"] == class_id].copy()

            quadrant_file = (
                quadrant_csv_dir /
                f"{yr}_{style['name'].replace(' ', '_')}.csv"
            )

            quad_df.to_csv(quadrant_file, index=False)

        # ============================
        # Class counts for the year
        # ============================
        class_counts = (
            sub.groupby(["Year", "Class", "Class_Name"])
            .size()
            .reset_index(name="Count")
        )

        all_class_counts.append(class_counts)

        # ============================
        # Plot limits with padding
        # Same as original code
        # ============================
        x_min = max(0, sub["Stage1_Efficiency"].min() - 0.05)
        x_max = min(1.1, sub["Stage1_Efficiency"].max() + 0.05)
        y_min = max(0, sub["Stage2_Efficiency"].min() - 0.05)
        y_max = min(1.1, sub["Stage2_Efficiency"].max() + 0.05)

        fig, ax = plt.subplots(figsize=(11, 8))

        # ============================
        # Plot each quadrant separately
        # This preserves different marker shapes
        # ============================
        for class_id, style in quadrant_styles.items():
            quad_df = sub[sub["Class"] == class_id]

            if len(quad_df) > 0:
                ax.scatter(
                    quad_df["Stage1_Efficiency"],
                    quad_df["Stage2_Efficiency"],
                    s=85,
                    color=style["color"],
                    marker=style["marker"],
                    alpha=0.9,
                    edgecolor="black",
                    linewidth=0.5,
                    label=style["name"]
                )

        # ============================
        # Mean lines
        # Same as original
        # ============================
        ax.axvline(mean_e1, color="gray", linewidth=1.6)
        ax.axhline(mean_e2, color="gray", linewidth=1.6)

        # ============================
        # Quadrant names at corners
        # Same text placement as original
        # ============================

        # Top-left
        ax.text(
            x_min + 0.015,
            y_max - 0.015,
            "Transformation\nSpecialists",
            fontsize=14,
            ha="left",
            va="top",
            alpha=0.9,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
                pad=2
            )
        )

        # Top-right
        ax.text(
            x_max - 0.015,
            y_max - 0.015,
            "Network\nLeaders",
            fontsize=14,
            ha="right",
            va="top",
            alpha=0.9,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
                pad=2
            )
        )

        # Bottom-left
        ax.text(
            x_min + 0.015,
            y_min + 0.015,
            "Network\nLaggards",
            fontsize=14,
            ha="left",
            va="bottom",
            alpha=0.9,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
                pad=2
            )
        )

        # Bottom-right
        ax.text(
            x_max - 0.015,
            y_min + 0.015,
            "Funding-Rich\nUnderperformers",
            fontsize=14,
            ha="right",
            va="bottom",
            alpha=0.9,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
                pad=2
            )
        )

        # ============================
        # Count boxes in each quadrant
        # Same as original
        # ============================
        count_q1 = (sub["Class"] == 1).sum()
        count_q2 = (sub["Class"] == 2).sum()
        count_q3 = (sub["Class"] == 3).sum()
        count_q4 = (sub["Class"] == 4).sum()

        # Top-left count
        ax.text(
            x_min + 0.015,
            mean_e2 + 0.015,
            f"n = {count_q2}",
            fontsize=14,
            ha="left",
            va="bottom",
            bbox=dict(
                facecolor="#f0f0f0",
                edgecolor="black",
                boxstyle="round,pad=0.25"
            )
        )

        # Top-right count
        ax.text(
            x_max - 0.015,
            mean_e2 + 0.015,
            f"n = {count_q1}",
            fontsize=14,
            ha="right",
            va="bottom",
            bbox=dict(
                facecolor="#f0f0f0",
                edgecolor="black",
                boxstyle="round,pad=0.25"
            )
        )

        # Bottom-left count
        ax.text(
            x_min + 0.015,
            mean_e2 - 0.015,
            f"n = {count_q3}",
            fontsize=14,
            ha="left",
            va="top",
            bbox=dict(
                facecolor="#f0f0f0",
                edgecolor="black",
                boxstyle="round,pad=0.25"
            )
        )

        # Bottom-right count
        ax.text(
            x_max - 0.015,
            mean_e2 - 0.015,
            f"n = {count_q4}",
            fontsize=14,
            ha="right",
            va="top",
            bbox=dict(
                facecolor="#f0f0f0",
                edgecolor="black",
                boxstyle="round,pad=0.25"
            )
        )

        # ============================
        # Axis labels and title
        # Same as original
        # ============================
        ax.set_title(
            f"Managerial Decision Matrix ({yr})",
            fontsize=18,
            pad=15
        )

        ax.set_xlabel(
            "Intermediation Efficiency (Stage 1)",
            fontsize=18
        )

        ax.set_ylabel(
            "Operating Efficiency (Stage 2)",
            fontsize=18
        )

        # Axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Legend
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=2,
            frameon=True,
            fontsize=16
        )

        # Style
        ax.tick_params(axis="both", labelsize=16)
        sns.despine()
        plt.tight_layout()

        # Save figure
        plot_file = output_dir / f"decision_matrix_{yr}.png"

        plt.savefig(
            plot_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved plot: {plot_file}")

    # ============================
    # Save combined classified dataset
    # Useful for ML stage
    # ============================
    classified_df = pd.concat(all_yearly_data, ignore_index=True)

    classified_file = output_dir / "dea_decision_matrix_classified.csv"
    classified_df.to_csv(classified_file, index=False)

    print(f"Saved combined classified decision matrix data: {classified_file}")

    # ============================
    # Save combined yearly class counts
    # ============================
    combined_class_counts = pd.concat(all_class_counts, ignore_index=True)

    combined_class_counts["Class_Name"] = pd.Categorical(
        combined_class_counts["Class_Name"],
        categories=class_order,
        ordered=True
    )

    combined_class_counts = combined_class_counts.sort_values(
        ["Year", "Class"]
    )

    counts_file = output_dir / "decision_matrix_class_counts.csv"
    combined_class_counts.to_csv(counts_file, index=False)

    print(f"Saved class counts: {counts_file}")


if __name__ == "__main__":
    run_decision_matrix()