from src.analysis.descriptive_stats import run_descriptive_stats
from src.analysis.correlation_analysis import run_correlation_heatmap
from src.analysis.efficiency_boxplots import run_efficiency_boxplots
from src.analysis.decision_matrix import run_decision_matrix


def main():
    print("Running descriptive statistics analysis...")
    run_descriptive_stats()

    print("Running correlation heatmap analysis...")
    run_correlation_heatmap()

    print("Running efficiency boxplots...")
    run_efficiency_boxplots()

    print("Running decision matrix analysis...")
    run_decision_matrix()

    print("Analysis completed successfully.")


if __name__ == "__main__":
    main()