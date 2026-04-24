import pandas as pd
from src.config import RAW_DATA_PATH, get_analysis_output_dir


def run_descriptive_stats():
    data = pd.read_csv(RAW_DATA_PATH)

    summary_vars = [
        "Assets",
        "Employee Expense",
        "Equity",
        "Deposits",
        "Borrowings",
        "NPAs (Previous Period)",
        "Performing Loans",
        "Investment",
        "Net Income",
        "Net-interest Income",
        "Non-interest Income",
        "NPAs"
    ]

    overall_summary = pd.DataFrame({
        "Variable": summary_vars,
        "Count": [data[col].count() for col in summary_vars],
        "Mean": [data[col].mean() for col in summary_vars],
        "Std_Dev": [data[col].std() for col in summary_vars],
        "Variance": [data[col].var() for col in summary_vars],
        "Min": [data[col].min() for col in summary_vars],
        "Max": [data[col].max() for col in summary_vars],
        "Skewness": [data[col].skew() for col in summary_vars],
        "Kurtosis": [data[col].kurt() for col in summary_vars],
    }).round(3)

    out_dir = get_analysis_output_dir("descriptive")
    output_file = out_dir / "statistical_summary.csv"
    overall_summary.to_csv(output_file, index=False)

    print(f"Saved descriptive statistics to: {output_file}")