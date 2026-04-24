import pandas as pd
from itertools import combinations
from src.config import OUTPUTS_DIR

COMPARISON_DIR = OUTPUTS_DIR / "model_comparison"
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)



MODEL_FILES = {
    "AdaBoost Optuna": OUTPUTS_DIR / "ml" / "adaboost_optuna" / "ada_evaluation_metrics.csv",
    "Bagging Optuna": OUTPUTS_DIR / "ml" / "bagging_optuna" / "bag_evaluation_metrics.csv",
    "CatBoost Optuna": OUTPUTS_DIR / "ml" / "catboost_optuna" / "cat_evaluation_metrics.csv",
    "Extra Trees Optuna": OUTPUTS_DIR / "ml" / "extra_trees_optuna" / "et_evaluation_metrics.csv",
    "LightGBM Optuna": OUTPUTS_DIR / "ml" / "lightgbm_optuna" / "lgbm_evaluation_metrics.csv",
    "Random Forest Optuna": OUTPUTS_DIR / "ml" / "random_forest_optuna" / "rf_evaluation_metrics.csv",
    "Stacking Classifier": OUTPUTS_DIR / "ml" / "stacking_optimized_classifier" / "stacking_evaluation_metrics.csv",
    "Voting Classifier": OUTPUTS_DIR / "ml" / "voting_optimized_classifier" / "voting_evaluation_metrics.csv",
    "XGBoost Optuna": OUTPUTS_DIR / "ml" / "xgboost_optuna" / "xgb_evaluation_metrics.csv",
}

METRICS = [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Weighted Precision",
    "Weighted Recall",
    "Weighted F1",
    "Avg AUC",
]


def load_model_metrics(model_name, file_path):
    if not file_path.exists():
        print(f"[WARNING] File not found for {model_name}: {file_path}")
        return None

    df = pd.read_csv(file_path)

    if "Metric" not in df.columns or "Value" not in df.columns:
        print(f"[WARNING] Invalid file format for {model_name}: {file_path}")
        return None

    row = {"Model": model_name}

    for _, r in df.iterrows():
        metric_name = str(r["Metric"]).strip()
        metric_value = pd.to_numeric(r["Value"], errors="coerce")

        if metric_name in METRICS:
            row[metric_name] = metric_value

    return row


def build_combined_metrics_table():
    rows = []

    for model_name, file_path in MODEL_FILES.items():
        row = load_model_metrics(model_name, file_path)
        if row is not None:
            rows.append(row)

    if not rows:
        raise ValueError(
            "No valid model evaluation files were found. "
            "Check your outputs/ml/<model_name>/ folders and metric filenames."
        )

    combined_df = pd.DataFrame(rows)

    ordered_cols = ["Model"] + [m for m in METRICS if m in combined_df.columns]
    combined_df = combined_df[ordered_cols]

    return combined_df


def round_metrics_for_comparison(df, decimals=3):
    rounded_df = df.copy()

    for metric in METRICS:
        if metric in rounded_df.columns:
            rounded_df[metric] = pd.to_numeric(
                rounded_df[metric], errors="coerce"
            ).round(decimals)

    return rounded_df


def add_metric_ranks(comparison_df):
    ranked_df = comparison_df.copy()

    for metric in METRICS:
        if metric in ranked_df.columns:
            ranked_df[f"Rank_{metric}"] = ranked_df[metric].rank(
                ascending=False,
                method="min"
            ).astype("Int64")

    return ranked_df


def compute_copeland_scores(comparison_df):
    model_names = comparison_df["Model"].tolist()

    stats = {
        model: {
            "Wins": 0,
            "Losses": 0,
            "Ties": 0,
            "Copeland Score": 0
        }
        for model in model_names
    }

    pairwise_records = []

    for model_a, model_b in combinations(model_names, 2):
        row_a = comparison_df[comparison_df["Model"] == model_a].iloc[0]
        row_b = comparison_df[comparison_df["Model"] == model_b].iloc[0]

        for metric in METRICS:
            if metric not in comparison_df.columns:
                continue

            value_a = row_a[metric]
            value_b = row_b[metric]

            if pd.isna(value_a) or pd.isna(value_b):
                continue

            if value_a > value_b:
                stats[model_a]["Wins"] += 1
                stats[model_b]["Losses"] += 1
                stats[model_a]["Copeland Score"] += 1
                stats[model_b]["Copeland Score"] -= 1
                outcome = f"{model_a} wins"

            elif value_a < value_b:
                stats[model_b]["Wins"] += 1
                stats[model_a]["Losses"] += 1
                stats[model_b]["Copeland Score"] += 1
                stats[model_a]["Copeland Score"] -= 1
                outcome = f"{model_b} wins"

            else:
                stats[model_a]["Ties"] += 1
                stats[model_b]["Ties"] += 1
                outcome = "Draw"

            pairwise_records.append({
                "Model_A": model_a,
                "Model_B": model_b,
                "Metric": metric,
                "Value_A": value_a,
                "Value_B": value_b,
                "Outcome": outcome
            })

    copeland_df = pd.DataFrame([
        {
            "Model": model,
            "Wins": stats[model]["Wins"],
            "Losses": stats[model]["Losses"],
            "Ties": stats[model]["Ties"],
            "Copeland Score": stats[model]["Copeland Score"]
        }
        for model in model_names
    ])

    copeland_df = copeland_df.sort_values(
        by=["Copeland Score", "Wins", "Ties"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    copeland_df["Rank"] = range(1, len(copeland_df) + 1)

    pairwise_df = pd.DataFrame(pairwise_records)

    return copeland_df, pairwise_df


def main():
    print("=" * 80)
    print("MODEL COMPARISON USING EVALUATION METRICS + COPELAND")
    print("=" * 80)

    # Step 1: Raw metrics
    raw_df = build_combined_metrics_table()
    raw_df.to_csv(COMPARISON_DIR / "all_model_metrics_comparison_raw.csv", index=False)

    print("\n[1] Raw Combined Metrics Table")
    print(raw_df)

    # Step 2: Rounded metrics
    rounded_df = round_metrics_for_comparison(raw_df, decimals=3)
    rounded_df.to_csv(COMPARISON_DIR / "all_model_metrics_comparison_rounded_3dp.csv", index=False)

    print("\n[2] Rounded Combined Metrics Table (3 d.p.)")
    print(rounded_df)

    # Step 3: Per-metric ranks
    ranked_df = add_metric_ranks(rounded_df)
    ranked_df.to_csv(COMPARISON_DIR / "all_model_metrics_with_ranks_3dp.csv", index=False)

    print("\n[3] Rounded Metrics Table with Per-Metric Ranks")
    print(ranked_df)

    # Step 4: Copeland
    copeland_df, pairwise_df = compute_copeland_scores(rounded_df)
    copeland_df.to_csv(COMPARISON_DIR / "copeland_ranking_table_3dp.csv", index=False)
    pairwise_df.to_csv(COMPARISON_DIR / "copeland_pairwise_results_3dp.csv", index=False)

    print("\n[4] Copeland Ranking Table (based on 3 d.p. rounded metrics)")
    print(copeland_df)

    # Step 5: Best model summary
    best_model = copeland_df.iloc[0]["Model"]
    best_score = copeland_df.iloc[0]["Copeland Score"]

    summary_text = (
        f"Best model by Copeland ranking (3 d.p. rounded metrics): {best_model}\n"
        f"Copeland Score: {best_score}\n"
    )

    summary_file = COMPARISON_DIR / "best_model_summary_3dp.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\n[5] Best Model Summary")
    print(summary_text)

    print(f"Saved all comparison outputs to: {COMPARISON_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()