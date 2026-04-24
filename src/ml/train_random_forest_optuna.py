import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import optuna
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
    auc,
    precision_score,
    recall_score,
)

from sklearn.preprocessing import label_binarize

from src.config import DEA_PROCESSED_PATH, get_model_output_dir, get_model_dir

OUTPUT_DIR = get_model_output_dir("random_forest_optuna")
MODEL_DIR = get_model_dir("random_forest_optuna")


def assign_class(row):
    e1 = row["Stage1_Efficiency"]
    e2 = row["Stage2_Efficiency"]
    e1_bar = row["Stage1_Mean"]
    e2_bar = row["Stage2_Mean"]

    if e1 >= e1_bar and e2 >= e2_bar:
        return 1  # Network Leaders
    elif e1 < e1_bar and e2 >= e2_bar:
        return 2  # Transformation Specialists
    elif e1 < e1_bar and e2 < e2_bar:
        return 3  # Network Laggards
    else:
        return 4  # Funding-Rich Underperformers

def main():
    print("=" * 70)
    print("RUNNING RANDOM FOREST OPTUNA TRAINING")
    print("=" * 70)

    # ==========================================================
    # Load DEA-processed data
    # ==========================================================
    df = pd.read_csv(DEA_PROCESSED_PATH)
    df.columns = df.columns.str.strip()

    df = df.dropna(subset=["Year", "Stage1_Efficiency", "Stage2_Efficiency"]).copy()
    df["Year"] = df["Year"].astype(int)

    # ==========================================================
    # Compute yearly mean-based class labels
    # ==========================================================
    yearly_means = df.groupby("Year").agg(
        Stage1_Mean=("Stage1_Efficiency", "mean"),
        Stage2_Mean=("Stage2_Efficiency", "mean"),
    ).reset_index()

    df = df.merge(yearly_means, on="Year", how="left")
    df["Class"] = df.apply(assign_class, axis=1)

    class_name_map = {
        1: "Network Leaders",
        2: "Transformation Specialists",
        3: "Network Laggards",
        4: "Funding-Rich Underperformers",
    }
    df["Class_Name"] = df["Class"].map(class_name_map)

    print("\nClass distribution:")
    print(df["Class_Name"].value_counts().sort_index())
    
    # ==========================================================
    # Plot class distribution
    # ==========================================================
    class_counts = df["Class_Name"].value_counts().sort_index()

    custom_colors = {
        "Network Leaders": "#1b9e77",
        "Transformation Specialists": "#7570b3",
        "Network Laggards": "#d95f02",
        "Funding-Rich Underperformers": "#e7298a"
    }

    palette_colors = [custom_colors[class_name] for class_name in class_counts.index]

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=class_counts.index,
        y=class_counts.values,
        hue=class_counts.index,
        palette=palette_colors,
        legend=False
    )

    plt.title("Distribution of Class Names", fontsize=14)
    plt.xlabel("Class Name", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(rotation=30, ha="right", fontsize=14)
    plt.yticks(fontsize=14)

    for i, v in enumerate(class_counts.values):
        plt.text(i, v + 0.5, str(v), ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ==========================================================
    # Define features
    # ==========================================================
    feature_cols = [
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
        "NPAs",
    ]

    # Keep only rows with complete feature values
    df_model = df.dropna(subset=feature_cols + ["Class"]).copy()

    X = df_model[feature_cols]
    y = df_model["Class"]

    # ==========================================================
    # Train-test split
    # ==========================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # ==========================================================
    # 5-Fold Cross-Validation setup
    # ==========================================================
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scoring = {
        "accuracy": "accuracy",
        "macro_precision": "precision_macro",
        "macro_recall": "recall_macro",
        "macro_f1": "f1_macro",
        "weighted_precision": "precision_weighted",
        "weighted_recall": "recall_weighted",
        "weighted_f1": "f1_weighted",
        "roc_auc_ovr": "roc_auc_ovr"
    }

    # ==========================================================
    # Optuna Objective Function
    # ==========================================================
    def objective(trial):
        rf_model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 10, 500),
            max_depth=trial.suggest_int("max_depth", 5, 30),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 8),
            random_state=42,
            n_jobs=1
        )

        cv_results = cross_validate(
            rf_model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            return_train_score=False,
            error_score="raise"
        )

        return cv_results["test_score"].mean()

    # ==========================================================
    # Run Optuna Optimization
    # ==========================================================
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        study_name="rf_multiclass_optuna",
        sampler=sampler
    )
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print("\nBest Trial:")
    print(f"  Value (Macro F1): {study.best_value:.6f}")
    print("  Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # Save Optuna trials
    optuna_trials_df = study.trials_dataframe()
    optuna_trials_df.to_csv(
        OUTPUT_DIR / "rf_optuna_trials.csv",
        index=False
    )

    # Save best params
    best_params_df = pd.DataFrame({
        "Parameter": list(study.best_params.keys()),
        "Value": list(study.best_params.values())
    })
    best_params_df.to_csv(
        OUTPUT_DIR / "rf_best_params.csv",
        index=False
    )

    # ==========================================================
    # Train best Random Forest classifier
    # ==========================================================
    rf_model = RandomForestClassifier(
        **study.best_params,
        random_state=42,
        n_jobs=-1
    )

    # ==========================================================
    # Fit Final Model on Full Training Set
    # ==========================================================
    rf_model.fit(X_train, y_train)

    # ==========================================================
    # Save Final Trained Model
    # ==========================================================
    joblib.dump(rf_model, MODEL_DIR / "rf_final_optimized_model.joblib")
    print("\nFinal model saved successfully.")

    # ==========================================================
    # Predictions
    # ==========================================================
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)

    # ==========================================================
    # Evaluation
    # ==========================================================
    acc = accuracy_score(y_test, y_pred)

    macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    weighted_precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    weighted_recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\nClassification Report (Random Forest Classifier):")
    print(classification_report(
        y_test,
        y_pred,
        digits=4,
        target_names=[class_name_map[i] for i in sorted(class_name_map.keys())],
        zero_division=0
    ))

    # ==========================================================
    # Confusion Matrix
    # ==========================================================
    cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[class_name_map[i] for i in [1, 2, 3, 4]],
        yticklabels=[class_name_map[i] for i in [1, 2, 3, 4]],
        annot_kws={"size": 14}
    )
    plt.title("Random Forest Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=30, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rf_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ==========================================================
    # Multiclass ROC Curves (One-vs-Rest)
    # ==========================================================
    classes = [1, 2, 3, 4]
    y_test_bin = label_binarize(y_test, classes=classes)

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i, cls in enumerate(classes):
        fpr[cls], tpr[cls], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[cls] = auc(fpr[cls], tpr[cls])

    avg_auc = np.mean([roc_auc[cls] for cls in classes])

    print(f"Avg AUC             : {avg_auc:.4f}")

    plt.figure(figsize=(9, 7))
    colors = ["#1b9e77", "#7570b3", "#d95f02", "#e7298a"]

    for i, cls in enumerate(classes):
        plt.plot(
            fpr[cls],
            tpr[cls],
            color=colors[i],
            lw=2,
            label=f"{class_name_map[cls]} (AUC = {roc_auc[cls]:.3f})"
        )

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=16)
    plt.ylabel("True Positive Rate", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("Multiclass ROC Curves (One-vs-Rest)", fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rf_multiclass_roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ==========================================================
    # Save evaluation metrics table
    # ==========================================================
    evaluation_metrics_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1",
            "Weighted Precision",
            "Weighted Recall",
            "Weighted F1",
            "Avg AUC"
        ],
        "Value": [
            acc,
            macro_precision,
            macro_recall,
            macro_f1,
            weighted_precision,
            weighted_recall,
            weighted_f1,
            avg_auc
        ]
    })

    print("\nEvaluation Metrics Table:")
    print(evaluation_metrics_df)

    evaluation_metrics_df.to_csv(
        OUTPUT_DIR / "rf_evaluation_metrics.csv",
        index=False
    )

    # ==========================================================
    # Feature Importance
    # ==========================================================
    feature_importance = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\nTop Feature Importances:")
    print(feature_importance)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=feature_importance,
        x="Importance",
        y="Feature",
        hue="Feature",
        legend=False
    )
    plt.title("Feature Importance (Aggregate)", fontsize=16)
    plt.xlabel("Aggregate Importance", fontsize=16)
    plt.ylabel("Feature", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rf_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ==========================================================
    # SHAP Analysis
    # ==========================================================
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_by_class = {cls: shap_values[i] for i, cls in enumerate(classes)}
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_by_class = {cls: shap_values[:, :, i] for i, cls in enumerate(classes)}
    else:
        raise ValueError("Unexpected SHAP output format. Please inspect shap_values.")

    # ==========================================================
    # Stacked mean absolute SHAP importance table and plot
    # ==========================================================
    mean_abs_shap_df = pd.DataFrame({"Feature": feature_cols})

    for cls in classes:
        mean_abs_shap_df[class_name_map[cls]] = np.abs(shap_by_class[cls]).mean(axis=0)

    mean_abs_shap_df["Total"] = mean_abs_shap_df[[class_name_map[c] for c in classes]].sum(axis=1)
    mean_abs_shap_df = mean_abs_shap_df.sort_values(by="Total", ascending=True)

    mean_abs_shap_df.to_csv(
        OUTPUT_DIR / "rf_shap_stacked_mean_importance.csv",
        index=False
    )

    plt.figure(figsize=(11, 7))
    left = np.zeros(len(mean_abs_shap_df))

    stack_colors = {
        "Network Leaders": "#1b9e77",
        "Transformation Specialists": "#7570b3",
        "Network Laggards": "#d95f02",
        "Funding-Rich Underperformers": "#e7298a"
    }

    for cls_name in [class_name_map[c] for c in classes]:
        plt.barh(
            mean_abs_shap_df["Feature"],
            mean_abs_shap_df[cls_name],
            left=left,
            label=cls_name,
            color=stack_colors[cls_name],
            alpha=0.9
        )
        left += mean_abs_shap_df[cls_name].values

    plt.title("Stacked mean absolute SHAP importance across classes", fontsize=16)
    plt.xlabel("mean(|SHAP value|) (average impact on model output magnitude)", fontsize=16)
    plt.ylabel("Feature", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(title="Class", loc="lower right", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "rf_shap_stacked_mean_importance.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # ==========================================================
    # Save classified dataset
    # ==========================================================
    df_model.to_csv(OUTPUT_DIR / "dea_multiclass_dataset.csv", index=False)

    # ==========================================================
    # Save predictions
    # ==========================================================
    pred_df = X_test.copy()
    pred_df["True_Class"] = y_test.values
    pred_df["Predicted_Class"] = y_pred
    pred_df["True_Class_Name"] = pred_df["True_Class"].map(class_name_map)
    pred_df["Predicted_Class_Name"] = pred_df["Predicted_Class"].map(class_name_map)
    pred_df.to_csv(OUTPUT_DIR / "rf_test_predictions.csv", index=False)

    # ==========================================================
    # Save feature importance
    # ==========================================================
    feature_importance.to_csv(OUTPUT_DIR / "rf_feature_importance.csv", index=False)

    # ==========================================================
    # Save ROC AUC table
    # ==========================================================
    roc_auc_df = pd.DataFrame({
        "Class": classes,
        "Class_Name": [class_name_map[c] for c in classes],
        "AUC": [roc_auc[c] for c in classes]
    })
    roc_auc_df.to_csv(OUTPUT_DIR / "rf_multiclass_roc_auc.csv", index=False)

    print(f"\nSaved ML outputs to: {OUTPUT_DIR}")
    print("=" * 70)
    print("RANDOM FOREST OPTIMIZED TRAINING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()