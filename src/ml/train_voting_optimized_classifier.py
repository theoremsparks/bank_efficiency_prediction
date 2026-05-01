import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import VotingClassifier
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

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.config import DEA_PROCESSED_PATH, get_model_output_dir, get_model_dir

# ==========================================================
# Set Directory
# ==========================================================
OUTPUT_DIR = get_model_output_dir("voting_optimized_classifier")
MODEL_DIR = get_model_dir("voting_optimized_classifier")

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
    print("RUNNING VOTING OPTIMIZED CLASSIFIER TRAINING")
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
        "Funding-Rich Underperformers": "#e7298a",
    }

    palette_colors = [custom_colors[class_name] for class_name in class_counts.index]

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=class_counts.index,
        y=class_counts.values,
        hue=class_counts.index,
        palette=palette_colors,
        legend=False,
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
        "Lagged NPAs",
        "Performing Loans",
        "Investment",
        "Net Income",
        "Net-interest Income",
        "Non-interest Income",
        "NPAs",
    ]

    df_model = df.dropna(subset=feature_cols + ["Class"]).copy()

    X = df_model[feature_cols]
    y_original = df_model["Class"]
    y = y_original - 1   # convert 1,2,3,4 to 0,1,2,3

    # ==========================================================
    # Train-test split
    # ==========================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # ==========================================================
    # Base models
    # ==========================================================
    xgb_model = XGBClassifier(
        n_estimators=1421,
        max_depth=10,
        learning_rate=0.421819565879543,
        min_child_weight=6,
        subsample=0.964495024889279,
        colsample_bytree=0.87418490203239,
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    lgbm_model = LGBMClassifier(
        n_estimators=1833,
        learning_rate=0.239648825080941,
        max_depth=5,
        subsample=0.733263852402479,
        colsample_bytree=0.581051855067085,
        objective="multiclass",
        num_class=4,
        random_state=42,
        n_jobs=-1
    )

    cat_model = CatBoostClassifier(
        n_estimators=1076,
        max_depth=5,
        learning_rate=0.103771812635294,
        l2_leaf_reg=0.198475873721383,
        subsample=0.844920991818226,
        bootstrap_type="Bernoulli",
        loss_function="MultiClass",
        random_state=42,
        verbose=0
    )

    # ==========================================================
    # Voting Classifier
    # ==========================================================
    voting_model = VotingClassifier(
        estimators=[
            ("xgb", xgb_model),
            ("lgbm", lgbm_model),
            ("cat", cat_model)
        ],
        voting="soft",
        weights=[1, 1, 1],
        n_jobs=-1
    )

    # ==========================================================
    # 5-Fold Cross-Validation
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
        "roc_auc_ovr": "roc_auc_ovr",
    }

    cv_results = cross_validate(
        voting_model,
        X_train,
        y_train,
        cv=cv,
        scoring=cv_scoring,
        n_jobs=1,
        return_train_score=False,
    )

    cv_metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Macro Precision",
                "Macro Recall",
                "Macro F1",
                "Weighted Precision",
                "Weighted Recall",
                "Weighted F1",
                "Avg AUC",
            ],
            "Mean_CV_Score": [
                cv_results["test_accuracy"].mean(),
                cv_results["test_macro_precision"].mean(),
                cv_results["test_macro_recall"].mean(),
                cv_results["test_macro_f1"].mean(),
                cv_results["test_weighted_precision"].mean(),
                cv_results["test_weighted_recall"].mean(),
                cv_results["test_weighted_f1"].mean(),
                cv_results["test_roc_auc_ovr"].mean(),
            ],
            "Std_CV_Score": [
                cv_results["test_accuracy"].std(),
                cv_results["test_macro_precision"].std(),
                cv_results["test_macro_recall"].std(),
                cv_results["test_macro_f1"].std(),
                cv_results["test_weighted_precision"].std(),
                cv_results["test_weighted_recall"].std(),
                cv_results["test_weighted_f1"].std(),
                cv_results["test_roc_auc_ovr"].std(),
            ],
        }
    )

    print("\n===== 5-Fold Cross-Validation Results (Training Set) =====")
    print(cv_metrics_df)

    cv_metrics_df.to_csv(
        OUTPUT_DIR / "voting_cross_validation_metrics.csv",
        index=False,
    )

    # ==========================================================
    # Fit final model
    # ==========================================================
    voting_model.fit(X_train, y_train)

    # ==========================================================
    # Save final trained model
    # ==========================================================
    joblib.dump(voting_model, MODEL_DIR / "voting_final_model.joblib")
    print("\nFinal voting model saved successfully.")

    # ==========================================================
    # Predictions
    # ==========================================================
    y_pred = voting_model.predict(X_test)
    y_pred_proba = voting_model.predict_proba(X_test)

    y_test_original = y_test + 1
    y_pred_original = y_pred + 1

    # ==========================================================
    # Evaluation
    # ==========================================================
    acc = accuracy_score(y_test_original, y_pred_original)

    macro_precision = precision_score(y_test_original, y_pred_original, average="macro", zero_division=0)
    macro_recall = recall_score(y_test_original, y_pred_original, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test_original, y_pred_original, average="macro", zero_division=0)

    weighted_precision = precision_score(y_test_original, y_pred_original, average="weighted", zero_division=0)
    weighted_recall = recall_score(y_test_original, y_pred_original, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_test_original, y_pred_original, average="weighted", zero_division=0)

    print("\nClassification Report (Voting Classifier):")
    print(
        classification_report(
            y_test_original,
            y_pred_original,
            digits=4,
            target_names=[class_name_map[i] for i in sorted(class_name_map.keys())],
            zero_division=0,
        )
    )

    # ==========================================================
    # Confusion Matrix
    # ==========================================================
    cm = confusion_matrix(y_test_original, y_pred_original, labels=[1, 2, 3, 4])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[class_name_map[i] for i in [1, 2, 3, 4]],
        yticklabels=[class_name_map[i] for i in [1, 2, 3, 4]],
        annot_kws={"size": 14},
    )
    plt.title("Voting Classifier Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=30, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "voting_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ==========================================================
    # Multiclass ROC Curves
    # ==========================================================
    classes = [1, 2, 3, 4]
    y_test_bin = label_binarize(y_test_original, classes=classes)

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
            label=f"{class_name_map[cls]} (AUC = {roc_auc[cls]:.3f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=16)
    plt.ylabel("True Positive Rate", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("Voting Classifier ROC Curves (One-vs-Rest)", fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "voting_multiclass_roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ==========================================================
    # Save evaluation metrics table
    # ==========================================================
    evaluation_metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Macro Precision",
                "Macro Recall",
                "Macro F1",
                "Weighted Precision",
                "Weighted Recall",
                "Weighted F1",
                "Avg AUC",
            ],
            "Value": [
                acc,
                macro_precision,
                macro_recall,
                macro_f1,
                weighted_precision,
                weighted_recall,
                weighted_f1,
                avg_auc,
            ],
        }
    )

    print("\nEvaluation Metrics Table:")
    print(evaluation_metrics_df)

    evaluation_metrics_df.to_csv(
        OUTPUT_DIR / "voting_evaluation_metrics.csv",
        index=False,
    )

    # ==========================================================
    # Save classified dataset
    # ==========================================================
    df_model.to_csv(OUTPUT_DIR / "dea_multiclass_dataset.csv", index=False)

    # ==========================================================
    # Save predictions
    # ==========================================================
    pred_df = X_test.copy()
    pred_df["True_Class"] = y_test_original
    pred_df["Predicted_Class"] = y_pred_original
    pred_df["True_Class_Name"] = pred_df["True_Class"].map(class_name_map)
    pred_df["Predicted_Class_Name"] = pred_df["Predicted_Class"].map(class_name_map)
    pred_df.to_csv(OUTPUT_DIR / "voting_test_predictions.csv", index=False)

    # ==========================================================
    # Save ROC AUC table
    # ==========================================================
    roc_auc_df = pd.DataFrame(
        {
            "Class": classes,
            "Class_Name": [class_name_map[c] for c in classes],
            "AUC": [roc_auc[c] for c in classes],
        }
    )
    roc_auc_df.to_csv(OUTPUT_DIR / "voting_multiclass_roc_auc.csv", index=False)

    print(f"\nSaved ML outputs to: {OUTPUT_DIR}")
    print(f"Saved model to: {MODEL_DIR}")

    # ==========================================================
    # SHAP Analysis for Voting Classifier
    # ==========================================================
    background_size = min(100, len(X_train))
    background = X_train.sample(n=background_size, random_state=42)
    X_shap = X_test.copy()

    # Build KernelExplainer on the ensemble probability output
    explainer = shap.KernelExplainer(voting_model.predict_proba, background)
    shap_values = explainer.shap_values(X_shap, nsamples=100)

    # ==========================================================
    # Map SHAP outputs to original class labels
    # ==========================================================
    # Internal classes used by the fitted voting model
    model_classes = list(voting_model.classes_)   # expected: [0, 1, 2, 3]

    # Map internal labels back to original labels used in reporting
    internal_to_original = {0: 1, 1: 2, 2: 3, 3: 4}

    # Final class labels for reporting
    classes = [1, 2, 3, 4]

    if isinstance(shap_values, list):
        # common multiclass format: one array per internal class
        shap_by_class = {
            internal_to_original[internal_cls]: shap_values[i]
            for i, internal_cls in enumerate(model_classes)
        }
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # shape: (n_samples, n_features, n_internal_classes)
        shap_by_class = {
            internal_to_original[internal_cls]: shap_values[:, :, i]
            for i, internal_cls in enumerate(model_classes)
        }
    else:
        raise ValueError("Unexpected SHAP output format. Please inspect shap_values.")

    # Optional sanity check
    print("Voting model internal classes:", model_classes)
    print("SHAP mapped to original classes:", list(shap_by_class.keys()))

    # ==========================================================
    # Stacked mean absolute SHAP importance table and plot
    # ==========================================================
    mean_abs_shap_df = pd.DataFrame({"Feature": feature_cols})

    for cls in classes:
        mean_abs_shap_df[class_name_map[cls]] = np.abs(shap_by_class[cls]).mean(axis=0)

    mean_abs_shap_df["Total"] = mean_abs_shap_df[[class_name_map[c] for c in classes]].sum(axis=1)
    mean_abs_shap_df = mean_abs_shap_df.sort_values(by="Total", ascending=True)

    # Save SHAP importance table - using OUTPUT_DIR (Path object)
    mean_abs_shap_df.to_csv(
        OUTPUT_DIR / "voting_shap_stacked_mean_importance.csv",
        index=False
    )

    # Save stacked SHAP plot
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

    plt.title("Stacked mean |SHAP value| across classes", fontsize=16)
    plt.xlabel("mean(|SHAP value|) (average impact on model output magnitude)", fontsize=16)
    plt.ylabel("Feature", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(title="Class", loc="lower right", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "voting_shap_stacked_mean_importance.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(f"\nSHAP analysis saved to: {OUTPUT_DIR}")
    print("=" * 70)
    print("VOTING OPTIMIZED TRAINING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
