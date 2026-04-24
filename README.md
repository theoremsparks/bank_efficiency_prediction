# Bank Efficiency Class Prediction App

This project combines Data Envelopment Analysis (DEA) and machine learning to predict the managerial efficiency class of a bank. It includes the full research pipeline, model comparison workflow, and a deployed Streamlit app for bank prediction.

## Features

- Two-stage DEA processing
- Descriptive and decision-matrix analysis
- Multiple ML models and optimized variants
- Model comparison using Copeland-based ranking
- Streamlit app for interactive prediction

## Predicted Classes

- Network Leaders
- Transformation Specialists
- Network Laggards
- Funding-Rich Underperformers

## Project Structure

```text
bank_efficiency_prediction/
│
├── .gitignore
├── app.py
├── .venv/
│
├── data/
│   ├── raw/
│   │   └── data.csv
│   └── processed/
│       └── dea_data_for_ml.csv
│
├── models/
│   └── ml/
│       ├── xgboost/
│       ├── xgboost_optuna/
│       ├── random_forest/
│       ├── random_forest_optuna/
│       ├── lightgbm/
│       ├── lightgbm_optuna/
│       ├── catboost/
│       ├── catboost_optuna/
│       ├── extra_trees/
│       ├── extra_trees_optuna/
│       ├── adaboost/
│       ├── adaboost_optuna/
│       ├── bagging/
│       ├── bagging_optuna/
│       ├── voting/
│       ├── voting_optimized/
│       ├── stacking/
│       └── stacking_optimized/
│
├── outputs/
│   ├── analysis/
│   │   ├── descriptive/
│   │   ├── boxplots/
│   │   └── decision_matrix/
│   ├── model_comparison/
│   └── ml/
│       ├── xgboost/
│       ├── xgboost_optuna/
│       ├── random_forest/
│       ├── random_forest_optuna/
│       ├── lightgbm/
│       ├── lightgbm_optuna/
│       ├── catboost/
│       ├── catboost_optuna/
│       ├── extra_trees/
│       ├── extra_trees_optuna/
│       ├── adaboost/
│       ├── adaboost_optuna/
│       ├── bagging/
│       ├── bagging_optuna/
│       ├── voting/
│       ├── voting_optimized/
│       ├── stacking/
│       └── stacking_optimized/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── compare_models.py
│   ├── run_dea.py
│   ├── run_analysis.py
│   ├── run_pipeline.py
│   ├── run_full_pipeline.py
│   │
│   ├── dea/
│   │   ├── __init__.py
│   │   └── compute_dea.py
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── descriptive_stats.py
│   │   ├── correlation_analysis.py
│   │   ├── efficiency_boxplots.py
│   │   └── decision_matrix.py
│   │
│   └── ml/
│       ├── __init__.py
│       ├── train_xgboost.py
│       ├── train_xgboost_optuna.py
│       ├── train_random_forest.py
│       ├── train_random_forest_optuna.py
│       ├── train_lightgbm.py
│       ├── train_lightgbm_optuna.py
│       ├── train_catboost.py
│       ├── train_catboost_optuna.py
│       ├── train_extra_trees.py
│       ├── train_extra_trees_optuna.py
│       ├── train_adaboost.py
│       ├── train_adaboost_optuna.py
│       ├── train_bagging.py
│       ├── train_bagging_optuna.py
│       ├── train_voting_classifier.py
│       ├── train_voting_optimized_classifier.py
│       ├── train_stacking_classifier.py
│       └── train_stacking_optimized_classifier.py
│
├── requirements.txt
└── README.md

Run Locally

Install dependencies:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run app.py

Pipeline Commands

Run DEA only:
python -m src.run_dea

Run DEA and analysis:
python -m src.run_pipeline

Run the full pipeline:
python -m src.run_full_pipeline

Compare models:
python -m src.compare_models

Deployment
Streamlit App: https://bankefficiencyprediction.streamlit.app/
GitHub Repo: https://github.com/theoremsparks/bank_efficiency_prediction