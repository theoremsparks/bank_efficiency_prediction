# ***Bank Efficiency Class Prediction App***

This repository is an end-to-end machine learning application for predicting the managerial efficiency class of a bank using financial and operational variables derived from a directional distance function Network Data Envelopment Analysis (DDF-NDEA) framework. It translates a research pipeline into an interactive decision-support tool deployed with Streamlit.

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Ensemble-green)
![Optimization](https://img.shields.io/badge/Optimization-DEA-orange)

## ***1. Overview***
The workflow begins with DEA computation and analysis, after which the processed dataset is used for machine learning classification. The final deployed application allows a user to enter the financial values of one bank at a time and receive a predicted managerial class, class probabilities, and a plain-language interpretation of the result.

    - Two-stage DEA processing
    - Descriptive and decision-matrix analysis
    - Multiple ML models and optimized variants
    - Model comparison using Copeland-based ranking
    - Streamlit app for interactive prediction
    - Prediction of bank managerial efficiency class
    - Probability scores across all four classes
    - Visual display of class probabilities
    - Plain-language interpretation of the predicted class

The deployed model predicts one of four managerial classes:

    - Network Leaders
    - Transformation Specialists
    - Network Laggards
    - Funding-Rich Underperformers


## ***2. Input Variables***

The deployed model uses the following variables.

### Stage 1
- **Controllable inputs:** Assets, Employee Expense
- **Quasi-fixed input:** Equity
- **Intermediate output:** Deposits

### Stage 2
- **Controllable input:** Borrowings
- **Undesirable input:** NPAs (Previous Period)
- **Desirable outputs:**
  - Performing Loans
  - Investment
  - Net Income
  - Net-interest Income
  - Non-interest Income
- **Undesirable output:** NPAs

## ***3. Methodological Context***

This project is based on a two-stage DEA plus machine learning workflow:

    1. Raw bank data is processed through a two-stage DEA model.
    2. DEA-related outputs are merged into a processed dataset for downstream analysis and classification.
    3. Multiple machine learning models are trained and compared.
    4. A voting ensemble classifier is selected as the final deployed model.
    5. The selected model is served through a Streamlit application for interactive prediction.

## ***4. Final Deployed Model***
The deployed application uses a **Voting Classifier** selected after comparative evaluation against multiple alternative models. The voting optimized classifier was chosen as the most suitable deployed model based on model comparison results.

## ***5. Project Structure***
```
    bank_efficiency_prediction/
├── 📄 .gitignore
├── 📄 app.py                          # Streamlit deployment app
├── 📁 data/
│   ├── 📁 raw/
│   │   └── 📄 data.csv                # Raw bank dataset
│   └── 📁 processed/
│       └── 📄 dea_data_for_ml.csv     # DEA-processed dataset for ML
│
├── 📁 models/
│   └── 📁 ml/                         # Saved trained model files
│       ├── 📁 xgboost/
│       ├── 📁 xgboost_optuna/
│       ├── 📁 random_forest/
│       ├── 📁 random_forest_optuna/
│       ├── 📁 lightgbm/
│       ├── 📁 lightgbm_optuna/
│       ├── 📁 catboost/
│       ├── 📁 catboost_optuna/
│       ├── 📁 extra_trees/
│       ├── 📁 extra_trees_optuna/
│       ├── 📁 adaboost/
│       ├── 📁 adaboost_optuna/
│       ├── 📁 bagging/
│       ├── 📁 bagging_optuna/
│       ├── 📁 voting/
│       ├── 📁 voting_optimized/
│       ├── 📁 stacking/
│       └── 📁 stacking_optimized/
│
├── 📁 outputs/
│   ├── 📁 analysis/                   # Analysis results and plots
│   │   ├── 📁 descriptive/
│   │   ├── 📁 boxplots/
│   │   └── 📁 decision_matrix/
│   ├── 📁 model_comparison/           # Model comparison reports
│   └── 📁 ml/                         # Model-specific outputs
│       ├── 📁 xgboost/
│       ├── 📁 xgboost_optuna/
│       ├── 📁 random_forest/
│       ├── 📁 random_forest_optuna/
│       ├── 📁 lightgbm/
│       ├── 📁 lightgbm_optuna/
│       ├── 📁 catboost/
│       ├── 📁 catboost_optuna/
│       ├── 📁 extra_trees/
│       ├── 📁 extra_trees_optuna/
│       ├── 📁 adaboost/
│       ├── 📁 adaboost_optuna/
│       ├── 📁 bagging/
│       ├── 📁 bagging_optuna/
│       ├── 📁 voting/
│       ├── 📁 voting_optimized/
│       ├── 📁 stacking/
│       └── 📁 stacking_optimized/
│
├── 📁 src/
│   ├── 📄 __init__.py
│   ├── 📄 config.py                   # Central path and folder configuration
│   ├── 📄 compare_models.py           # Copeland-based model comparison
│   ├── 📄 run_dea.py                  # Runs DEA stage
│   ├── 📄 run_analysis.py             # Runs analysis stage
│   ├── 📄 run_pipeline.py             # Runs DEA + analysis pipeline
│   ├── 📄 run_full_pipeline.py        # Runs DEA + analysis + final ML pipeline
│   │
│   ├── 📁 dea/
│   │   ├── 📄 __init__.py
│   │   └── 📄 compute_dea.py          # DEA model computation
│   │
│   ├── 📁 analysis/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 descriptive_stats.py    # Descriptive statistics
│   │   ├── 📄 correlation_analysis.py # Correlation analysis
│   │   ├── 📄 efficiency_boxplots.py  # Efficiency boxplots
│   │   └── 📄 decision_matrix.py      # Decision matrix classification
│   │
│   └── 📁 ml/
│       ├── 📄 __init__.py
│       ├── 📄 train_xgboost.py
│       ├── 📄 train_xgboost_optuna.py
│       ├── 📄 train_random_forest.py
│       ├── 📄 train_random_forest_optuna.py
│       ├── 📄 train_lightgbm.py
│       ├── 📄 train_lightgbm_optuna.py
│       ├── 📄 train_catboost.py
│       ├── 📄 train_catboost_optuna.py
│       ├── 📄 train_extra_trees.py
│       ├── 📄 train_extra_trees_optuna.py
│       ├── 📄 train_adaboost.py
│       ├── 📄 train_adaboost_optuna.py
│       ├── 📄 train_bagging.py
│       ├── 📄 train_bagging_optuna.py
│       ├── 📄 train_voting_classifier.py
│       ├── 📄 train_voting_optimized_classifier.py
│       ├── 📄 train_stacking_classifier.py
│       └── 📄 train_stacking_optimized_classifier.py
│
├── 📄 requirements.txt
└── 📄 README.md
```

## ***6. Pipeline Commands***

```bash
Run DEA only: python -m src.run_dea
Run DEA and analysis: python -m src.run_pipeline
Run the end-to-end model: python -m src.run_full_pipeline
Compare models: python -m src.compare_models
```

## ***🚀 7. Deployment***
 **[bankefficiencyprediction.streamlit.app](https://bankefficiencyprediction.streamlit.app/)** 

Repository: [github.com/theoremsparks/bank_efficiency_prediction](https://github.com/theoremsparks/bank_efficiency_prediction)