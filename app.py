import streamlit as st
import pandas as pd
import joblib
import altair as alt
from pathlib import Path

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(page_title="Bank Efficiency Predictor", layout="centered")

st.title("Bank Efficiency Class Predictor")

# ---------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    BASE_DIR / "models" / "ml" / "voting_optimized_classifier" / "voting_final_model.joblib"
)

FEATURE_COLS = [
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

CLASS_NAME_MAP = {
    1: "Network Leaders",
    2: "Transformation Specialists",
    3: "Network Laggards",
    4: "Funding-Rich Underperformers",
}

CLASS_INTERPRETATION_MAP = {
    "Network Leaders": (
        "This bank appears strong in both stages of the network. "
        "It shows good deposit-generation performance and strong operating transformation."
    ),
    "Transformation Specialists": (
        "This bank appears relatively weaker in deposit mobilization, "
        "but stronger in converting resources into desirable outputs."
    ),
    "Network Laggards": (
        "This bank appears weak across both stages. "
        "Managerially, this suggests the clearest need for improvement."
    ),
    "Funding-Rich Underperformers": (
        "This bank appears relatively strong in funding or deposit-side position, "
        "but weak in operational transformation into final outputs."
    ),
}

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
with st.expander("Enter Bank Input Variables", expanded=True):
    with st.form("prediction_form"):
        st.markdown("### Stage 1 Variables")
        st.markdown(
            """
            - **Controllable inputs:** Assets, Employee Expense  
            - **Quasi-fixed input:** Equity  
            - **Intermediate output:** Deposits
            """
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            assets = st.number_input("Assets", min_value=0.01, value=100.0, step=1.0, format="%.4f")
            deposits = st.number_input("Deposits", min_value=0.01, value=100.0, step=1.0, format="%.4f")
        with c2:
            employee_expense = st.number_input("Employee Expense", min_value=0.01, value=100.0, step=1.0, format="%.4f")
        with c3:
            equity = st.number_input("Equity", min_value=0.01, value=100.0, step=1.0, format="%.4f")

        st.markdown("---")

        st.markdown("### Stage 2 Variables")
        st.markdown(
            """
            - **Controllable input:** Borrowings  
            - **Undesirable input:** NPAs (Previous Period)  
            - **Desirable outputs:** Performing Loans, Investment, Net Income, Net-interest Income, Non-interest Income  
            - **Undesirable output:** NPAs
            """
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            borrowings = st.number_input("Borrowings", min_value=0.01, value=100.0, step=1.0, format="%.4f")
            performing_loans = st.number_input("Performing Loans", min_value=0.01, value=100.0, step=1.0, format="%.4f")
            net_interest_income = st.number_input("Net-interest Income", min_value=-100000.0, value=100.0, step=1.0, format="%.4f")
        with c2:
            npas_previous = st.number_input("NPAs (Previous Period)", min_value=0.01, value=100.0, step=1.0, format="%.4f")
            investment = st.number_input("Investment", min_value=0.01, value=100.0, step=1.0, format="%.4f")
            non_interest_income = st.number_input("Non-interest Income", min_value=-100000.0, value=100.0, step=1.0, format="%.4f")
        with c3:
            net_income = st.number_input("Net Income", min_value=-100000.0, value=100.0, step=1.0, format="%.4f")
            npas = st.number_input("NPAs", min_value=0.01, value=100.0, step=1.0, format="%.4f")

        submitted = st.form_submit_button("Predict Class")

# ---------------------------------------------------------
# Prediction and results
# ---------------------------------------------------------
if submitted:
    input_data = {
        "Assets": assets,
        "Employee Expense": employee_expense,
        "Equity": equity,
        "Deposits": deposits,
        "Borrowings": borrowings,
        "NPAs (Previous Period)": npas_previous,
        "Performing Loans": performing_loans,
        "Investment": investment,
        "Net Income": net_income,
        "Net-interest Income": net_interest_income,
        "Non-interest Income": non_interest_income,
        "NPAs": npas,
    }

    input_df = pd.DataFrame([input_data], columns=FEATURE_COLS)

    try:
        raw_pred = model.predict(input_df)[0]

        if raw_pred in [0, 1, 2, 3]:
            pred_class = int(raw_pred) + 1
        else:
            pred_class = int(raw_pred)

        pred_label = CLASS_NAME_MAP.get(pred_class, f"Class {pred_class}")

        st.success(f"Predicted Class: {pred_label}")

        st.subheader("Interpretation of Prediction")
        st.write(
            CLASS_INTERPRETATION_MAP.get(
                pred_label,
                "No interpretation is available for this class."
            )
        )

        st.subheader("Entered Input Data")
        st.dataframe(input_df, width="stretch")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]

            display_classes = []
            for cls in model.classes_:
                cls_int = int(cls)
                if cls_int in [0, 1, 2, 3]:
                    cls_int += 1
                display_classes.append(CLASS_NAME_MAP.get(cls_int, f"Class {cls_int}"))

            prob_df = pd.DataFrame({
                "Class": display_classes,
                "Probability (%)": [round(float(p) * 100, 2) for p in probs]
            }).sort_values("Probability (%)", ascending=False)

            st.subheader("Prediction Probabilities")
            st.dataframe(prob_df, width="stretch")

            top_class = prob_df.iloc[0]["Class"]
            top_prob = prob_df.iloc[0]["Probability (%)"]

            if top_prob >= 80:
                confidence_text = (
                    f"The model is highly confident that this bank belongs to "
                    f"'{top_class}' with probability {top_prob:.2f}%."
                )
            elif top_prob >= 60:
                confidence_text = (
                    f"The model is moderately confident that this bank belongs to "
                    f"'{top_class}' with probability {top_prob:.2f}%."
                )
            else:
                confidence_text = (
                    f"The prediction is less certain. Although '{top_class}' has the "
                    f"highest probability at {top_prob:.2f}%, other classes are also plausible."
                )

            st.subheader("Interpretation of Probabilities")
            st.write(confidence_text)

            st.subheader("Probability Chart")

            custom_colors = {
                "Network Leaders": "#1b9e77",
                "Transformation Specialists": "#7570b3",
                "Network Laggards": "#d95f02",
                "Funding-Rich Underperformers": "#e7298a",
            }

            chart = (
                alt.Chart(prob_df)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "Class:N",
                        sort="-y",
                        title="Managerial Decision Matrix Class",
                        axis=alt.Axis(labelAngle=-25),
                    ),
                    y=alt.Y(
                        "Probability (%):Q",
                        title="Probability (%)",
                        scale=alt.Scale(domain=[0, 100]),
                    ),
                    color=alt.Color(
                        "Class:N",
                        scale=alt.Scale(
                            domain=list(custom_colors.keys()),
                            range=list(custom_colors.values()),
                        ),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("Class:N", title="Class"),
                        alt.Tooltip("Probability (%):Q", title="Probability (%)", format=".2f"),
                    ],
                )
                .properties(height=400)
            )

            st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")