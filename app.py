import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib


st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
)

st.title("🚢 Titanic Survival Prediction")
st.caption("Explore the Titanic dataset, train a Logistic Regression model, and predict passenger survival.")

EXPECTED_TARGET = "survived"

def load_and_clean_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    original_rows = len(data)

    # Match the cleaning steps in the supplied training script.
    if "deck" in data.columns:
        data = data.drop(columns=["deck"])

    duplicate_count = int(data.duplicated().sum())
    data = data.drop_duplicates().copy()

    if "age" in data.columns:
        data["age"] = data["age"].fillna(data["age"].median())

    for col in ["embarked", "embark_town"]:
        if col in data.columns and data[col].isna().any():
            mode = data[col].mode(dropna=True)
            if not mode.empty:
                data[col] = data[col].fillna(mode.iloc[0])

    # These columns were excluded in the original script.
    drop_cols = [c for c in ["embarked", "class", "alive"] if c in data.columns]
    data = data.drop(columns=drop_cols)

    return data, original_rows, duplicate_count


def build_model(X_train, y_train, use_smote):
    categorical_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical_cols = [c for c in X_train.columns if c not in categorical_cols]

    numeric_pipeline = SklearnPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = SklearnPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numerical_cols:
        transformers.append(("numeric", numeric_pipeline, numerical_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    steps = [("preprocessor", preprocessor)]
    if use_smote:
        class_counts = y_train.value_counts()
        if len(class_counts) != 2 or class_counts.min() < 2:
            raise ValueError("SMOTE needs at least two examples in each class in the training split.")
        k_neighbors = min(5, int(class_counts.min()) - 1)
        steps.append(("smote", SMOTE(random_state=42, k_neighbors=k_neighbors)))

    steps.append(("model", LogisticRegression(max_iter=1000, random_state=42)))
    return ImbPipeline(steps=steps)


def show_metric(label, value):
    st.metric(label, value)


with st.sidebar:
    st.header("⚙️ Model settings")
    test_size = st.slider("Test set size", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
    use_smote = st.toggle("Apply SMOTE to training data", value=True)
    st.caption("SMOTE is applied only to the training split to avoid test-data leakage.")
    st.divider()
    st.markdown("**Model:** Logistic Regression")
    st.markdown("**Split:** Stratified train/test split")
    st.markdown("**Preprocessing:** Median/mode imputation, scaling and one-hot encoding")


uploaded_file = st.file_uploader("Upload Titanic CSV dataset", type=["csv"])

if uploaded_file is None:
    st.info("Upload your `titanic_dataset.csv` file to get started.")
    st.markdown(
        """
        **Expected target column:** `survived` (0 = did not survive, 1 = survived)

        The app follows the cleaning and Logistic Regression workflow in your
        `titanic_model.py` script. You can also choose whether to apply SMOTE.
        """
    )
    st.stop()

try:
    data, original_rows, duplicate_count = load_and_clean_data(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

if EXPECTED_TARGET not in data.columns:
    st.error("The uploaded CSV must contain a `survived` target column.")
    st.stop()

if data[EXPECTED_TARGET].isna().any():
    st.error("The `survived` column contains missing values. Please clean these target rows and upload the CSV again.")
    st.stop()

if data[EXPECTED_TARGET].nunique() != 2:
    st.error("This app expects a binary `survived` target (0 and 1).")
    st.stop()

# Convert target to integer labels where possible.
try:
    data[EXPECTED_TARGET] = data[EXPECTED_TARGET].astype(int)
except (ValueError, TypeError):
    st.error("The `survived` column must contain numeric binary labels (0 and 1).")
    st.stop()

if not set(data[EXPECTED_TARGET].unique()).issubset({0, 1}):
    st.error("The `survived` column must use 0 for did not survive and 1 for survived.")
    st.stop()

X = data.drop(columns=[EXPECTED_TARGET])
y = data[EXPECTED_TARGET]

if X.empty:
    st.error("No feature columns remain after preprocessing.")
    st.stop()

tab_data, tab_train, tab_predict = st.tabs(
    ["📊 Data overview", "🤖 Train & evaluate", "🔮 Predict survival"]
)

with tab_data:
    st.subheader("Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows after cleaning", f"{len(data):,}")
    c2.metric("Features", len(X.columns))
    c3.metric("Duplicates removed", duplicate_count)
    c4.metric("Rows removed", original_rows - len(data))

    st.markdown("#### Dataset preview")
    st.dataframe(data.head(20), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Missing values")
        missing = pd.DataFrame({
            "Missing values": data.isna().sum(),
            "Missing percentage": (data.isna().mean() * 100).round(2),
        }).sort_values("Missing values", ascending=False)
        st.dataframe(missing, use_container_width=True)
    with right:
        st.markdown("#### Survival distribution")
        distribution = y.value_counts().sort_index().rename(index={0: "Did not survive", 1: "Survived"})
        st.bar_chart(distribution)
        st.dataframe(
            pd.DataFrame({
                "Class": ["Did not survive (0)", "Survived (1)"],
                "Count": [(y == 0).sum(), (y == 1).sum()],
                "Percentage": [round((y == 0).mean() * 100, 2), round((y == 1).mean() * 100, 2)],
            }),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("#### Feature types")
    feature_info = pd.DataFrame({
        "Feature": X.columns,
        "Data type": [str(X[c].dtype) for c in X.columns],
        "Unique values": [X[c].nunique(dropna=True) for c in X.columns],
    })
    st.dataframe(feature_info, hide_index=True, use_container_width=True)


with tab_train:
    st.subheader("Train Logistic Regression")
    st.write("The data is split using stratification so both sets retain a similar survival-class distribution.")

    if st.button("🚀 Train model", type="primary", use_container_width=True):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=1, shuffle=True, stratify=y
            )
            model = build_model(X_train, y_train, use_smote)
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, test_pred)
            report_dict = classification_report(
                y_test, test_pred, output_dict=True, zero_division=0
            )
            cm = confusion_matrix(y_test, test_pred, labels=[0, 1])

            st.session_state["titanic_model"] = model
            st.session_state["feature_columns"] = X.columns.tolist()
            st.session_state["metrics"] = {
                "accuracy": accuracy,
                "train_accuracy": accuracy_score(y_train, train_pred),
                "report": report_dict,
                "confusion_matrix": cm,
                "test_size": len(X_test),
                "train_size": len(X_train),
                "smote": use_smote,
            }
            st.success("Model trained successfully.")
        except Exception as exc:
            st.error(f"Model training failed: {exc}")

    if "metrics" in st.session_state:
        metrics = st.session_state["metrics"]
        st.markdown("#### Evaluation results")
        m1, m2, m3 = st.columns(3)
        m1.metric("Test accuracy", f"{metrics['accuracy']:.2%}")
        m2.metric("Training accuracy", f"{metrics['train_accuracy']:.2%}")
        m3.metric("Test samples", metrics["test_size"])

        st.caption(
            f"Training rows: {metrics['train_size']} · "
            f"Test rows: {metrics['test_size']} · "
            f"SMOTE: {'Enabled' if metrics['smote'] else 'Disabled'}"
        )

        left, right = st.columns(2)
        with left:
            st.markdown("#### Confusion matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                metrics["confusion_matrix"],
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Did not survive", "Survived"],
                yticklabels=["Did not survive", "Survived"],
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close(fig)

        with right:
            st.markdown("#### Classification report")
            report_df = pd.DataFrame(metrics["report"]).transpose()
            st.dataframe(report_df.round(3), use_container_width=True)

        model_bytes = None
        import io
        buffer = io.BytesIO()
        joblib.dump(st.session_state["titanic_model"], buffer)
        model_bytes = buffer.getvalue()
        st.download_button(
            "⬇️ Download trained model",
            data=model_bytes,
            file_name="titanic_logistic_regression.joblib",
            mime="application/octet-stream",
        )
    else:
        st.info("Choose your settings in the sidebar, then click **Train model**.")


with tab_predict:
    st.subheader("Predict a passenger's survival")
    if "titanic_model" not in st.session_state:
        st.info("Train the model in the **Train & evaluate** tab before making predictions.")
    else:
        model = st.session_state["titanic_model"]
        st.write("Enter passenger details below. The form uses the feature columns available in your uploaded dataset.")

        with st.form("passenger_prediction_form"):
            values = {}
            cols = st.columns(2)
            for idx, col in enumerate(X.columns):
                series = X[col]
                with cols[idx % 2]:
                    if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 10:
                        median = float(series.median()) if series.notna().any() else 0.0
                        min_value = float(series.min()) if series.notna().any() else median - 100
                        max_value = float(series.max()) if series.notna().any() else median + 100
                        if min_value == max_value:
                            max_value = min_value + 1.0
                        values[col] = st.number_input(
                            col.replace("_", " ").title(),
                            min_value=min_value,
                            max_value=max_value,
                            value=min(max(median, min_value), max_value),
                        )
                    elif pd.api.types.is_numeric_dtype(series):
                        options = sorted(series.dropna().unique().tolist())
                        if options:
                            values[col] = st.selectbox(col.replace("_", " ").title(), options)
                        else:
                            values[col] = 0
                    else:
                        options = sorted(series.dropna().astype(str).unique().tolist())
                        if options:
                            values[col] = st.selectbox(col.replace("_", " ").title(), options)
                        else:
                            values[col] = ""

            submitted = st.form_submit_button("Predict survival", type="primary", use_container_width=True)

        if submitted:
            input_df = pd.DataFrame([values], columns=X.columns)
            try:
                prediction = int(model.predict(input_df)[0])
                probabilities = model.predict_proba(input_df)[0]
                survival_probability = float(probabilities[list(model.classes_).index(1)])
                result_col, probability_col = st.columns(2)
                with result_col:
                    if prediction == 1:
                        st.success("Prediction: Passenger is likely to survive.")
                    else:
                        st.warning("Prediction: Passenger is unlikely to survive.")
                with probability_col:
                    st.metric("Estimated survival probability", f"{survival_probability:.1%}")
                st.progress(survival_probability)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
