import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)

# =====================================
# 1. Config
# =====================================
st.set_page_config(page_title="Credit Risk Prediction", page_icon="🏦", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "credit_risk_model.keras"
SCALER_PATH = BASE_DIR / "scaler_pipeline.pkl"
DATA_PATH = BASE_DIR / "credit_risk_cleaned.csv"
TEST_DATA_PATH = BASE_DIR / "credit_risk_test.csv"  # Features + target (held-out test set)

TARGET = "SeriousDlqin2yrs"
DEFAULT_THRESHOLD = 0.38

FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

LATE_COLS = {
    "NumberOfTime30-59DaysPastDueNotWorse": "30-59 Days",
    "NumberOfTime60-89DaysPastDueNotWorse": "60-89 Days",
    "NumberOfTimes90DaysLate": "90+ Days",
}

STATUS_MAP = {0: "Non-Default", 1: "Default"}


# =====================================
# 2. Loaders (no st.* calls inside cache)
# =====================================
@st.cache_resource
def load_artifacts():
    for p in (MODEL_PATH, SCALER_PATH):
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p.name}")
    return tf.keras.models.load_model(MODEL_PATH), joblib.load(SCALER_PATH)


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"File not found: {DATA_PATH.name}")
    data = pd.read_csv(DATA_PATH)
    return data.loc[:, ~data.columns.str.contains("^Unnamed")]


@st.cache_data
def load_test_data():
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"File not found: {TEST_DATA_PATH.name}")
    t = pd.read_csv(TEST_DATA_PATH)
    cols = FEATURES + [TARGET]
    missing = [c for c in cols if c not in t.columns]
    if missing:
        raise ValueError(f"Test dataset is missing columns: {missing}")
    t = t[cols].apply(pd.to_numeric, errors="coerce")
    t = t.replace([np.inf, -np.inf], np.nan).dropna()
    return t.reset_index(drop=True)


try:
    model, scaler = load_artifacts()
    df = load_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

missing_cols = [c for c in FEATURES if c not in df.columns]
if missing_cols:
    st.error(f"Dataset is missing columns: {missing_cols}")
    st.stop()


# =====================================
# 3. Helpers
# =====================================
def predict_proba(frame: pd.DataFrame) -> np.ndarray:
    """Return default probabilities for one or many rows."""
    scaled = scaler.transform(frame[FEATURES])
    return np.clip(model(scaled, training=False).numpy().ravel(), 0, 1)


@st.cache_data
def get_test_scores(data: pd.DataFrame) -> np.ndarray:
    return predict_proba(data[FEATURES])


def make_gauge(prob: float, threshold: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#c0392b" if prob >= threshold else "#27ae60"},
                "steps": [
                    {"range": [0, threshold * 100], "color": "#e8f8f0"},
                    {"range": [threshold * 100, 100], "color": "#fdecea"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(t=30, b=10, l=30, r=30))
    return fig


def count_label(series: pd.Series, cap: int = 5) -> pd.Series:
    """Cap counts (dataset may have placeholder codes like 96/98) and label as '5+'."""
    s = series.clip(upper=cap).astype(int).astype(str)
    return s.replace(str(cap), f"{cap}+")


# =====================================
# 4. Sidebar
# =====================================
st.sidebar.title("🏦 Credit Risk System")
page = st.sidebar.radio(
    "Navigation",
    ["Customer Assessment", "EDA Dashboard", "Model Performance"],
)


# =====================================
# PAGE 1: CUSTOMER ASSESSMENT
# =====================================
if page == "Customer Assessment":
    st.title("🏦 Credit Risk Prediction System")
    st.write("An ANN-based system for predicting customer credit risk.")
    st.divider()

    st.subheader("Customer Information")
    c1, c2 = st.columns(2)

    with c1:
        revolving = st.number_input("Revolving Utilization", 0.0, value=0.5, step=0.01)
        age = st.number_input("Customer Age", 18, 100, 35)
        late_30_59 = st.number_input("Times 30-59 Days Past Due", 0, 98, 0)
        debt_ratio = st.number_input("Debt Ratio", 0.0, value=0.4, step=0.01)
        income = st.number_input("Monthly Income", 0.0, value=5000.0, step=500.0)

    with c2:
        open_credit = st.number_input("Open Credit Lines and Loans", 0, 60, 5)
        late_90 = st.number_input("Times 90 Days Late", 0, 98, 0)
        real_estate = st.number_input("Real Estate Loans or Lines", 0, 60, 1)
        late_60_89 = st.number_input("Times 60-89 Days Past Due", 0, 98, 0)
        dependents = st.number_input("Number of Dependents", 0, 20, 0)

    threshold = st.slider(
        "Classification Threshold", 0.10, 0.90, DEFAULT_THRESHOLD, 0.01, key="assess_threshold"
    )
    st.caption("Probability required to classify a customer as high risk.")

    customer = pd.DataFrame(
        [[revolving, age, late_30_59, debt_ratio, income, open_credit,
          late_90, real_estate, late_60_89, dependents]],
        columns=FEATURES,
    )

    # Warn when inputs fall outside the range the model was trained on
    warnings = []
    for col in ("RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome"):
        lo, hi = df[col].quantile([0.001, 0.999])
        val = customer[col].iloc[0]
        if val < lo or val > hi:
            warnings.append(
                f"**{col}** = {val:,.2f} is outside the typical training range ({lo:,.2f} – {hi:,.2f})."
            )

    if st.button("Predict Credit Risk", type="primary"):
        st.session_state["last_customer"] = customer
        st.session_state["last_threshold"] = threshold

    if "last_customer" in st.session_state:
        cust = st.session_state["last_customer"]
        thr = st.session_state["last_threshold"]
        prob = float(predict_proba(cust)[0])
        is_high = prob >= thr

        st.divider()
        st.subheader("Prediction Results")

        left, right = st.columns([1, 1])
        with left:
            st.plotly_chart(make_gauge(prob, thr), use_container_width=True)
        with right:
            m1, m2 = st.columns(2)
            m1.metric("Default Probability", f"{prob:.2%}")
            m2.metric("Threshold", f"{thr:.2f}")

            if is_high:
                st.error("⚠️ High Credit Risk: likely to default.")
                st.info("Recommendation: Enhanced Credit Review (not an automatic rejection).")
            else:
                st.success("✅ Low Credit Risk: unlikely to default.")
                st.info("Recommendation: Further Credit Review (not an automatic approval).")

            if abs(prob - thr) <= 0.05:
                st.warning("The probability is very close to the threshold, so treat this result as borderline.")

        for w in warnings:
            st.warning(w)

        # What-if analysis
        with st.expander("🔍 What-if analysis: how does one feature change the risk?"):
            feat = st.selectbox("Feature to vary", FEATURES, index=4)
            lo, hi = df[feat].quantile([0.01, 0.99])
            grid = np.linspace(lo, hi, 40)
            sim = pd.concat([cust] * len(grid), ignore_index=True)
            sim[feat] = grid
            sim_probs = predict_proba(sim)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=grid, y=sim_probs, mode="lines", name="Default probability"))
            fig.add_hline(y=thr, line_dash="dash", annotation_text="Threshold")
            fig.add_trace(go.Scatter(
                x=[cust[feat].iloc[0]], y=[prob], mode="markers",
                marker=dict(size=12, color="red"), name="Current customer"))
            fig.update_layout(xaxis_title=feat, yaxis_title="Probability",
                              yaxis_tickformat=".0%", height=400)
            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "This is a model prediction, not a final lending decision. "
            "The probability should be validated and calibrated before real-world use."
        )


# =====================================
# PAGE 2: EDA DASHBOARD
# =====================================
elif page == "EDA Dashboard":
    st.title("📊 Exploratory Data Analysis")
    st.write("Explore the credit dataset and understand default patterns.")

    has_target = TARGET in df.columns
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    feature_cols = [c for c in numeric_cols if c != TARGET]

    tab_over, tab_uni, tab_bi, tab_multi = st.tabs(
        ["Overview", "Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"]
    )

    # ---------------------------------
    # 1. OVERVIEW
    # ---------------------------------
    with tab_over:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", f"{len(df):,}")
        c2.metric("Columns", len(df.columns))
        c3.metric("Missing Values", f"{int(df.isnull().sum().sum()):,}")

        if has_target:
            c4.metric("Default Rate", f"{df[TARGET].mean():.2%}")

            counts = (
                df[TARGET]
                .map(STATUS_MAP)
                .value_counts()
                .reindex(["Non-Default", "Default"], fill_value=0)
                .rename_axis("Class")
                .reset_index(name="Customers")
            )

            a, b = st.columns(2)
            a.plotly_chart(
                px.pie(counts, names="Class", values="Customers", hole=0.45,
                       title="Default Distribution"),
                use_container_width=True,
            )
            b.plotly_chart(
                px.bar(counts, x="Class", y="Customers", text_auto=",", color="Class",
                       title="Customers per Class"),
                use_container_width=True,
            )
            st.caption(
                "The target classes are imbalanced. Therefore, accuracy alone "
                "may be misleading when evaluating the model."
            )
        else:
            st.warning(f"{TARGET} column was not found.")

        st.subheader("Missing Values")
        miss = df.isnull().sum()
        miss = miss[miss > 0].sort_values(ascending=False)

        if not miss.empty:
            missing_table = miss.to_frame("Missing Count")
            missing_table["Missing Percentage"] = (
                missing_table["Missing Count"] / len(df) * 100
            ).round(2)
            st.dataframe(missing_table, use_container_width=True)
        else:
            st.success("No missing values found.")

    # ---------------------------------
    # 2. UNIVARIATE ANALYSIS
    # ---------------------------------
    with tab_uni:
        st.subheader("Analyze One Feature")
        st.write(
            "Select one feature to examine its descriptive statistics, "
            "distribution, missing values, and possible outliers."
        )

        if not numeric_cols:
            st.warning("No numeric columns are available for Univariate Analysis.")
        else:
            default_feature = (
                numeric_cols.index("MonthlyIncome") if "MonthlyIncome" in numeric_cols else 0
            )
            feat = st.selectbox("Select Feature", numeric_cols,
                                index=default_feature, key="univariate_feature")

            series = df[feat]
            non_null = series.dropna()

            st.markdown("#### Descriptive Statistics")
            stats = non_null.describe().to_frame("Value")
            stats.loc["Missing Values"] = series.isna().sum()
            stats.loc["Missing Percentage"] = series.isna().mean() * 100
            st.dataframe(stats, use_container_width=True)

            st.markdown("#### Distribution Analysis")
            c1, c2 = st.columns(2)
            trim = c1.checkbox("Exclude values above the 99th percentile",
                               value=True, key="uni_trim")
            n_bins = c2.slider("Number of bins", min_value=10, max_value=100,
                               value=40, key="uni_bins")

            plot_data = df.dropna(subset=[feat]).copy()
            if trim and not plot_data.empty:
                upper_limit = plot_data[feat].quantile(0.99)
                plot_data = plot_data[plot_data[feat] <= upper_limit]

            if plot_data.empty:
                st.warning("There are no values to display for this feature.")
            else:
                color_by_target = has_target and st.checkbox(
                    "Compare distribution by default status", value=False, key="uni_by_target"
                )

                if color_by_target:
                    plot_data["Default Status"] = plot_data[TARGET].map(STATUS_MAP)
                    fig = px.histogram(
                        plot_data, x=feat, nbins=n_bins, color="Default Status",
                        barmode="overlay", opacity=0.70, marginal="box",
                        title=f"Distribution of {feat} by Default Status",
                    )
                else:
                    fig = px.histogram(
                        plot_data, x=feat, nbins=n_bins, marginal="box",
                        title=f"Distribution of {feat}",
                    )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Outlier Inspection")
                st.plotly_chart(
                    px.box(plot_data, x=feat, points="outliers", title=f"Box Plot of {feat}"),
                    use_container_width=True,
                )

    # ---------------------------------
    # 3. BIVARIATE ANALYSIS
    # ---------------------------------
    with tab_bi:
        st.subheader("Analyze the Relationship Between Two Features")
        st.write(
            "Choose two columns to investigate their relationship. "
            "The chart type depends on the selected variables."
        )

        if len(feature_cols) < 2:
            st.warning("At least two numeric features are required.")
        else:
            c1, c2 = st.columns(2)
            x_default = feature_cols.index("age") if "age" in feature_cols else 0
            x_feat = c1.selectbox("First Feature (X-axis)", feature_cols,
                                  index=x_default, key="bivariate_x")

            y_options = [c for c in numeric_cols if c != x_feat]
            y_default = y_options.index(TARGET) if has_target and TARGET in y_options else 0
            y_feat = c2.selectbox("Second Feature (Y-axis)", y_options,
                                  index=y_default, key="bivariate_y")

            plot_data = df[[x_feat, y_feat]].copy()
            if has_target and TARGET not in (x_feat, y_feat):
                if st.checkbox("Color points by default status", value=True, key="bi_color_target"):
                    plot_data[TARGET] = df[TARGET]

            plot_data = plot_data.dropna(subset=[x_feat, y_feat])

            if plot_data.empty:
                st.warning("No valid rows are available for this pair of features.")

            elif has_target and TARGET in (x_feat, y_feat):
                value_col = x_feat if y_feat == TARGET else y_feat
                plot_data["Default Status"] = plot_data[TARGET].map(STATUS_MAP)
                fig = px.box(
                    plot_data, x="Default Status", y=value_col, color="Default Status",
                    points="outliers", title=f"{value_col} by Default Status",
                )
                st.plotly_chart(fig, use_container_width=True)

            else:
                color_col = None
                if has_target and TARGET in plot_data.columns:
                    color_col = "Default Status"
                    plot_data[color_col] = plot_data[TARGET].map(STATUS_MAP)

                chart_data = plot_data.sample(min(5000, len(plot_data)), random_state=42)
                fig = px.scatter(
                    chart_data, x=x_feat, y=y_feat, color=color_col,
                    opacity=0.55, title=f"{x_feat} vs {y_feat}",
                )
                st.plotly_chart(fig, use_container_width=True)

                correlation = plot_data[[x_feat, y_feat]].corr().iloc[0, 1]
                st.metric("Pearson Correlation", f"{correlation:.3f}")

        st.divider()
        st.markdown("#### Credit Risk Patterns")

        if has_target and "age" in df.columns:
            age_bins = [17, 25, 35, 45, 55, 65, 75, 120]
            age_labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "66-75", "76+"]
            age_data = df[["age", TARGET]].dropna().copy()
            age_data["Age Group"] = pd.cut(age_data["age"], bins=age_bins, labels=age_labels)
            age_summary = (
                age_data.groupby("Age Group", observed=True)[TARGET]
                .agg(["mean", "size"])
                .reset_index()
            )
            age_summary.columns = ["Age Group", "Default Rate", "Customers"]

            fig = px.bar(
                age_summary, x="Age Group", y="Default Rate",
                text=age_summary["Default Rate"].map("{:.1%}".format),
                hover_data=["Customers"], title="Default Rate by Age Group",
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        late_cols_available = [c for c in LATE_COLS if c in df.columns]
        if has_target and late_cols_available:
            rows = []
            for col in late_cols_available:
                temp = df[[col, TARGET]].dropna().copy()
                temp["Times Late"] = count_label(temp[col])
                grouped = (
                    temp.groupby("Times Late")[TARGET]
                    .agg(["mean", "size"])
                    .reset_index()
                )
                grouped["Late Type"] = LATE_COLS[col]
                rows.append(grouped)

            late_summary = pd.concat(rows, ignore_index=True)
            late_summary.columns = ["Times Late", "Default Rate", "Customers", "Late Type"]

            fig = px.bar(
                late_summary, x="Times Late", y="Default Rate", color="Late Type",
                barmode="group",
                category_orders={"Times Late": ["0", "1", "2", "3", "4", "5+"]},
                hover_data=["Customers"], title="Default Rate by Late-Payment Count",
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------
    # 4. MULTIVARIATE ANALYSIS
    # ---------------------------------
    with tab_multi:
        st.subheader("Analyze Multiple Features Together")
        st.write(
            "Explore patterns across several numeric features. "
            "These visualizations help reveal relationships that may not be "
            "obvious when examining columns individually."
        )

        if len(feature_cols) < 2:
            st.warning("At least two numeric features are required.")
        else:
            selected_features = st.multiselect(
                "Select features for correlation analysis",
                feature_cols,
                default=feature_cols[: min(6, len(feature_cols))],
                key="multi_features",
            )

            if len(selected_features) >= 2:
                corr = df[selected_features].corr()
                fig = px.imshow(
                    corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Correlation Heatmap",
                )
                fig.update_layout(height=650)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select at least two features to display the correlation heatmap.")

            st.divider()
            st.markdown("#### Multivariate Scatter Matrix")

            pair_features = st.multiselect(
                "Select features for scatter matrix (2 to 5 recommended)",
                feature_cols,
                default=feature_cols[: min(3, len(feature_cols))],
                key="multi_pair_features",
            )

            if len(pair_features) >= 2:
                matrix_cols = pair_features.copy()
                if has_target:
                    matrix_cols.append(TARGET)

                matrix_data = df[matrix_cols].dropna()
                matrix_data = matrix_data.sample(min(1500, len(matrix_data)), random_state=42)

                if has_target:
                    matrix_data["Default Status"] = matrix_data[TARGET].map(STATUS_MAP)
                    fig = px.scatter_matrix(
                        matrix_data, dimensions=pair_features, color="Default Status",
                        title="Relationships Among Selected Features",
                    )
                else:
                    fig = px.scatter_matrix(
                        matrix_data, dimensions=pair_features,
                        title="Relationships Among Selected Features",
                    )

                fig.update_layout(height=850)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "The scatter matrix displays pairwise relationships among "
                    "the selected features. It is exploratory and does not "
                    "establish causation."
                )
            else:
                st.info("Select at least two features to display the scatter matrix.")


# =====================================
# PAGE 3: MODEL PERFORMANCE
# =====================================
elif page == "Model Performance":
    st.title("📈 Model Performance")
    st.write(
        "Evaluate the model using an independent test dataset "
        "and explore how the classification threshold affects performance."
    )

    # ---- Load & validate ----
    try:
        evaluation_data = load_test_data()
    except (FileNotFoundError, ValueError) as e:
        st.error(str(e))
        st.stop()

    if evaluation_data.empty:
        st.error("No valid records found in the test dataset.")
        st.stop()

    y_true = evaluation_data[TARGET].astype(int).to_numpy()

    if not set(np.unique(y_true)).issubset({0, 1}):
        st.error("Target column must contain only 0 and 1.")
        st.stop()
    if len(np.unique(y_true)) < 2:
        st.error("Test dataset must contain both target classes.")
        st.stop()

    # ---- Data overview ----
    st.subheader("Data Overview")
    total = len(evaluation_data)
    non_default = int((y_true == 0).sum())
    default = int((y_true == 1).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Total Features", len(FEATURES))
    c3.metric("Non-Default", f"{non_default:,}", f"{non_default / total:.1%}", delta_color="off")
    c4.metric("Default", f"{default:,}", f"{default / total:.1%}", delta_color="off")

    dist = pd.DataFrame({
        "Customer Type": ["Non-Default", "Default"],
        "Percentage": [non_default / total * 100, default / total * 100],
    })
    st.plotly_chart(
        px.bar(dist, x="Customer Type", y="Percentage", color="Customer Type",
               text_auto=".1f", title="Target Distribution (%)"),
        use_container_width=True,
    )
    st.divider()

    # ---- Predictions ----
    try:
        y_scores = np.asarray(get_test_scores(evaluation_data)).reshape(-1)
    except Exception as e:
        st.error(f"Error generating predictions: {e}")
        st.stop()

    if len(y_scores) != len(y_true) or not np.isfinite(y_scores).all():
        st.error("Model generated invalid predictions.")
        st.stop()

    # ---- Threshold ----
    st.subheader("Classification Threshold")
    threshold = st.slider(
        "Adjust Threshold", 0.05, 0.95, float(DEFAULT_THRESHOLD), 0.01, key="perf_threshold"
    )
    st.caption("The threshold controls the probability required to classify a customer as high credit risk.")
    y_pred = (y_scores >= threshold).astype(int)

    # ---- Metrics ----
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy:.2%}")
    m2.metric("Precision", f"{precision:.2%}")
    m3.metric("Recall", f"{recall:.2%}")
    m4.metric("F1-Score", f"{f1:.2%}")
    st.divider()

    # ---- Confusion matrix ----
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    fig_cm = go.Figure(go.Heatmap(
        z=cm,
        x=["Predicted Non-Default", "Predicted Default"],
        y=["Actual Non-Default", "Actual Default"],
        text=cm, texttemplate="%{text}", colorscale="Blues", showscale=True,
    ))
    fig_cm.update_layout(xaxis_title="Predicted Class", yaxis_title="Actual Class", height=450)
    st.plotly_chart(fig_cm, use_container_width=True)

    a, b = st.columns(2)
    a.metric("True Negatives (TN)", f"{tn:,}")
    a.metric("False Positives (FP)", f"{fp:,}")
    b.metric("False Negatives (FN)", f"{fn:,}")
    b.metric("True Positives (TP)", f"{tp:,}")
    st.divider()

    # ---- Precision-Recall curve ----
    st.subheader("Precision-Recall Curve")
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, y_scores)

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=pr_recall, y=pr_precision, mode="lines", name="Precision-Recall"))
    if len(pr_thresholds) > 0:
        i = int(np.argmin(np.abs(pr_thresholds - threshold)))
        fig_pr.add_trace(go.Scatter(
            x=[pr_recall[i]], y=[pr_precision[i]], mode="markers",
            marker=dict(size=12, color="red"), name="Selected Threshold"))
    fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision",
                         xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]), height=450)
    st.plotly_chart(fig_pr, use_container_width=True)
    st.divider()

    # ---- Threshold analysis ----
    st.subheader("Threshold Analysis")
    rows = []
    for t in np.arange(0.05, 0.96, 0.01):
        p = (y_scores >= t).astype(int)
        rows.append({
            "Threshold": t,
            "Precision": precision_score(y_true, p, zero_division=0),
            "Recall": recall_score(y_true, p, zero_division=0),
            "F1-Score": f1_score(y_true, p, zero_division=0),
            "False Positives": int(((y_true == 0) & (p == 1)).sum()),
            "False Negatives": int(((y_true == 1) & (p == 0)).sum()),
        })
    threshold_df = pd.DataFrame(rows)

    fig_t = go.Figure()
    for m in ["Precision", "Recall", "F1-Score"]:
        fig_t.add_trace(go.Scatter(x=threshold_df["Threshold"], y=threshold_df[m], mode="lines", name=m))
    fig_t.add_vline(x=threshold, line_dash="dash", annotation_text=f"Selected: {threshold:.2f}")
    fig_t.update_layout(xaxis_title="Threshold", yaxis_title="Score", yaxis=dict(range=[0, 1]), height=450)
    st.plotly_chart(fig_t, use_container_width=True)

    st.markdown("### Classification Errors")
    fig_e = go.Figure()
    for m in ["False Positives", "False Negatives"]:
        fig_e.add_trace(go.Scatter(x=threshold_df["Threshold"], y=threshold_df[m], mode="lines", name=m))
    fig_e.add_vline(x=threshold, line_dash="dash", annotation_text=f"Selected: {threshold:.2f}")
    fig_e.update_layout(xaxis_title="Threshold", yaxis_title="Number of Errors", height=450)
    st.plotly_chart(fig_e, use_container_width=True)

    # ---- Selected threshold table ----
    st.markdown("### Selected Threshold Results")
    selected_result = pd.DataFrame([{
        "Threshold": threshold, "Precision": precision, "Recall": recall,
        "F1-Score": f1, "Accuracy": accuracy,
        "False Positives": fp, "False Negatives": fn,
    }])
    st.dataframe(
        selected_result.style.format({
            "Threshold": "{:.2f}", "Precision": "{:.2%}", "Recall": "{:.2%}",
            "F1-Score": "{:.2%}", "Accuracy": "{:.2%}",
        }),
        use_container_width=True, hide_index=True,
    )

    st.info(
        "This model is a decision-support tool, not an automatic "
        "credit approval or rejection system. Final decisions should "
        "consider additional customer information and human review."
    )