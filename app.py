# ============================================================
# Employee Attrition Analysis - Streamlit Web App
# Run with: streamlit run app.py
# ============================================================
 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              classification_report, confusion_matrix,
                              recall_score, f1_score)
from imblearn.over_sampling import RandomOverSampler
 
# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Employee Attrition Analysis",
    page_icon="📊",
    layout="wide"
)
 
# ============================================================
# LOAD DATA (cached so it only runs once)
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Employee_Attrition.csv")
    return df
 
@st.cache_data
def prepare_data(df):
    df_encoded = df.copy()
    df_encoded.drop(
        ["EmployeeCount", "EmployeeNumber", "StandardHours", "Over18"],
        axis=1, inplace=True
    )
    le = LabelEncoder()
    for col in df_encoded.select_dtypes(include="object").columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    return df_encoded
 
@st.cache_data
def train_models(df_encoded):
    X = df_encoded.drop("Attrition", axis=1)
    y = df_encoded["Attrition"]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
 
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
 
    dt  = DecisionTreeClassifier(max_depth=3, random_state=42)
    rf  = RandomForestClassifier(n_estimators=100, random_state=42)
    lr  = LogisticRegression(max_iter=1000, random_state=42)
 
    dt.fit(X_train_res, y_train_res)
    rf.fit(X_train_res, y_train_res)
    lr.fit(X_train_res, y_train_res)
 
    preds = {
        "Decision Tree":       dt.predict(X_test),
        "Random Forest":       rf.predict(X_test),
        "Logistic Regression": lr.predict(X_test),
    }
 
    return dt, rf, lr, X_train, X_test, y_test, preds
 
# ============================================================
# LOAD EVERYTHING
# ============================================================
df         = load_data()
df_encoded = prepare_data(df)
dt, rf, lr, X_train, X_test, y_test, preds = train_models(df_encoded)
 
# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Home",
    "📊 Dataset Overview",
    "🔍 EDA - Univariate",
    "📈 EDA - Bivariate",
    "🔗 Correlation Heatmap",
    "🤖 ML Models",
    "📋 Model Comparison"
])
 
# ============================================================
# PAGE: HOME
# ============================================================
if page == "🏠 Home":
    st.title("Employee Attrition Prediction")
    st.write("**Dataset:** IBM HR Analytics Employee Attrition & Performance")
    st.write("**Source:** Kaggle (pavansubhash)")
    st.write("**Target Variable:** Attrition (Yes / No)")
 
    st.markdown("---")
 
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Employees", "1,470")
    col2.metric("Attrited (Yes)",  "237")
    col3.metric("Stayed (No)",     "1,233")
    col4.metric("Attrition Rate",  "16.1%")
 
    st.markdown("---")
    st.subheader("Research Questions")
    st.write("1. Can we predict whether an employee will leave based on HR data?")
    st.write("2. Which features have the most impact on employee attrition?")
    st.write("3. Can we identify high-risk employees before they leave?")
    st.write("4. How effective are different ML models in predicting attrition?")
 
# ============================================================
# PAGE: DATASET OVERVIEW
# ============================================================
elif page == "📊 Dataset Overview":
    st.title("Dataset Overview")
 
    st.subheader("First 5 Rows")
    st.dataframe(df.head())
 
    st.subheader("Shape")
    st.write(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
 
    st.subheader("Data Types & Missing Values")
    info_df = pd.DataFrame({
        "Data Type":     df.dtypes,
        "Missing Values": df.isnull().sum(),
        "Unique Values": df.nunique()
    })
    st.dataframe(info_df)
 
    st.subheader("Basic Statistics")
    st.dataframe(df.describe())
 
    st.subheader("Attrition Distribution")
    attrition_counts = df["Attrition"].value_counts()
    attrition_rate   = attrition_counts["Yes"] / len(df) * 100
    st.write(f"**Attrition Rate: {attrition_rate:.2f}%**")
 
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
 
    axes[0].bar(attrition_counts.index, attrition_counts.values,
                color=["steelblue", "salmon"], edgecolor="black", width=0.5)
    axes[0].set_title("Attrition Count")
    axes[0].set_xlabel("Attrition")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(attrition_counts.values):
        axes[0].text(i, v + 10, str(v), ha="center", fontsize=12)
 
    axes[1].pie(attrition_counts.values, labels=attrition_counts.index,
                autopct="%1.1f%%", colors=["steelblue", "salmon"],
                startangle=90)
    axes[1].set_title("Attrition Proportion")
 
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
 
# ============================================================
# PAGE: EDA UNIVARIATE
# ============================================================
elif page == "🔍 EDA - Univariate":
    st.title("Exploratory Data Analysis — Univariate")
    st.write("Looking at each variable individually.")
 
    # Numeric distributions
    st.subheader("Numeric Features Distribution")
    numeric_cols = df.select_dtypes(include="number").drop(
        columns=["EmployeeCount", "EmployeeNumber", "StandardHours"],
        errors="ignore"
    ).columns
 
    fig = df[numeric_cols].hist(figsize=(18, 15), bins=20,
                                color="steelblue", edgecolor="white")
    plt.suptitle("Numeric Features Distribution", fontsize=14)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()
 
    # Categorical distributions
    st.subheader("Categorical Features Distribution")
    cat_cols = ["Department", "JobRole", "MaritalStatus",
                "Gender", "BusinessTravel", "EducationField", "OverTime"]
 
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
 
    for i, col in enumerate(cat_cols):
        counts = df[col].value_counts()
        axes[i].bar(counts.index, counts.values,
                    color="steelblue", edgecolor="black")
        axes[i].set_title(col, fontweight="bold")
        axes[i].tick_params(axis="x", rotation=20)
        for j, v in enumerate(counts.values):
            axes[i].text(j, v + 2, str(v), ha="center", fontsize=9)
 
    for k in range(len(cat_cols), len(axes)):
        axes[k].set_visible(False)
 
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
 
    # Outlier detection
    st.subheader("Outlier Detection — Box Plots")
    outlier_cols = ["Age", "MonthlyIncome", "DistanceFromHome",
                    "TotalWorkingYears", "YearsAtCompany"]
 
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
 
    for i, col in enumerate(outlier_cols):
        axes[i].boxplot(df[col].dropna(), patch_artist=True,
                        boxprops=dict(facecolor="steelblue", color="navy"),
                        medianprops=dict(color="red", linewidth=2),
                        flierprops=dict(marker="o", color="salmon",
                                        alpha=0.5, markersize=5))
        axes[i].set_title(col, fontweight="bold")
 
    for k in range(len(outlier_cols), len(axes)):
        axes[k].set_visible(False)
 
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
 
    # Outlier count using IQR
    st.subheader("Outlier Count (IQR Method)")
    outlier_data = []
    for col in outlier_cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        n   = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        outlier_data.append({"Feature": col, "Outlier Count": n})
    st.dataframe(pd.DataFrame(outlier_data))
 
# ============================================================
# PAGE: EDA BIVARIATE
# ============================================================
elif page == "📈 EDA - Bivariate":
    st.title("Exploratory Data Analysis — Bivariate")
    st.write("Comparing each feature against the target variable (Attrition).")
 
    # Attrition by Department
    st.subheader("Attrition by Department")
    dept     = df.groupby(["Department", "Attrition"]).size().unstack()
    dept_pct = dept.div(dept.sum(axis=1), axis=0) * 100
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    dept.plot(kind="bar", ax=axes[0], color=["steelblue", "salmon"],
              edgecolor="black", rot=15)
    axes[0].set_title("Attrition Count by Department", fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].legend(title="Attrition")
 
    dept_pct["Yes"].plot(kind="bar", ax=axes[1], color="salmon",
                         edgecolor="black", rot=15)
    axes[1].set_title("Attrition Rate (%) by Department", fontweight="bold")
    axes[1].set_ylabel("Attrition %")
    for i, v in enumerate(dept_pct["Yes"]):
        axes[1].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontweight="bold")
 
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
 
    # Attrition by Age and Monthly Income
    st.subheader("Attrition by Age and Monthly Income")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
    for label, color in zip(["No", "Yes"], ["steelblue", "salmon"]):
        axes[0].hist(df[df["Attrition"] == label]["Age"], bins=15,
                     alpha=0.7, label=label, color=color, edgecolor="white")
    axes[0].set_title("Age vs Attrition", fontweight="bold")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Count")
    axes[0].legend(title="Attrition")
 
    sns.boxplot(data=df, x="Attrition", y="MonthlyIncome",
                palette={"No": "steelblue", "Yes": "salmon"}, ax=axes[1])
    axes[1].set_title("Monthly Income vs Attrition", fontweight="bold")
 
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
 
    # Attrition by OverTime and Job Role
    st.subheader("Attrition by OverTime and Job Role")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
    ot     = df.groupby(["OverTime", "Attrition"]).size().unstack()
    ot_pct = ot.div(ot.sum(axis=1), axis=0) * 100
    ot_pct["Yes"].plot(kind="bar", ax=axes[0], color=["steelblue", "salmon"],
                       edgecolor="black", rot=0)
    axes[0].set_title("OverTime vs Attrition (%)", fontweight="bold")
    axes[0].set_ylabel("Attrition %")
    for i, v in enumerate(ot_pct["Yes"]):
        axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")
 
    jr     = df.groupby(["JobRole", "Attrition"]).size().unstack().fillna(0)
    jr_pct = jr.div(jr.sum(axis=1), axis=0) * 100
    jr_pct["Yes"].sort_values().plot(kind="barh", ax=axes[1],
                                      color="salmon", edgecolor="black")
    axes[1].set_title("Job Role vs Attrition (%)", fontweight="bold")
    axes[1].set_xlabel("Attrition %")
 
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
 
    # Satisfaction Scores
    st.subheader("Satisfaction Scores vs Attrition")
    satisfaction_cols = ["JobSatisfaction", "EnvironmentSatisfaction",
                         "RelationshipSatisfaction", "WorkLifeBalance"]
 
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
 
    for i, col in enumerate(satisfaction_cols):
        sat     = df.groupby([col, "Attrition"]).size().unstack().fillna(0)
        sat_pct = sat.div(sat.sum(axis=1), axis=0) * 100
        sat_pct["Yes"].plot(kind="bar", ax=axes[i], color="salmon",
                            edgecolor="black", rot=0, width=0.55)
        axes[i].set_title(f"{col}", fontweight="bold")
        axes[i].set_xlabel(f"{col} (1=Low → 4=High)")
        axes[i].set_ylabel("Attrition %")
        axes[i].set_ylim(0, 30)
        for j, v in enumerate(sat_pct["Yes"]):
            axes[i].text(j, v + 0.3, f"{v:.1f}%", ha="center",
                         fontsize=10, fontweight="bold")
 
    plt.suptitle("Satisfaction Scores vs Attrition", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
 
# ============================================================
# PAGE: CORRELATION HEATMAP
# ============================================================
elif page == "🔗 Correlation Heatmap":
    st.title("Correlation Heatmap")
    st.write("Shows how strongly every numeric feature is related to every other feature.")
 
    numeric_df = df.select_dtypes(include="number").drop(
        columns=["EmployeeCount", "EmployeeNumber", "StandardHours"],
        errors="ignore"
    )
 
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
 
    fig, ax = plt.subplots(figsize=(18, 14))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", linewidths=0.5,
                annot_kws={"size": 9, "weight": "bold"},
                vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title("Correlation Heatmap — All Numeric Features",
                 fontsize=16, fontweight="bold", pad=20)
    plt.xticks(fontsize=10, rotation=45, ha="right")
    plt.yticks(fontsize=10, rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
 
# ============================================================
# PAGE: ML MODELS
# ============================================================
elif page == "🤖 ML Models":
    st.title("Machine Learning Models")
 
    model_choice = st.selectbox(
        "Select a model to view:",
        ["Decision Tree", "Random Forest", "Logistic Regression"]
    )
 
    y_pred_map = {
        "Decision Tree":       preds["Decision Tree"],
        "Random Forest":       preds["Random Forest"],
        "Logistic Regression": preds["Logistic Regression"],
    }
    cmap_map = {
        "Decision Tree":       "Blues",
        "Random Forest":       "Greens",
        "Logistic Regression": "Oranges",
    }
 
    selected_pred = y_pred_map[model_choice]
    acc = accuracy_score(y_test, selected_pred)
    bal = balanced_accuracy_score(y_test, selected_pred)
 
    col1, col2 = st.columns(2)
    col1.metric("Accuracy",          f"{acc*100:.2f}%")
    col2.metric("Balanced Accuracy", f"{bal*100:.2f}%")
 
    st.subheader("Classification Report")
    report = classification_report(y_test, selected_pred,
                                   target_names=["Stayed", "Left"])
    st.text(report)
 
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, selected_pred),
                annot=True, fmt="d", cmap=cmap_map[model_choice],
                xticklabels=["Stayed", "Left"],
                yticklabels=["Stayed", "Left"],
                linewidths=1, ax=ax, annot_kws={"size": 14})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_choice} — Confusion Matrix", fontweight="bold")
    st.pyplot(fig)
    plt.close()
 
    # Decision Tree visualisation
    if model_choice == "Decision Tree":
        st.subheader("Decision Tree Visualisation")
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(dt, feature_names=X_train.columns.tolist(),
                  class_names=["Stayed", "Left"],
                  filled=True, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
 
    # Random Forest feature importance
    if model_choice == "Random Forest":
        st.subheader("Feature Importance")
        importances = rf.feature_importances_
        feat_df     = pd.DataFrame({
            "Feature":    X_train.columns,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(10)
 
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feat_df["Feature"][::-1], feat_df["Importance"][::-1],
                color="salmon", edgecolor="black")
        ax.set_title("Top 10 Most Important Features", fontweight="bold")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
 
# ============================================================
# PAGE: MODEL COMPARISON
# ============================================================
elif page == "📋 Model Comparison":
    st.title("Model Comparison")
 
    # Results table
    results_df = pd.DataFrame({
        "Model": list(preds.keys()),
        "Accuracy":          [round(accuracy_score(y_test, p)*100, 2)
                               for p in preds.values()],
        "Balanced Accuracy": [round(balanced_accuracy_score(y_test, p)*100, 2)
                               for p in preds.values()],
        "Recall (Left)":     [round(recall_score(y_test, p)*100, 2)
                               for p in preds.values()],
        "F1 Score (Left)":   [round(f1_score(y_test, p)*100, 2)
                               for p in preds.values()],
    })
 
    st.subheader("Results Table")
    st.dataframe(results_df, use_container_width=True)
 
    # Side-by-side confusion matrices
    st.subheader("Confusion Matrices — All 3 Models")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cmaps = ["Blues", "Greens", "Oranges"]
 
    for ax, (name, pred), cmap in zip(axes, preds.items(), cmaps):
        sns.heatmap(confusion_matrix(y_test, pred),
                    annot=True, fmt="d", cmap=cmap,
                    xticklabels=["Stayed", "Left"],
                    yticklabels=["Stayed", "Left"],
                    linewidths=1, ax=ax, annot_kws={"size": 13})
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
 
    plt.suptitle("Confusion Matrices — All Models",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
 
    # Comparison bar chart
    st.subheader("Performance Bar Chart")
    metrics     = ["Accuracy", "Balanced Accuracy", "Recall (Left)", "F1 Score (Left)"]
    model_names = results_df["Model"].tolist()
    x           = np.arange(len(metrics))
    width       = 0.25
    colors      = ["steelblue", "mediumseagreen", "tomato"]
 
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (m, c) in enumerate(zip(model_names, colors)):
        vals = results_df[results_df["Model"] == m][metrics].values[0]
        bars = ax.bar(x + i * width, vals, width, label=m,
                      color=c, edgecolor="white", alpha=0.9)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}",
                    ha="center", va="bottom", fontsize=9)
 
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
 
    # Conclusion
    st.subheader("Conclusion")
    st.success("**Random Forest** achieved the best overall performance — highest accuracy (92.9%) and best F1 score (70.6%) for the Left class.")
    st.warning("**Decision Tree** is the most interpretable model but has the lowest recall (42.4%) — misses the most at-risk employees.")
    st.info("**Logistic Regression** has the highest recall (63.6%) but lower overall accuracy (78.2%).")
 
    st.markdown("---")
    st.write("**Key Insight:** OverTime, Monthly Income and Years at Company are the strongest predictors of employee attrition.")
