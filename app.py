import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              classification_report, confusion_matrix,
                              recall_score, f1_score)
from imblearn.over_sampling import RandomOverSampler

# ─────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────
st.set_page_config(page_title="Employee Attrition", layout="wide")

st.title("Employee Attrition Prediction")
st.write("Dataset: IBM HR Analytics Employee Attrition & Performance")
st.write("Source: Kaggle (pavansubhash)")
st.write("Target Variable: Attrition (Yes / No)")
st.markdown("---")

# ─────────────────────────────────────────
# SIDEBAR MENU
# ─────────────────────────────────────────
page = st.sidebar.radio("Go to section", [
    "1. Dataset Overview",
    "2. Univariate Analysis",
    "3. Outlier Detection",
    "4. Bivariate Analysis",
    "5. Correlation Heatmap",
    "6. Data Preparation",
    "7. Decision Tree",
    "8. Random Forest",
    "9. Logistic Regression",
    "10. Model Comparison"
])

# ─────────────────────────────────────────
# LOAD + PREPARE DATA (cached)
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Employee_Attrition.csv")
    return df

@st.cache_data
def get_prepared_data():
    df = pd.read_csv("Employee_Attrition.csv")

    # Encoding — Cell 31
    df_encoded = df.copy()
    df_encoded.drop(
        ['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18'],
        axis=1, inplace=True
    )
    le = LabelEncoder()
    for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])

    # Split — Cell 33
    X = df_encoded.drop('Attrition', axis=1)
    y = df_encoded['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Balance — Cell 36
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    # Train models — Cells 42-60
    dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_model.fit(X_train_resampled, y_train_resampled)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_resampled, y_train_resampled)

    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_resampled, y_train_resampled)

    # Predictions
    y_pred    = dt_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)
    y_pred_lr = lr_model.predict(X_test)

    return (df_encoded, X_train, X_test, y_train, y_test,
            X_train_resampled, y_train_resampled,
            dt_model, rf_model, lr_model,
            y_pred, y_pred_rf, y_pred_lr)

df = load_data()

(df_encoded, X_train, X_test, y_train, y_test,
 X_train_resampled, y_train_resampled,
 dt_model, rf_model, lr_model,
 y_pred, y_pred_rf, y_pred_lr) = get_prepared_data()


# ═════════════════════════════════════════
# 1. DATASET OVERVIEW
# ═════════════════════════════════════════
if page == "1. Dataset Overview":
    st.header("Dataset Overview")

    st.subheader("First 5 Rows")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(f"Rows: **{df.shape[0]}**  |  Columns: **{df.shape[1]}**")

    st.subheader("Missing Values")
    st.write(df.isnull().sum())
    st.write("**No missing values**")

    # Cell 11 — Attrition count + pie
    st.subheader("Target Variable Distribution (Attrition)")
    attrition_counts = df['Attrition'].value_counts()
    attrition_rate   = (attrition_counts['Yes'] / len(df)) * 100
    st.write(f"Attrition Rate: **{attrition_rate:.2f}%**")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].bar(attrition_counts.index, attrition_counts.values,
                color=['steelblue', 'salmon'])
    axes[0].set_title("Attrition Count")
    axes[0].set_xlabel("Attrition")
    axes[0].set_ylabel("Count")

    axes[1].pie(attrition_counts.values,
                labels=attrition_counts.index,
                autopct='%1.1f%%',
                colors=['steelblue', 'salmon'])
    axes[1].set_title("Attrition Proportion")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═════════════════════════════════════════
# 2. UNIVARIATE ANALYSIS
# ═════════════════════════════════════════
elif page == "2. Univariate Analysis":
    st.header("Univariate Analysis")
    st.write("Looking at each variable on its own.")

    # Cell 13 — Numeric histograms
    st.subheader("Numeric Features Distribution")
    numeric_cols = df.select_dtypes(include='number').drop(
        columns=['EmployeeCount', 'EmployeeNumber', 'StandardHours'],
        errors='ignore'
    ).columns

    fig, axes = plt.subplots(5, 5, figsize=(18, 15))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=20,
                     color='steelblue', edgecolor='white')
        axes[i].set_title(col, fontsize=9)
    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Numeric Features Distribution", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Cell 15 — Categorical bar charts
    st.subheader("Categorical Features Distribution")
    cat_cols = ['Department', 'JobRole', 'MaritalStatus',
                'Gender', 'BusinessTravel', 'EducationField', 'OverTime']

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        counts = df[col].value_counts()
        axes[i].bar(counts.index, counts.values, color='steelblue')
        axes[i].set_title(col)
        axes[i].tick_params(axis='x', rotation=30)

    for j in range(len(cat_cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═════════════════════════════════════════
# 3. OUTLIER DETECTION
# ═════════════════════════════════════════
elif page == "3. Outlier Detection":
    st.header("Outlier Detection")
    st.write("Box plots show outliers — the dots outside the whiskers.")

    outlier_cols = [
        'Age', 'MonthlyIncome', 'DistanceFromHome',
        'TotalWorkingYears', 'YearsAtCompany'
    ]

    # Cell 17 — Box plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, col in enumerate(outlier_cols):
        axes[i].boxplot(df[col].dropna())
        axes[i].set_title(col)

    for j in range(len(outlier_cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Cell 19 — IQR outlier count
    st.subheader("Outlier Count (IQR Method)")
    outlier_data = []
    for col in outlier_cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        n   = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
        outlier_data.append({"Feature": col, "Outlier Count": int(n)})

    st.table(pd.DataFrame(outlier_data))


# ═════════════════════════════════════════
# 4. BIVARIATE ANALYSIS
# ═════════════════════════════════════════
elif page == "4. Bivariate Analysis":
    st.header("Bivariate Analysis")
    st.write("Comparing each feature against Attrition (Yes / No).")

    # Cell 21 — Department
    st.subheader("Attrition by Department")
    dept     = df.groupby(['Department', 'Attrition']).size().unstack()
    dept_pct = dept.div(dept.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    dept_pct.plot(kind='bar', ax=axes[0], color=['steelblue', 'salmon'])
    axes[0].set_title("Attrition Count by Department")
    axes[0].set_ylabel("Count")

    dept_pct['Yes'].plot(kind='bar', ax=axes[1], color='salmon')
    axes[1].set_title("Attrition Rate by Department (%)")
    axes[1].set_ylabel("Attrition %")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Cell 23 — Age and Monthly Income
    st.subheader("Attrition by Age and Monthly Income")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for label, color in zip(['No', 'Yes'], ['steelblue', 'salmon']):
        df[df['Attrition'] == label]['Age'].plot.hist(
            bins=15, alpha=0.6, ax=axes[0], label=label, color=color
        )
    axes[0].set_title("Age vs Attrition")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    sns.boxplot(data=df, x='Attrition', y='MonthlyIncome',
                palette={'No': 'steelblue', 'Yes': 'salmon'}, ax=axes[1])
    axes[1].set_title("Monthly Income vs Attrition")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Cell 25 — OverTime and JobRole
    st.subheader("Attrition by OverTime and Job Role")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ot     = df.groupby(['OverTime', 'Attrition']).size().unstack()
    ot_pct = ot.div(ot.sum(axis=1), axis=0) * 100
    ot_pct['Yes'].plot(kind='bar', ax=axes[0], color='salmon')
    axes[0].set_title("OverTime vs Attrition (%)")
    axes[0].set_ylabel("Attrition %")

    jr     = df.groupby(['JobRole', 'Attrition']).size().unstack().fillna(0)
    jr_pct = jr.div(jr.sum(axis=1), axis=0) * 100
    jr_pct['Yes'].sort_values().plot(kind='barh', ax=axes[1], color='salmon')
    axes[1].set_title("JobRole vs Attrition (%)")
    axes[1].set_xlabel("Attrition %")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Cell 27 — Satisfaction scores
    st.subheader("Satisfaction Scores vs Attrition")
    satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction',
                         'RelationshipSatisfaction', 'WorkLifeBalance']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, col in enumerate(satisfaction_cols):
        sat     = df.groupby([col, 'Attrition']).size().unstack().fillna(0)
        sat_pct = sat.div(sat.sum(axis=1), axis=0) * 100
        sat_pct['Yes'].plot(kind='bar', ax=axes[i], color='salmon')
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Attrition %")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═════════════════════════════════════════
# 5. CORRELATION HEATMAP
# ═════════════════════════════════════════
elif page == "5. Correlation Heatmap":
    st.header("Correlation Heatmap")
    st.write("Shows how strongly every numeric feature is related to every other.")

    # Cell 29
    numeric_df = df.select_dtypes(include='number').drop(
        columns=['EmployeeCount', 'EmployeeNumber', 'StandardHours'],
        errors='ignore'
    )
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                linewidths=0.5, annot_kws={'size': 8}, ax=ax)
    ax.set_title("Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═════════════════════════════════════════
# 6. DATA PREPARATION
# ═════════════════════════════════════════
elif page == "6. Data Preparation":
    st.header("Data Preparation")

    # Cell 31
    st.subheader("Step 1 — Encoding Categorical Variables")
    st.write("Dropped: EmployeeCount, EmployeeNumber, StandardHours, Over18")
    st.write("All text columns converted to numbers using LabelEncoder.")
    st.write("Attrition: No = 0, Yes = 1")
    st.dataframe(df_encoded.head())

    # Cell 33
    st.subheader("Step 2 — Train / Test Split (80% / 20%)")
    st.write(f"Training set: **{X_train.shape[0]} rows**")
    st.write(f"Test set:     **{X_test.shape[0]} rows**")

    # Cells 35-37
    st.subheader("Step 3 — Balancing the Data (RandomOverSampler)")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Before oversampling:**")
        st.write(y_train.value_counts().rename({0: "Stayed (0)", 1: "Left (1)"}))
    with col2:
        st.write("**After oversampling:**")
        st.write(y_train_resampled.value_counts().rename({0: "Stayed (0)", 1: "Left (1)"}))

    st.success("The data is balanced")


# ═════════════════════════════════════════
# 7. DECISION TREE
# ═════════════════════════════════════════
elif page == "7. Decision Tree":
    st.header("Decision Tree Classifier")

    # Cells 44-45
    acc     = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    col1, col2 = st.columns(2)
    col1.metric("Accuracy",          f"{acc:.4f}")
    col2.metric("Balanced Accuracy", f"{bal_acc:.4f}")

    # Cell 47
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred,
                                  target_names=['Stayed', 'Left']))

    # Cell 48
    y_train_pred = dt_model.predict(X_train_resampled)
    train_acc    = accuracy_score(y_train_resampled, y_train_pred)
    st.write(f"Training Accuracy: **{train_acc:.4f}**")

    # Cell 49
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Stayed', 'Left'],
                yticklabels=['Stayed', 'Left'], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    plt.close()

    # Cell 51
    st.subheader("Decision Tree Visualisation")
    fig, ax = plt.subplots(figsize=(16, 10))
    plot_tree(dt_model,
              feature_names=X_train.columns.tolist(),
              class_names=['Stayed', 'Left'],
              filled=True, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═════════════════════════════════════════
# 8. RANDOM FOREST
# ═════════════════════════════════════════
elif page == "8. Random Forest":
    st.header("Random Forest Classifier")

    # Cell 56
    accuracy = accuracy_score(y_test, y_pred_rf)
    st.metric("Random Forest Accuracy", f"{round(accuracy, 2)}")

    # Cell 57
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred_rf,
                                  target_names=['Stayed', 'Left']))

    # Cell 58
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred_rf),
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Stayed', 'Left'],
                yticklabels=['Stayed', 'Left'], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    plt.close()


# ═════════════════════════════════════════
# 9. LOGISTIC REGRESSION
# ═════════════════════════════════════════
elif page == "9. Logistic Regression":
    st.header("Logistic Regression")

    # Cell 60
    accuracy = accuracy_score(y_test, y_pred_lr)
    st.write(f"Logistic Regression Accuracy: **{round(accuracy, 4)}**")

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred_lr))

    st.subheader("Logistic Regression Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred_lr)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Stayed', 'Left'],
                yticklabels=['Stayed', 'Left'], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Logistic Regression Confusion Matrix")
    st.pyplot(fig)
    plt.close()


# ═════════════════════════════════════════
# 10. MODEL COMPARISON
# ═════════════════════════════════════════
elif page == "10. Model Comparison":
    st.header("Model Scores Comparison")

    preds = {
        'Decision Tree':       y_pred,
        'Random Forest':       y_pred_rf,
        'Logistic Regression': y_pred_lr
    }

    # Cell 63 — side-by-side confusion matrices
    st.subheader("Confusion Matrices — All 3 Models")
    cmaps = ['Blues', 'Greens', 'Oranges']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, pred), cmap in zip(axes, preds.items(), cmaps):
        sns.heatmap(confusion_matrix(y_test, pred),
                    annot=True, fmt='d', cmap=cmap,
                    xticklabels=['Stayed', 'Left'],
                    yticklabels=['Stayed', 'Left'],
                    ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Cell 64 — results table
    st.subheader("Results Table")
    results_df = pd.DataFrame({
        'Model':             list(preds.keys()),
        'Accuracy':          [round(accuracy_score(y_test, p), 4)
                               for p in preds.values()],
        'Balanced Accuracy': [round(balanced_accuracy_score(y_test, p), 4)
                               for p in preds.values()],
        'Recall (Left)':     [round(recall_score(y_test, p), 4)
                               for p in preds.values()],
        'F1 Score (Left)':   [round(f1_score(y_test, p), 4)
                               for p in preds.values()],
    })
    st.dataframe(results_df, use_container_width=True)

    # Cell 65 — bar chart
    st.subheader("Performance Bar Chart")
    metrics     = ['Accuracy', 'Balanced Accuracy', 'Recall (Left)', 'F1 Score (Left)']
    model_names = results_df['Model'].tolist()
    x           = np.arange(len(metrics))
    width       = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, model in enumerate(model_names):
        values = results_df.loc[i, metrics]
        ax.bar(x + i * width, values, width, label=model)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
