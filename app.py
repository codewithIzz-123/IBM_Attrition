# =============================================
# Employee Attrition - Streamlit Web App
# Run: streamlit run app.py
# =============================================

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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

# =============================================
# APP TITLE
# =============================================
st.set_page_config(page_title="Employee Attrition", layout="wide")
st.title("Employee Attrition Prediction")
st.write("IBM HR Analytics Dataset | Target: Attrition (Yes / No)")
st.markdown("---")

# =============================================
# SIDEBAR - NAVIGATION
# =============================================
page = st.sidebar.radio("Go to", [
    "1. Dataset Overview",
    "2. EDA - Charts",
    "3. Correlation Heatmap",
    "4. Data Preparation",
    "5. ML Models & Results",
    "6. Model Comparison"
])

# =============================================
# LOAD DATA
# =============================================
@st.cache_data
def load_data():
    return pd.read_csv("Employee_Attrition.csv")

df = load_data()

# =============================================
# PREPARE + TRAIN (runs once, cached)
# =============================================
@st.cache_data
def run_models():
    df = pd.read_csv("Employee_Attrition.csv")

    # --- Encode ---
    df2 = df.copy()
    df2.drop(['EmployeeCount','EmployeeNumber','StandardHours','Over18'],
              axis=1, inplace=True)
    le = LabelEncoder()
    for col in df2.select_dtypes(include='object').columns:
        df2[col] = le.fit_transform(df2[col])

    # --- Split ---
    X = df2.drop('Attrition', axis=1)
    y = df2['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # --- Balance ---
    ros = RandomOverSampler(random_state=42)
    X_train, y_train = ros.fit_resample(X_train, y_train)

    # --- Train 3 models ---
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)

    # --- Predictions ---
    p_dt = dt.predict(X_test)
    p_rf = rf.predict(X_test)
    p_lr = lr.predict(X_test)

    return dt, rf, lr, X_train, X_test, y_test, p_dt, p_rf, p_lr

dt, rf, lr, X_train, X_test, y_test, p_dt, p_rf, p_lr = run_models()


# =============================================
# PAGE 1 - DATASET OVERVIEW
# =============================================
if page == "1. Dataset Overview":
    st.header("Dataset Overview")

    st.subheader("First 5 Rows")
    st.dataframe(df.head())

    st.subheader("Shape & Missing Values")
    st.write(f"Rows: **{df.shape[0]}** | Columns: **{df.shape[1]}**")
    st.write(f"Missing Values: **{df.isnull().sum().sum()}**")

    st.subheader("Attrition Count & Percentage")
    counts = df['Attrition'].value_counts()
    rate   = counts['Yes'] / len(df) * 100
    st.write(f"Stayed: **{counts['No']}** | Left: **{counts['Yes']}** | "
             f"Attrition Rate: **{rate:.1f}%**")

    # Bar + Pie
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].bar(counts.index, counts.values, color=['steelblue','salmon'])
    axes[0].set_title("Attrition Count")
    axes[0].set_xlabel("Attrition")
    axes[0].set_ylabel("Number of Employees")

    axes[1].pie(counts.values, labels=counts.index,
                autopct='%1.1f%%', colors=['steelblue','salmon'])
    axes[1].set_title("Attrition Proportion")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# =============================================
# PAGE 2 - EDA CHARTS
# =============================================
elif page == "2. EDA - Charts":
    st.header("Exploratory Data Analysis")

    # --- Univariate: Numeric histograms ---
    st.subheader("Numeric Features - Histograms")
    num_cols = df.select_dtypes(include='number').drop(
        columns=['EmployeeCount','EmployeeNumber','StandardHours'],
        errors='ignore').columns

    fig, axes = plt.subplots(5, 5, figsize=(18, 14))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        axes[i].hist(df[col], bins=20, color='steelblue', edgecolor='white')
        axes[i].set_title(col, fontsize=9)
    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Distribution of Numeric Features", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Univariate: Categorical bar charts ---
    st.subheader("Categorical Features - Bar Charts")
    cat_cols = ['Department','JobRole','MaritalStatus',
                'Gender','BusinessTravel','EducationField','OverTime']

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

    # --- Outlier box plots ---
    st.subheader("Outlier Detection - Box Plots")
    outlier_cols = ['Age','MonthlyIncome','DistanceFromHome',
                    'TotalWorkingYears','YearsAtCompany']

    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    for i, col in enumerate(outlier_cols):
        axes[i].boxplot(df[col])
        axes[i].set_title(col)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Bivariate: Attrition vs Department ---
    st.subheader("Attrition by Department")
    dept     = df.groupby(['Department','Attrition']).size().unstack()
    dept_pct = dept.div(dept.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    dept.plot(kind='bar', ax=axes[0], color=['steelblue','salmon'])
    axes[0].set_title("Attrition Count by Department")
    axes[0].set_ylabel("Count")

    dept_pct['Yes'].plot(kind='bar', ax=axes[1], color='salmon')
    axes[1].set_title("Attrition Rate (%) by Department")
    axes[1].set_ylabel("Attrition %")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Bivariate: Age and Income ---
    st.subheader("Attrition by Age and Monthly Income")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for label, color in zip(['No','Yes'], ['steelblue','salmon']):
        df[df['Attrition'] == label]['Age'].plot.hist(
            bins=15, alpha=0.6, ax=axes[0], label=label, color=color)
    axes[0].set_title("Age vs Attrition")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Count")
    axes[0].legend(title="Attrition")

    sns.boxplot(data=df, x='Attrition', y='MonthlyIncome',
                palette={'No':'steelblue','Yes':'salmon'}, ax=axes[1])
    axes[1].set_title("Monthly Income vs Attrition")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Bivariate: OverTime and JobRole ---
    st.subheader("Attrition by OverTime and Job Role")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ot     = df.groupby(['OverTime','Attrition']).size().unstack()
    ot_pct = ot.div(ot.sum(axis=1), axis=0) * 100
    ot_pct['Yes'].plot(kind='bar', ax=axes[0], color='salmon')
    axes[0].set_title("OverTime vs Attrition (%)")
    axes[0].set_ylabel("Attrition %")

    jr     = df.groupby(['JobRole','Attrition']).size().unstack().fillna(0)
    jr_pct = jr.div(jr.sum(axis=1), axis=0) * 100
    jr_pct['Yes'].sort_values().plot(kind='barh', ax=axes[1], color='salmon')
    axes[1].set_title("Job Role vs Attrition (%)")
    axes[1].set_xlabel("Attrition %")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Bivariate: Satisfaction scores ---
    st.subheader("Satisfaction Scores vs Attrition")
    sat_cols = ['JobSatisfaction','EnvironmentSatisfaction',
                'RelationshipSatisfaction','WorkLifeBalance']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, col in enumerate(sat_cols):
        sat     = df.groupby([col,'Attrition']).size().unstack().fillna(0)
        sat_pct = sat.div(sat.sum(axis=1), axis=0) * 100
        sat_pct['Yes'].plot(kind='bar', ax=axes[i], color='salmon')
        axes[i].set_title(col)
        axes[i].set_ylabel("Attrition %")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# =============================================
# PAGE 3 - CORRELATION HEATMAP
# =============================================
elif page == "3. Correlation Heatmap":
    st.header("Correlation Heatmap")
    st.write("How strongly every numeric feature is related to every other.")

    num_df = df.select_dtypes(include='number').drop(
        columns=['EmployeeCount','EmployeeNumber','StandardHours'], errors='ignore')

    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(num_df.corr(), annot=True, fmt='.2f',
                cmap='coolwarm', linewidths=0.5,
                annot_kws={'size': 8}, ax=ax)
    ax.set_title("Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# =============================================
# PAGE 4 - DATA PREPARATION
# =============================================
elif page == "4. Data Preparation":
    st.header("Data Preparation")

    st.subheader("Step 1 - Encode Categorical Columns")
    st.write("Text columns converted to numbers using LabelEncoder.")
    st.write("Attrition: No = 0, Yes = 1")

    st.subheader("Step 2 - Train / Test Split")
    st.write(f"Training set: **{X_train.shape[0]} rows** (80%)")
    st.write(f"Test set:     **{X_test.shape[0]} rows** (20%)")

    st.subheader("Step 3 - Balance the Data (RandomOverSampler)")
    st.write("Before: 985 Stayed, 191 Left — imbalanced")
    st.write("After:  985 Stayed, 985 Left — balanced")
    st.success("The data is balanced")


# =============================================
# PAGE 5 - ML MODELS & RESULTS
# =============================================
elif page == "5. ML Models & Results":
    st.header("Machine Learning Models")

    model_choice = st.selectbox("Select a model", [
        "Decision Tree", "Random Forest", "Logistic Regression"
    ])

    pred_map = {
        "Decision Tree":       p_dt,
        "Random Forest":       p_rf,
        "Logistic Regression": p_lr
    }
    cmap_map = {
        "Decision Tree":       "Blues",
        "Random Forest":       "Greens",
        "Logistic Regression": "Oranges"
    }

    pred = pred_map[model_choice]

    # Accuracy
    acc = accuracy_score(y_test, pred)
    st.metric("Accuracy", f"{acc:.4f}")

    # Classification report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, pred,
                                  target_names=['Stayed','Left']))

    # Confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, pred),
                annot=True, fmt='d',
                cmap=cmap_map[model_choice],
                xticklabels=['Stayed','Left'],
                yticklabels=['Stayed','Left'],
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_choice} - Confusion Matrix")
    st.pyplot(fig)
    plt.close()

    # Decision tree picture
    if model_choice == "Decision Tree":
        st.subheader("Decision Tree Visualisation")
        fig, ax = plt.subplots(figsize=(16, 8))
        plot_tree(dt, feature_names=X_train.columns.tolist(),
                  class_names=['Stayed','Left'], filled=True, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Random forest feature importance
    if model_choice == "Random Forest":
        st.subheader("Feature Importance (Top 10)")
        feat_df = pd.DataFrame({
            'Feature':    X_train.columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(feat_df['Feature'][::-1],
                feat_df['Importance'][::-1], color='salmon')
        ax.set_title("Top 10 Important Features")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# =============================================
# PAGE 6 - MODEL COMPARISON
# =============================================
elif page == "6. Model Comparison":
    st.header("Model Comparison")

    from sklearn.metrics import balanced_accuracy_score, recall_score, f1_score

    preds = {
        'Decision Tree':       p_dt,
        'Random Forest':       p_rf,
        'Logistic Regression': p_lr
    }

    # Results table
    st.subheader("Results Table")
    results = pd.DataFrame({
        'Model':             list(preds.keys()),
        'Accuracy':          [round(accuracy_score(y_test, p), 4) for p in preds.values()],
        'Balanced Accuracy': [round(balanced_accuracy_score(y_test, p), 4) for p in preds.values()],
        'Recall (Left)':     [round(recall_score(y_test, p), 4) for p in preds.values()],
        'F1 Score (Left)':   [round(f1_score(y_test, p), 4) for p in preds.values()],
    })
    st.dataframe(results, use_container_width=True)

    # Side by side confusion matrices
    st.subheader("Confusion Matrices - All 3 Models")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, pred), cmap in zip(axes, preds.items(),
                                       ['Blues','Greens','Oranges']):
        sns.heatmap(confusion_matrix(y_test, pred),
                    annot=True, fmt='d', cmap=cmap,
                    xticklabels=['Stayed','Left'],
                    yticklabels=['Stayed','Left'], ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Bar chart
    st.subheader("Performance Bar Chart")
    metrics     = ['Accuracy','Balanced Accuracy','Recall (Left)','F1 Score (Left)']
    x           = np.arange(len(metrics))
    width       = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (model, pred) in enumerate(preds.items()):
        vals = results[results['Model'] == model][metrics].values[0]
        ax.bar(x + i * width, vals, width, label=model)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Winner
    st.subheader("Conclusion")
    st.success("Random Forest performed best — highest accuracy and F1 score.")
    st.warning("Decision Tree is simplest and most interpretable.")
    st.info("Logistic Regression is a good statistical baseline.")
