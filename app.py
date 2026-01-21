import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    plot_tree
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ======================
# App Title
# ======================
st.set_page_config(page_title="Decision Tree App", layout="wide")
st.title("Decision Tree: Classification & Regression")

# ======================
# Sidebar
# ======================
task = st.sidebar.radio(
    "Select Task",
    ["Classification", "Regression"]
)

max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
splitter = st.sidebar.selectbox("Splitter", ["best", "random"])

# ======================
# Load Dataset
# ======================
iris = load_iris()
df = pd.DataFrame(
    iris.data,
    columns=[
        'Sepal Length',
        'Sepal Width',
        'Petal Length',
        'Petal Width'
    ]
)

# ======================
# CLASSIFICATION
# ======================
if task == "Classification":
    st.header("Decision Tree Classification")

    X = df
    y = iris.target

    criterion = st.sidebar.selectbox(
        "Criterion", ["gini", "entropy", "log_loss"]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    st.subheader("Model Performance")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Tree Plot
    st.subheader("Decision Tree Visualization")
    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(
        model,
        feature_names=X.columns,
        class_names=iris.target_names,
        filled=True,
        ax=ax
    )
    st.pyplot(fig)

# ======================
# REGRESSION
# ======================
else:
    st.header("Decision Tree Regression")

    X = df[['Sepal Width', 'Petal Length', 'Petal Width']]
    y = df['Sepal Length']

    criterion = st.sidebar.selectbox(
        "Criterion",
        ["squared_error", "friedman_mse", "absolute_error"]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeRegressor(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    st.subheader("Model Performance")
    st.write("MAE:", mean_absolute_error(y_test, y_pred))
    st.write("MSE:", mean_squared_error(y_test, y_pred))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    st.write("R2 Score:", r2_score(y_test, y_pred))

    # Tree Plot
    st.subheader("Decision Tree Visualization")
    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(
        model,
        feature_names=X.columns,
        filled=True,
        ax=ax
    )
    st.pyplot(fig)
