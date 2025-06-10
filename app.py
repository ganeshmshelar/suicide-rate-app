import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Set page config
st.set_page_config(page_title="Suicide Logistic Regression App", layout="wide")

st.title("🧠 Suicide Dataset Logistic Regression App")
st.markdown(
    "Upload your dataset to explore suicide causes and predict cases using Logistic Regression.")

# Upload file
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset Loaded Successfully")
    st.dataframe(df.head())

    # Create Target
    df["Target"] = df["Total"].apply(lambda x: 1 if x > 0 else 0)

    # Show class distribution
    st.subheader("🎯 Target Class Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="Target", data=df, ax=ax1)
    st.pyplot(fig1)

    # Show Gender Distribution
    st.subheader("🧍 Gender vs Target")
    fig2, ax2 = plt.subplots()
    sns.countplot(x="Gender", hue="Target", data=df, ax=ax2)
    st.pyplot(fig2)

    # Age Group
    st.subheader("📊 Age Group vs Target")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    sns.countplot(x="Age_group", hue="Target", data=df, ax=ax3,
                  order=sorted(df["Age_group"].unique()))
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    # Cause Types
    st.subheader("📌 Top 10 Suicide Causes")
    top_types = df["Type"].value_counts().nlargest(10).index
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    sns.countplot(y="Type", data=df[df["Type"].isin(
        top_types)], hue="Target", ax=ax4)
    st.pyplot(fig4)

    # Encoding
    df_encoded = pd.get_dummies(
        df[["State", "Type", "Gender", "Age_group"]], drop_first=True)
    X = df_encoded
    y = df["Target"]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    # Model Training
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Results
    st.subheader("📈 Model Evaluation")
    st.write("**Accuracy:**", accuracy_score(y_test, y_pred))

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

else:
    st.info("👆 Please upload a CSV file to continue.")
