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
st.markdown("This app uses a fixed CSV file. Upload functionality has been removed.")

# === Load your CSV directly here ===
DATA_PATH = "Suicides in India 2001-2012.csv"  # Change if needed

try:
    df = pd.read_csv(DATA_PATH)
    st.success("✅ Dataset Loaded Successfully")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"❌ Error loading data from {DATA_PATH}: {e}")
    st.stop()

# Create Target column
df["Target"] = df["Total"].apply(lambda x: 1 if x > 0 else 0)

# Show class distribution
st.subheader("🎯 Target Class Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x="Target", data=df, ax=ax1)
st.pyplot(fig1)

st.subheader("📉 Target Class Distribution Stats")
class_counts = df["Target"].value_counts()
st.write(class_counts)

# Gender Distribution
st.subheader("🧍 Gender vs Target")
fig2, ax2 = plt.subplots()
sns.countplot(x="Gender", hue="Target", data=df, ax=ax2)
st.pyplot(fig2)

# Age Group
st.subheader("📊 Age Group vs Target")
fig3, ax3 = plt.subplots(figsize=(10, 4))
sns.countplot(
    x="Age_group", hue="Target", data=df, ax=ax3, order=sorted(df["Age_group"].unique())
)
plt.xticks(rotation=45)
st.pyplot(fig3)

# Cause Types
st.subheader("📌 Top 10 Suicide Causes")
top_types = df["Type"].value_counts().nlargest(10).index
fig4, ax4 = plt.subplots(figsize=(10, 5))
sns.countplot(y="Type", data=df[df["Type"].isin(top_types)], hue="Target", ax=ax4)
st.pyplot(fig4)

# Encode features
features = ["State", "Type", "Gender", "Age_group"]
df_encoded = pd.get_dummies(df[features], drop_first=True)
df_encoded["Year"] = df["Year"]

X = df_encoded
y = df["Target"]

# Scale features
scaler = StandardScaler()
dummy_cols = df_encoded.drop(columns=["Year"]).columns
X_dummy_scaled = scaler.fit_transform(df_encoded[dummy_cols])
X_scaled = np.hstack([X_dummy_scaled, df_encoded[["Year"]].values])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train logistic regression model with class weight
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)

st.subheader("📈 Model Evaluation")
st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# --- Prediction Section ---
st.subheader("🔮 Make a Prediction")

state_options = sorted(df["State"].unique())
type_options = sorted(df["Type"].unique())
gender_options = sorted(df["Gender"].unique())
age_group_options = sorted(df["Age_group"].unique())

with st.form("prediction_form"):
    state_input = st.selectbox("Select State", options=state_options)
    year_input = st.slider(
        "Select Year",
        min_value=int(df["Year"].min()),
        max_value=int(df["Year"].max()),
        value=int(df["Year"].min()),
        step=1,
    )
    type_input = st.selectbox("Select Cause Type", options=type_options)
    gender_input = st.selectbox("Select Gender", options=gender_options)
    age_group_input = st.selectbox("Select Age Group", options=age_group_options)

    submitted = st.form_submit_button("Predict Suicide Risk")

if submitted:
    input_dict = {
        "State": [state_input],
        "Type": [type_input],
        "Gender": [gender_input],
        "Age_group": [age_group_input],
    }
    input_df = pd.DataFrame(input_dict)

    # One-hot encode input
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # Align input with training features
    for col in df_encoded.drop(columns=["Year"]).columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[df_encoded.drop(columns=["Year"]).columns]

    # Add year
    input_encoded["Year"] = year_input

    # Scale and predict
    input_dummy_scaled = scaler.transform(input_encoded.drop(columns=["Year"]))
    input_scaled = np.hstack([input_dummy_scaled, np.array([[year_input]])])

    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    # Display prediction
    st.write("🔍 **Prediction Details:**")
    st.json({
        "Probability (Low Risk - Class 0)": f"{proba[0]:.4f}",
        "Probability (High Risk - Class 1)": f"{proba[1]:.4f}",
        "Predicted Class": int(prediction)
    })

    if prediction == 1:
        st.error(f"⚠️ High suicide risk predicted with probability {proba[1]:.2f}")
    else:
        st.success(f"✅ Low suicide risk predicted with probability {proba[0]:.2f}")
