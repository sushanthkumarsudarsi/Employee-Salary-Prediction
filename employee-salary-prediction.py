import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Streamlit setup
st.set_page_config(page_title="Employee Salary Classification", layout="wide")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("adult 3.csv")
    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)

    # Add experience column (for UI use only)
    df["experience"] = (df["age"] - 18).clip(lower=0)

    return df

df = load_data()

# Encode categorical variables
def preprocess_data(df):
    df_copy = df.copy()
    encoders = {}
    for col in df_copy.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_copy[col] = le.fit_transform(df_copy[col])
        encoders[col] = le
    return df_copy, encoders

df_encoded, encoders = preprocess_data(df)

# Sidebar navigation
option = st.sidebar.radio("Navigate", ["📊 EDA", "🤖 Model Evaluation", "🔮 Predict Income"])

# EDA Section
if option == "📊 EDA":
    st.subheader("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Income Distribution")
        st.bar_chart(df["income"].value_counts())

    with col2:
        st.write("### Education Levels")
        st.bar_chart(df["education"].value_counts())

    st.write("### Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["age"], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    st.write("### Hours-per-week Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df["hours-per-week"], bins=30, kde=True, ax=ax2)
    st.pyplot(fig2)

    st.write("### Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_encoded.corr(), cmap="coolwarm", annot=False, ax=ax3)
    st.pyplot(fig3)

# Model Evaluation Section
elif option == "🤖 Model Evaluation":
    st.subheader("🤖 Model Evaluation: Logistic Regression, Random Forest, SVM")

    X = df_encoded.drop("income", axis=1)
    y = df_encoded["income"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=False)
        st.markdown(f"### {name}")
        st.text(report)

# Prediction Section
elif option == "🔮 Predict Income":
    st.subheader("🔮 Predict Employee Income Class")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", min_value=18, max_value=70, value=30)
        education = st.selectbox("Education", sorted(df["education"].unique()))
        occupation = st.selectbox("Occupation", sorted(df["occupation"].unique()))

    with col2:
        hours_per_week = st.slider("Hours per week", min_value=1, max_value=99, value=40)
        experience = st.slider("Experience (Years)", min_value=0, max_value=40, value=5)

    # Form DataFrame
    user_input_df = pd.DataFrame({
        "age": [age],
        "education": [education],
        "occupation": [occupation],
        "hours-per-week": [hours_per_week],
        "experience": [experience]
    })

    st.write("### Input Data")
    st.dataframe(user_input_df)

    # Encode input
    input_encoded = user_input_df.copy()
    for col in input_encoded.columns:
        if col in encoders:
            input_encoded[col] = encoders[col].transform(input_encoded[col])

    # Ensure columns match training data
    for col in df_encoded.drop("income", axis=1).columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[df_encoded.drop("income", axis=1).columns]

    # Train model (RandomForest)
    model = RandomForestClassifier()
    model.fit(df_encoded.drop("income", axis=1), df_encoded["income"])
    prediction = model.predict(input_encoded)[0]
    income_label = encoders["income"].inverse_transform([prediction])[0]

    if st.button("🎯 Predict Salary Class"):
        if income_label == "<=50K":
            st.success(f"🟩 Predicted Income: {income_label}")
        else:
            st.error(f"🟥 Predicted Income: {income_label}")

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit and Scikit-learn")
