from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Freshlytics-AI", page_icon="🍕", layout="centered")

st.title("Freshlytics-AI: Predicting Food Spoilage Risk")
st.caption("This app predicts food spoilage risk using various machine learning algorithms.")

f = st.file_uploader("Upload your food dataset (CSV)", type=["csv"])

if f is not None:

    # Read dataset
    df = pd.read_csv(f)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    data = df.copy()

    # Remove extra spaces from text columns
    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].str.strip()

    # Label Encoding
    enc = {}

    for col in data.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        enc[col] = le

    # Features and Target
    X = data.drop("Risk", axis=1)
    y = data["Risk"]

    # Train-Test Split
    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Feature Scaling
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    # Models
    models = {
        "Gaussian Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Support Vector Machine": SVC(probability=True)
    }

    trained = {}
    rows = []

    # Train Models
    for name, model in models.items():
        model.fit(Xtr, ytr)

        trained[name] = model

        pred = model.predict(Xte)

        acc = accuracy_score(yte, pred)

        rows.append({
            "Algorithm": name,
            "Accuracy": acc
        })

    rdf = pd.DataFrame(rows)

    st.subheader("Model Comparison")
    st.dataframe(rdf, use_container_width=True)

    st.bar_chart(rdf.set_index("Algorithm"))

    best = rdf.sort_values("Accuracy", ascending=False).iloc[0]

    c1, c2, c3 = st.columns(3)

    c1.metric("Best Model", best["Algorithm"])
    c2.metric("Accuracy", f"{best['Accuracy']*100:.2f}%")
    c3.metric("Models Trained", len(models))

    st.subheader("Predict Food Risk")

    col1, col2 = st.columns(2)

    with col1:
        food = st.selectbox(
            "Food Category",
            enc["Food_Category"].classes_
        )

        temp = st.number_input(
            "Temperature (°C)",
            min_value=0.0,
            max_value=40.0,
            value=5.0
        )

        hum = st.slider(
            "Humidity (%)",
            0,
            100,
            70
        )

        days = st.slider(
            "Storage Days",
            1,
            60,
            5
        )

        pack = st.selectbox(
            "Packaging",
            enc["Packaging"].classes_
        )

    with col2:

        moist = st.slider(
            "Moisture (%)",
            0.0,
            100.0,
            50.0
        )

        ph = st.slider(
            "pH",
            3.0,
            8.0,
            6.5
        )

        ref = st.selectbox(
            "Refrigerated",
            enc["Refrigerated"].classes_
        )

        trans = st.slider(
            "Transport Hours",
            1,
            24,
            5
        )

        alg = st.selectbox(
            "Algorithm",
            list(models.keys())
        )

    if st.button("Predict"):

        row = [[
            enc["Food_Category"].transform([food])[0],
            temp,
            hum,
            days,
            enc["Packaging"].transform([pack])[0],
            moist,
            ph,
            enc["Refrigerated"].transform([ref])[0],
            trans
        ]]

        row = scaler.transform(row)

        pred = trained[alg].predict(row)[0]

        risk = enc["Risk"].inverse_transform([pred])[0]

        st.success(f"Predicted Risk: **{risk}**")

else:
    st.info("Please upload a CSV dataset.")