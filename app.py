import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ================================
# LOAD DATA
# ================================
df = pd.read_csv("dataset.csv")

symptom_cols = [col for col in df.columns if "Symptom" in col]

for col in symptom_cols:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].str.replace(" ", "_")
    df[col] = df[col].str.replace("__", "_")

df[symptom_cols] = df[symptom_cols].replace("nan", "none")
df[symptom_cols] = df[symptom_cols].fillna("none")
df["Disease"] = df["Disease"].str.strip()

# ================================
# FILTER DISEASES
# ================================
top_diseases = df["Disease"].value_counts().head(15).index
df = df[df["Disease"].isin(top_diseases)]

# ================================
# ENCODING
# ================================
X = pd.get_dummies(df[symptom_cols])
y = df["Disease"]

le = LabelEncoder()
y = le.fit_transform(y)

# ================================
# TRAIN MODELS
# ================================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

xgb = XGBClassifier(eval_metric="mlogloss", random_state=42)
xgb.fit(X, y)

# ================================
# LOAD EXTRA DATA
# ================================
precautions_df = pd.read_csv("Disease precaution.csv")
desc_df = pd.read_csv("symptom_Description.csv")

precautions_df["Disease"] = precautions_df["Disease"].str.strip().str.lower().str.replace(" ", "")
desc_df["Disease"] = desc_df["Disease"].str.strip().str.lower().str.replace(" ", "")

precautions_map = {
    row["Disease"]: row[1:].dropna().astype(str).str.strip().values.tolist()
    for _, row in precautions_df.iterrows()
}

desc_df["Description"] = desc_df["Description"].astype(str).str.strip()
desc_map = dict(zip(desc_df["Disease"], desc_df["Description"]))

# ================================
# CLEAN SYMPTOMS FOR UI (FIX)
# ================================
clean_symptoms = []

for col in X.columns:
    symptom = col

    if "_" in symptom:
        symptom = symptom.split("_", 2)[-1]

    symptom = symptom.replace("_", " ")

    if symptom != "none":
        clean_symptoms.append(symptom)

clean_symptoms = sorted(list(set(clean_symptoms)))

# ================================
# UI
# ================================
st.title("Disease Prediction System")

selected_symptoms = st.multiselect("Select Symptoms", clean_symptoms)

# ================================
# PREDICTION
# ================================
if st.button("Predict"):
    input_dict = {col: 0 for col in X.columns}

    for symptom in selected_symptoms:
        formatted = symptom.replace(" ", "_")
        for col in X.columns:
            if formatted in col:
                input_dict[col] = 1

    input_df = pd.DataFrame([input_dict])

    pred_rf = rf.predict(input_df)[0]

    disease_key = (
        le.inverse_transform([pred_rf])[0]
        .strip()
        .lower()
        .replace(" ", "")
    )

    disease_display = le.inverse_transform([pred_rf])[0]

    description = desc_map.get(disease_key, "No description available")
    precautions = precautions_map.get(disease_key, [])

    st.subheader(f"Predicted Disease: {disease_display}")

    st.write("Description")
    st.write(description)

    st.write("Precautions")
    for p in precautions:
        st.write("- ", p)

    st.success("Model Confidence: High")