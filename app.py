import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Muat model
model = joblib.load("model.pkl")

# Muat urutan fitur
feature_order = joblib.load("feature_order.pkl")

# Muat encoder
encoders = joblib.load("encoders.pkl")

# Input manual
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0, step=1)
height = st.number_input("Height (m)", min_value=0.0, step=0.01)
weight = st.number_input("Weight (kg)", min_value=0.0, step=0.01)
calc = st.selectbox("CALC (Sometimes/No)", ["Sometimes", "no"])
favc = st.selectbox("FAVC (Yes/No)", ["yes", "no"])
fcvc = st.number_input("FCVC (Frequency of consuming vegetables)", min_value=0, step=1)
ncp = st.number_input("NCP (Number of main meals)", min_value=0.0, step=0.1)
scc = st.selectbox("SCC (Consumption of food between meals)", ["Sometimes", "no"])
smoke = st.selectbox("SMOKE (Smoking habit)", ["yes", "no"])
ch2o = st.number_input("CH2O (Daily consumption of water)", min_value=0.0, step=0.1)
family_history = st.selectbox("Family History with Overweight (Yes/No)", ["yes", "no"])
faf = st.number_input("FAF (Physical Activity Frequency)", min_value=0.0, step=0.1)
tue = st.number_input("TUE (Time Using Technology for Entertainment)", min_value=0, step=1)
caec = st.selectbox("CAEC (Consumption of alcohol)", ["Sometimes", "no"])
mtrans = st.selectbox("MTRANS (Mode of Transportation)", ["Automobile", "Motorbike", "Public_Transportation", "Walking"])

# Buat DataFrame dari input
inputs = {
    "Gender": gender,
    "Age": age,
    "Height": height,
    "Weight": weight,
    "CALC": calc,
    "FAVC": favc,
    "FCVC": fcvc,
    "NCP": ncp,
    "SCC": scc,
    "SMOKE": smoke,
    "CH2O": ch2o,
    "family_history_with_overweight": family_history,
    "FAF": faf,
    "TUE": tue,
    "CAEC": caec,
    "MTRANS": mtrans
}
X = pd.DataFrame([inputs], columns=feature_order)

# Encode fitur kategorikal
for col, encoder in encoders.items():
    if col in X.columns:
        X[col] = encoder.transform(X[col])

# Prediksi
if st.button("Prediksi"):
    yhat = model.predict(X)[0]
    st.success(f"Prediksi obesitas: **{yhat}**")
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        st.write("Probabilitas per kelas:")
        st.json(dict(zip(model.classes_, [float(p) for p in probs])))
