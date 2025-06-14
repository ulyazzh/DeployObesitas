import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set config halaman
st.set_page_config(page_title="Prediksi Obesitas", layout="centered")
st.title("Prediksi Tingkat Obesitas")

# Load model
@st.cache_data
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# Upload file CSV
uploaded = st.file_uploader("Upload file CSV data pasien", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)

    st.write("Data preview:")
    st.dataframe(df.head())

    # Kolom yang digunakan saat training
    EXPECTED_FEATURES = [
        'Age', 'Gender', 'Height', 'Weight', 'CALC', 'FAVC', 'FCVC', 'NCP', 'SCC',
        'SMOKE', 'CH2O', 'family_history_with_overweight', 'FAF', 'TUE', 'CAEC', 'MTRANS'
    ]

    dtype_map = {
        "Age": np.float64,
        "Gender": "object",
        "Height": np.float64,
        "Weight": np.float64,
        "CALC": "object",
        "FAVC": "object",
        "FCVC": np.int64,
        "NCP": np.float64,
        "SCC": "object",
        "SMOKE": "object",
        "CH2O": np.float64,
        "family_history_with_overweight": "object",
        "FAF": np.float64,
        "TUE": np.int64,
        "CAEC": "object",
        "MTRANS": "object"
    }

    inputs = {}

    # Isi input manual
    st.subheader("Isi input manual:")

    # Age
    age = st.number_input("Age", min_value=0, step=1, format="%d")
    inputs["Age"] = age
    
    # Gender
    gender = st.selectbox("Gender (Male/Female)", ["Male", "Female"])
    inputs["Gender"] = gender

    # Height
    height = st.number_input("Height (in meters)", min_value=0.0, step=0.01, format="%.2f")
    inputs["Height"] = height

    # Weight
    weight = st.number_input("Weight (in kg)", min_value=0.0, step=0.01, format="%.2f")
    inputs["Weight"] = weight

    # CALC
    calc = st.selectbox("CALC (Sometimes/No)", ["Sometimes", "no"])
    inputs["CALC"] = calc

    # FAVC
    favc = st.selectbox("FAVC (Yes/No)", ["yes", "no"])
    inputs["FAVC"] = favc

    # FCVC
    fcvc = st.number_input("FCVC (Frequency of consuming vegetables)", min_value=0, step=1, format="%d")
    inputs["FCVC"] = fcvc

    # NCP
    ncp = st.number_input("NCP (Number of main meals)", min_value=0.0, step=0.1, format="%.1f")
    inputs["NCP"] = ncp

    # SCC
    scc = st.selectbox("SCC (Consumption of food between meals)", ["Sometimes", "no"])
    inputs["SCC"] = scc

    # SMOKE
    smoke = st.selectbox("SMOKE (Smoking habit)", ["yes", "no"])
    inputs["SMOKE"] = smoke

    # CH2O
    ch2o = st.number_input("CH2O (Daily consumption of water)", min_value=0.0, step=0.1, format="%.1f")
    inputs["CH2O"] = ch2o

    # family_history_with_overweight
    family_history = st.selectbox("Family History with Overweight (Yes/No)", ["yes", "no"])
    inputs["family_history_with_overweight"] = family_history

    # FAF
    faf = st.number_input("FAF (Physical Activity Frequency)", min_value=0.0, step=0.1, format="%.1f")
    inputs["FAF"] = faf

    # TUE
    tue = st.number_input("TUE (Time Using Technology for Entertainment)", min_value=0, step=1, format="%d")
    inputs["TUE"] = tue

    # CAEC
    caec = st.selectbox("CAEC (Consumption of alcohol)", ["Sometimes", "no"])
    inputs["CAEC"] = caec

    # MTRANS
    mtrans = st.selectbox("MTRANS (Mode of Transportation)", ["Automobile", "Motorbike", "Public_Transportation", "Walking"])
    inputs["MTRANS"] = mtrans

    # Buat DataFrame dari input
    X = pd.DataFrame([inputs])

    # Encode fitur kategorikal
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = X[col].astype('category').cat.codes

    st.subheader("Input untuk prediksi:")
    st.json(inputs)

    if st.button("Prediksi"):
        yhat = model.predict(X)[0]
        st.success(f"Prediksi obesitas: **{yhat}**")
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            st.write("Probabilitas per kelas:")
            st.json(dict(zip(model.classes_, [float(p) for p in probs])))
