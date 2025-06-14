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

# Muat urutan fitur
feature_order = joblib.load("feature_order.pkl")

# Pastikan urutan fitur sama
X = pd.DataFrame([inputs], columns=feature_order)

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
    Age = st.number_input("Age", min_value=0, step=1, format="%d")
    inputs["Age"] = Age
    
    # Gender
    Gender = st.selectbox("Gender (Male/Female)", ["Male", "Female"])
    inputs["Gender"] = Gender

    # Height
    Height = st.number_input("Height (in meters)", min_value=0.0, step=0.01, format="%.2f")
    inputs["Height"] = Height

    # Weight
    Weight = st.number_input("Weight (in kg)", min_value=0.0, step=0.01, format="%.2f")
    inputs["Weight"] = Weight

    # CALC
    CALC = st.selectbox("CALC (Sometimes/No)", ["Sometimes", "no"])
    inputs["CALC"] = CALC

    # FAVC
    FAVC = st.selectbox("FAVC (Yes/No)", ["yes", "no"])
    inputs["FAVC"] = FAVC

    # FCVC
    FCVC = st.number_input("FCVC (Frequency of consuming vegetables)", min_value=0, step=1, format="%d")
    inputs["FCVC"] = FCVC

    # NCP
    NCP = st.number_input("NCP (Number of main meals)", min_value=0.0, step=0.1, format="%.1f")
    inputs["NCP"] = NCP

    # SCC
    SCC = st.selectbox("SCC (Consumption of food between meals)", ["Sometimes", "no"])
    inputs["SCC"] = SCC

    # SMOKE
    SMOKE = st.selectbox("SMOKE (Smoking habit)", ["yes", "no"])
    inputs["SMOKE"] = SMOKE

    # CH2O
    CH2O = st.number_input("CH2O (Daily consumption of water)", min_value=0.0, step=0.1, format="%.1f")
    inputs["CH2O"] = CH2O

    # family_history_with_overweight
    family_history_with_overweight = st.selectbox("Family History with Overweight (Yes/No)", ["yes", "no"])
    inputs["family_history_with_overweight"] = family_history_with_overweight

    # FAF
    FAF = st.number_input("FAF (Physical Activity Frequency)", min_value=0.0, step=0.1, format="%.1f")
    inputs["FAF"] = FAF

    # TUE
    TUE = st.number_input("TUE (Time Using Technology for Entertainment)", min_value=0, step=1, format="%d")
    inputs["TUE"] = TUE

    # CAEC
    CAEC = st.selectbox("CAEC (Consumption of alcohol)", ["Sometimes", "no"])
    inputs["CAEC"] = CAEC

    # MTRANS
    MTRANS = st.selectbox("MTRANS (Mode of Transportation)", ["Automobile", "Motorbike", "Public_Transportation", "Walking"])
    inputs["MTRANS"] = MTRANS

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
