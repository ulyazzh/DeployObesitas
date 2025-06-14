import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set config halaman
st.set_page_config(page_title="Prediksi Obesitas", layout="centered")
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    .stTextInput, .stNumberInput, .stSelectbox {
        border-radius: 8px;
        background-color: #ffffff;
        padding: 0.25em;
    }
    </style>
""", unsafe_allow_html=True)

st.title("\U0001F4CA Aplikasi Prediksi Tingkat Obesitas")
st.markdown("""
Gunakan aplikasi ini untuk memprediksi tingkat obesitas berdasarkan data pribadi dan kebiasaan hidup.
Silakan unggah file CSV atau isi data secara manual.
""")

# Load model
@st.cache_data
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# Upload file CSV
uploaded = st.file_uploader("\U0001F4C2 Upload file CSV data pasien", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.markdown("**Data Preview:**")
    st.dataframe(df.head(), use_container_width=True)

st.markdown("---")
st.subheader("\U0001F4DD Input Data Manual")

EXPECTED_FEATURES = [
    'Age', 'Gender', 'Height', 'Weight', 'CALC', 'FAVC', 'FCVC', 'NCP', 'SCC',
    'SMOKE', 'CH2O', 'family_history_with_overweight', 'FAF', 'TUE', 'CAEC', 'MTRANS'
]

inputs = {}

cols = st.columns(2)
inputs["Age"] = cols[0].number_input("Usia", min_value=0, step=1)
inputs["Gender"] = cols[1].selectbox("Jenis Kelamin", ["Male", "Female"])

cols = st.columns(2)
inputs["Height"] = cols[0].number_input("Tinggi Badan (m)", min_value=0.0, step=0.01)
inputs["Weight"] = cols[1].number_input("Berat Badan (kg)", min_value=0.0, step=0.01)

cols = st.columns(2)
inputs["CALC"] = cols[0].selectbox("Konsumsi Alkohol", ["Sometimes", "no"])
inputs["FAVC"] = cols[1].selectbox("Konsumsi Makanan Tinggi Kalori", ["yes", "no"])

cols = st.columns(2)
inputs["FCVC"] = cols[0].number_input("Frekuensi Konsumsi Sayur", min_value=0, step=1)
inputs["NCP"] = cols[1].number_input("Jumlah Makan Utama per Hari", min_value=0.0, step=0.1)

cols = st.columns(2)
inputs["SCC"] = cols[0].selectbox("Makan di Luar Waktu Makan", ["Sometimes", "no"])
inputs["SMOKE"] = cols[1].selectbox("Merokok", ["yes", "no"])

cols = st.columns(2)
inputs["CH2O"] = cols[0].number_input("Konsumsi Air per Hari (liter)", min_value=0.0, step=0.1)
inputs["family_history_with_overweight"] = cols[1].selectbox("Riwayat Keluarga Obesitas", ["yes", "no"])

cols = st.columns(2)
inputs["FAF"] = cols[0].number_input("Aktivitas Fisik (jam/minggu)", min_value=0.0, step=0.1)
inputs["TUE"] = cols[1].number_input("Waktu Hiburan Teknologi (jam)", min_value=0, step=1)

cols = st.columns(2)
inputs["CAEC"] = cols[0].selectbox("Kebiasaan Konsumsi Alkohol", ["Sometimes", "no"])
inputs["MTRANS"] = cols[1].selectbox("Transportasi Utama", ["Automobile", "Motorbike", "Public_Transportation", "Walking"])

X = pd.DataFrame([inputs])
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    X[col] = X[col].astype('category').cat.codes

st.markdown("---")
st.markdown("### \U0001F50E Ringkasan Input")
st.json(inputs)

if st.button("\U0001F52E Prediksi Obesitas"):
    yhat = model.predict(X)[0]
    
    # Mapping hasil prediksi ke nama kelas
    if hasattr(model, "classes_"):
        class_names = model.classes_
        predicted_class = class_names[yhat]
    else:
        predicted_class = str(yhat)

    # Deskripsi hasil prediksi dalam Bahasa Indonesia
    prediction_description = {
        'Insufficient_Weight': "Berat badan anda kurang",
        'Normal_Weight': "Berat badan anda Normal",
        'Overweight_Level_I': "Anda Kelebihan berat badan level I",
        'Overweight_Level_II': "Anda Kelebihan berat badan level II",
        'Obesity_Type_I': "Anda mengalami Obesitas Tipe I",
        'Obesity_Type_II': "Anda mengalami Obesitas Tipe II",
        'Obesity_Type_III': "Anda mengalami Obesitas Tipe III"
    }

    result_text = prediction_description.get(predicted_class, f"Kelas tidak dikenali: {predicted_class}")

    st.markdown("### 🧾 Hasil Prediksi")
    st.success(f"**{result_text}**")

    # Jika model support probabilitas
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        st.markdown("**Probabilitas Kelas:**")
        st.json(dict(zip(class_names, [float(p) for p in probs])))
