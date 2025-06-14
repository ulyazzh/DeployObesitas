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
    EXPECTED_FEATURES = ['Gender', 'Age', 'Height', 'Weight']
    dtype_map = {
    "Gender": "object",
    "Age": np.float64,
    "Height": np.float64,
    "Weight": np.float64
}
    inputs = {}

    import streamlit as st

st.title("Isi input manual:")

gender = st.text_input("Gender (Male/Female)")
age = st.number_input("Age", min_value=0, step=1, format="%d")
height = st.number_input("Height (in meters)", step=0.01, format="%.2f")
weight = st.number_input("Weight (in kg)", step=0.01, format="%.2f")

# Tampilkan hasil input
st.subheader("Input untuk prediksi:")
st.json({
    "Gender": gender,
    "Age": age,
    "Height": height,
    "Weight": weight
})

# Buat DataFrame dari input
X = pd.DataFrame([inputs])

    # Encode fitur kategorikal
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = X[col].astype('category').cat.codes

    st.write("Input untuk prediksi:")
    st.json(inputs)

    if st.button("Prediksi"):
        yhat = model.predict(X)[0]
        st.success(f"Prediksi obesitas: **{yhat}**")
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            st.write("Probabilitas per kelas:")
            st.json(dict(zip(model.classes_, [float(p) for p in probs])))
