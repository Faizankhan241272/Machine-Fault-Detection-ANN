import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras

st.set_page_config(page_title="Predictive Maintenance", page_icon="🛠️", layout="centered")

# ---------- Styling ----------
st.markdown("""
<style>
.main { background-color: #f7f9fc; }
.stButton>button {
    background: linear-gradient(90deg,#4facfe,#00f2fe);
    color: white; border-radius: 10px; height: 3em; width: 100%;
    font-size: 18px; font-weight: 600; border: none;
}
.result-card {
    padding: 1.5rem; border-radius: 15px; text-align: center;
    margin-top: 1rem; font-size: 22px; font-weight: 700;
}
.safe { background-color: #d4f8e8; color: #0f9d58; }
.danger { background-color: #fde8e8; color: #d93025; }
</style>
""", unsafe_allow_html=True)

st.title("🛠️ Predictive Maintenance System")
st.write("Machine ki readings daalein, model batayega ke failure ka risk hai ya nahi.")

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("ColumnTransformer.pkl")
    model = keras.models.load_model("model.keras")
    return preprocessor, model

preprocessor, model = load_artifacts()

col1, col2 = st.columns(2)

with col1:
    machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
    air_temp = st.number_input("Air Temperature [K]", value=298.0, step=0.1, format="%.1f")
    process_temp = st.number_input("Process Temperature [K]", value=308.5, step=0.1, format="%.1f")

with col2:
    rot_speed = st.number_input("Rotational Speed [rpm]", value=1500, step=1)
    torque = st.number_input("Torque [Nm]", value=40.0, step=0.1, format="%.1f")
    tool_wear = st.number_input("Tool Wear [min]", value=0, step=1)

st.write("")

if st.button("🔍 Predict Failure"):
    input_df = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }])
    input_df["Power"] = input_df["Torque [Nm]"] * input_df["Rotational speed [rpm]"]
    input_df["Temp_diff"] = input_df["Process temperature [K]"] - input_df["Air temperature [K]"]
    input_df["Torque_per_wear"] = input_df["Torque [Nm]"] / (input_df["Tool wear [min]"] + 1)

    X_processed = preprocessor.transform(input_df)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    prob = float(model.predict(X_processed, verbose=0)[0][0])
    pred = prob >= 0.5

    if pred:
        st.markdown(f"<div class='result-card danger'>⚠️ Failure Likely<br>Risk Score: {prob*100:.1f}%</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='result-card safe'>✅ Machine Healthy<br>Risk Score: {prob*100:.1f}%</div>", unsafe_allow_html=True)

    st.progress(min(max(prob, 0.0), 1.0))

st.markdown("---")
st.caption("Model: Neural Network trained on AI4I 2020 Predictive Maintenance Dataset")
