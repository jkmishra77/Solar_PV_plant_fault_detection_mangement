import streamlit as st
import pandas as pd
import joblib
import numpy as np

# App title and description
st.set_page_config(page_title="Solar PV Fault Detection", page_icon="☀️", layout="wide")
st.title("☀️ Solar PV Plant Fault Detection System")
st.markdown("""
This application predicts faults in solar PV systems using machine learning. 
Select a sample from the dataset to get predictions.
""")

# Load model and artifacts
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/xgboost_model.joblib")
        scaler = joblib.load("models/scaler.joblib")
        label_encoder = joblib.load("models/label_encoder.joblib")
        return model, scaler, label_encoder
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, scaler, label_encoder = load_model()

# Fault type mapping with emojis
FAULT_MAPPING = {
    0: "✅ Normal (no fault)",
    1: "⚠️ Line-line fault type 1", 
    2: "⚠️ Line-line fault type 2",
    3: "🌥️ Partial shading"
}

# Load sample data
@st.cache_data
def load_sample_data():
    try:
        sample_data = pd.read_csv("data/processed/processed_data.csv")
        return sample_data.sample(10)
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return pd.DataFrame()

# Initialize session state
if 'sample_df' not in st.session_state:
    st.session_state.sample_df = load_sample_data()

# Sample Prediction UI
st.header("📊 Sample Prediction")

# Display analysis image with expandable view
st.subheader("📈 Combined Fault Analysis")

try:
    from PIL import Image
    image_path = "models/combined_analysis.png"
    image = Image.open(image_path)

    # Show large preview
    st.image(image, caption="Combined Fault Analysis", use_column_width=True)

    # Expandable full view
    with st.expander("🔍 Click to expand full analysis"):
        st.image(image, caption="Expanded View", use_column_width=True)

except Exception as e:
    st.error(f"Error loading analysis image: {e}")


if not st.session_state.sample_df.empty:
    st.subheader("Samples from Dataset")
    st.dataframe(st.session_state.sample_df[['V', 'I', 'G', 'P', 'fault_type']], width='stretch' , height=200)

    sample_options = [f"Sample {i+1} (Fault: {row['fault_type']})" 
                      for i, row in st.session_state.sample_df.iterrows()]
    selected_index = st.selectbox("Choose a sample to test:", range(len(st.session_state.sample_df)), 
                                  format_func=lambda x: sample_options[x])

    selected_sample = st.session_state.sample_df.iloc[selected_index]

    st.subheader("Selected Sample Details")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Voltage (V)", f"{selected_sample['V']:.2f} V")
        st.metric("Current (I)", f"{selected_sample['I']:.2f} A")
        st.metric("Irradiance (G)", f"{selected_sample['G']:.2f} W/m²")
    with col2:
        st.metric("Power (P)", f"{selected_sample['P']:.2f} W")
        actual_fault = FAULT_MAPPING.get(selected_sample['fault_type'], "Unknown")
        st.metric("Actual Fault", actual_fault)

    if st.button("🔍 Predict Fault", type="primary"):
        if model is not None:
            features = selected_sample[['V', 'I', 'G', 'P', 'P_actual', 'I_G_ratio', 'V_G_ratio', 'efficiency']].values.reshape(1, -1)
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]

            st.subheader("Prediction Result")
            predicted_fault = FAULT_MAPPING.get(prediction, "Unknown")
            if prediction == 0:
                st.success(f"**Prediction:** {predicted_fault}")
            else:
                st.warning(f"**Prediction:** {predicted_fault}")

            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features_scaled)[0]
                st.write("**Confidence Scores:**")
                for i, prob in enumerate(proba):
                    fault_name = FAULT_MAPPING.get(i, f"Fault {i}")
                    st.progress(int(prob * 100), text=f"{fault_name}: {prob:.1%}")


    if st.button("🔄 Refresh Samples"):
        st.session_state.sample_df = load_sample_data()
        st.rerun()
else:
    st.warning("No sample data available. Please check your data files.")
