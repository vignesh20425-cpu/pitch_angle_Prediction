# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. Configuration and Model Loading ---

MODEL_FILENAME = 'multi_output_wind_model_small.pkl'

# Input features (must match training order)
FEATURE_COLUMNS = [
    'Wind Speed (m/s)', 
    'Wind Direction (°)', 
    'Temperature (°C)', 
    'Altitude (m)', 
    'Latitude', 
    'Longitude'
]

# Output variables
OUTPUT_COLUMNS = [
    'Predicted Pitch Angle (°)',  
    'Predicted Max Output (kW)'
]

# Use Streamlit's cache mechanism to load the model only once
@st.cache_resource
def load_model():
    """Loads the pre-trained MultiOutputRegressor model."""
    if not os.path.exists(MODEL_FILENAME):
        st.error(f"Error: Model file '{MODEL_FILENAME}' not found. Please run the training script first.")
        return None
    try:
        model = joblib.load(MODEL_FILENAME)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- 2. Streamlit UI Setup ---

# Set wide layout and title
st.set_page_config(
    page_title="Wind Turbine Performance Predictor",
    layout="wide",
    initial_sidebar_state="auto"
)

st.title("⚡ Wind Turbine Performance Predictor")
st.markdown("---")

# --- 3. Prediction Function ---

def predict_wind_performance(data):
    """
    Takes input data (DataFrame) and returns prediction results.
    """
    if model is None:
        return None
        
    try:
        prediction = model.predict(data)
        
        # Format results
        results = {
            'pitch': f"{prediction[0][0]:.2f}",
            'output': f"{prediction[0][1]:.2f}",
        }
        return results
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# --- 4. Sidebar for Input (The Form) ---

st.sidebar.header("Input Parameters")

# Create sliders/number inputs for all 6 features
with st.sidebar.form("input_form"):
    
    # Use default/example values that make sense for a turbine
    st.markdown("Enter environmental and location data:")
    
    wind_speed = st.number_input(
        "Wind Speed (m/s)", 
        min_value=0.0, max_value=40.0, value=12.5, step=0.1
    )
    wind_direction = st.number_input(
        "Wind Direction (°)", 
        min_value=0.0, max_value=360.0, value=180.0, step=1.0
    )
    temperature = st.number_input(
        "Temperature (°C)", 
        min_value=-50.0, max_value=60.0, value=25.0, step=0.1
    )
    altitude = st.number_input(
        "Altitude (m)", 
        min_value=0.0, max_value=5000.0, value=500.0, step=10.0
    )
    latitude = st.number_input(
        "Latitude (°)", 
        min_value=-90.0, max_value=90.0, value=-35.0, step=0.01
    )
    longitude = st.number_input(
        "Longitude (°)", 
        min_value=-180.0, max_value=180.0, value=150.0, step=0.01
    )
    
    submitted = st.form_submit_button("Get Prediction")

# --- 5. Main Content Display (Results) ---

if submitted:
    if model is None:
        st.error("Cannot run prediction: Model failed to load.")
    else:
        # Create a DataFrame from the user input (model requires DataFrame)
        input_data = pd.DataFrame([[
            wind_speed, wind_direction, temperature, altitude, latitude, longitude
        ]], columns=FEATURE_COLUMNS)
        
        # Run prediction
        results = predict_wind_performance(input_data)
        
        if results:
            st.success("✅ Prediction Successful!")
            
            # Display Outputs (Using Streamlit's metrics for emphasis)
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label=f"1. {OUTPUT_COLUMNS[0]}",
                    value=f"{results['pitch']} °"
                )
            with col2:
                st.metric(
                    label=f"2. {OUTPUT_COLUMNS[1]}",
                    value=f"{results['output']} kW"
                )
            
            st.markdown("---")
            
            # Display Input Confirmation
            st.subheader("Input Parameters Used:")
            
            # Format inputs nicely for display
            display_inputs = pd.DataFrame(
                [[f"{v:.2f}" for v in input_data.iloc[0].values]],
                columns=FEATURE_COLUMNS
            ).T.rename(columns={0: 'Value'})
            
            st.dataframe(display_inputs)