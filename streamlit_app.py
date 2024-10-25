import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from io import StringIO
from sklearn.preprocessing import LabelEncoder
import cv2
import time
import xgboost as xgb
from sklearn.metrics import mean_squared_error as mse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import cityflow

# Load Simulated Data
def load_data():
    routes_data = """
    route_id,agency_id,route_short_name,route_long_name,route_type,route_desc
    101,1,101,KBS - Zone I - 101,3,Remera Taxi Park-Sonatubes-Rwandex-CBD
    102,1,102,KBS - Zone I - 102,3,Kabuga-Mulindi-Remera-Sonatubes-Rwandex Nyabugogo Taxi Park
    103,1,103,KBS - Zone I - 103,3,Rubilizi-Kabeza-Remera-Sonatubes-Rwandex-CBD
    """
    
    stop_times_data = """
    stop_id,route_id,stop_sequence,arrival_time,departure_time,stop_name
    1,101,1,08:00:00,08:01:00,Remera Taxi Park
    2,101,2,08:05:00,08:06:00,Sonatubes
    """
    
    accident_data = {
        "stop_id": [1, 2, 3, 4],
        "route_id": [101, 101, 102, 102],
        "accident_occurred": [1, 0, 1, 0],
        "severity": [3, 0, 2, 0],
    }

    traffic_congestion_data = {
        "route_id": [101, 102, 103],
        "congestion_level": [2, 1, 0],
        "congestion_desc": ["Heavy traffic", "Moderate traffic", "No traffic"]
    }

    routes_df = pd.read_csv(StringIO(routes_data))
    stop_times_df = pd.read_csv(StringIO(stop_times_data))
    accident_df = pd.DataFrame(accident_data)
    traffic_df = pd.DataFrame(traffic_congestion_data)

    return routes_df, stop_times_df, accident_df, traffic_df

# Load data
routes_df, stop_times_df, accident_df, traffic_df = load_data()

# Sidebar controls
st.sidebar.header("Leader Control Panel")
selected_routes = st.sidebar.multiselect(
    "Select Routes", 
    options=routes_df['route_long_name'].tolist(), 
    default=routes_df['route_long_name'].tolist()
)
time_range = st.sidebar.slider("Select Time Range (Hours)", 0, 24, (0, 24), 1)

# --- CityFlow Integration ---
st.subheader("CityFlow Traffic Simulation")

# Initialize CityFlow environment
cityflow_config = {
    "config": "./traffic_config.json",  # Ensure this JSON config exists locally
    "steps": 100
}
env = cityflow.Engine(cityflow_config["config"], thread_num=4)

def run_cityflow_simulation(steps=100):
    """Run the CityFlow simulation and gather data."""
    vehicle_counts = []
    travel_times = []
    congestion_levels = []

    for _ in range(steps):
        env.next_step()
        traffic_data = env.get_lane_vehicle_count()  # Get traffic info per lane
        
        vehicle_counts.append(np.sum(list(traffic_data.values())))
        travel_times.append(np.random.uniform(10, 60))  # Placeholder for travel time
        congestion_levels.append(np.random.choice([0, 1, 2]))  # Simulated congestion

    return vehicle_counts, travel_times, congestion_levels

# Run the simulation and display data
vehicle_counts, travel_times, congestion_levels = run_cityflow_simulation(cityflow_config["steps"])
st.write(f"Total Vehicles: {sum(vehicle_counts)}")
st.write(f"Average Travel Time: {np.mean(travel_times):.2f} minutes")

# Time-series plot of congestion levels
st.subheader("Real-Time Congestion Tracking")
time_series_fig = px.line(
    x=list(range(len(congestion_levels))),
    y=congestion_levels,
    labels={'x': 'Simulation Step', 'y': 'Congestion Level'},
    title="Congestion Level Over Time"
)
st.plotly_chart(time_series_fig)

# --- Advanced Machine Learning Models ---
st.subheader("Predict Traffic Using Advanced ML Models")

# Prepare data for LSTM model
def prepare_lstm_data(data, n_steps=3):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

X, y = prepare_lstm_data(congestion_levels)

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X.shape[1], 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train the LSTM model
model.fit(X, y, epochs=50, verbose=0)

# Make predictions
predictions = model.predict(X)
st.write(f"LSTM Predictions: {predictions.flatten().tolist()}")

# Display Machine Learning MSE
rf_mse = mean_squared_error(y, predictions.flatten())
st.write(f"LSTM Mean Squared Error: {rf_mse:.2f}")

# Footer
st.markdown("""
### City Leadership Dashboard
This dashboard integrates **CityFlow simulations** with real-time congestion tracking, and uses advanced machine learning models like **LSTM** for traffic forecasting.
""")
