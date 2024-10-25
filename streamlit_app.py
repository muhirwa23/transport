# Import necessary libraries
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from io import StringIO
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.arima.model import ARIMA
import cityflow
import time
import cv2

# Load Simulated Data
def load_data():
    routes_data = """
    route_id,agency_id,route_short_name,route_long_name,route_type,route_desc
    101,1,101,KBS - Zone I - 101,3,Remera Taxi Park-Sonatubes-Rwandex-CBD
    102,1,102,KBS - Zone I - 102,3,Kabuga-Mulindi-Remera-Sonatubes-Rwandex
    103,1,103,KBS - Zone I - 103,3,Rubilizi-Kabeza-Remera-Sonatubes-CBD
    104,1,104,KBS - Zone I - 104,3,Kanombe-Airport-Sonatubes-CBD
    """
    
    stop_times_data = """
    stop_id,route_id,stop_sequence,arrival_time,departure_time,stop_name
    1,101,1,08:00:00,08:01:00,Remera Taxi Park
    2,101,2,08:05:00,08:06:00,Sonatubes
    3,102,1,08:10:00,08:11:00,Kabuga
    4,102,2,08:15:00,08:16:00,Remera
    """
    
    accident_data = {
        "stop_id": [1, 2, 3, 4],
        "route_id": [101, 102, 103, 104],
        "accident_occurred": [1, 0, 1, 0],
        "severity": [3, 0, 2, 1]
    }

    traffic_congestion_data = {
        "route_id": [101, 102, 103, 104],
        "congestion_level": [2, 1, 0, 2],
        "avg_vehicle_speed": [15, 30, 45, 12],  # Average speed in km/h
        "congestion_desc": ["Heavy traffic", "Moderate traffic", "No traffic", "Heavy traffic"]
    }

    routes_df = pd.read_csv(StringIO(routes_data))
    stop_times_df = pd.read_csv(StringIO(stop_times_data))
    accident_df = pd.DataFrame(accident_data)
    traffic_df = pd.DataFrame(traffic_congestion_data)

    return routes_df, stop_times_df, accident_df, traffic_df

# Load Data
routes_df, stop_times_df, accident_df, traffic_df = load_data()

# Sidebar for route selection and time filtering
st.sidebar.header("Control Panel")
selected_routes = st.sidebar.multiselect(
    "Select Routes", 
    options=routes_df['route_long_name'].tolist(), 
    default=routes_df['route_long_name'].tolist()
)
time_range = st.sidebar.slider("Select Time Range (Hours)", 0, 24, (0, 24), 1)

# Filter stop times based on user inputs
filtered_stop_times = stop_times_df[stop_times_df['route_id'].isin(
    [int(r.split('-')[0]) for r in selected_routes]
)]
filtered_stop_times = filtered_stop_times[
    filtered_stop_times['arrival_time'].str.slice(0, 2).astype(int).between(time_range[0], time_range[1])
]

# Display filtered data
st.subheader("Filtered Stop Times")
st.dataframe(filtered_stop_times)

# --- CityFlow Simulation Integration ---
st.subheader("CityFlow Traffic Simulation")

cityflow_config_path = "./traffic_config.json"  # Ensure this config exists locally
env = cityflow.Engine(cityflow_config_path, thread_num=4)

def run_simulation(steps=100):
    """Run CityFlow simulation and collect metrics."""
    vehicle_counts, travel_times, speeds = [], [], []

    for _ in range(steps):
        env.next_step()
        traffic_data = env.get_lane_vehicle_count()
        
        # Collect data
        vehicle_counts.append(np.sum(list(traffic_data.values())))
        travel_times.append(np.random.uniform(10, 50))  # Placeholder travel times
        speeds.append(np.random.uniform(5, 50))  # Placeholder for vehicle speeds

    return vehicle_counts, travel_times, speeds

vehicle_counts, travel_times, speeds = run_simulation(steps=100)

# Plot simulation results
st.subheader("Simulation Metrics")
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(100)), y=vehicle_counts, mode='lines', name='Vehicle Count'))
fig.add_trace(go.Scatter(x=list(range(100)), y=speeds, mode='lines', name='Speed (km/h)'))
st.plotly_chart(fig)

# --- Advanced Machine Learning Models ---

# Prepare data for LSTM and ARIMA
def prepare_lstm_data(data, n_steps=3):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

X, y = prepare_lstm_data(vehicle_counts)

# Build LSTM Model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, verbose=0)

# ARIMA Model for Comparison
arima_model = ARIMA(vehicle_counts, order=(2, 1, 2))
arima_result = arima_model.fit()

# Predictions
lstm_predictions = model.predict(X)
arima_predictions = arima_result.forecast(steps=10)

st.subheader("ML Model Predictions")
st.write(f"LSTM Predictions: {lstm_predictions.flatten().tolist()}")
st.write(f"ARIMA Predictions: {arima_predictions.tolist()}")

# --- Route Optimization ---
st.subheader("Suggested Route Adjustments")
optimized_routes = traffic_df.sort_values(by='avg_vehicle_speed', ascending=False)
st.dataframe(optimized_routes)

# Footer
st.markdown("""
### City Leadership Dashboard
This dashboard integrates **CityFlow traffic simulations** and **ML models** (LSTM, ARIMA) to predict congestion and suggest route optimizations.
""")
