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

# Load Simulated Data
def load_data():
    routes_data = """
    route_id,agency_id,route_short_name,route_long_name,route_type,route_desc
    101,1,101,KBS - Zone I - 101,3,Remera Taxi Park-Sonatubes-Rwandex-CBD
    102,1,102,KBS - Zone I - 102,3,Kabuga-Mulindi-Remera-Sonatubes-Rwandex Nyabugogo Taxi Park
    103,1,103,KBS - Zone I - 103,3,Rubilizi-Kabeza-Remera-Sonatubes-Rwandex-CBD
    104,1,104,KBS - Zone I - 104,3,Kibaya-Kanombe MH-Airport-Remera-Sonatubes-Rwandex-CBD
    105,1,105,KBS - Zone I - 105,3,Remera Taxi Park-Chez Lando-Kacyiru-Nyabugogo Taxi Park
    """
    
    stop_times_data = """
    stop_id,route_id,stop_sequence,arrival_time,departure_time,stop_name
    1,101,1,08:00:00,08:01:00,Remera Taxi Park
    2,101,2,08:05:00,08:06:00,Sonatubes
    3,101,3,08:10:00,08:11:00,Rwandex
    4,101,4,08:15:00,08:16:00,CBD
    5,102,1,08:00:00,08:01:00,Kabuga
    6,102,2,08:05:00,08:06:00,Remera
    7,102,3,08:10:00,08:11:00,Sonatubes
    8,102,4,08:15:00,08:16:00,Rwandex Nyabugogo Taxi Park
    """
    
    accident_data = {
        "stop_id": [1, 2, 3, 4],
        "route_id": [101, 101, 102, 102],
        "accident_occurred": [1, 0, 1, 0],
        "severity": [3, 0, 2, 0],
    }

    traffic_congestion_data = {
        "route_id": [101, 102, 103, 104, 105],
        "congestion_level": [2, 1, 0, 2, 0],
        "congestion_desc": ["Heavy traffic", "Moderate traffic", "No traffic", "Heavy traffic", "No traffic"]
    }

    routes_df = pd.read_csv(StringIO(routes_data))
    stop_times_df = pd.read_csv(StringIO(stop_times_data))
    accident_df = pd.DataFrame(accident_data)
    traffic_df = pd.DataFrame(traffic_congestion_data)

    return routes_df, stop_times_df, accident_df, traffic_df

# Load the data
routes_df, stop_times_df, accident_df, traffic_df = load_data()

# Convert time columns to datetime
stop_times_df['arrival_time'] = pd.to_datetime(stop_times_df['arrival_time'], format='%H:%M:%S').dt.time
stop_times_df['departure_time'] = pd.to_datetime(stop_times_df['departure_time'], format='%H:%M:%S').dt.time

# Sidebar inputs
st.sidebar.header("Leader Control Panel")
selected_routes = st.sidebar.multiselect("Select Routes", options=routes_df['route_long_name'].tolist(), default=routes_df['route_long_name'].tolist())
selected_stops = st.sidebar.multiselect("Select Stops", options=stop_times_df['stop_name'].unique().tolist(), default=stop_times_df['stop_name'].unique().tolist())
time_range = st.sidebar.slider("Select Time Range (Hours)", min_value=0, max_value=24, value=(0, 24), step=1)

# Filter stop times based on inputs
filtered_stop_times = stop_times_df[stop_times_df['route_id'].isin([int(r.split('-')[0]) for r in selected_routes])]
filtered_stop_times = filtered_stop_times[filtered_stop_times['stop_name'].isin(selected_stops)]

# Time range filtering
filtered_stop_times['arrival_hour'] = pd.to_datetime(filtered_stop_times['arrival_time'].astype(str)).dt.hour
filtered_stop_times = filtered_stop_times[(filtered_stop_times['arrival_hour'] >= time_range[0]) & (filtered_stop_times['arrival_hour'] <= time_range[1])]

# Show filtered stop times
st.subheader("Filtered Stop Times")
st.dataframe(filtered_stop_times)

# Correlation matrix (accident severity, stop sequence, congestion)
st.subheader("Correlation Matrix")
correlation_data = pd.merge(accident_df, traffic_df, on='route_id')
correlation_matrix = correlation_data[['severity', 'congestion_level']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", cbar=True)
st.pyplot()

# Real-time traffic jam tracking using Waze Map
st.subheader("Real-Time Traffic Jam Tracking")
st.components.v1.iframe("https://embed.waze.com/iframe?zoom=10&lat=-1.934712&lon=29.974184&ct=livemap", width=600, height=450)

# Live Congestion Tracking
st.subheader("Real-Time Congestion Chart")
traffic_fig = px.bar(traffic_df, x='route_id', y='congestion_level', color='congestion_desc', title='Real-Time Congestion Tracking', 
                     labels={"congestion_level": "Congestion Level (0: None, 1: Medium, 2: High)"})
st.plotly_chart(traffic_fig)

# Prediction with RandomForest and XGBoost
st.subheader("Predict Traffic Congestion Using Machine Learning")

# Prepare data for ML
encoder = LabelEncoder()
traffic_df['congestion_desc_encoded'] = encoder.fit_transform(traffic_df['congestion_desc'])
X = traffic_df[['route_id']]
y = traffic_df['congestion_desc_encoded']

# Split data and train RandomForest model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, y_pred_rf)

# Train XGBoost model
xg_model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
xg_model.fit(X_train, y_train)
y_pred_xg = xg_model.predict(X_test)
xg_mse = mse(y_test, y_pred_xg)

st.subheader(f"RandomForest MSE: {rf_mse:.2f}, XGBoost MSE: {xg_mse:.2f}")

# Using OpenCV for Traffic Detection (Simulated)
st.subheader("AI-Powered Traffic Density Detection using OpenCV")

# Simulated traffic footage analysis (since we don’t have real video footage, we use placeholders)
st.markdown("**Simulated Traffic Footage Analysis**")
uploaded_file = st.file_uploader("Upload a traffic footage file", type=["mp4", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)

    # Assuming real-time processing happens here (OpenCV logic)
    cap = cv2.VideoCapture(uploaded_file)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect traffic density (Placeholder: count cars, analyze pixel movement, etc.)
        traffic_density = np.random.randint(0, 100)  # Random density for simulation
        frame_count += 1
        st.text(f"Frame {frame_count}: Detected Traffic Density = {traffic_density}")
        if frame_count > 10:
            break
    
    cap.release()

# Footer for City Leadership Insights
st.markdown("""
### City Leadership Dashboard
This application provides city leaders with real-time insights into traffic congestion, stop times, and accidents.
It predicts future congestion using machine learning models and provides AI-powered traffic density detection from live footage.
""")
