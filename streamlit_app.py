import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.graph_objects as go
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import time

# --- Initialize Session State ---
if 'traffic_data' not in st.session_state:
    st.session_state.traffic_data = pd.DataFrame(columns=[
        'route', 'timestamp', 'latitude', 'longitude', 'vehicle_count', 'event', 'average_speed'
    ])

# --- Load Route Data ---
@st.cache_data
def load_route_data():
    data = """... (same CSV data as above) ..."""
    from io import StringIO
    return pd.read_csv(StringIO(data))

routes_df = load_route_data()

# --- Simulate Live Traffic Data ---
def simulate_event():
    route = np.random.choice(routes_df['route_short_name'])
    vehicle_count = np.random.randint(10, 100)
    average_speed = np.random.uniform(10, 60)
    latitude, longitude = np.random.uniform(-1.96, -1.93), np.random.uniform(30.05, 30.10)
    event = np.random.choice(['Accident', 'Traffic Jam', 'Closed Road', 'Damaged Road'])

    return {
        'route': route,
        'timestamp': pd.Timestamp.now(),
        'latitude': latitude,
        'longitude': longitude,
        'vehicle_count': vehicle_count,
        'event': event,
        'average_speed': average_speed
    }

# --- Create 3D Simulation ---
def create_3d_simulation(route_suggestions):
    view_state = pdk.ViewState(latitude=-1.9499, longitude=30.0589, zoom=13, pitch=50)

    # Color map for events
    color_map = {'Accident': [255, 0, 0], 'Traffic Jam': [255, 165, 0], 'Closed Road': [0, 0, 255], 'Damaged Road': [128, 128, 128]}

    scatter_data = st.session_state.traffic_data.to_dict('records')

    scatter_layer = pdk.Layer("ScatterplotLayer", data=scatter_data, get_position=["longitude", "latitude"],
                              get_color=lambda d: color_map.get(d["event"], [0, 255, 0]), get_radius=300, pickable=True, auto_highlight=True)

    text_layer = pdk.Layer("TextLayer", data=scatter_data, get_position=["longitude", "latitude"], get_text="event",
                           get_size=16, get_color=[0, 0, 0], pickable=True)

    route_layer = pdk.Layer("PathLayer", data=route_suggestions, get_path="path", get_width=5, get_color=[0, 255, 0], width_min_pixels=2)

    return pdk.Deck(layers=[scatter_layer, text_layer, route_layer], initial_view_state=view_state)

# --- Predict Traffic Jam ---
def predict_traffic_jam():
    # Data for predictions
    features = st.session_state.traffic_data[['vehicle_count', 'average_speed']].dropna()
    target = np.where(features['vehicle_count'] > 50, 1, 0)  # Define high traffic jam based on vehicle count

    if len(features) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        prediction = model.predict(X_test[-1:])[0]
        return prediction
    return None

# --- Sidebar Features ---
st.sidebar.header("Settings")
st.sidebar.markdown("Control simulation parameters:")
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 30, 10)
vehicle_threshold = st.sidebar.slider("Vehicle Threshold for Jam Prediction", 10, 100, 50)

# --- User Interface ---
st.title("Kigali Transport Optimization Dashboard")

# Real-time Data Cards
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Latest Vehicle Count", int(st.session_state.traffic_data['vehicle_count'].iloc[-1]) if not st.session_state.traffic_data.empty else 0)
with col2:
    st.metric("Average Speed", f"{st.session_state.traffic_data['average_speed'].mean():.2f} km/h" if not st.session_state.traffic_data.empty else "N/A")
with col3:
    traffic_prediction = predict_traffic_jam()
    st.metric("Traffic Jam Prediction", "High" if traffic_prediction and traffic_prediction > 0.5 else "Low")

# Route Suggestion
start_location = st.text_input("Start Location", placeholder="Enter starting point")
end_location = st.text_input("End Location", placeholder="Enter destination")
if st.button("Suggest Route"):
    route_suggestions = [{"path": [[30.0589, -1.9499], [30.0590, -1.9500]]}, {"path": [[30.0589, -1.9499], [30.0592, -1.9502]]}]
    st.pydeck_chart(create_3d_simulation(route_suggestions))

# Update Traffic Data
new_data = simulate_event()
st.session_state.traffic_data = st.session_state.traffic_data.append(new_data, ignore_index=True)

# Visualization of Vehicle Count and Speed
if not st.session_state.traffic_data.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st.session_state.traffic_data['timestamp'], y=st.session_state.traffic_data['vehicle_count'],
                             mode='lines+markers', name='Vehicle Count'))
    fig.add_trace(go.Scatter(x=st.session_state.traffic_data['timestamp'], y=st.session_state.traffic_data['average_speed'],
                             mode='lines+markers', name='Average Speed'))
    fig.update_layout(title="Traffic Analysis Over Time", xaxis_title="Time", yaxis_title="Count / Speed")
    st.plotly_chart(fig)

# New Bar Plot for Events
event_counts = st.session_state.traffic_data['event'].value_counts()
fig2 = go.Figure([go.Bar(x=event_counts.index, y=event_counts.values)])
fig2.update_layout(title="Event Distribution", xaxis_title="Event Type", yaxis_title="Frequency")
st.plotly_chart(fig2)

# Refresh dashboard
time.sleep(refresh_rate)
st.experimental_rerun()
