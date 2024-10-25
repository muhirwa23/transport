import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.graph_objects as go
import requests
from sklearn.ensemble import RandomForestRegressor
import time

# --- Initialize Session State ---
if 'traffic_data' not in st.session_state:
    st.session_state.traffic_data = pd.DataFrame(columns=[
        'route', 'timestamp', 'latitude', 'longitude', 'vehicle_count', 'event', 'average_speed'
    ])

# --- Load Route Data ---
@st.cache_data
def load_route_data():
    data = """route_id,agency_id,route_short_name,route_long_name,route_type,route_desc
    101,1,101,KBS - Zone I - 101,3,Remera Taxi Park-Sonatubes-Rwandex-CBD
    102,1,102,KBS - Zone I - 102,3,Kabuga-Mulindi-Remera-Sonatubes-Rwandex Nyabugogo Taxi Park
    103,1,103,KBS - Zone I - 103,3,Rubilizi-Kabeza-Remera-Sonatubes-Rwandex-CBD
    104,1,104,KBS - Zone I - 104,3,Kibaya-Kanombe MH-Airport-Remera-Sonatubes-Rwandex-CBD
    105,1,105,KBS - Zone I - 105,3,Remera Taxi Park-Chez Lando-Kacyiru-Nyabugogo Taxi Park
    106,1,106,KBS - Zone I - 106,3,Remera Taxi Park-15-Ndera-Musave
    107,1,107,KBS - Zone I - 107,3,Remera Taxi Park-Mulindi-Masaka
    108,1,108,KBS - Zone I - 108,3,Remera Taxi Park-Sonatubes-Nyanza Taxi Park
    109,1,109,KBS - Zone I - 109,3,Remera Taxi Park-Sonatubes-Rwandex-Gikondo (Nyenyeli)-Bwerankoli
    111,1,111,KBS - Zone I - 111,3,Kabuga-Mulindi-Remera Taxi Park
    112,1,112,KBS - Zone I - 112,3,Remera Taxi Park-Sonatubes-Rwandex-Nyabugogo Taxi Park
    113,1,113,KBS - Zone I - 113,3,Busanza-Rubilizi-Kabeza-Remera Taxi Park
    114,1,114,KBS - Zone I - 114,3,Kibaya-Kanombe MH-Airport-Remera Taxi Park
    115,1,115,KBS - Zone I - 115,3,Remera Taxi Park-Airport-Busanza
    """
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
    view_state = pdk.ViewState(
        latitude=-1.9499, longitude=30.0589, zoom=13, pitch=50
    )

    # Color map for events
    color_map = {
        'Accident': [255, 0, 0],
        'Traffic Jam': [255, 165, 0],
        'Closed Road': [0, 0, 255],
        'Damaged Road': [128, 128, 128]
    }

    scatter_data = st.session_state.traffic_data.to_dict('records')

    # Scatter layer
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=scatter_data,
        get_position=["longitude", "latitude"],
        get_color=lambda d: color_map.get(d["event"], [0, 255, 0]),
        get_radius=300,
        pickable=True,
        auto_highlight=True
    )

    # TextLayer to label events
    text_layer = pdk.Layer(
        "TextLayer",
        data=scatter_data,
        get_position=["longitude", "latitude"],
        get_text="event",
        get_size=16,
        get_color=[0, 0, 0],
        pickable=True
    )

    # Layer for suggested routes
    route_layer = pdk.Layer(
        "PathLayer",
        data=route_suggestions,
        get_path="path",
        get_width=5,
        get_color=[0, 255, 0],
        width_min_pixels=2
    )

    return pdk.Deck(layers=[scatter_layer, text_layer, route_layer], initial_view_state=view_state)

# --- Predict Traffic Jam ---
def predict_traffic_jam():
    # Mock prediction model
    model = RandomForestRegressor()
    model.fit(np.array([[10, 20], [20, 30], [30, 40]]), np.array([0, 1, 1]))  # Example data
    return model.predict(np.array([[40, 50]]))

# --- User Interface ---
st.title("Kigali Transport Optimization Dashboard")

# Create statistics cards for real-time data
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Latest Vehicle Count", int(st.session_state.traffic_data['vehicle_count'].iloc[-1]) if not st.session_state.traffic_data.empty else 0)
with col2:
    st.metric("Average Speed", f"{st.session_state.traffic_data['average_speed'].mean():.2f} km/h" if not st.session_state.traffic_data.empty else "N/A")
with col3:
    st.metric("Traffic Jam Prediction", "Yes" if predict_traffic_jam()[0] > 0 else "No")

# Input for route suggestions
start_location = st.text_input("Start Location", placeholder="Enter starting point")
end_location = st.text_input("End Location", placeholder="Enter destination")

# Button to generate suggested routes
if st.button("Suggest Route"):
    # Generate mock route suggestions
    route_suggestions = [
        {"path": [[30.0589, -1.9499], [30.0590, -1.9500]]},  # Mock path coordinates
        {"path": [[30.0589, -1.9499], [30.0592, -1.9502]]}   # Mock path coordinates
    ]
    
    # Display the 3D simulation with route suggestions
    st.pydeck_chart(create_3d_simulation(route_suggestions))

# Simulate traffic data and update the session state
new_data = simulate_event()
st.session_state.traffic_data = st.session_state.traffic_data.append(new_data, ignore_index=True)

# Multi-plot visualization for traffic analysis
if not st.session_state.traffic_data.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st.session_state.traffic_data['timestamp'], 
                             y=st.session_state.traffic_data['vehicle_count'],
                             mode='lines+markers', name='Vehicle Count'))
    fig.add_trace(go.Scatter(x=st.session_state.traffic_data['timestamp'], 
                             y=st.session_state.traffic_data['average_speed'],
                             mode='lines+markers', name='Average Speed'))
    fig.update_layout(title="Traffic Analysis Over Time", xaxis_title="Time", yaxis_title="Count / Speed")
    st.plotly_chart(fig)

# Refresh the dashboard periodically
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 30, 10)
time.sleep(refresh_rate)
st.experimental_rerun()
