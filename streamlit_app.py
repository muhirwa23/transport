import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk  # For 3D visualization
import plotly.express as px
import requests
import time
from sklearn.ensemble import RandomForestRegressor

# --- Initialize Session State ---
if 'traffic_data' not in st.session_state:
    st.session_state.traffic_data = pd.DataFrame(columns=[
        'route', 'timestamp', 'latitude', 'longitude', 'vehicle_count', 'event'
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

# --- Simulate Live Traffic Data with Multiple Events ---
def simulate_event():
    route = np.random.choice(routes_df['route_short_name'])
    vehicle_count = np.random.randint(10, 100)
    latitude, longitude = np.random.uniform(-1.96, -1.93), np.random.uniform(30.05, 30.10)
    event = np.random.choice(['Accident', 'Traffic Jam', 'Closed Road', 'Damaged Road'])

    return {
        'route': route,
        'timestamp': pd.Timestamp.now(),
        'latitude': latitude,
        'longitude': longitude,
        'vehicle_count': vehicle_count,
        'event': event
    }

# --- Create 3D Simulation with Enhanced Layer ---
def create_3d_simulation():
    view_state = pdk.ViewState(
        latitude=-1.9499, longitude=30.0589, zoom=13, pitch=50
    )

    # Define color map for events
    color_map = {
        'Accident': [255, 0, 0],      # Red
        'Traffic Jam': [255, 165, 0], # Orange
        'Closed Road': [0, 0, 255],   # Blue
        'Damaged Road': [128, 128, 128]  # Gray
    }

    # Prepare data for scatter layer
    scatter_data = st.session_state.traffic_data.to_dict('records')
    
    # Scatter layer to visualize events with colors
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=scatter_data,
        get_position=["longitude", "latitude"],
        get_color=lambda d: color_map.get(d["event"], [0, 255, 0]),
        get_radius=300,
        pickable=True,
        auto_highlight=True
    )

    # TextLayer to label each event on the map
    text_layer = pdk.Layer(
        "TextLayer",
        data=scatter_data,
        get_position=["longitude", "latitude"],
        get_text="event",
        get_size=16,
        get_color=[0, 0, 0],
        pickable=True
    )

    return pdk.Deck(layers=[scatter_layer, text_layer], initial_view_state=view_state)

# --- User Interface ---
st.title("🚦 Kigali Traffic Monitoring System with Real-Time 3D Event Simulation")

# Add new simulated traffic data to the session state
new_data = simulate_event()
st.session_state.traffic_data = pd.concat(
    [st.session_state.traffic_data, pd.DataFrame([new_data])], ignore_index=True
).tail(100)  # Keep the latest 100 events

# Display 3D Map Simulation
st.subheader("🗺️ Real-Time 3D Simulation of Traffic Events")
st.pydeck_chart(create_3d_simulation())

# Plot Real-Time Vehicle Count Trends
st.subheader("📈 Real-Time Vehicle Count Trends")
fig = px.line(
    st.session_state.traffic_data, 
    x='timestamp', y='vehicle_count', 
    title="Vehicle Count over Time", markers=True
)
st.plotly_chart(fig, use_container_width=True)

# Predict Traffic Congestion Using Random Forest
def predict_traffic():
    data = st.session_state.traffic_data[['vehicle_count']]
    if len(data) > 10:
        X = np.array(data.index).reshape(-1, 1)  # Use index as feature
        y = data['vehicle_count']
        model = RandomForestRegressor().fit(X, y)
        prediction = model.predict([[len(data) + 1]])  # Predict next value
        return prediction[0]
    return None

st.subheader("🔮 Traffic Prediction")
prediction = predict_traffic()
if prediction:
    st.write(f"Predicted Vehicle Count: {int(prediction)} vehicles")

# Refresh the dashboard periodically
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 30, 10)
time.sleep(refresh_rate)
st.experimental_rerun()
