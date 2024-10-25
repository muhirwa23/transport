import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import time

# --- Initialize Session State ---
if 'traffic_data' not in st.session_state:
    st.session_state.traffic_data = pd.DataFrame(columns=[
        'route', 'timestamp', 'latitude', 'longitude', 'vehicle_count', 'event', 'average_speed'
    ])

# --- Load Route Data with Coordinates ---
@st.cache_data
def load_route_data():
    # Sample routes with actual coordinates for each path point
    routes_data = [
        {
            "route": "Route A",
            "coordinates": [
                [30.0605, -1.9441], [30.0615, -1.9451], [30.0625, -1.9461], [30.0635, -1.9471]
            ],
            "description": "Test Route A - Sample Path"
        },
        {
            "route": "Route B",
            "coordinates": [
                [30.0689, -1.9425], [30.0699, -1.9435], [30.0709, -1.9445], [30.0719, -1.9455]
            ],
            "description": "Test Route B - Sample Path"
        },
        {
            "route": "Route C",
            "coordinates": [
                [30.0550, -1.9500], [30.0560, -1.9510], [30.0570, -1.9520], [30.0580, -1.9530]
            ],
            "description": "Test Route C - Sample Path"
        },
    ]
    return routes_data

routes_df = load_route_data()

# --- Simulate Live Traffic Data ---
def simulate_event():
    route = np.random.choice([route["route"] for route in routes_df])
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

# --- Create 3D Simulation with Actual Routes ---
def create_3d_simulation():
    view_state = pdk.ViewState(latitude=-1.9499, longitude=30.0589, zoom=13, pitch=50)

    color_map = {'Accident': [255, 0, 0], 'Traffic Jam': [255, 165, 0], 'Closed Road': [0, 0, 255], 'Damaged Road': [128, 128, 128]}
    scatter_data = st.session_state.traffic_data.to_dict('records')

    scatter_layer = pdk.Layer(
        "ScatterplotLayer", data=scatter_data, get_position=["longitude", "latitude"],
        get_color=lambda d: color_map.get(d["event"], [0, 255, 0]), get_radius=300,
        pickable=True, auto_highlight=True
    )

    text_layer = pdk.Layer(
        "TextLayer", data=scatter_data, get_position=["longitude", "latitude"], get_text="event",
        get_size=16, get_color=[0, 0, 0], pickable=True
    )

    # Route Layer with Actual Coordinates
    route_layers = []
    for route in routes_df:
        route_layer = pdk.Layer(
            "PathLayer", data=[{"path": route["coordinates"]}], get_path="path",
            get_width=5, get_color=[0, 255, 0], width_min_pixels=2
        )
        route_layers.append(route_layer)

    return pdk.Deck(layers=[scatter_layer, text_layer] + route_layers, initial_view_state=view_state)

# --- Predict Traffic Jam ---
def predict_traffic_jam():
    if len(st.session_state.traffic_data) < 10:
        return None

    X = st.session_state.traffic_data[['vehicle_count', 'average_speed']].dropna()
    y = np.where(X['vehicle_count'] > 50, 1, 0)

    model = RandomForestRegressor()
    model.fit(X, y)
    return model.predict([[60, 40]])[0]  # Example input

# --- User Interface ---
st.title("Kigali Transport Optimization Dashboard")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Latest Vehicle Count", int(st.session_state.traffic_data['vehicle_count'].iloc[-1]) if not st.session_state.traffic_data.empty else 0)
with col2:
    st.metric("Average Speed", f"{st.session_state.traffic_data['average_speed'].mean():.2f} km/h" if not st.session_state.traffic_data.empty else "N/A")
with col3:
    traffic_prediction = predict_traffic_jam()
    st.metric("Traffic Jam Prediction", "High" if traffic_prediction and traffic_prediction > 0.5 else "Low")

# Display Routes on Map
st.pydeck_chart(create_3d_simulation())

new_data = simulate_event()
st.session_state.traffic_data = st.session_state.traffic_data.append(new_data, ignore_index=True)

# --- Seaborn Visualization: FacetGrid, Swarmplot, and Relplot ---
if not st.session_state.traffic_data.empty:
    st.subheader("Vehicle Count by Event Type")
    facet_grid_fig = sns.FacetGrid(st.session_state.traffic_data, col="event", height=3, aspect=1)
    facet_grid_fig.map_dataframe(sns.histplot, x="vehicle_count", kde=True)
    st.pyplot(facet_grid_fig)

    st.subheader("Swarmplot of Average Speed per Event Type")
    plt.figure(figsize=(10, 5))
    swarm_fig = sns.swarmplot(x="event", y="average_speed", data=st.session_state.traffic_data)
    st.pyplot(swarm_fig.figure)

    st.subheader("Vehicle Count Over Time by Event Type")
    rel_fig = sns.relplot(x="timestamp", y="vehicle_count", hue="event", kind="line", data=st.session_state.traffic_data, height=5, aspect=2)
    st.pyplot(rel_fig)
