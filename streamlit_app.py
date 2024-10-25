import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import time

# --- Initialize Session State ---
if 'traffic_data' not in st.session_state:
    st.session_state.traffic_data = pd.DataFrame(columns=[
        'route', 'timestamp', 'latitude', 'longitude', 'vehicle_count', 'event', 'severity'
    ])
    st.session_state.selected_event = None

# --- Load Route Data ---
@st.cache_data
def load_route_data():
    data = """route_id,agency_id,route_short_name,route_long_name,route_type,route_desc
    101,1,101,Remera Taxi Park-Sonatubes-Rwandex-CBD,3,Zone I
    ...(more routes here)...
    """
    from io import StringIO
    return pd.read_csv(StringIO(data))

routes_df = load_route_data()

# --- Simulate Live Traffic Data with Severity Level ---
def simulate_event():
    route = np.random.choice(routes_df['route_short_name'])
    vehicle_count = np.random.randint(10, 100)
    latitude, longitude = np.random.uniform(-1.96, -1.93), np.random.uniform(30.05, 30.10)
    event = np.random.choice(['Accident', 'Traffic Jam', 'Closed Road', 'Damaged Road'])
    
    severity = "Minor" if vehicle_count < 30 else "Moderate" if vehicle_count < 70 else "Severe"
    return {
        'route': route,
        'timestamp': pd.Timestamp.now(),
        'latitude': latitude,
        'longitude': longitude,
        'vehicle_count': vehicle_count,
        'event': event,
        'severity': severity
    }

# --- Create GPS-based Route Simulation ---
def get_gps_route(start, end, num_points=10):
    """Simulate GPS waypoints between start and end locations."""
    lats = np.linspace(start[1], end[1], num_points)
    lons = np.linspace(start[0], end[0], num_points)
    return [[lon, lat] for lon, lat in zip(lons, lats)]

def create_3d_simulation():
    view_state = pdk.ViewState(
        latitude=-1.9499, longitude=30.0589, zoom=13, pitch=50
    )

    color_map = {
        'Accident': [255, 0, 0],          # Red
        'Traffic Jam': [255, 165, 0],     # Orange
        'Closed Road': [0, 0, 255],       # Blue
        'Damaged Road': [128, 128, 128],  # Gray
    }
    severity_map = {
        "Minor": [100, 100, 100],         # Light gray
        "Moderate": [150, 150, 0],        # Yellow
        "Severe": [255, 0, 0],            # Red
    }

    scatter_data = st.session_state.traffic_data.to_dict('records')

    # Scatter layer with tooltip and click event
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=scatter_data,
        get_position=["longitude", "latitude"],
        get_color=lambda d: color_map.get(d["event"], severity_map[d["severity"]]),
        get_radius=300,
        pickable=True,
        auto_highlight=True,
        tooltip={"text": "{route}\nEvent: {event}\nSeverity: {severity}\nVehicle Count: {vehicle_count}"}
    )

    # Cluster layer for high-density areas
    cluster_layer = pdk.Layer(
        "HeatmapLayer",
        data=scatter_data,
        get_position=["longitude", "latitude"],
        opacity=0.9,
        threshold=0.1,
        aggregation="SUM",
        get_weight="vehicle_count",
    )

    # Display GPS route if an event is selected
    if st.session_state.selected_event:
        event = st.session_state.selected_event
        start = [event["longitude"], event["latitude"]]
        end = [event["longitude"] + 0.01, event["latitude"] + 0.01]  # Example endpoint
        
        # Generate GPS waypoints
        gps_route = get_gps_route(start, end)
        
        path_layer = pdk.Layer(
            "PathLayer",
            data=[{"path": gps_route, "color": [0, 255, 0]}],
            get_path="path",
            get_color="color",
            width_scale=10,
            width_min_pixels=5
        )
        layers = [scatter_layer, cluster_layer, path_layer]
    else:
        layers = [scatter_layer, cluster_layer]

    return pdk.Deck(layers=layers, initial_view_state=view_state)

# --- Handle Event Clicks and Route Simulation ---
def on_event_click(event):
    st.session_state.selected_event = event

# --- User Interface ---
st.title("🚦 Kigali Traffic Monitoring System with Real-Time 3D Event Simulation")

new_data = simulate_event()
st.session_state.traffic_data = pd.concat(
    [st.session_state.traffic_data, pd.DataFrame([new_data])], ignore_index=True
).tail(100)

# Display 3D Map Simulation
st.subheader("🗺️ Real-Time 3D Simulation of Traffic Events")
map_3d = create_3d_simulation()
st.pydeck_chart(map_3d)

# --- KPI Cards ---
st.subheader("🚥 Traffic Key Performance Indicators (KPIs)")
total_events = st.session_state.traffic_data['event'].count()
avg_vehicle_count = st.session_state.traffic_data['vehicle_count'].mean()
most_common_event = st.session_state.traffic_data['event'].mode()[0]

col1, col2, col3 = st.columns(3)
col1.metric("Total Traffic Events", total_events)
col2.metric("Average Vehicle Count", f"{avg_vehicle_count:.2f}")
col3.metric("Most Common Event", most_common_event)

# --- Additional Charts ---
st.subheader("📊 Detailed Traffic Analysis")

# Vehicle Count Histogram
st.write("### Vehicle Count Distribution")
fig_hist = px.histogram(st.session_state.traffic_data, x="vehicle_count", nbins=10, title="Vehicle Count Histogram")
st.plotly_chart(fig_hist, use_container_width=True)

# Event Type Distribution
st.write("### Event Type Breakdown")
fig_pie = px.pie(st.session_state.traffic_data, names="event", title="Event Type Distribution")
st.plotly_chart(fig_pie, use_container_width=True)

# Real-Time Vehicle Count Trend
st.write("### Real-Time Vehicle Count Trends")
fig_line = px.line(
    st.session_state.traffic_data, 
    x='timestamp', y='vehicle_count', 
    title="Vehicle Count over Time", markers=True
)
st.plotly_chart(fig_line, use_container_width=True)

# --- Enhanced Prediction Functionality ---
def predict_traffic():
    data = st.session_state.traffic_data[['vehicle_count', 'latitude', 'longitude']]
    if len(data) > 10:
        X = data[['vehicle_count']]
        y = data['vehicle_count'].rolling(2).mean().fillna(0)
        model = LinearRegression().fit(X, y)
        prediction = model.predict([[80]])  # Predict for a vehicle count of 80
        return prediction[0]
    return None

st.subheader("🔮 Traffic Prediction")
prediction = predict_traffic()
if prediction:
    st.write(f"Predicted Vehicle Count: {int(prediction)} vehicles")

# Periodic Refresh
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 30, 10)
time.sleep(refresh_rate)
st.experimental_rerun()
