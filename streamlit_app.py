import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from io import StringIO

# --- Load Route Data ---
@st.cache_data
def load_route_data():
    """Load the complete route data."""
    data = """route_id,agency_id,route_short_name,route_long_name,route_type,route_desc
    101,1,101,KBS - Zone I - 101,3,Remera Taxi Park-Sonatubes-Rwandex-CBD
    102,1,102,Kabuga-Mulindi-Remera-Sonatubes-Rwandex-Nyabugogo Taxi Park
    212,2,212,ROYAL - Zone II - 212,3,St. Joseph-Kicukiro Centre-Sonatubes-Rwandex-Nyabugogo Taxi Park
    """
    return pd.read_csv(StringIO(data))

routes_df = load_route_data()

# --- Initialize Session State ---
if 'traffic_data' not in st.session_state:
    st.session_state.traffic_data = pd.DataFrame(columns=[
        'route', 'timestamp', 'vehicle_count', 'travel_time', 'latitude', 'longitude'
    ])

if 'event_data' not in st.session_state:
    st.session_state.event_data = pd.DataFrame(columns=['latitude', 'longitude', 'event_time', 'event_type'])

# --- Generate Live Traffic Data ---
def generate_live_traffic_data():
    route = np.random.choice(routes_df['route_short_name'])
    vehicle_count = np.random.randint(10, 100)
    travel_time = np.random.uniform(10, 60)
    timestamp = pd.Timestamp.now()
    latitude = -1.9499 + np.random.uniform(-0.01, 0.01)
    longitude = 30.0589 + np.random.uniform(-0.01, 0.01)
    return {
        'route': route, 'timestamp': timestamp, 
        'vehicle_count': vehicle_count, 'travel_time': travel_time,
        'latitude': latitude, 'longitude': longitude
    }

def generate_event_data():
    event_types = ['Traffic Jam', 'Accident', 'Road Break', 'Congestion']
    latitude = -1.9499 + np.random.uniform(-0.02, 0.02)
    longitude = 30.0589 + np.random.uniform(-0.02, 0.02)
    event_time = pd.Timestamp.now()
    event_type = np.random.choice(event_types)
    return {'latitude': latitude, 'longitude': longitude, 'event_time': event_time, 'event_type': event_type}

# --- UI Display ---
st.title("🚦 Kigali Traffic Monitoring and Prediction System")

# --- Sidebar Filters ---
selected_routes = st.sidebar.multiselect(
    "Select Routes", routes_df['route_short_name'].unique(), default=[]
)
min_vehicle_count = st.sidebar.slider("Min Vehicle Count", 0, 100, 10)
max_travel_time = st.sidebar.slider("Max Travel Time (minutes)", 10, 60, 30)

# --- Update Data ---
new_traffic_data = generate_live_traffic_data()
st.session_state.traffic_data = pd.concat(
    [st.session_state.traffic_data, pd.DataFrame([new_traffic_data])], ignore_index=True
).tail(50)

new_event = generate_event_data()
st.session_state.event_data = pd.concat(
    [st.session_state.event_data, pd.DataFrame([new_event])], ignore_index=True
).tail(20)

# --- KPI Cards ---
st.header("Key Performance Indicators")
col1, col2, col3 = st.columns(3)

with col1:
    avg_vehicle_count = st.session_state.traffic_data['vehicle_count'].mean() if not st.session_state.traffic_data.empty else 0
    st.metric("Avg Vehicle Count", f"{avg_vehicle_count:.2f}")

with col2:
    avg_travel_time = st.session_state.traffic_data['travel_time'].mean() if not st.session_state.traffic_data.empty else 0
    st.metric("Avg Travel Time (min)", f"{avg_travel_time:.2f}")

with col3:
    congestion_level = "High" if avg_vehicle_count > 50 else "Low"
    st.metric("Congestion Level", congestion_level)

# --- 3D Map of Kigali ---
# Create the map figure
fig_map = go.Figure()

# Add markers for each event type
for i in range(len(st.session_state.event_data)):
    event_type = st.session_state.event_data['event_type'].iloc[i]
    fig_map.add_trace(go.Scattermapbox(
        lat=[st.session_state.event_data['latitude'].iloc[i]],
        lon=[st.session_state.event_data['longitude'].iloc[i]],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=10,
            color={
                'Traffic Jam': 'rgb(255,0,0)',  # Red for traffic jams
                'Accident': 'rgb(255,69,0)',    # Orange for accidents
                'Road Break': 'rgb(255,255,0)',  # Yellow for road breaks
                'Congestion': 'rgb(0,255,0)'     # Green for congestion
            }.get(event_type, 'rgb(0,0,255)'),  # Default to blue if unknown
            opacity=0.7
        ),
        text=f"{event_type} at {st.session_state.event_data['event_time'].iloc[i]}",
        hoverinfo='text'
    ))

# Map layout
fig_map.update_layout(
    mapbox=dict(
        style="mapbox://styles/mapbox/streets-v11",  # 3D map style
        center=dict(lat=-1.9499, lon=30.0589),  # Kigali's coordinates
        zoom=12,
        accesstoken="your_mapbox_access_token",  # Add your Mapbox access token
        pitch=60,  # Set the pitch for a 3D effect
        bearing=0,  # Set the bearing angle
    ),
    margin=dict(r=10, l=10, b=10, t=10),
    title="Live 3D Map of Kigali Traffic Events"
)

# Display the map
st.plotly_chart(fig_map, use_container_width=True)

# --- Real-Time Vehicle Count Chart ---
st.subheader("📈 Real-Time Vehicle Count")
line_fig = px.line(
    st.session_state.traffic_data, x='timestamp', y='vehicle_count',
    title="Vehicle Count Over Time", markers=True,
    color_discrete_sequence=["rgb(52,152,219)"]
)
st.plotly_chart(line_fig, use_container_width=True)

# --- Prediction of Future Vehicle Counts ---
st.subheader("🔮 Vehicle Count Prediction")

# Prepare data for prediction
if not st.session_state.traffic_data.empty:
    timestamps = np.arange(len(st.session_state.traffic_data)).reshape(-1, 1)
    vehicle_counts = st.session_state.traffic_data['vehicle_count'].values

    poly = PolynomialFeatures(degree=2)
    timestamps_poly = poly.fit_transform(timestamps)
    model = LinearRegression().fit(timestamps_poly, vehicle_counts)

    # Predict future vehicle counts
    future_timestamps = np.arange(len(timestamps), len(timestamps) + 10).reshape(-1, 1)
    future_timestamps_poly = poly.transform(future_timestamps)
    future_vehicle_counts = model.predict(future_timestamps_poly)

    # Prepare plot data
    pred_fig = px.line(
        x=np.concatenate([timestamps.flatten(), future_timestamps.flatten()]),
        y=np.concatenate([vehicle_counts, future_vehicle_counts]),
        labels={'x': 'Time', 'y': 'Vehicle Count'},
        title="Predicted vs Observed Vehicle Counts"
    )
    pred_fig.add_scatter(x=timestamps.flatten(), y=vehicle_counts, mode='markers', name='Observed')
    pred_fig.add_scatter(x=future_timestamps.flatten(), y=future_vehicle_counts, mode='lines', name='Predicted')
    st.plotly_chart(pred_fig, use_container_width=True)

# --- Dynamic Travel Time Chart ---
st.subheader("⏱️ Avg Travel Time per Route")
avg_travel_time_fig = px.bar(
    st.session_state.traffic_data.groupby("route")['travel_time'].mean().reset_index(),
    x='route', y='travel_time',
    title="Avg Travel Time per Route",
    labels={'travel_time': 'Avg Travel Time (min)'},
    color_discrete_sequence=["rgb(46,204,113)"]
)
st.plotly_chart(avg_travel_time_fig, use_container_width=True)

# --- Suggested Routes ---
st.sidebar.subheader("🚍 Suggested Routes")
selected_route = st.sidebar.selectbox("Select Route for Suggestions", routes_df['route_short_name'])
congested_routes = st.session_state.traffic_data[st.session_state.traffic_data['vehicle_count'] > 50]['route'].unique()
suggestions = routes_df[~routes_df['route_short_name'].isin(congested_routes)]

st.sidebar.write(f"🛣️ **Alternate routes for {selected_route}:**")
st.sidebar.table(suggestions[['route_short_name', 'route_long_name']])

# --- Auto Refresh Logic ---
time.sleep(1)
st.experimental_rerun()
