import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import plotly.express as px
import time

# --- Initialize Session State ---
if 'traffic_data' not in st.session_state:
    st.session_state.traffic_data = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-10-24 10:00:00', '2024-10-24 10:05:00', '2024-10-24 10:10:00']),
        'latitude': [-1.9441, -1.9493, -1.9535],
        'longitude': [30.0619, 30.0595, 30.0647],
        'type': ['Accident', 'Congestion', 'Congestion'],
        'severity': [3, 2, 4],
        'route': ['Route A', 'Route B', 'Route C'],
        'vehicle_count': [10, 20, 30],
        'travel_time': [15, 25, 10]
    })

routes_df = pd.DataFrame({'route_short_name': ['Route A', 'Route B', 'Route C']})

# --- Generate Folium Map ---
def generate_folium_map(data):
    """Generate a dynamic Folium map with accident and congestion markers."""
    # Initialize map centered at Kigali with a fixed zoom level
    m = folium.Map(location=[-1.9499, 30.0589], zoom_start=13, control_scale=True)

    marker_cluster = MarkerCluster().add_to(m)

    for _, row in data.iterrows():
        icon_color = 'red' if row['type'] == 'Accident' else 'orange'
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"{row['type']} - Severity: {row['severity']}",
            icon=folium.Icon(color=icon_color, icon='info-sign')
        ).add_to(marker_cluster)

    return m

# --- Generate Live Traffic Data ---
def generate_live_data():
    """Simulate new traffic data."""
    new_data = {
        'timestamp': pd.Timestamp.now(),
        'latitude': -1.9441 + (0.001 * pd.np.random.randn()),
        'longitude': 30.0619 + (0.001 * pd.np.random.randn()),
        'type': pd.np.random.choice(['Accident', 'Congestion']),
        'severity': pd.np.random.randint(1, 5),
        'route': pd.np.random.choice(['Route A', 'Route B', 'Route C']),
        'vehicle_count': pd.np.random.randint(5, 50),
        'travel_time': pd.np.random.randint(5, 30)
    }
    return new_data

# --- Suggest Alternate Routes ---
def suggest_alternate_routes(selected_route):
    """Provide alternate routes based on current route selection."""
    st.sidebar.markdown("### Suggested Alternate Routes:")
    other_routes = routes_df[routes_df['route_short_name'] != selected_route]
    for route in other_routes['route_short_name']:
        st.sidebar.write(f"- {route}")

# --- Main Application Logic ---
st.title("Kigali Traffic Optimization System")

# Generate and append new traffic data
new_data = generate_live_data()
st.session_state.traffic_data = pd.concat(
    [st.session_state.traffic_data, pd.DataFrame([new_data])], ignore_index=True
).tail(50)

# User Inputs for Filtering Data
selected_routes = st.multiselect("Select Routes", routes_df['route_short_name'], default=routes_df['route_short_name'])
min_vehicle_count = st.slider("Minimum Vehicle Count", 0, 50, 10)
max_travel_time = st.slider("Maximum Travel Time (minutes)", 5, 30, 20)

# Filter Data Based on Inputs
filtered_data = st.session_state.traffic_data[
    (st.session_state.traffic_data['route'].isin(selected_routes)) &
    (st.session_state.traffic_data['vehicle_count'] >= min_vehicle_count) &
    (st.session_state.traffic_data['travel_time'] <= max_travel_time)
]

# Display the Embedded Folium Map
st.header("Live Map with Accidents and Congestion")
folium_map = generate_folium_map(filtered_data)

# Ensure the map stays embedded and doesn't pop out
st_folium(folium_map, width=700, height=500, returned_objects=[])

# Plot Real-Time Vehicle Count using Plotly
fig = px.line(
    filtered_data, x='timestamp', y='vehicle_count',
    title="Real-Time Vehicle Count", markers=True
)
st.plotly_chart(fig, use_container_width=True)

# Suggest Alternate Routes
selected_route = st.sidebar.selectbox("Select Route", routes_df['route_short_name'])
suggest_alternate_routes(selected_route)

# Refresh the Dashboard Periodically
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 30, 10)
if st.sidebar.button("Refresh Map"):
    st.experimental_rerun()

# Periodic Refresh Logic
time.sleep(refresh_rate)
st.experimental_rerun()
