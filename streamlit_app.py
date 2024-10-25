import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import plotly.express as px
from datetime import datetime
from io import StringIO

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Kigali Traffic Optimization", layout="wide")

# --- LOAD FULL ROUTE DATA ---
@st.cache_data
def load_route_data():
    """Load the full route data."""
    data = """route_id,agency_id,route_short_name,route_long_name,route_type,route_desc
    101,1,101,KBS - Zone I - 101,3,Remera Taxi Park-Sonatubes-Rwandex-CBD
    ...  # (TRIMMED for brevity)
    412,3,412,RFTC - Zone III and IV - 412,3,Nyabugogo Taxi Park-Giticyinyoni
    """
    return pd.read_csv(StringIO(data))

routes_df = load_route_data()

# --- Initialize Traffic Data in Session State ---
if 'traffic_data' not in st.session_state:
    def generate_initial_data():
        np.random.seed(42)
        data = []
        for _ in range(100):
            data.append(generate_live_data())
        return pd.DataFrame(data)

    st.session_state.traffic_data = generate_initial_data()

# --- LIVE DATA GENERATION FUNCTION ---
def generate_live_data():
    vehicle_count = np.random.randint(20, 100)
    travel_time = np.random.uniform(5, 25)
    route = np.random.choice(routes_df['route_short_name'])
    congestion = np.random.choice(["Low", "Moderate", "High"])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "timestamp": timestamp,
        "vehicle_count": vehicle_count,
        "travel_time": travel_time,
        "route": route,
        "congestion": congestion,
    }

# --- Sidebar Filters ---
st.sidebar.header("Control Panel")
selected_routes = st.sidebar.multiselect("Select Routes", routes_df['route_short_name'], default=routes_df['route_short_name'].tolist())
min_vehicle_count = st.sidebar.slider("Minimum Vehicle Count", 0, 100, 20)
max_travel_time = st.sidebar.slider("Maximum Travel Time (minutes)", 5, 30, 20)

# --- Filter Traffic Data ---
filtered_data = st.session_state.traffic_data[
    (st.session_state.traffic_data['route'].isin(selected_routes)) & 
    (st.session_state.traffic_data['vehicle_count'] >= min_vehicle_count) & 
    (st.session_state.traffic_data['travel_time'] <= max_travel_time)
]

# --- Correlation Heatmap ---
st.subheader("Traffic Data Correlation Heatmap")
corr = filtered_data[['vehicle_count', 'travel_time']].corr()
corr_fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
st.plotly_chart(corr_fig, use_container_width=True)

# --- Route-Specific Statistics ---
st.subheader("Route-Specific Statistics")
route_stats = filtered_data.groupby('route').agg({
    'vehicle_count': 'mean',
    'travel_time': 'mean',
}).rename(columns={'vehicle_count': 'Avg Vehicle Count', 'travel_time': 'Avg Travel Time'}).reset_index()

route_stats_fig = px.bar(route_stats, x='route', y=['Avg Vehicle Count', 'Avg Travel Time'],
                         barmode='group', title="Average Vehicle Count & Travel Time by Route")
st.plotly_chart(route_stats_fig, use_container_width=True)

# --- Route Map Visualization ---
st.subheader("Kigali Route Map")
map_center = [-1.9579, 30.0594]
m = folium.Map(location=map_center, zoom_start=12)

# Add MarkerCluster for route points
marker_cluster = MarkerCluster().add_to(m)

# Placeholder route coordinates, to be replaced with actual lat/lon for each route
route_coords = {route: [map_center[0] + np.random.uniform(-0.02, 0.02), map_center[1] + np.random.uniform(-0.02, 0.02)] 
                for route in routes_df['route_short_name']}

for route_id, coord in route_coords.items():
    folium.Marker(
        location=coord,
        popup=f"Route: {route_id}",
        tooltip=f"Route: {route_id}"
    ).add_to(marker_cluster)

# Display the map in Streamlit
st_folium(m, width=700, height=500)

# --- Display Traffic Data ---
st.subheader("Real-time Traffic Data")
st.dataframe(filtered_data)  # Better visualization with st.dataframe for larger tables
