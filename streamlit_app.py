import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta
import time
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import plotly.express as px
from io import StringIO

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Kigali Traffic Optimization",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD ROUTE DATA ---
route_data = """
route_id,agency_id,route_short_name,route_long_name,route_type,route_desc
101,1,101,KBS - Zone I - 101,3,Remera Taxi Park-Sonatubes-Rwandex-CBD
102,1,102,KBS - Zone I - 102,3,Kabuga-Mulindi-Remera-Sonatubes-Rwandex Nyabugogo Taxi Park
103,1,103,KBS - Zone I - 103,3,Rubilizi-Kabeza-Remera-Sonatubes-Rwandex-CBD
104,1,104,KBS - Zone I - 104,3,Kibaya-Kanombe MH-Airport-Remera-Sonatubes-Rwandex-CBD
105,1,105,KBS - Zone I - 105,3,Remera Taxi Park-Chez Lando-Kacyiru-NyabugogoTaxi Park
"""

@st.cache_data
def load_route_data():
    return pd.read_csv(StringIO(route_data))

# Call the function
routes_df = load_route_data()
st.dataframe(routes_df)

# --- GENERATE LIVE TRAFFIC DATA ---
def generate_live_data():
    """Simulate real-time traffic data with congestion and incidents."""
    np.random.seed(int(datetime.now().timestamp()))
    vehicle_count = np.random.randint(20, 100)
    travel_time = np.random.uniform(5, 25)
    route = np.random.choice(routes_df['route_short_name'])
    congestion = np.random.choice(["Low", "Moderate", "High"])
    incident = np.random.choice(["None", "Accident", "Roadblock"], p=[0.8, 0.15, 0.05])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "timestamp": timestamp,
        "vehicle_count": vehicle_count,
        "travel_time": travel_time,
        "route": route,
        "congestion": congestion,
        "incident": incident
    }

# --- INITIALIZE SESSION STATE ---
if 'traffic_data' not in st.session_state:
    st.session_state.traffic_data = pd.DataFrame([generate_live_data() for _ in range(10)])

# --- SIDEBAR FILTERS ---
st.sidebar.header("Control Panel")
selected_routes = st.sidebar.multiselect(
    "Select Routes", routes_df['route_short_name'], default=routes_df['route_short_name'].tolist()
)
min_vehicle_count = st.sidebar.slider("Minimum Vehicle Count", 0, 100, 20)
max_travel_time = st.sidebar.slider("Maximum Travel Time (minutes)", 5, 30, 20)
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 10, 5)
congestion_threshold = st.sidebar.slider("Set Congestion Threshold", 0, 100, 50)

# --- SUGGEST ALTERNATE ROUTES ---
def suggest_alternate_routes(route):
    if st.session_state.traffic_data['vehicle_count'].mean() > congestion_threshold:
        st.warning(f"High congestion detected on {route}. Suggesting alternate routes...")
        suggestions = routes_df[routes_df['route_short_name'] != route].head(3)
        st.table(suggestions)
    else:
        st.success(f"Traffic on {route} is under control.")

# --- Sample Data Simulation for Dynamic Map ---
def generate_random_data():
    """Generate random data for congestion and accidents."""
    latitudes = np.random.uniform(-1.96, -1.92, 10)  # Latitude range for Kigali
    longitudes = np.random.uniform(29.9, 30.0, 10)   # Longitude range for Kigali
    congestion = np.random.randint(20, 100, 10)      # Congestion percentage
    accident = np.random.choice([0, 1], 10, p=[0.8, 0.2])  # Accident presence
    routes = np.random.choice(routes_df['route_short_name'], 10)
    timestamps = [datetime.now() - timedelta(minutes=i) for i in range(10)]

    return pd.DataFrame({
        'latitude': latitudes,
        'longitude': longitudes,
        'congestion': congestion,
        'accident': accident,
        'route': routes,
        'timestamp': timestamps
    })

# --- Load or Generate Data ---
if 'map_data' not in st.session_state:
    st.session_state.map_data = generate_random_data()

# --- Dynamic Map Function ---
def create_dynamic_map(data):
    """Create a dynamic Folium map with congestion and accident markers."""
    # Initialize a Folium map centered around Kigali
    folium_map = folium.Map(location=[-1.95, 30.06], zoom_start=12)

    # Use a MarkerCluster for better marker management
    marker_cluster = MarkerCluster().add_to(folium_map)

    # Add markers to the map
    for _, row in data.iterrows():
        popup_text = f"""
        <b>Route:</b> {row['route']}<br>
        <b>Congestion:</b> {row['congestion']}%<br>
        <b>Accident:</b> {"Yes" if row['accident'] else "No"}<br>
        <b>Time:</b> {row['timestamp']}
        """
        icon_color = "red" if row['accident'] else "blue"
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=popup_text,
            icon=folium.Icon(color=icon_color, icon="info-sign")
        ).add_to(marker_cluster)

    return folium_map

# --- Streamlit App Layout ---
st.title("Kigali Traffic Optimization and Monitoring System")

# Display real-time traffic data
st.subheader("Live Traffic Data")
st.dataframe(st.session_state.traffic_data)

# Suggest alternate routes for high congestion
for route in selected_routes:
    suggest_alternate_routes(route)

# Dynamic map display
st.subheader("Real-time Traffic Map")
map_ = create_dynamic_map(st.session_state.map_data)
st_folium(map_, width=700, height=500)

# Auto-refresh data at the set interval
if st.button("Refresh Traffic Data"):
    st.session_state.traffic_data = pd.DataFrame([generate_live_data() for _ in range(10)])
    st.session_state.map_data = generate_random_data()
    st.experimental_rerun()
