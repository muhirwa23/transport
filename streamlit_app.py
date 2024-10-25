import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta
import time
import plotly.express as px
import altair as alt
import folium
from streamlit_folium import st_folium  # For embedding Folium in Streamlit
from folium.plugins import MarkerCluster

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Kigali Traffic Optimization",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SAMPLE ROUTE DATA ---
route_data = """
route_id,agency_id,route_short_name,route_long_name,route_type,route_desc
101,1,101,KBS - Zone I - 101,3,Remera Taxi Park-Sonatubes-Rwandex-CBD
102,1,102,Kabuga-Mulindi-Remera-Sonatubes-Rwandex Nyabugogo Taxi Park
103,1,103,KBS - Zone I - 103,3,Rubilizi-Kabeza-Remera-Sonatubes-Rwandex-CBD
104,1,104,KBS - Zone I - 104,3,Kibaya-Kanombe MH-Airport-Remera-Sonatubes-Rwandex-CBD
105,1,105,KBS - Zone I - 105,3,Remera Taxi Park-Chez Lando-Kacyiru-NyabugogoTaxi Park
"""

@st.cache_data
def load_route_data():
    return pd.read_csv(pd.compat.StringIO(route_data))

routes_df = load_route_data()

# --- GENERATE LIVE TRAFFIC DATA ---
def generate_live_data():
    """Simulate live traffic data for congestion and incidents."""
    np.random.seed(int(datetime.now().timestamp()))
    vehicle_count = np.random.randint(20, 100)
    travel_time = np.random.uniform(5, 25)
    route = np.random.choice(routes_df['route_short_name'])
    congestion_level = np.random.choice(["Low", "Moderate", "High"])
    incident = np.random.choice(["None", "Accident", "Roadblock"], p=[0.8, 0.15, 0.05])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "timestamp": timestamp,
        "vehicle_count": vehicle_count,
        "travel_time": travel_time,
        "route": route,
        "congestion_level": congestion_level,
        "incident": incident
    }

# --- SESSION STATE FOR LIVE DATA ---
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

# --- OPTIMIZATION: SUGGEST ALTERNATE ROUTES ---
def suggest_alternate_routes(selected_route):
    """Suggest alternate routes if congestion exceeds threshold."""
    if st.session_state.traffic_data['vehicle_count'].mean() > congestion_threshold:
        st.warning(f"High congestion on Route {selected_route}. Suggesting alternate routes...")
        suggestions = routes_df[routes_df['route_short_name'] != selected_route].head(3)
        st.table(suggestions)
    else:
        st.success(f"Traffic on Route {selected_route} is manageable.")

# --- DYNAMIC MAP SETUP USING FOLIUM ---
def generate_folium_map(data):
    """Generate a Folium map with congestion and incidents marked."""
    folium_map = folium.Map(location=[-1.9577, 30.1127], zoom_start=12)  # Centered in Kigali

    marker_cluster = MarkerCluster().add_to(folium_map)
    for _, row in data.iterrows():
        popup_info = f"""
            <b>Route:</b> {row['route']}<br>
            <b>Vehicles:</b> {row['vehicle_count']}<br>
            <b>Travel Time:</b> {row['travel_time']:.2f} mins<br>
            <b>Congestion:</b> {row['congestion_level']}<br>
            <b>Incident:</b> {row['incident']}
        """
        icon_color = "green" if row["congestion_level"] == "Low" else \
                     "orange" if row["congestion_level"] == "Moderate" else "red"
        folium.Marker(
            location=[-1.9577 + np.random.uniform(-0.01, 0.01), 30.1127 + np.random.uniform(-0.01, 0.01)],
            popup=popup_info,
            icon=folium.Icon(color=icon_color, icon="info-sign"),
        ).add_to(marker_cluster)

    return folium_map

# --- MAIN APP LOGIC: CHARTS AND MAP ---
st.title("Kigali Traffic Optimization System")

# Generate new traffic data in real-time
while True:
    new_data = generate_live_data()
    st.session_state.traffic_data = pd.concat(
        [st.session_state.traffic_data, pd.DataFrame([new_data])], ignore_index=True
    ).tail(50)

    # Filter data based on user selections
    filtered_data = st.session_state.traffic_data[
        (st.session_state.traffic_data['route'].isin(selected_routes)) &
        (st.session_state.traffic_data['vehicle_count'] >= min_vehicle_count) &
        (st.session_state.traffic_data['travel_time'] <= max_travel_time)
    ]

    # Display Folium map with dynamic markers
    folium_map = generate_folium_map(filtered_data)
    st_folium(folium_map, width=700, height=500)

    # Plot real-time vehicle count with Plotly
    fig = px.line(
        filtered_data, x='timestamp', y='vehicle_count',
        title="Real-Time Vehicle Count", markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # Suggest alternate routes based on congestion level
    selected_route = st.sidebar.selectbox("Select Route", routes_df['route_short_name'])
    suggest_alternate_routes(selected_route)

    # Pause before next refresh
    time.sleep(refresh_rate)
    st.experimental_rerun()
