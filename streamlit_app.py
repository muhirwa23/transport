import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import time
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
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
102,1,102,Kabuga-Mulindi-Remera-Sonatubes-Rwandex Nyabugogo Taxi Park
103,1,103,Rubilizi-Kabeza-Remera-Sonatubes-Rwandex-CBD
104,1,104,Kibaya-Kanombe MH-Airport-Remera-Sonatubes-Rwandex-CBD
105,1,105,Remera Taxi Park-Chez Lando-Kacyiru-Nyabugogo Taxi Park
"""

@st.cache_data
def load_route_data():
    return pd.read_csv(StringIO(route_data))

# Call the function
routes_df = load_route_data()

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

# --- LIVE UPDATING DATA ---
st.write("Live data updates every", refresh_rate, "seconds.")

# Filter traffic data based on the selected routes and conditions
filtered_data = st.session_state.traffic_data[
    (st.session_state.traffic_data['route'].isin(selected_routes)) & 
    (st.session_state.traffic_data['vehicle_count'] >= min_vehicle_count) & 
    (st.session_state.traffic_data['travel_time'] <= max_travel_time)
]

# --- PREDICTION FEATURE (Simple Moving Average) ---
def predict_traffic(data, window=3):
    """Predict future traffic using a simple moving average."""
    data['vehicle_count_pred'] = data['vehicle_count'].rolling(window=window).mean().shift(-1)
    data['congestion_pred'] = data['congestion'].map({"Low": 1, "Moderate": 2, "High": 3}).rolling(window=window).mean().shift(-1)
    data['congestion_pred'] = data['congestion_pred'].round().map({1: "Low", 2: "Moderate", 3: "High"})
    return data

# Apply prediction
predicted_data = predict_traffic(filtered_data)

# --- DISPLAY CHARTS ---
st.subheader("Live Traffic Data and Predictions")

# Plot real-time and predicted vehicle count
vehicle_fig = px.line(filtered_data, x='timestamp', y='vehicle_count', color='route',
                      title="Vehicle Count Over Time", markers=True)
vehicle_fig.add_scatter(x=predicted_data['timestamp'], y=predicted_data['vehicle_count_pred'], mode='lines+markers', 
                        name='Predicted Vehicle Count', line=dict(dash='dash'))

# Plot real-time and predicted congestion
congestion_fig = px.line(filtered_data, x='timestamp', y='congestion', color='route',
                         title="Congestion Level Over Time", markers=True)
congestion_fig.add_scatter(x=predicted_data['timestamp'], y=predicted_data['congestion_pred'], mode='lines+markers', 
                           name='Predicted Congestion', line=dict(dash='dash'))

# Display charts
st.plotly_chart(vehicle_fig, use_container_width=True)
st.plotly_chart(congestion_fig, use_container_width=True)

# --- DYNAMIC MAP ---
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

if 'map_data' not in st.session_state:
    st.session_state.map_data = generate_random_data()

def create_dynamic_map(data):
    """Create a dynamic Folium map with congestion and accident markers."""
    folium_map = folium.Map(location=[-1.95, 30.06], zoom_start=12)
    marker_cluster = MarkerCluster().add_to(folium_map)

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

# --- MAP DISPLAY ---
st.subheader("Real-time Traffic Map")
map_ = create_dynamic_map(st.session_state.map_data)
st_folium(map_, width=700, height=500)

# --- AUTO-REFRESH TRAFFIC DATA ---
def auto_refresh_traffic_data():
    for _ in range(100):
        new_data = pd.DataFrame([generate_live_data()])
        st.session_state.traffic_data = pd.concat([st.session_state.traffic_data, new_data]).tail(100)
        st.session_state.map_data = generate_random_data()
        time.sleep(refresh_rate)
        st.experimental_rerun()

st.write("Starting the auto-refresh of live data...")
auto_refresh_traffic_data()
