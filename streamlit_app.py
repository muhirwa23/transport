import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster, HeatMap
from io import StringIO
import time

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

routes_df = load_route_data()

# --- GENERATE LIVE TRAFFIC DATA ---
def generate_live_data():
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
st.sidebar.info("Use the filters to analyze traffic conditions in real-time.")

selected_routes = st.sidebar.multiselect(
    "Select Routes", routes_df['route_short_name'], default=routes_df['route_short_name'].tolist()
)
min_vehicle_count = st.sidebar.slider("Minimum Vehicle Count", 0, 100, 20)
max_travel_time = st.sidebar.slider("Maximum Travel Time (minutes)", 5, 30, 20)
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 10, 5)

# --- LIVE UPDATING DATA ---
st.write(f"Data updates every {refresh_rate} seconds. Adjust the refresh rate from the control panel.")

# --- FILTER AND PREDICTION ---
filtered_data = st.session_state.traffic_data[
    (st.session_state.traffic_data['route'].isin(selected_routes)) & 
    (st.session_state.traffic_data['vehicle_count'] >= min_vehicle_count) & 
    (st.session_state.traffic_data['travel_time'] <= max_travel_time)
]

def predict_traffic(data, window=3):
    data['vehicle_count_pred'] = data['vehicle_count'].rolling(window=window).mean().shift(-1)
    data['congestion_pred'] = data['congestion'].map({"Low": 1, "Moderate": 2, "High": 3}).rolling(window=window).mean().shift(-1)
    data['congestion_pred'] = data['congestion_pred'].round().map({1: "Low", 2: "Moderate", 3: "High"})
    return data

predicted_data = predict_traffic(filtered_data)

# --- DASHBOARD SUMMARY ---
st.subheader("Traffic Summary")
total_incidents = filtered_data['incident'].value_counts().drop("None", errors='ignore').sum()
avg_congestion = filtered_data['congestion'].map({"Low": 1, "Moderate": 2, "High": 3}).mean()

st.metric("Total Incidents Reported", total_incidents)
st.metric("Average Congestion Level", round(avg_congestion, 2), delta=None)

# --- DISPLAY CHARTS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Vehicle Count Over Time")
    vehicle_fig = px.line(filtered_data, x='timestamp', y='vehicle_count', color='route',
                          title="Vehicle Count Over Time", markers=True)
    vehicle_fig.add_scatter(x=predicted_data['timestamp'], y=predicted_data['vehicle_count_pred'], mode='lines+markers', 
                            name='Predicted Vehicle Count', line=dict(dash='dash'))
    st.plotly_chart(vehicle_fig, use_container_width=True)

with col2:
    st.subheader("Congestion Level Over Time")
    congestion_fig = px.line(filtered_data, x='timestamp', y='congestion', color='route',
                             title="Congestion Level Over Time", markers=True)
    congestion_fig.add_scatter(x=predicted_data['timestamp'], y=predicted_data['congestion_pred'], mode='lines+markers', 
                               name='Predicted Congestion', line=dict(dash='dash'))
    st.plotly_chart(congestion_fig, use_container_width=True)

# --- ADDITIONAL INSIGHTS ---
st.subheader("Additional Insights")

col3, col4, col5 = st.columns(3)

with col3:
    incident_counts = filtered_data['incident'].value_counts().drop("None", errors='ignore')
    incident_pie = px.pie(values=incident_counts.values, names=incident_counts.index, title="Incident Types Distribution")
    st.plotly_chart(incident_pie, use_container_width=True)

with col4:
    travel_time_hist = px.histogram(filtered_data, x='travel_time', title="Travel Time Distribution",
                                    nbins=20, color='route')
    st.plotly_chart(travel_time_hist, use_container_width=True)

with col5:
    congestion_heatmap = px.density_heatmap(filtered_data, x='timestamp', y='route', z='vehicle_count',
                                            title="Vehicle Count Heatmap", color_continuous_scale='Viridis')
    st.plotly_chart(congestion_heatmap, use_container_width=True)

# --- DYNAMIC MAP ---
def generate_random_data():
    latitudes = np.random.uniform(-1.96, -1.92, 10)
    longitudes = np.random.uniform(29.9, 30.0, 10)
    congestion = np.random.randint(20, 100, 10)
    accident = np.random.choice([0, 1], 10, p=[0.8, 0.2])
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
    folium_map = folium.Map(location=[-1.95, 30.06], zoom_start=12)
    marker_cluster = MarkerCluster().add_to(folium_map)

    for _, row in data.iterrows():
        color = "red" if row['accident'] else "blue"
        congestion_level = row['congestion']
        
        # Change marker color intensity based on congestion percentage
        color_intensity = "darkred" if congestion_level > 75 else "orange" if congestion_level > 50 else "green"
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=7,
            color=color_intensity,
            fill=True,
            fill_color=color_intensity,
            fill_opacity=0.7
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
