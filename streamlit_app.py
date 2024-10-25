import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import plotly.express as px
import time

# --- Load Route Data ---
@st.cache_data
def load_route_data():
    """Load the complete route data."""
    data = """route_id,agency_id,route_short_name,route_long_name,route_type,route_desc
    101,1,101,KBS - Zone I - 101,3,Remera Taxi Park-Sonatubes-Rwandex-CBD
    102,1,102,Kabuga-Mulindi-Remera-Sonatubes-Rwandex-Nyabugogo Taxi Park
    ... (shortened for brevity, include your full dataset here)
    212,2,212,ROYAL - Zone II - 212,3,St. Joseph-Kicukiro Centre-Sonatubes-Rwandex-Nyabugogo Taxi Park
    """
    from io import StringIO
    return pd.read_csv(StringIO(data))

routes_df = load_route_data()

# --- Initialize Session State ---
if 'traffic_data' not in st.session_state:
    st.session_state.traffic_data = pd.DataFrame(columns=[
        'route', 'timestamp', 'vehicle_count', 'travel_time'
    ])

# --- Generate Live Data ---
def generate_live_data():
    """Simulate live traffic data."""
    route = np.random.choice(routes_df['route_short_name'])
    vehicle_count = np.random.randint(10, 100)
    travel_time = np.random.uniform(10, 60)
    timestamp = pd.Timestamp.now()
    return {'route': route, 'timestamp': timestamp, 
            'vehicle_count': vehicle_count, 'travel_time': travel_time}

# --- Generate Folium Map ---
def generate_folium_map(data):
    """Generate a map with traffic congestion markers."""
    m = folium.Map(location=[-1.9499, 30.0589], zoom_start=13)
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in data.iterrows():
        folium.Marker(
            location=[-1.9499 + np.random.uniform(-0.01, 0.01),
                      30.0589 + np.random.uniform(-0.01, 0.01)],
            popup=f"Route: {row['route']}<br>Vehicles: {row['vehicle_count']}<br>Travel Time: {row['travel_time']} min",
            icon=folium.Icon(color='red' if row['vehicle_count'] > 50 else 'blue', icon='info-sign')
        ).add_to(marker_cluster)

    return m

# --- Display UI ---
st.title("Kigali Traffic Monitoring and Optimization System")

# --- Route Selection ---
selected_routes = st.sidebar.multiselect(
    "Select Routes", routes_df['route_short_name'].unique(), default=[]
)
min_vehicle_count = st.sidebar.slider("Min Vehicle Count", 0, 100, 10)
max_travel_time = st.sidebar.slider("Max Travel Time (minutes)", 10, 60, 30)

# --- Generate and Update Traffic Data ---
new_data = generate_live_data()
st.session_state.traffic_data = pd.concat(
    [st.session_state.traffic_data, pd.DataFrame([new_data])], ignore_index=True
).tail(50)

# --- Filter Data Based on User Inputs ---
filtered_data = st.session_state.traffic_data[
    (st.session_state.traffic_data['route'].isin(selected_routes)) &
    (st.session_state.traffic_data['vehicle_count'] >= min_vehicle_count) &
    (st.session_state.traffic_data['travel_time'] <= max_travel_time)
]

# --- Display the Dynamic Map ---
st.subheader("Live Traffic Map")
folium_map = generate_folium_map(filtered_data)
st_folium(folium_map, width=700, height=500)

# --- Plot Real-Time Vehicle Count ---
st.subheader("Real-Time Vehicle Count")
fig = px.line(
    filtered_data, x='timestamp', y='vehicle_count', 
    title="Real-Time Vehicle Count per Route", markers=True
)
st.plotly_chart(fig, use_container_width=True)

# --- Suggest Alternate Routes ---
st.sidebar.subheader("Suggest Alternate Routes")
selected_route = st.sidebar.selectbox("Select Route", routes_df['route_short_name'])
st.sidebar.write(f"Alternate routes for {selected_route}:")
alternate_routes = routes_df[routes_df['route_short_name'] != selected_route]
st.sidebar.write(alternate_routes[['route_short_name', 'route_long_name']])

# --- Refresh Dashboard ---
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 30, 10)
if st.sidebar.button("Refresh Now"):
    st.experimental_rerun()

# --- Periodic Refresh Logic ---
time.sleep(refresh_rate)
st.experimental_rerun()
