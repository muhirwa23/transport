import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster, HeatMap
import plotly.express as px
import requests
import time

# --- Initialize Session State ---
if 'traffic_data' not in st.session_state:
    st.session_state.traffic_data = pd.DataFrame(columns=[
        'route', 'timestamp', 'vehicle_count', 'travel_time'
    ])

if 'weather_data' not in st.session_state:
    st.session_state.weather_data = None

# --- Load Route Data ---
@st.cache_data
def load_route_data():
    """Load and cache the full route dataset."""
    data = """route_id,agency_id,route_short_name,route_long_name,route_type,route_desc
    101,1,101,KBS - Zone I - 101,3,Remera Taxi Park-Sonatubes-Rwandex-CBD
    102,1,102,Kabuga-Mulindi-Remera-Sonatubes-Rwandex-Nyabugogo Taxi Park
    ...(additional routes here)...
    212,2,212,ROYAL - Zone II - 212,3,St. Joseph-Kicukiro Centre-Sonatubes-Rwandex-Nyabugogo Taxi Park
    """
    from io import StringIO
    return pd.read_csv(StringIO(data))

routes_df = load_route_data()

# --- Fetch Real-Time Weather Data ---
def fetch_weather():
    """Fetch weather data for Kigali using OpenWeather API."""
    api_key = "your_openweather_api_key"  # Replace with your actual API key
    url = f"https://api.openweathermap.org/data/2.5/weather?q=Kigali&units=metric&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    st.sidebar.error("Failed to fetch weather data.")
    return None

# --- Generate Live Traffic Data ---
def generate_live_data():
    """Simulate live traffic data with random metrics."""
    route = np.random.choice(routes_df['route_short_name'])
    vehicle_count = np.random.randint(10, 100)
    travel_time = np.random.uniform(10, 60)
    timestamp = pd.Timestamp.now()
    return {'route': route, 'timestamp': timestamp, 
            'vehicle_count': vehicle_count, 'travel_time': travel_time}

# --- Generate Folium Map with Heatmap & Markers ---
def generate_folium_map(data):
    """Generate map with congestion heatmap and markers."""
    m = folium.Map(location=[-1.9499, 30.0589], zoom_start=13)
    marker_cluster = MarkerCluster().add_to(m)

    heat_data = [
        [row['latitude'] + np.random.uniform(-0.005, 0.005), 
         row['longitude'] + np.random.uniform(-0.005, 0.005), row['vehicle_count']] 
        for _, row in data.iterrows()
    ]

    # Add heatmap layer
    HeatMap(heat_data).add_to(m)

    # Add congestion markers
    for _, row in data.iterrows():
        folium.Marker(
            location=[-1.9499 + np.random.uniform(-0.01, 0.01),
                      30.0589 + np.random.uniform(-0.01, 0.01)],
            popup=f"Route: {row['route']}<br>Vehicles: {row['vehicle_count']}<br>Travel Time: {row['travel_time']} min",
            icon=folium.Icon(color='red' if row['vehicle_count'] > 50 else 'blue', icon='info-sign')
        ).add_to(marker_cluster)

    return m

# --- UI and Layout ---
st.title("🚦 Kigali Traffic Monitoring and Optimization System")

# Display Weather Information
if st.sidebar.button("Fetch Weather"):
    st.session_state.weather_data = fetch_weather()

if st.session_state.weather_data:
    weather = st.session_state.weather_data
    st.sidebar.write(f"🌦️ **Weather:** {weather['weather'][0]['description'].capitalize()}")
    st.sidebar.write(f"🌡️ **Temperature:** {weather['main']['temp']} °C")
    st.sidebar.write(f"💧 **Humidity:** {weather['main']['humidity']}%")
    st.sidebar.write(f"🌬️ **Wind Speed:** {weather['wind']['speed']} m/s")

# Route Selection and Filters
selected_routes = st.sidebar.multiselect(
    "Select Routes", routes_df['route_short_name'].unique(), default=[]
)
min_vehicle_count = st.sidebar.slider("Min Vehicle Count", 0, 100, 10)
max_travel_time = st.sidebar.slider("Max Travel Time (minutes)", 10, 60, 30)

# Generate and Append New Traffic Data
new_data = generate_live_data()
st.session_state.traffic_data = pd.concat(
    [st.session_state.traffic_data, pd.DataFrame([new_data])], ignore_index=True
).tail(50)

# Filter Data Based on User Inputs
filtered_data = st.session_state.traffic_data[
    (st.session_state.traffic_data['route'].isin(selected_routes)) &
    (st.session_state.traffic_data['vehicle_count'] >= min_vehicle_count) &
    (st.session_state.traffic_data['travel_time'] <= max_travel_time)
]

# Display the Live Map with Heatmap
st.subheader("📍 Live Traffic Map")
folium_map = generate_folium_map(filtered_data)
st_folium(folium_map, width=700, height=500)

# Real-Time Plot of Vehicle Count
st.subheader("📈 Real-Time Vehicle Count per Route")
fig = px.line(
    filtered_data, x='timestamp', y='vehicle_count', 
    title="Vehicle Count Trends", markers=True
)
st.plotly_chart(fig, use_container_width=True)

# Alternate Routes Suggestions
st.sidebar.subheader("🔀 Suggest Alternate Routes")
selected_route = st.sidebar.selectbox("Select Route", routes_df['route_short_name'])
st.sidebar.write(f"Alternate routes for {selected_route}:")
alternate_routes = routes_df[routes_df['route_short_name'] != selected_route]
st.sidebar.write(alternate_routes[['route_short_name', 'route_long_name']])

# Refresh Logic
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 30, 10)
if st.sidebar.button("Refresh Now"):
    st.experimental_rerun()

time.sleep(refresh_rate)
st.experimental_rerun()
