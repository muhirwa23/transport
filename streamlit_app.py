import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from datetime import datetime, timedelta
import numpy as np
import time

# Weather API key for Kigali (replace with your OpenWeatherMap API key)
WEATHER_API_KEY = 'c80a258e17ec49ad85a101108242410'

# Function to get weather data from OpenWeatherMap for Kigali
def get_weather_data(city_name="Kigali"):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "q=" + city_name + "&appid=" + WEATHER_API_KEY + "&units=metric"
    response = requests.get(complete_url)
    return response.json()

# Simulate live bus data for the dashboard with moving buses
def simulate_bus_data():
    base_lat, base_lon = -1.94407, 30.06147
    return pd.DataFrame({
        'bus_id': ['Bus_1', 'Bus_2', 'Bus_3', 'Bus_4', 'Bus_5'],
        'latitude': [base_lat + np.random.uniform(-0.01, 0.01) for _ in range(5)],
        'longitude': [base_lon + np.random.uniform(-0.01, 0.01) for _ in range(5)],
        'passenger_count': np.random.randint(10, 50, size=5),
        'last_updated': [datetime.now() - timedelta(minutes=np.random.randint(1, 10)) for _ in range(5)]
    })

# Simulate traffic data for different routes
def simulate_traffic_for_route(route_id):
    traffic_conditions = np.random.choice(['Heavy Traffic', 'Moderate Traffic', 'Light Traffic'], p=[0.3, 0.5, 0.2])
    delay = np.random.randint(5, 30) if traffic_conditions == 'Heavy Traffic' else np.random.randint(0, 10)
    return {
        'route_id': route_id,
        'traffic_condition': traffic_conditions,
        'estimated_delay_minutes': delay
    }

# Function to suggest alternative routes based on traffic
def suggest_alternative_route(routes, traffic_data):
    heavy_traffic_routes = [r['route_id'] for r in traffic_data if r['traffic_condition'] == 'Heavy Traffic']
    
    if heavy_traffic_routes:
        st.write("**Heavy Traffic detected on the following routes:**")
        st.write(heavy_traffic_routes)
        st.write("Suggesting alternative routes...")
        
        # Suggest other routes that do not have heavy traffic
        alternative_routes = routes[~routes['route_id'].isin(heavy_traffic_routes)]
        st.write(alternative_routes[['route_id', 'origin', 'destination']])
    else:
        st.write("No heavy traffic detected. All routes are running smoothly.")

# Simulate demand prediction (without model)
def simulate_demand_prediction(data):
    future_time_units = np.arange(10)
    demand_forecast = np.cumsum(np.random.randint(10, 100, size=10))
    return future_time_units, demand_forecast

# Function to update bus data and traffic information every few seconds
def update_data_periodically():
    while True:
        st.write("**Updating data in real-time...**")
        time.sleep(5)
        # Simulated updates for buses and traffic
        bus_data = simulate_bus_data()
        st.dataframe(bus_data)
        
        # Updating map with new positions
        fig = px.scatter_mapbox(bus_data, lat="latitude", lon="longitude", hover_name="bus_id", hover_data=["passenger_count"],
                                zoom=12, height=500)
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig)
        # Auto-refresh every 10 seconds

# Main App Layout
st.set_page_config(layout="wide", page_title="Kigali Public Transport Optimization Dashboard")

# Sidebar - File Upload
st.sidebar.title("Upload Historical Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Load Data
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.sidebar.write("**Data Loaded Successfully!**")
else:
    st.sidebar.write("Please upload historical data to enable demand prediction.")

# Main Layout - Tabs for different sections
tabs = st.tabs(["Demand Prediction", "Traffic Rerouting", "Real-time Dashboard", "Weather Info", "Safety & Incident Alerts"])

# --- Tab 1: Demand Prediction ---
with tabs[0]:
    st.title("Demand Prediction")
    
    if uploaded_file is not None:
        st.write("### Historical Data")
        st.dataframe(data.head())
        
        # Simulate demand prediction
        future_time, demand_forecast = simulate_demand_prediction(data)
        
        st.write("### Simulated Passenger Demand Forecast")
        demand_df = pd.DataFrame({
            'Time Units': future_time,
            'Predicted Demand': demand_forecast
        })
        st.line_chart(demand_df.set_index('Time Units'))
    else:
        st.write("Upload historical data to predict demand.")

# --- Tab 2: Traffic Rerouting Simulation ---
with tabs[1]:
    st.title("Traffic Rerouting Simulation")
    st.write("## Simulate Traffic Incidents and Optimize Routes")

    # Input boxes for origin and destination
    origin = st.text_input("Origin", "Kigali Convention Centre")
    destination = st.text_input("Destination", "Nyabugogo Bus Terminal")

    if st.button("Simulate Traffic"):
        # Simulate traffic conditions for different routes
        traffic_info = [simulate_traffic_for_route(f"Route_{i+1}") for i in range(3)]
        
        st.write("### Traffic Conditions for Routes:")
        for t in traffic_info:
            st.write(f"**Route {t['route_id']}:** {t['traffic_condition']} (Estimated Delay: {t['estimated_delay_minutes']} mins)")
        
        # Route suggestions
        routes = pd.DataFrame({
            'route_id': ['Route_1', 'Route_2', 'Route_3'],
            'origin': ['KCC', 'Remera', 'Nyabugogo'],
            'destination': ['Nyabugogo', 'Kimironko', 'Kicukiro']
        })
        suggest_alternative_route(routes, traffic_info)

# --- Tab 3: Real-time Dashboard ---
with tabs[2]:
    st.title("Real-time Public Transport Dashboard")
    st.write("## Live Updates on Bus Locations and Routes")

    # Real-time updates with simulated data
    update_data_periodically()

# --- Tab 4: Weather Info ---
with tabs[3]:
    st.title("Real-time Weather Info for Kigali")
    
    city = "Kigali"
    if st.button("Get Weather"):
        weather_data = get_weather_data(city)
        
        if weather_data["cod"] != "404":
            weather_main = weather_data["main"]
            weather_description = weather_data["weather"][0]["description"]
            temp = weather_main["temp"]
            humidity = weather_main["humidity"]
            wind_speed = weather_data["wind"]["speed"]

            st.write(f"### Weather in {city}")
            st.write(f"**Temperature**: {temp} °C")
            st.write(f"**Weather**: {weather_description}")
            st.write(f"**Humidity**: {humidity}%")
            st.write(f"**Wind Speed**: {wind_speed} m/s")
        else:
            st.write("City Not Found")

# --- Tab 5: Safety & Incident Alerts ---
with tabs[4]:
    st.title("Safety & Incident Alerts")
    st.write("## Monitor Safety and Incident Reports in Kigali")

    # Simulated incident reports for safety
    incident_data = pd.DataFrame({
        'incident_id': ['Incident_1', 'Incident_2', 'Incident_3'],
        'location': ['Nyabugogo', 'Remera', 'Kicukiro'],
        'incident_type': ['Accident', 'Protest', 'Road Closure'],
        'reported_time': [datetime.now() - timedelta(minutes=i*15) for i in range(3)]
    })
    st.write("### Recent Incidents")
    st.dataframe(incident_data)
    
    st.write("Monitor incidents to reroute buses and deploy emergency services efficiently.")
