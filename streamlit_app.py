import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from datetime import datetime, timedelta
import googlemaps
import numpy as np
from tensorflow.keras.models import load_model

# Initialize Google Maps Client
#gmaps = googlemaps.Client(key='YOUR_GOOGLE_MAPS_API_KEY')
# Weather API key
WEATHER_API_KEY = 'c80a258e17ec49ad85a101108242410'

# Function to get weather data from OpenWeatherMap
def get_weather_data(city_name):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "q=" + city_name + "&appid=" + WEATHER_API_KEY + "&units=metric"
    response = requests.get(complete_url)
    return response.json()

# Simulated bus data
def simulate_bus_data():
    return pd.DataFrame({
        'bus_id': ['Bus_1', 'Bus_2', 'Bus_3', 'Bus_4', 'Bus_5'],
        'latitude': [-1.94407, -1.95265, -1.94946, -1.95007, -1.95330],
        'longitude': [30.06147, 30.08210, 30.07964, 30.06750, 30.05430],
        'passenger_count': np.random.randint(10, 50, size=5),
        'last_updated': [datetime.now() - timedelta(minutes=np.random.randint(1, 10)) for _ in range(5)]
    })

# Simulated traffic data for each route
def simulate_traffic_for_route(route_id):
    traffic_conditions = np.random.choice(['Heavy Traffic', 'Moderate Traffic', 'Light Traffic'], p=[0.3, 0.5, 0.2])
    delay = np.random.randint(5, 30) if traffic_conditions == 'Heavy Traffic' else np.random.randint(0, 10)
    return {
        'route_id': route_id,
        'traffic_condition': traffic_conditions,
        'estimated_delay_minutes': delay
    }

# Suggest alternative routes based on traffic conditions
def suggest_alternative_route(routes, traffic_data):
    heavy_traffic_routes = [r['route_id'] for r in traffic_data if r['traffic_condition'] == 'Heavy Traffic']
    
    if heavy_traffic_routes:
        st.write("Heavy Traffic detected on the following routes:")
        st.write(heavy_traffic_routes)
        st.write("Suggesting alternative routes...")
        
        # Suggest other routes that do not have heavy traffic
        alternative_routes = routes[~routes['route_id'].isin(heavy_traffic_routes)]
        st.write(alternative_routes[['route_id', 'origin', 'destination']])
    else:
        st.write("No heavy traffic detected. All routes are running smoothly.")

# Main App Layout
st.set_page_config(layout="wide", page_title="Public Transport Optimization - Kigali")

# Sidebar - File Upload
st.sidebar.title("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Load Data
@st.cache
def load_data(file):
    return pd.read_csv(file)

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.sidebar.write("Data Loaded Successfully!")
else:
    st.sidebar.write("Please upload data to begin")

# Main Layout - Tabs for different sections
tabs = st.tabs(["Demand Prediction", "Traffic Rerouting", "Real-time Dashboard", "Weather Info"])

# --- Tab 1: Demand Prediction ---
with tabs[0]:
    st.title("Demand Prediction")
    
    # Pretrained LSTM Model Loading (Assume it's a mock model for now)
    st.write("## Predict Future Passenger Demand")
    model = load_model('lstm_demand_model.h5')  # Replace with actual path
    
    if uploaded_file is not None:
        st.write("### Historical Data")
        st.dataframe(data.head())

        # Simulate preprocessed data for demand prediction
        def preprocess_data(data):
            # Perform any necessary data preprocessing here (scaling, reshaping, etc.)
            return data

        processed_data = preprocess_data(data)
        predictions = model.predict(processed_data)  # Apply LSTM model for demand prediction
        
        st.write("### Demand Forecast")
        st.line_chart(predictions)
    else:
        st.write("Upload data to predict demand")

# --- Tab 2: Traffic Rerouting Simulation ---
with tabs[1]:
    st.title("Traffic Rerouting Simulation")
    st.write("## Simulate Traffic Incidents and Optimize Routes")

    # Example: Predefined locations for rerouting simulation (you can use input boxes to make it dynamic)
    st.write("### Simulated Incident Locations")
    origin = st.text_input("Origin", "Kigali Convention Centre")
    destination = st.text_input("Destination", "Nyabugogo Bus Terminal")

    if st.button("Get Traffic Info"):
        traffic_info = simulate_traffic_for_route("Route_1")  # Simulated traffic condition
        st.write("Traffic Condition:", traffic_info['traffic_condition'])
        st.write("Estimated Delay (minutes):", traffic_info['estimated_delay_minutes'])

        # Suggest alternative route based on traffic condition
        routes = pd.DataFrame({
            'route_id': ['Route_1', 'Route_2', 'Route_3'],
            'origin': ['KCC', 'Remera', 'Nyabugogo'],
            'destination': ['Nyabugogo', 'Kimironko', 'Kicukiro']
        })
        
        suggest_alternative_route(routes, [traffic_info])

# --- Tab 3: Real-time Dashboard ---
with tabs[2]:
    st.title("Real-time Public Transport Dashboard")
    st.write("## Live Updates on Bus Locations and Routes")

    # Simulated bus data
    bus_data = simulate_bus_data()
    st.write("### Live Bus Locations")
    st.dataframe(bus_data)

    # Display bus locations on map
    fig = px.scatter_mapbox(bus_data, lat="latitude", lon="longitude", hover_name="bus_id", hover_data=["passenger_count"],
                            zoom=12, height=500)
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig)

# --- Tab 4: Weather Info ---
with tabs[3]:
    st.title("Real-time Weather Info")
    st.write("## Get Current Weather Data for Specific Locations")

    city = st.text_input("Enter City", "Kigali")
    
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
