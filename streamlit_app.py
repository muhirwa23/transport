import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from datetime import datetime
import numpy as np
import googlemaps
from tensorflow.keras.models import load_model

# Initialize Google Maps Client (use your own API key here)
# gmaps = googlemaps.Client(key='YOUR_GOOGLE_MAPS_API_KEY')

# Weather API key
WEATHER_API_KEY = 'c80a258e17ec49ad85a101108242410'

# Function to get weather data from OpenWeatherMap
def get_weather_data(city_name):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "q=" + city_name + "&appid=" + WEATHER_API_KEY + "&units=metric"
    response = requests.get(complete_url)
    return response.json()

# Simulate historical demand data (for example, for 24 hours)
def simulate_historical_demand(hours=24):
    np.random.seed(42)
    time_index = pd.date_range(start="2024-10-01 00:00", periods=hours, freq="H")
    passenger_demand = np.random.randint(50, 150, size=(hours,))  # Random passenger counts between 50 and 150
    simulated_data = pd.DataFrame({'timestamp': time_index, 'passenger_count': passenger_demand})
    return simulated_data

# Simulate traffic data for rerouting
def simulate_traffic_data(origin, destination):
    traffic_conditions = np.random.choice(['Heavy Traffic', 'Moderate Traffic', 'Light Traffic'], p=[0.3, 0.5, 0.2])
    delay = np.random.randint(5, 30) if traffic_conditions == 'Heavy Traffic' else np.random.randint(0, 10)
    simulated_traffic_info = {
        'origin': origin,
        'destination': destination,
        'traffic_condition': traffic_conditions,
        'estimated_delay_minutes': delay
    }
    return simulated_traffic_info

# Simulate bus location and passenger data
def simulate_bus_data():
    np.random.seed(42)
    bus_data = pd.DataFrame({
        'bus_id': ['Bus_1', 'Bus_2', 'Bus_3'],
        'latitude': [-1.94407 + np.random.uniform(-0.01, 0.01) for _ in range(3)],
        'longitude': [30.06147 + np.random.uniform(-0.01, 0.01) for _ in range(3)],
        'passenger_count': np.random.randint(10, 50, size=3),
        'last_updated': [datetime.now() for _ in range(3)]
    })
    return bus_data

# Simulate weather data
def simulate_weather_data(city):
    weather_conditions = ['Clear', 'Cloudy', 'Rainy', 'Stormy']
    weather_data = {
        'main': {
            'temp': np.random.uniform(18, 30),  # Simulated temperature in Celsius
            'humidity': np.random.randint(40, 90)  # Simulated humidity percentage
        },
        'weather': [{'description': np.random.choice(weather_conditions)}],
        'wind': {'speed': np.random.uniform(0, 5)}  # Simulated wind speed in m/s
    }
    return weather_data

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
    
    st.write("## Predict Future Passenger Demand")

    if uploaded_file is not None:
        st.write("### Historical Data")
        st.dataframe(data.head())

        # Simulate data preprocessing for LSTM
        def preprocess_data(data):
            return data['passenger_count'].values.reshape(-1, 1)

        processed_data = preprocess_data(data)
        predictions = np.random.randint(60, 160, size=(24,))  # Simulating random predictions
        st.write("### Demand Forecast")
        st.line_chart(predictions)

    else:
        # Use simulated data if no file is uploaded
        st.write("### Simulated Historical Data")
        simulated_data = simulate_historical_demand()
        st.dataframe(simulated_data)

        processed_data = preprocess_data(simulated_data)
        predictions = np.random.randint(60, 160, size=(24,))
        st.line_chart(predictions)

# --- Tab 2: Traffic Rerouting Simulation ---
with tabs[1]:
    st.title("Traffic Rerouting Simulation")
    st.write("## Simulate Traffic Incidents and Optimize Routes")

    origin = st.text_input("Origin", "Kigali Convention Centre")
    destination = st.text_input("Destination", "Nyabugogo Bus Terminal")

    if st.button("Simulate Traffic Info"):
        traffic_info = simulate_traffic_data(origin, destination)
        st.write("Simulated Traffic Data:")
        st.json(traffic_info)
        
        fig = px.scatter_mapbox(
            pd.DataFrame({
                'lat': [-1.94407, -1.94946],
                'lon': [30.06147, 30.07964],
                'name': ['Origin', 'Destination']
            }),
            lat='lat', lon='lon', hover_name='name', zoom=12
        )
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig)

# --- Tab 3: Real-time Dashboard ---
with tabs[2]:
    st.title("Real-time Public Transport Dashboard")
    st.write("## Live Updates on Bus Locations and Routes")

    bus_data = simulate_bus_data()
    st.write("### Simulated Live Bus Locations")
    st.dataframe(bus_data)

    fig = px.scatter_mapbox(bus_data, lat="latitude", lon="longitude", hover_name="bus_id", hover_data=["passenger_count"],
                            zoom=12, height=500)
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig)

# --- Tab 4: Weather Info ---
with tabs[3]:
    st.title("Real-time Weather Info")
    st.write("## Get Current Weather Data for Specific Locations")

    city = st.text_input("Enter City", "Kigali")

    if st.button("Simulate Weather"):
        weather_data = simulate_weather_data(city)
        st.write(f"### Simulated Weather in {city}")
        st.write(f"**Temperature**: {weather_data['main']['temp']} °C")
        st.write(f"**Weather**: {weather_data['weather'][0]['description']}")
        st.write(f"**Humidity**: {weather_data['main']['humidity']}%")
        st.write(f"**Wind Speed**: {weather_data['wind']['speed']} m/s")
