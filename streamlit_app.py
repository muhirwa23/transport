import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from datetime import datetime
import googlemaps
from tensorflow.keras.models import load_model

# Initialize Google Maps Client
gmaps = googlemaps.Client(key='YOUR_GOOGLE_MAPS_API_KEY')

# Weather API key
WEATHER_API_KEY = 'c80a258e17ec49ad85a101108242410'

# Function to get weather data from OpenWeatherMap
def get_weather_data(city_name):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "q=" + city_name + "&appid=" + WEATHER_API_KEY + "&units=metric"
    response = requests.get(complete_url)
    return response.json()

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
    
    # Pretrained LSTM Model Loading
    st.write("## Predict Future Passenger Demand")

    # Placeholder: Replace with your own LSTM model loading
    model = load_model('lstm_demand_model.h5')  # Replace with actual path

    if uploaded_file is not None:
        # Assuming the uploaded file has the necessary historical data (preprocessed)
        st.write("### Historical Data")
        st.dataframe(data.head())

        # Placeholder for data preprocessing before feeding into the LSTM
        def preprocess_data(data):
            # Perform any necessary data preprocessing here (scaling, reshaping, etc.)
            return data

        # Apply preprocessing and make predictions
        processed_data = preprocess_data(data)
        predictions = model.predict(processed_data)  # Apply LSTM model for demand prediction

        # Display predictions as a line chart
        st.write("### Demand Forecast")
        st.line_chart(predictions)

    else:
        st.write("Upload data to predict demand")

# --- Tab 2: Traffic Rerouting Simulation ---
with tabs[1]:
    st.title("Traffic Rerouting Simulation")
    st.write("## Simulate Traffic Incidents and Optimize Routes")
    
    # Function to get traffic data from Google Maps
    def get_traffic_data(origin, destination):
        now = datetime.now()
        directions_result = gmaps.directions(origin, destination, mode="driving", departure_time=now)
        return directions_result

    # Example: Predefined locations for rerouting simulation (you can use input boxes to make it dynamic)
    st.write("### Simulated Incident Locations")
    origin = st.text_input("Origin", "Kigali Convention Centre")
    destination = st.text_input("Destination", "Nyabugogo Bus Terminal")

    if st.button("Get Traffic Info"):
        traffic_info = get_traffic_data(origin, destination)
        st.write("Traffic Data:")
        st.json(traffic_info)

        # Display the traffic on the map (use directions_result to show routes)
        # This is placeholder logic for visualization
        st.write("Simulated Rerouted Path Based on Traffic:")
        # For now, just showing the path directly on map (can be improved based on traffic conditions)
        fig = px.scatter_mapbox(
            pd.DataFrame({
                'lat': [traffic_info[0]['legs'][0]['start_location']['lat'], traffic_info[0]['legs'][0]['end_location']['lat']],
                'lon': [traffic_info[0]['legs'][0]['start_location']['lng'], traffic_info[0]['legs'][0]['end_location']['lng']],
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

    # Placeholder bus location data (replace with real-time streaming data)
    bus_data = pd.DataFrame({
        'bus_id': ['Bus_1', 'Bus_2', 'Bus_3'],
        'latitude': [-1.94407, -1.95265, -1.94946],
        'longitude': [30.06147, 30.08210, 30.07964],
        'passenger_count': [25, 40, 15],
        'last_updated': [datetime.now(), datetime.now(), datetime.now()]
    })

    st.write("### Live Bus Locations")
    st.dataframe(bus_data)

    # Display on map
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
