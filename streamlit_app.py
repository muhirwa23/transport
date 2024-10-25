import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde
from transformers import pipeline
import time

# --- Initialize Session State ---
if 'traffic_data' not in st.session_state:
    st.session_state.traffic_data = pd.DataFrame(columns=[
        'route', 'timestamp', 'latitude', 'longitude', 'vehicle_count', 'event', 'severity'
    ])
    st.session_state.selected_event = None

# --- Load Route Data ---
@st.cache_data
def load_route_data():
    data = """route_id,agency_id,route_short_name,route_long_name,route_type,route_desc
    101,1,101,Remera Taxi Park-Sonatubes-Rwandex-CBD,3,Zone I
    102,1,102,Gikondo-CBD-Kigali City,3,Zone II
    ...(more routes here)...
    """
    from io import StringIO
    return pd.read_csv(StringIO(data))

routes_df = load_route_data()

# --- Simulate Enhanced Live Traffic Data ---
@st.cache_data
def simulate_event():
    route = np.random.choice(routes_df['route_short_name'])
    vehicle_count = np.random.randint(10, 500)
    latitude, longitude = np.random.uniform(-1.96, -1.93), np.random.uniform(30.05, 30.10)
    event = np.random.choice(['Accident', 'Traffic Jam', 'Closed Road', 'Damaged Road', 'Road Work', 'Congestion'])
    
    severity = "Minor" if vehicle_count < 50 else "Moderate" if vehicle_count < 200 else "Severe"
    return {
        'route': route,
        'timestamp': pd.Timestamp.now(),
        'latitude': latitude,
        'longitude': longitude,
        'vehicle_count': vehicle_count,
        'event': event,
        'severity': severity
    }

# --- Hugging Face Model for Prediction ---
@st.cache_resource
def load_hf_model():
    model = pipeline("text-generation", model="gpt2", framework="pt")  # Explicitly specify PyTorch
    return model

hf_model = load_hf_model()

def predict_traffic_trends():
    if len(st.session_state.traffic_data) > 10:
        input_text = "Predict traffic for the next hour based on recent events."
        response = hf_model(input_text, max_length=50, num_return_sequences=1)
        return response[0]["generated_text"]
    return "Not enough data to make a prediction."

# --- Display KDE Plot ---
def display_kde_plot():
    st.subheader("📈 KDE Plot of Vehicle Counts")
    if len(st.session_state.traffic_data) > 10:
        vehicle_counts = st.session_state.traffic_data['vehicle_count'].values
        kde = gaussian_kde(vehicle_counts)
        x = np.linspace(vehicle_counts.min(), vehicle_counts.max(), 100)
        fig_kde = px.line(x=x, y=kde(x), title="KDE Plot of Vehicle Count Density")
        st.plotly_chart(fig_kde, use_container_width=True)
    else:
        st.write("Not enough data for KDE Plot.")

# --- Display Suggested Routes ---
def suggest_routes(event_severity):
    st.subheader("🚗 Suggested Routes")
    if event_severity == "Severe":
        st.write("We suggest alternative routes due to severe traffic on this route.")
        st.write(routes_df.sample(2))
    elif event_severity == "Moderate":
        st.write("Moderate traffic detected; consider using the following routes:")
        st.write(routes_df.sample(3))
    else:
        st.write("Traffic is light, proceed with the selected route.")
    
# --- User Interface ---
st.title("🚦 Kigali Traffic Monitoring System with Enhanced Data Simulation")

# Sidebar: Route Selection
st.sidebar.header("🚗 Available Routes in Kigali")
selected_route = st.sidebar.selectbox("Select a route to view traffic events:", options=routes_df['route_short_name'].unique())

# Simulate and add new data
for _ in range(5):  # Increase simulated events for richer demo
    new_data = simulate_event()
    st.session_state.traffic_data = pd.concat(
        [st.session_state.traffic_data, pd.DataFrame([new_data])], ignore_index=True
    ).tail(500)

# --- Display KDE Plot ---
display_kde_plot()

# --- Prediction using Hugging Face Model ---
st.subheader("🔮 Traffic Prediction")
hf_prediction = predict_traffic_trends()
st.write(f"Model Prediction: {hf_prediction}")

# --- Display Suggested Routes ---
if st.session_state.selected_event:
    suggest_routes(st.session_state.selected_event['severity'])

# --- Display Traffic KPIs ---
st.subheader("🚥 Traffic Key Performance Indicators (KPIs)")
total_events = st.session_state.traffic_data['event'].count()
avg_vehicle_count = st.session_state.traffic_data['vehicle_count'].mean()
most_common_event = st.session_state.traffic_data['event'].mode()[0]

col1, col2, col3 = st.columns(3)
col1.metric("Total Traffic Events", total_events)
col2.metric("Average Vehicle Count", f"{avg_vehicle_count:.2f}")
col3.metric("Most Common Event", most_common_event)

# --- Main 3D Map ---
st.subheader("🗺️ Kigali Real-Time 3D Traffic Map")
main_3d_map = create_main_3d_map(selected_route=selected_route)
st.pydeck_chart(main_3d_map)

# --- Real-Time Refresh ---
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 30, 10)
time.sleep(refresh_rate)
st.experimental_rerun()
