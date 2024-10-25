import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk  # For 3D visualization
import plotly.express as px
import requests
from sklearn.linear_model import LinearRegression
import time

# --- Initialize Session State ---
if 'traffic_data' not in st.session_state:
    st.session_state.traffic_data = pd.DataFrame(columns=[
        'route', 'timestamp', 'latitude', 'longitude', 'vehicle_count', 'event'
    ])

if 'predicted_data' not in st.session_state:
    st.session_state.predicted_data = pd.DataFrame()

# --- Load Route Data ---
@st.cache_data
def load_route_data():
    data = """route_id,agency_id,route_short_name,route_long_name,route_type,route_desc
    101,1,101,Remera Taxi Park-Sonatubes-Rwandex-CBD,3,Zone I
    ...(additional routes here)...
    """
    from io import StringIO
    return pd.read_csv(StringIO(data))

routes_df = load_route_data()

# --- Simulate Live Traffic Data ---
def simulate_event():
    route = np.random.choice(routes_df['route_short_name'])
    vehicle_count = np.random.randint(10, 100)
    latitude, longitude = np.random.uniform(-1.96, -1.93), np.random.uniform(30.05, 30.10)
    event = np.random.choice(['Accident', 'Congestion', 'Clear'])

    return {
        'route': route,
        'timestamp': pd.Timestamp.now(),
        'latitude': latitude,
        'longitude': longitude,
        'vehicle_count': vehicle_count,
        'event': event
    }

# --- Predict Traffic Congestion ---
def predict_traffic():
    data = st.session_state.traffic_data[['vehicle_count']]
    if len(data) > 10:
        X = data[['vehicle_count']]
        y = data['vehicle_count'].rolling(2).mean().fillna(0)  # Simplified predictor
        model = LinearRegression().fit(X, y)
        prediction = model.predict([[80]])  # Predict for a vehicle count of 80
        st.session_state.predicted_data = pd.DataFrame({
            'Predicted Vehicle Count': prediction, 'Timestamp': [pd.Timestamp.now()]
        })

# --- Create 3D Map Simulation with Pydeck ---
def create_3d_simulation():
    # View state settings for the 3D map
    view_state = pdk.ViewState(
        latitude=-1.9499, longitude=30.0589, zoom=13, pitch=50
    )

    # Scatter plot layer simulating events
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=st.session_state.traffic_data,
        get_position=["longitude", "latitude"],
        get_color="[255, 0, 0, 160]" if st.session_state.traffic_data['event'].eq('Accident').any() 
                  else "[0, 128, 0, 160]",  # Red for accidents, green otherwise
        get_radius=200,
        pickable=True,
        auto_highlight=True
    )

    # Text layer showing live event labels
    text_layer = pdk.Layer(
        "TextLayer",
        data=st.session_state.traffic_data,
        get_position=["longitude", "latitude"],
        get_text="event",
        get_size=20,
        get_color="[0, 0, 0]",
        get_angle=0,
        pickable=True
    )

    return pdk.Deck(layers=[scatter_layer, text_layer], initial_view_state=view_state)

# --- User Interface ---
st.title("🚦 Kigali Traffic Monitoring System with Real-Time 3D Simulation")

# Add new traffic data to the session state
new_data = simulate_event()
st.session_state.traffic_data = pd.concat(
    [st.session_state.traffic_data, pd.DataFrame([new_data])], ignore_index=True
).tail(100)  # Keep the latest 100 events

# Display 3D Map Simulation
st.subheader("🗺️ Real-Time 3D Simulation of Traffic Events")
st.pydeck_chart(create_3d_simulation())

# Plot Real-Time Vehicle Count Trends
st.subheader("📈 Real-Time Vehicle Count Trends")
fig = px.line(
    st.session_state.traffic_data, 
    x='timestamp', y='vehicle_count', 
    title="Vehicle Count over Time", markers=True
)
st.plotly_chart(fig, use_container_width=True)

# Predict and Display Congestion Forecast
st.subheader("🔮 Traffic Prediction")
predict_traffic()
if not st.session_state.predicted_data.empty:
    st.write(st.session_state.predicted_data)

# Refresh the dashboard periodically
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 30, 10)
time.sleep(refresh_rate)
st.experimental_rerun()
