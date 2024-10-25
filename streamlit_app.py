import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Kigali Traffic Optimization", layout="wide")

# --- LOAD FULL ROUTE DATA ---
@st.cache_data
def load_route_data():
    """Load the full route data."""
    data = """route_id,agency_id,route_short_name,route_long_name,route_type,route_desc
    101,1,101,KBS - Zone I - 101,3,Remera Taxi Park-Sonatubes-Rwandex-CBD
    ...  # Trimmed for brevity
    412,3,412,RFTC - Zone III and IV - 412,3,Nyabugogo Taxi Park-Giticyinyoni
    """
    return pd.read_csv(StringIO(data))

routes_df = load_route_data()

# --- LIVE DATA GENERATION FUNCTION ---
def generate_live_data():
    """Generates one row of live traffic data with consistent keys."""
    try:
        vehicle_count = np.random.randint(20, 100)
        travel_time = np.random.uniform(5, 25)
        route = np.random.choice(routes_df['route_short_name'])
        congestion = np.random.choice(["Low", "Moderate", "High"])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {
            "timestamp": timestamp,
            "vehicle_count": vehicle_count,
            "travel_time": travel_time,
            "route": route,
            "congestion": congestion,
        }
    except Exception as e:
        st.error(f"Error generating live data: {str(e)}")
        return None

# --- Initialize Traffic Data in Session State ---
if 'traffic_data' not in st.session_state:
    def generate_initial_data():
        np.random.seed(42)
        data = []
        for _ in range(100):
            live_data = generate_live_data()  # Generate one row of live data
            if live_data:  # Check if live_data is not None
                data.append(live_data)
        # Convert list of dictionaries to DataFrame with consistent columns
        return pd.DataFrame(data, columns=["timestamp", "vehicle_count", "travel_time", "route", "congestion"])

    st.session_state.traffic_data = generate_initial_data()

# --- Sidebar Navigation ---
st.sidebar.title("Traffic Dashboard")
analysis_type = st.sidebar.selectbox("Choose Analysis", ["Overview", "Time Series Analysis", "Predictive Modeling", "3D Traffic Map"])
selected_routes = st.sidebar.multiselect("Select Routes", routes_df['route_short_name'], default=routes_df['route_short_name'].tolist())
min_vehicle_count = st.sidebar.slider("Minimum Vehicle Count", 0, 100, 20)
max_travel_time = st.sidebar.slider("Maximum Travel Time (minutes)", 5, 30, 20)

# --- Filter Traffic Data ---
filtered_data = st.session_state.traffic_data[
    (st.session_state.traffic_data['route'].isin(selected_routes)) & 
    (st.session_state.traffic_data['vehicle_count'] >= min_vehicle_count) & 
    (st.session_state.traffic_data['travel_time'] <= max_travel_time)
]

# --- Deep Learning Model Setup ---
def prepare_lstm_data(data, feature, n_past=5):
    """Prepare data for LSTM modeling with lookback."""
    X, y = [], []
    for i in range(n_past, len(data)):
        X.append(data[i-n_past:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def train_lstm_model(data, feature='vehicle_count'):
    """Train an LSTM model to predict vehicle count."""
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[[feature]])
    
    X, y = prepare_lstm_data(data_scaled, feature)
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, verbose=0)
    
    return model, scaler

# --- Display Analysis Based on Menu Selection ---
if analysis_type == "Overview":
    st.subheader("Real-time Traffic Data Overview")
    st.dataframe(filtered_data)
    
elif analysis_type == "Time Series Analysis":
    st.subheader("Time Series Analysis of Vehicle Count and Travel Time")
    filtered_data['timestamp'] = pd.to_datetime(filtered_data['timestamp'])
    time_series_fig = px.line(
        filtered_data,
        x="timestamp",
        y=["vehicle_count", "travel_time"],
        title="Time Series of Vehicle Count and Travel Time",
        labels={"timestamp": "Timestamp", "value": "Count / Travel Time"},
    )
    st.plotly_chart(time_series_fig, use_container_width=True)
    
elif analysis_type == "Predictive Modeling":
    st.subheader("Predictive Analysis using Deep Learning")
    # Train LSTM model
    lstm_model, lstm_scaler = train_lstm_model(filtered_data, feature='vehicle_count')
    
    # Predict future values
    last_n_values = filtered_data['vehicle_count'].values[-5:]
    last_n_values_scaled = lstm_scaler.transform(last_n_values.reshape(-1, 1))
    last_n_values_scaled = last_n_values_scaled.reshape(1, 5, 1)
    
    future_prediction = lstm_model.predict(last_n_values_scaled)
    predicted_value = lstm_scaler.inverse_transform(future_prediction).flatten()[0]
    
    st.write(f"Predicted Future Vehicle Count: {predicted_value:.2f}")
    
    # Plot predictions
    prediction_fig = go.Figure()
    prediction_fig.add_trace(go.Scatter(
        x=filtered_data['timestamp'],
        y=filtered_data['vehicle_count'],
        mode='lines',
        name='Actual Vehicle Count'
    ))
    prediction_fig.add_trace(go.Scatter(
        x=[filtered_data['timestamp'].max() + timedelta(minutes=10)],
        y=[predicted_value],
        mode='markers',
        name='Predicted Vehicle Count',
        marker=dict(color='red', size=10)
    ))

    prediction_fig.update_layout(
        title="Vehicle Count Prediction using LSTM Model",
        xaxis_title="Timestamp",
        yaxis_title="Vehicle Count"
    )
    
    st.plotly_chart(prediction_fig, use_container_width=True)

elif analysis_type == "3D Traffic Map":
    st.subheader("3D Traffic Visualization on Satellite Map")
    # Create a 3D plot of traffic congestion based on selected routes
    latitudes = np.random.uniform(-1.97, -1.95, len(filtered_data))
    longitudes = np.random.uniform(30.05, 30.1, len(filtered_data))
    congestion_levels = np.where(filtered_data['congestion'] == "High", 3, np.where(filtered_data['congestion'] == "Moderate", 2, 1))

    fig = go.Figure(data=[go.Scatter3d(
        x=longitudes, y=latitudes, z=congestion_levels,
        mode='markers',
        marker=dict(
            size=6,
            color=congestion_levels,
            colorscale='Viridis',
            opacity=0.8
        )
    )])

    fig.update_layout(
        title="3D Traffic Visualization on Satellite Map",
        scene=dict(
            xaxis=dict(title="Longitude"),
            yaxis=dict(title="Latitude"),
            zaxis=dict(title="Congestion Level")
        )
    )

    st.plotly_chart(fig, use_container_width=True)
