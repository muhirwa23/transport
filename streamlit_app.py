# --- Import Required Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time

# --- Simulate Data ---
@st.cache_data
def simulate_data(n_samples=1000):
    """
    Simulate traffic data for analysis and prediction purposes.
    
    Args:
        n_samples (int): Number of samples to generate.
    
    Returns:
        pandas.DataFrame: A DataFrame with simulated traffic data.
    """
    np.random.seed(42)
    
    # Simulate traffic data: vehicle count, speed, time, route, lat/lon for mapping
    routes = ['Kigali - Kanombe', 'Kigali - Nyabugogo', 'Kigali - Kicukiro', 'Kigali - Kimironko', 'Kigali - Remera']
    latitudes = [1.96, 1.94, 1.92, 1.93, 1.95]
    longitudes = [30.11, 30.08, 30.12, 30.10, 30.09]
    
    data = {
        'route': np.random.choice(routes, n_samples),
        'latitude': np.random.choice(latitudes, n_samples),
        'longitude': np.random.choice(longitudes, n_samples),
        'vehicle_count': np.random.randint(10, 100, n_samples),
        'average_speed': np.random.uniform(10, 60, n_samples),
        'event': np.random.choice(['Accident', 'Traffic Jam', 'Closed Road', 'Damaged Road'], n_samples),
        'timestamp': pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
    }
    
    return pd.DataFrame(data)

# Load simulated data into session state
if 'traffic_data' not in st.session_state:
    st.session_state.traffic_data = simulate_data()

# --- Sidebar Settings ---
st.sidebar.header("Settings")
n_samples = st.sidebar.slider("Number of Samples to Simulate", min_value=500, max_value=5000, value=1000, step=100)
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 30, 10)
vehicle_threshold = st.sidebar.slider("Vehicle Threshold for Traffic Jam Prediction", 10, 100, 50)

# --- Title ---
st.title("Kigali Transport Optimization & Prediction Dashboard")

# --- Resimulate Data if Samples Changed ---
if st.sidebar.button("Resimulate Data"):
    st.session_state.traffic_data = simulate_data(n_samples)

# --- Display Dataframe ---
st.subheader("Simulated Traffic Data")
st.dataframe(st.session_state.traffic_data.head())

# --- Predictive Model: Random Forest Regression ---
def predict_vehicle_count():
    """
    Predict vehicle count using a Random Forest Regressor based on speed and event type.
    
    Returns:
        pandas.Series: Predictions for vehicle count.
        sklearn.ensemble.RandomForestRegressor: Trained model.
    """
    # Feature encoding: Convert categorical event type to numeric
    traffic_data = st.session_state.traffic_data.copy()
    traffic_data['event_code'] = traffic_data['event'].astype('category').cat.codes
    
    X = traffic_data[['average_speed', 'event_code']]
    y = traffic_data['vehicle_count']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return pd.Series(y_pred, index=X_test.index), model

# --- Prediction ---
st.subheader("Traffic Prediction: Vehicle Count")
predictions, model = predict_vehicle_count()

# --- Display Model Evaluation ---
st.markdown("### Model Evaluation")
mse = mean_squared_error(st.session_state.traffic_data['vehicle_count'][predictions.index], predictions)
r2 = r2_score(st.session_state.traffic_data['vehicle_count'][predictions.index], predictions)
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R-Squared: {r2:.2f}")

# --- Dynamic Predictive Model Chart ---
st.subheader("Dynamic Prediction vs Actual: Vehicle Count")
fig = go.Figure()

# Dropdown for dynamic filtering
selected_route = st.selectbox("Select a Route to Display:", st.session_state.traffic_data['route'].unique())

# Filter by selected route
filtered_data = st.session_state.traffic_data[st.session_state.traffic_data['route'] == selected_route]
predictions_filtered = predictions[filtered_data.index]

fig.add_trace(go.Scatter(
    x=filtered_data['timestamp'],
    y=filtered_data['vehicle_count'],
    mode='lines+markers', name='Actual'
))
fig.add_trace(go.Scatter(
    x=filtered_data['timestamp'],
    y=predictions_filtered, mode='lines+markers', name='Predicted'
))
fig.update_layout(title=f"Actual vs Predicted Vehicle Count for {selected_route}", 
                  xaxis_title="Time", yaxis_title="Vehicle Count")
st.plotly_chart(fig)

# --- Dynamic 3D Map Visualization ---
st.subheader("3D Map Visualization of Routes")
fig_map = go.Figure()

# Add 3D scatter for lat/lon of routes
fig_map.add_trace(go.Scatter3d(
    x=st.session_state.traffic_data['longitude'], 
    y=st.session_state.traffic_data['latitude'], 
    z=st.session_state.traffic_data['vehicle_count'],
    mode='markers',
    marker=dict(size=5, color=st.session_state.traffic_data['vehicle_count'], colorscale='Viridis', opacity=0.8),
    text=st.session_state.traffic_data['route'], 
    hoverinfo='text+x+y+z'
))

fig_map.update_layout(
    scene=dict(
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        zaxis_title='Vehicle Count'
    ),
    title="3D Map Visualization of Traffic Routes"
)

st.plotly_chart(fig_map)

# --- Additional Exploratory Charts ---
st.subheader("Exploratory Data Analysis (EDA)")

# Histogram of Vehicle Counts
st.markdown("### Histogram of Vehicle Counts")
fig_hist = px.histogram(st.session_state.traffic_data, x="vehicle_count", nbins=30)
st.plotly_chart(fig_hist)

# Event Distribution
st.markdown("### Event Type Distribution")
fig_bar = px.bar(st.session_state.traffic_data['event'].value_counts(), 
                 labels={'index': 'Event Type', 'value': 'Count'}, title="Event Distribution")
st.plotly_chart(fig_bar)

# Scatter Plot: Speed vs Vehicle Count
st.markdown("### Scatter Plot: Speed vs Vehicle Count")
fig_scatter = px.scatter(st.session_state.traffic_data, x='average_speed', y='vehicle_count', color='event', 
                         labels={'average_speed': 'Average Speed (km/h)', 'vehicle_count': 'Vehicle Count'},
                         title="Speed vs Vehicle Count")
st.plotly_chart(fig_scatter)

# Time Series of Vehicle Count and Speed
st.subheader("Time Series Analysis")
fig_time = go.Figure()
fig_time.add_trace(go.Scatter(
    x=st.session_state.traffic_data['timestamp'], y=st.session_state.traffic_data['vehicle_count'],
    mode='lines+markers', name='Vehicle Count'
))
fig_time.add_trace(go.Scatter(
    x=st.session_state.traffic_data['timestamp'], y=st.session_state.traffic_data['average_speed'],
    mode='lines+markers', name='Average Speed'
))
fig_time.update_layout(title="Time Series of Vehicle Count and Speed", xaxis_title="Time", yaxis_title="Count / Speed")
st.plotly_chart(fig_time)

# --- Refresh the Dashboard at the Defined Rate ---
time.sleep(refresh_rate)
st.experimental_rerun()
