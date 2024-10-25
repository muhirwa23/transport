import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- Initialize Session State ---
if 'traffic_data' not in st.session_state:
    st.session_state.traffic_data = pd.DataFrame(columns=[
        'route', 'timestamp', 'latitude', 'longitude', 'vehicle_count', 'event', 'average_speed'
    ])

# --- Load Route Data with Coordinates ---
@st.cache_data
def load_route_data():
    routes_data = [
        {
            "route": "Route A",
            "coordinates": [
                [30.0605, -1.9441], [30.0615, -1.9451], [30.0625, -1.9461], [30.0635, -1.9471]
            ],
            "description": "Test Route A - Sample Path"
        },
        {
            "route": "Route B",
            "coordinates": [
                [30.0689, -1.9425], [30.0699, -1.9435], [30.0709, -1.9445], [30.0719, -1.9455]
            ],
            "description": "Test Route B - Sample Path"
        },
    ]
    return routes_data

routes_df = load_route_data()

# --- Sidebar for Route Selection and Date Filtering ---
selected_route = st.sidebar.selectbox("Select a Route", [route["route"] for route in routes_df])
st.sidebar.subheader("Date Range Filter")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=7))
end_date = st.sidebar.date_input("End Date", datetime.now())

# --- Simulate Live Traffic Data ---
def simulate_event():
    route = np.random.choice([route["route"] for route in routes_df])
    vehicle_count = np.random.randint(10, 100)
    average_speed = np.random.uniform(10, 60)
    latitude, longitude = np.random.uniform(-1.96, -1.93), np.random.uniform(30.05, 30.10)
    event = np.random.choice(['Accident', 'Traffic Jam', 'Closed Road', 'Damaged Road'])

    return {
        'route': route,
        'timestamp': pd.Timestamp.now(),
        'latitude': latitude,
        'longitude': longitude,
        'vehicle_count': vehicle_count,
        'event': event,
        'average_speed': average_speed
    }

# --- Generate Real-Time Data ---
if st.sidebar.button("Simulate New Data"):
    new_data = simulate_event()
    st.session_state.traffic_data = st.session_state.traffic_data.append(new_data, ignore_index=True)

# Filter data based on selected route and date range
filtered_data = st.session_state.traffic_data[
    (st.session_state.traffic_data['route'] == selected_route) &
    (st.session_state.traffic_data['timestamp'] >= pd.to_datetime(start_date)) &
    (st.session_state.traffic_data['timestamp'] <= pd.to_datetime(end_date))
]

# --- Create 3D Simulation with Actual Routes ---
def create_3d_simulation():
    view_state = pdk.ViewState(latitude=-1.9499, longitude=30.0589, zoom=13, pitch=50)

    color_map = {'Accident': [255, 0, 0], 'Traffic Jam': [255, 165, 0], 'Closed Road': [0, 0, 255], 'Damaged Road': [128, 128, 128]}
    scatter_data = filtered_data.to_dict('records')

    scatter_layer = pdk.Layer(
        "ScatterplotLayer", data=scatter_data, get_position=["longitude", "latitude"],
        get_color=lambda d: color_map.get(d["event"], [0, 255, 0]), get_radius=300,
        pickable=True, auto_highlight=True
    )

    text_layer = pdk.Layer(
        "TextLayer", data=scatter_data, get_position=["longitude", "latitude"], get_text="event",
        get_size=16, get_color=[0, 0, 0], pickable=True
    )

    route_layers = []
    for route in routes_df:
        route_layer = pdk.Layer(
            "PathLayer", data=[{"path": route["coordinates"]}], get_path="path",
            get_width=5, get_color=[0, 255, 0], width_min_pixels=2
        )
        route_layers.append(route_layer)

    return pdk.Deck(layers=[scatter_layer, text_layer] + route_layers, initial_view_state=view_state)

# --- Predict Traffic Jam ---
def predict_traffic_jam():
    if len(st.session_state.traffic_data) < 10:
        return None

    X = st.session_state.traffic_data[['vehicle_count', 'average_speed']].dropna()
    y = np.where(X['vehicle_count'] > 50, 1, 0)

    model = RandomForestRegressor()
    model.fit(X, y)
    return model.predict([[60, 40]])[0]  # Example input

# --- User Interface ---
st.title("Kigali Transport Optimization Dashboard")

# --- Route-Specific Metrics ---
latest_vehicle_count = int(filtered_data['vehicle_count'].iloc[-1]) if not filtered_data.empty else 0
average_speed = filtered_data['average_speed'].mean() if not filtered_data.empty else "N/A"
latest_event = filtered_data['event'].iloc[-1] if not filtered_data.empty else "None"
traffic_prediction = predict_traffic_jam()

st.sidebar.metric("Route", selected_route)
st.sidebar.metric("Latest Vehicle Count", latest_vehicle_count)
st.sidebar.metric("Average Speed", f"{average_speed:.2f} km/h" if isinstance(average_speed, float) else average_speed)
st.sidebar.metric("Latest Event", latest_event)
st.sidebar.metric("Traffic Jam Prediction", "High" if traffic_prediction and traffic_prediction > 0.5 else "Low")

# Display Routes on Map
st.pydeck_chart(create_3d_simulation())

# --- Dynamic Plotly Plots ---
if not filtered_data.empty:
    # Time-Series Plot: Vehicle Count & Average Speed over Time
    st.subheader("Time-Series of Vehicle Count & Average Speed")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_data['timestamp'], y=filtered_data['vehicle_count'],
                             mode='lines+markers', name='Vehicle Count'))
    fig.add_trace(go.Scatter(x=filtered_data['timestamp'], y=filtered_data['average_speed'],
                             mode='lines+markers', name='Average Speed'))
    fig.update_layout(title=f"Traffic Analysis Over Time for {selected_route}", xaxis_title="Time", yaxis_title="Count / Speed")
    st.plotly_chart(fig)

    # Event Distribution: Bar Chart
    st.subheader("Event Distribution")
    event_counts = filtered_data['event'].value_counts()
    fig2 = px.bar(x=event_counts.index, y=event_counts.values, labels={'x': 'Event Type', 'y': 'Frequency'},
                  title="Distribution of Events")
    st.plotly_chart(fig2)

    # Scatter Plot: Average Speed vs Vehicle Count by Event
    st.subheader("Average Speed vs Vehicle Count by Event Type")
    fig3 = px.scatter(filtered_data, x="vehicle_count", y="average_speed", color="event",
                      title="Average Speed vs Vehicle Count", labels={"vehicle_count": "Vehicle Count", "average_speed": "Average Speed (km/h)"})
    st.plotly_chart(fig3)

    # Heatmap: Traffic Severity on Route
    st.subheader("Traffic Heatmap for Route Congestion")
    fig4 = px.density_mapbox(filtered_data, lat="latitude", lon="longitude", z="vehicle_count", radius=10,
                             center=dict(lat=-1.9499, lon=30.0589), zoom=12,
                             mapbox_style="carto-positron", title="Traffic Density by Vehicle Count")
    st.plotly_chart(fig4)
