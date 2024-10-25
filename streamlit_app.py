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

# --- Load Real Route Data for Kigali ---
@st.cache_data
def load_kigali_routes():
    # Real routes are simplified here for the demo; replace with actual routes
    routes_data = [
        {
            "route": "Kigali Route 1",
            "coordinates": [
                [30.0625, -1.9486], [30.0651, -1.9498], [30.0677, -1.9510], [30.0703, -1.9522]
            ],
            "description": "Main road between city center and suburbs"
        },
        {
            "route": "Kigali Route 2",
            "coordinates": [
                [30.0739, -1.9463], [30.0755, -1.9475], [30.0771, -1.9487], [30.0787, -1.9499]
            ],
            "description": "Popular route near major commercial areas"
        },
    ]
    return routes_data

routes_df = load_kigali_routes()

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

# --- Create 3D Map with Actual Routes ---
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
    # Traffic Event Timeline
    st.subheader("Traffic Event Timeline")
    fig_timeline = px.timeline(filtered_data, x_start="timestamp", x_end="timestamp", y="route",
                               color="event", title="Traffic Event Timeline by Route")
    st.plotly_chart(fig_timeline)

    # Event Type Distribution Pie Chart
    st.subheader("Event Type Distribution")
    event_counts = filtered_data['event'].value_counts()
    fig_pie = px.pie(names=event_counts.index, values=event_counts.values, title="Event Type Distribution")
    st.plotly_chart(fig_pie)

    # Histogram of Vehicle Count
    st.subheader("Vehicle Count Distribution")
    fig_histogram = px.histogram(filtered_data, x="vehicle_count", nbins=20, title="Histogram of Vehicle Counts")
    st.plotly_chart(fig_histogram)

    # Average Speed Heatmap
    st.subheader("Average Speed Heatmap")
    fig_heatmap = px.density_mapbox(filtered_data, lat="latitude", lon="longitude", z="average_speed", radius=10,
                                    center=dict(lat=-1.9499, lon=30.0589), zoom=12,
                                    mapbox_style="carto-positron", title="Average Speed Heatmap")
    st.plotly_chart(fig_heatmap)

    # Scatter Plot: Average Speed vs Vehicle Count by Event
    st.subheader("Average Speed vs Vehicle Count by Event Type")
    fig_scatter = px.scatter(filtered_data, x="vehicle_count", y="average_speed", color="event",
                             title="Average Speed vs Vehicle Count", labels={"vehicle_count": "Vehicle Count", "average_speed": "Average Speed (km/h)"})
    st.plotly_chart(fig_scatter)
