import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
from sklearn.linear_model import LinearRegression
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

# --- Simulate Live Traffic Data with Severity Level ---
@st.cache_data
def simulate_event():
    route = np.random.choice(routes_df['route_short_name'])
    vehicle_count = np.random.randint(10, 100)
    latitude, longitude = np.random.uniform(-1.96, -1.93), np.random.uniform(30.05, 30.10)
    event = np.random.choice(['Accident', 'Traffic Jam', 'Closed Road', 'Damaged Road'])
    
    severity = "Minor" if vehicle_count < 30 else "Moderate" if vehicle_count < 70 else "Severe"
    return {
        'route': route,
        'timestamp': pd.Timestamp.now(),
        'latitude': latitude,
        'longitude': longitude,
        'vehicle_count': vehicle_count,
        'event': event,
        'severity': severity
    }

# --- Generate Realistic GPS Route Waypoints ---
def get_gps_route(start, end, num_points=15):
    """Simulate GPS waypoints between start and end locations."""
    lats = np.linspace(start[1], end[1], num_points)
    lons = np.linspace(start[0], end[0], num_points)
    return [[lon, lat] for lon, lat in zip(lons, lats)]

# --- Create Main 3D Map with Event Markers ---
@st.cache_data
def create_main_3d_map(selected_route=None):
    view_state = pdk.ViewState(
        latitude=-1.9499, longitude=30.0589, zoom=13, pitch=50
    )

    color_map = {
        'Accident': [255, 0, 0],          # Red
        'Traffic Jam': [255, 165, 0],     # Orange
        'Closed Road': [0, 0, 255],       # Blue
        'Damaged Road': [128, 128, 128],  # Gray
    }
    severity_map = {
        "Minor": [100, 100, 100],         # Light gray
        "Moderate": [150, 150, 0],        # Yellow
        "Severe": [255, 0, 0],            # Red
    }

    filtered_data = st.session_state.traffic_data[
        st.session_state.traffic_data['route'] == selected_route
    ] if selected_route else st.session_state.traffic_data

    scatter_data = filtered_data.to_dict('records')

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=scatter_data,
        get_position=["longitude", "latitude"],
        get_color=lambda d: color_map.get(d["event"], severity_map[d["severity"]]),
        get_radius=300,
        pickable=True,
        auto_highlight=True,
        tooltip={"text": "{route}\nEvent: {event}\nSeverity: {severity}\nVehicle Count: {vehicle_count}"}
    )

    return pdk.Deck(layers=[scatter_layer], initial_view_state=view_state)

# --- Display Main Event Cards ---
def display_event_cards():
    st.subheader("🚧 Key Traffic Events")
    event_stats = (
        st.session_state.traffic_data
        .groupby("event")
        .agg(event_count=("event", "size"), latest_time=("timestamp", "max"), severity=("severity", "first"))
        .reset_index()
    )

    for idx, row in event_stats.iterrows():
        st.markdown(f"""
        <div style="border:1px solid #ddd; padding:10px; border-radius:5px; margin:5px;">
            <h3>{row['event']}</h3>
            <p><strong>Count:</strong> {row['event_count']}</p>
            <p><strong>Severity:</strong> {row['severity']}</p>
            <p><strong>Latest Occurrence:</strong> {row['latest_time']}</p>
        </div>
        """, unsafe_allow_html=True)

# --- Enhanced Prediction Functionality ---
@st.cache_data
def predict_traffic():
    data = st.session_state.traffic_data[['vehicle_count', 'latitude', 'longitude']]
    if len(data) > 10:
        X = data[['vehicle_count']]
        y = data['vehicle_count'].rolling(2).mean().fillna(0)
        model = LinearRegression().fit(X, y)
        return model.predict([[80]])[0]  # Predict for a vehicle count of 80
    return None

# --- User Interface ---
st.title("🚦 Kigali Traffic Monitoring System with Real-Time 3D Event Simulation")

# Sidebar: List all available routes and allow selection
st.sidebar.header("🚗 Available Routes in Kigali")
selected_route = st.sidebar.selectbox("Select a route to view traffic events:", options=routes_df['route_short_name'].unique())

# Simulate and add new data
new_data = simulate_event()
st.session_state.traffic_data = st.session_state.traffic_data.append(
    pd.DataFrame([new_data]), ignore_index=True
).tail(100)

# --- Display Event Cards for Main Events ---
display_event_cards()

# --- KPI Cards ---
st.subheader("🚥 Traffic Key Performance Indicators (KPIs)")
total_events = st.session_state.traffic_data['event'].count()
avg_vehicle_count = st.session_state.traffic_data['vehicle_count'].mean()
most_common_event = st.session_state.traffic_data['event'].mode()[0]

col1, col2, col3 = st.columns(3)
col1.metric("Total Traffic Events", total_events)
col2.metric("Average Vehicle Count", f"{avg_vehicle_count:.2f}")
col3.metric("Most Common Event", most_common_event)

# --- Additional Charts ---
st.subheader("📊 Detailed Traffic Analysis")
fig_hist = px.histogram(st.session_state.traffic_data, x="vehicle_count", nbins=10, title="Vehicle Count Histogram")
st.plotly_chart(fig_hist, use_container_width=True)

fig_pie = px.pie(st.session_state.traffic_data, names="event", title="Event Type Distribution")
st.plotly_chart(fig_pie, use_container_width=True)

fig_line = px.line(
    st.session_state.traffic_data, 
    x='timestamp', y='vehicle_count', 
    title="Vehicle Count over Time", markers=True
)
st.plotly_chart(fig_line, use_container_width=True)

# --- Main 3D Map in Footer ---
st.subheader("🗺️ Kigali Real-Time 3D Traffic Map")
main_3d_map = create_main_3d_map(selected_route=selected_route)
st.pydeck_chart(main_3d_map)

# --- Traffic Prediction ---
st.subheader("🔮 Traffic Prediction")
prediction = predict_traffic()
if prediction:
    st.write(f"Predicted Vehicle Count: {int(prediction)} vehicles")

# Periodic Refresh
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 30, 10)
time.sleep(refresh_rate)
st.experimental_rerun()
