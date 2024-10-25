import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# --- Load Route Data ---
@st.cache_data
def load_route_data():
    """Load the complete route data."""
    data = """route_id,agency_id,route_short_name,route_long_name,route_type,route_desc
    101,1,101,KBS - Zone I - 101,3,Remera Taxi Park-Sonatubes-Rwandex-CBD
    102,1,102,Kabuga-Mulindi-Remera-Sonatubes-Rwandex-Nyabugogo Taxi Park
    212,2,212,ROYAL - Zone II - 212,3,St. Joseph-Kicukiro Centre-Sonatubes-Rwandex-Nyabugogo Taxi Park
    """
    from io import StringIO
    return pd.read_csv(StringIO(data))

routes_df = load_route_data()

# --- Initialize Session State ---
if 'traffic_data' not in st.session_state:
    st.session_state.traffic_data = pd.DataFrame(columns=[
        'route', 'timestamp', 'vehicle_count', 'travel_time', 'latitude', 'longitude'
    ])

if 'event_data' not in st.session_state:
    st.session_state.event_data = pd.DataFrame(columns=['latitude', 'longitude', 'event_time'])

# --- Generate Live Traffic Data ---
def generate_live_traffic_data():
    """Simulate live traffic data."""
    route = np.random.choice(routes_df['route_short_name'])
    vehicle_count = np.random.randint(10, 100)
    travel_time = np.random.uniform(10, 60)
    timestamp = pd.Timestamp.now()
    latitude = -1.9499 + np.random.uniform(-0.01, 0.01)
    longitude = 30.0589 + np.random.uniform(-0.01, 0.01)
    return {
        'route': route, 'timestamp': timestamp, 
        'vehicle_count': vehicle_count, 'travel_time': travel_time,
        'latitude': latitude, 'longitude': longitude
    }

def generate_event_data():
    """Simulate event location tracking data."""
    latitude = -1.9499 + np.random.uniform(-0.02, 0.02)
    longitude = 30.0589 + np.random.uniform(-0.02, 0.02)
    event_time = pd.Timestamp.now()
    return {'latitude': latitude, 'longitude': longitude, 'event_time': event_time}

# --- Display UI ---
st.title("🚦 Kigali Traffic Monitoring and Prediction System")

# --- Route Selection ---
selected_routes = st.sidebar.multiselect(
    "Select Routes", routes_df['route_short_name'].unique(), default=[]
)
min_vehicle_count = st.sidebar.slider("Min Vehicle Count", 0, 100, 10)
max_travel_time = st.sidebar.slider("Max Travel Time (minutes)", 10, 60, 30)

# --- Generate and Update Traffic Data ---
new_traffic_data = generate_live_traffic_data()
st.session_state.traffic_data = pd.concat(
    [st.session_state.traffic_data, pd.DataFrame([new_traffic_data])], ignore_index=True
).tail(50)

# --- Generate and Update Event Data ---
new_event = generate_event_data()
st.session_state.event_data = pd.concat(
    [st.session_state.event_data, pd.DataFrame([new_event])], ignore_index=True
).tail(20)  # Keep only the last 20 events

# --- KPI Cards ---
st.header("🚗 Key Performance Indicators")
col1, col2, col3 = st.columns(3)

with col1:
    avg_vehicle_count = st.session_state.traffic_data['vehicle_count'].mean() if not st.session_state.traffic_data.empty else 0
    st.metric("Avg Vehicle Count", f"{avg_vehicle_count:.2f}")

with col2:
    avg_travel_time = st.session_state.traffic_data['travel_time'].mean() if not st.session_state.traffic_data.empty else 0
    st.metric("Avg Travel Time (min)", f"{avg_travel_time:.2f}")

with col3:
    congestion_level = "High" if avg_vehicle_count > 50 else "Low"
    st.metric("Congestion Level", congestion_level)

# --- 3D Event Tracking Map ---
st.subheader("🌍 Real-Time 3D Event Tracking in Kigali")

fig_map_3d = go.Figure(data=[go.Scatter3d(
    x=st.session_state.event_data['longitude'],
    y=st.session_state.event_data['latitude'],
    z=[0] * len(st.session_state.event_data),  # Ground level
    mode='markers',
    marker=dict(
        size=6,
        color='rgb(255,69,0)',  # Orange-Red for events
        opacity=0.8,
    ),
    text=st.session_state.event_data['event_time'],
    hoverinfo='text',
)])

fig_map_3d.update_layout(
    scene=dict(
        xaxis=dict(title="Longitude", backgroundcolor="rgb(230, 230, 230)", gridcolor="white"),
        yaxis=dict(title="Latitude", backgroundcolor="rgb(230, 230, 230)", gridcolor="white"),
        zaxis=dict(title="Elevation", range=[0, 100], visible=False),
    ),
    margin=dict(r=10, l=10, b=10, t=10),
    title="3D Map of Event Locations",
    paper_bgcolor="rgb(245,245,245)",
)

st.plotly_chart(fig_map_3d, use_container_width=True)

# --- Real-Time Vehicle Count Chart ---
st.subheader("📈 Real-Time Vehicle Count")

line_fig = px.line(
    st.session_state.traffic_data, x='timestamp', y='vehicle_count', 
    title="Vehicle Count Over Time", markers=True,
    color_discrete_sequence=["rgb(52,152,219)"]
)
st.plotly_chart(line_fig, use_container_width=True)

# --- Predict Future Vehicle Count ---
st.subheader("🔮 Vehicle Count Prediction")

timestamps = np.array([i for i in range(len(st.session_state.traffic_data))]).reshape(-1, 1)
vehicle_counts = st.session_state.traffic_data['vehicle_count'].values

poly = PolynomialFeatures(degree=2)
timestamps_poly = poly.fit_transform(timestamps)
model = LinearRegression().fit(timestamps_poly, vehicle_counts)
future_timestamps = np.array([i for i in range(len(timestamps), len(timestamps) + 10)]).reshape(-1, 1)
future_timestamps_poly = poly.transform(future_timestamps)
future_vehicle_counts = model.predict(future_timestamps_poly)

pred_fig = px.line(
    x=list(timestamps.flatten()) + list(future_timestamps.flatten()),
    y=list(vehicle_counts) + list(future_vehicle_counts),
    labels={'x': 'Time', 'y': 'Vehicle Count'},
    title="Predicted vs Observed Vehicle Counts"
)
pred_fig.add_scatter(x=timestamps.flatten(), y=vehicle_counts, mode='markers', name='Observed')
pred_fig.add_scatter(x=future_timestamps.flatten(), y=future_vehicle_counts, mode='lines', name='Predicted')
st.plotly_chart(pred_fig, use_container_width=True)

# --- Dynamic Average Travel Time per Route ---
st.subheader("⏱️ Average Travel Time per Route")

avg_travel_time_fig = px.bar(
    st.session_state.traffic_data.groupby("route")['travel_time'].mean().reset_index(),
    x='route', y='travel_time',
    title="Average Travel Time per Route",
    labels={'travel_time': 'Avg Travel Time (min)'},
    color_discrete_sequence=["rgb(46,204,113)"]
)
st.plotly_chart(avg_travel_time_fig, use_container_width=True)

# --- Suggested Routes ---
st.sidebar.subheader("🚍 Suggested Routes")

selected_route = st.sidebar.selectbox("Select Route for Suggestions", routes_df['route_short_name'])
congested_routes = st.session_state.traffic_data[st.session_state.traffic_data['vehicle_count'] > 50]['route'].unique()
suggestions = routes_df[~routes_df['route_short_name'].isin(congested_routes)]

st.sidebar.write(f"🛣️ **Alternate routes to avoid congestion on {selected_route}:**")
st.sidebar.table(suggestions[['route_short_name', 'route_long_name']])

# --- Refresh Dashboard ---
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 30, 10)
if st.sidebar.button("Refresh Now"):
    st.experimental_rerun()

# --- Periodic Refresh Logic ---
time.sleep(refresh_rate)
st.experimental_rerun()
