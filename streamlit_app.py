import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import time

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

# --- Generate Live Data ---
def generate_live_data():
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

# --- Display UI ---
st.title("Kigali Traffic Monitoring and Optimization System")

# --- Route Selection ---
selected_routes = st.sidebar.multiselect(
    "Select Routes", routes_df['route_short_name'].unique(), default=[]
)
min_vehicle_count = st.sidebar.slider("Min Vehicle Count", 0, 100, 10)
max_travel_time = st.sidebar.slider("Max Travel Time (minutes)", 10, 60, 30)

# --- Generate and Update Traffic Data ---
new_data = generate_live_data()
st.session_state.traffic_data = pd.concat(
    [st.session_state.traffic_data, pd.DataFrame([new_data])], ignore_index=True
).tail(50)

# --- Filter Data Based on User Inputs ---
filtered_data = st.session_state.traffic_data[
    (st.session_state.traffic_data['route'].isin(selected_routes)) &
    (st.session_state.traffic_data['vehicle_count'] >= min_vehicle_count) &
    (st.session_state.traffic_data['travel_time'] <= max_travel_time)
]

# --- KPI Cards ---
st.header("Key Performance Indicators")
col1, col2, col3 = st.columns(3)

with col1:
    avg_vehicle_count = filtered_data['vehicle_count'].mean() if not filtered_data.empty else 0
    st.metric("Average Vehicle Count", f"{avg_vehicle_count:.2f}")

with col2:
    avg_travel_time = filtered_data['travel_time'].mean() if not filtered_data.empty else 0
    st.metric("Average Travel Time (min)", f"{avg_travel_time:.2f}")

with col3:
    congestion_level = "High" if avg_vehicle_count > 50 else "Low"
    st.metric("Congestion Level", congestion_level)

# --- Dynamic 3D Map with Traffic Data ---
st.subheader("Live 3D Traffic Map")
fig = px.scatter_3d(
    filtered_data, 
    x='longitude', y='latitude', z='vehicle_count',
    color='vehicle_count',
    size='travel_time',
    hover_data=['route', 'timestamp', 'vehicle_count', 'travel_time'],
    color_continuous_scale=px.colors.sequential.Plasma,
    title="Traffic Congestion by Route"
)
fig.update_layout(scene=dict(
    xaxis_title='Longitude',
    yaxis_title='Latitude',
    zaxis_title='Vehicle Count'
))
st.plotly_chart(fig, use_container_width=True)

# --- Real-Time Vehicle Count Chart ---
st.subheader("Real-Time Vehicle Count")
line_fig = px.line(
    filtered_data, x='timestamp', y='vehicle_count', 
    title="Real-Time Vehicle Count per Route", markers=True
)
st.plotly_chart(line_fig, use_container_width=True)

# --- Average Travel Time per Route (Dynamic Bar Chart) ---
st.subheader("Dynamic Average Travel Time per Route")
avg_travel_time_fig = px.bar(
    filtered_data.groupby("route")['travel_time'].mean().reset_index(),
    x='route', y='travel_time',
    title="Dynamic Average Travel Time per Route",
    labels={'travel_time': 'Average Travel Time (minutes)'}
)
st.plotly_chart(avg_travel_time_fig, use_container_width=True)

# --- Vehicle Count Distribution ---
st.subheader("Vehicle Count Distribution")
vehicle_count_hist = px.histogram(
    filtered_data, x='vehicle_count', nbins=10,
    title="Vehicle Count Distribution",
    labels={'vehicle_count': 'Number of Vehicles'}
)
st.plotly_chart(vehicle_count_hist, use_container_width=True)

# --- Travel Time vs Vehicle Count ---
st.subheader("Travel Time vs Vehicle Count")
scatter_fig = px.scatter(
    filtered_data, x='vehicle_count', y='travel_time',
    title="Travel Time vs Vehicle Count",
    labels={'vehicle_count': 'Vehicle Count', 'travel_time': 'Travel Time (minutes)'},
    trendline='ols'
)
st.plotly_chart(scatter_fig, use_container_width=True)

# --- Suggest Alternate Routes ---
st.sidebar.subheader("Suggest Alternate Routes")
selected_route = st.sidebar.selectbox("Select Route for Suggestions", routes_df['route_short_name'])
alternate_routes = routes_df[routes_df['route_short_name'] != selected_route]
st.sidebar.write(f"Alternate routes for {selected_route}:")
st.sidebar.table(alternate_routes[['route_short_name', 'route_long_name']])

# --- Display Alternate Routes on 3D Map ---
for _, row in alternate_routes.iterrows():
    fig.add_trace(
        go.Scatter3d(
            x=[30.0589 + np.random.uniform(-0.01, 0.01)],
            y=[-1.9499 + np.random.uniform(-0.01, 0.01)],
            z=[np.random.randint(10, 100)],
            mode='markers',
            marker=dict(size=8, color='green'),
            name=row['route_short_name'],
            text=row['route_long_name']
        )
    )
st.plotly_chart(fig, use_container_width=True)

# --- Refresh Dashboard ---
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 30, 10)
if st.sidebar.button("Refresh Now"):
    st.experimental_rerun()

# --- Periodic Refresh Logic ---
time.sleep(refresh_rate)
st.experimental_rerun()
