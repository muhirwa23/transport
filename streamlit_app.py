import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from datetime import datetime, timedelta
import time
import plotly.express as px
import altair as alt
from io import StringIO

# --- SETTING PAGE CONFIG ---
st.set_page_config(
    page_title="Kigali Traffic Optimization",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SAMPLE ROUTE DATA ---
route_data = """
route_id,agency_id,route_short_name,route_long_name,route_type,route_desc
101,1,101,KBS - Zone I - 101,3,Remera Taxi Park-Sonatubes-Rwandex-CBD
102,1,102,Kabuga-Mulindi-Remera-Sonatubes-Rwandex Nyabugogo Taxi Park
103,1,103,KBS - Zone I - 103,3,Rubilizi-Kabeza-Remera-Sonatubes-Rwandex-CBD
104,1,104,KBS - Zone I - 104,3,Kibaya-Kanombe MH-Airport-Remera-Sonatubes-Rwandex-CBD
105,1,105,KBS - Zone I - 105,3,Remera Taxi Park-Chez Lando-Kacyiru-NyabugogoTaxi Park
"""

@st.cache_data
def load_route_data():
    return pd.read_csv(StringIO(route_data))

# --- INITIALIZE ROUTE DATA ---
routes_df = load_route_data()

# --- FUNCTION TO GENERATE LIVE DATA ---
def generate_live_data():
    """Simulate live data for traffic stats."""
    np.random.seed(int(datetime.now().timestamp()))
    vehicle_count = np.random.randint(20, 100)
    travel_time = np.random.uniform(5, 25)
    route = np.random.choice(routes_df['route_short_name'])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {"timestamp": timestamp, "vehicle_count": vehicle_count, "travel_time": travel_time, "route": route}

# --- INITIALIZE LIVE DATA SESSION STATE ---
if 'traffic_data' not in st.session_state:
    st.session_state.traffic_data = pd.DataFrame([generate_live_data() for _ in range(10)])

# --- SIDEBAR FILTERS ---
st.sidebar.header("Control Panel")

# Filters for route optimization
selected_routes = st.sidebar.multiselect(
    "Select Routes", routes_df['route_short_name'], default=routes_df['route_short_name'].tolist()
)
min_vehicle_count = st.sidebar.slider("Minimum Vehicle Count", 0, 100, 20)
max_travel_time = st.sidebar.slider("Maximum Travel Time (minutes)", 5, 30, 20)

# Refresh rate and view options
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 10, 5)
view_option = st.sidebar.selectbox("Select View", ["Traffic Chart", "Geo Map"])
show_table = st.sidebar.checkbox("Show Raw Data")

# Congestion threshold for optimization
congestion_threshold = st.sidebar.slider("Set Congestion Threshold", 0, 100, 50)

# --- BOKEH LIVE CHART FUNCTION ---
def bokeh_live_chart(data):
    """Generate a Bokeh chart for live traffic data."""
    source = ColumnDataSource(data)
    p = figure(
        title="Live Traffic Data",
        x_axis_type="datetime",
        height=400,
        x_axis_label="Timestamp",
        y_axis_label="Vehicle Count",
    )
    p.line(x="timestamp", y="vehicle_count", line_width=2, source=source, color="blue", legend_label="Vehicles")
    p.line(x="timestamp", y="travel_time", line_width=2, source=source, color="green", legend_label="Travel Time")
    hover = HoverTool(tooltips=[("Time", "@timestamp"), ("Vehicles", "@vehicle_count"), ("Travel Time", "@travel_time")])
    p.add_tools(hover)
    p.legend.title = "Metrics"
    return p

# --- MAIN LOGIC: CHART OR MAP VIEW ---
if view_option == "Traffic Chart":
    st.title("Live Traffic Monitoring")

    # Filter data based on user input
    filtered_data = st.session_state.traffic_data[
        (st.session_state.traffic_data['route'].isin(selected_routes)) &
        (st.session_state.traffic_data['vehicle_count'] >= min_vehicle_count) &
        (st.session_state.traffic_data['travel_time'] <= max_travel_time)
    ]

    # Placeholder for dynamic updates
    chart_placeholder = st.empty()

    while True:
        # Append new data
        new_data = generate_live_data()
        st.session_state.traffic_data = pd.concat([st.session_state.traffic_data, pd.DataFrame([new_data])], ignore_index=True)
        st.session_state.traffic_data = st.session_state.traffic_data.tail(50)  # Keep DataFrame manageable

        # Reapply filters on latest data
        filtered_data = st.session_state.traffic_data[
            (st.session_state.traffic_data['route'].isin(selected_routes)) &
            (st.session_state.traffic_data['vehicle_count'] >= min_vehicle_count) &
            (st.session_state.traffic_data['travel_time'] <= max_travel_time)
        ]

        # Update chart
        chart = bokeh_live_chart(filtered_data)
        chart_placeholder.bokeh_chart(chart, use_container_width=True)

        if show_table:
            st.dataframe(filtered_data)

        time.sleep(refresh_rate)
        st.experimental_rerun()

elif view_option == "Geo Map":
    st.title("Kigali Traffic Geo Map")
    geo_data = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    st.map(geo_data)

# --- TRAFFIC OPTIMIZATION SUGGESTIONS ---
st.sidebar.header("Optimization Options")
selected_route = st.sidebar.selectbox("Select Route", routes_df['route_short_name'])

def suggest_alternate_routes(selected_route, threshold):
    if st.session_state.traffic_data['vehicle_count'].mean() > threshold:
        st.warning(f"High congestion on Route {selected_route}. Suggesting alternate routes...")
        suggestions = routes_df[routes_df['route_short_name'] != selected_route].head(3)
        st.table(suggestions)
    else:
        st.success(f"Traffic on Route {selected_route} is under control.")

suggest_alternate_routes(selected_route, congestion_threshold)

# --- ALTAR VEHICLE CHART ---
def altair_vehicle_chart(data):
    chart = alt.Chart(data).mark_line().encode(
        x='timestamp:T',
        y='vehicle_count:Q',
        tooltip=['timestamp:T', 'vehicle_count:Q']
    ).properties(title='Vehicle Count Over Time')
    return chart

st.altair_chart(altair_vehicle_chart(st.session_state.traffic_data), use_container_width=True)

# --- PLOTLY TRAVEL TIME TREND ---
fig = px.line(st.session_state.traffic_data, x='timestamp', y='travel_time', title="Travel Time Trend")
st.plotly_chart(fig, use_container_width=True)

# --- FOOTER ---
st.markdown("""
---
**Developed by Kigali City Transport Team**  
*Optimizing traffic with real-time monitoring and route suggestions*
""")
