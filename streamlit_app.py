import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from datetime import datetime
import time

# --- Data Simulation Setup ---
def generate_live_data_kigali():
    """Simulate live traffic data for Kigali city."""
    np.random.seed(int(datetime.now().timestamp()))
    routes = [
        'Kigali-Rubavu', 'Kigali-Muhanga', 'Kigali-Huye', 'Kigali-Nyagatare',
        'Kigali-Kayonza', 'Kigali-Musanze', 'Kigali-Gatuna', 'Kigali-Bugesera'
    ]
    route = np.random.choice(routes)
    vehicle_count = np.random.randint(10, 150)
    travel_time = np.random.uniform(10, 60)  # Travel time in minutes
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {"timestamp": timestamp, "route": route, "vehicle_count": vehicle_count, "travel_time": travel_time}

# --- Initialize DataFrame for Live Data ---
if 'kigali_data' not in st.session_state:
    st.session_state.kigali_data = pd.DataFrame([generate_live_data_kigali() for _ in range(15)])

# --- Sidebar for Route Optimization Section ---
st.sidebar.header("Kigali City Route Optimization")

# Route Selection Filters
selected_kigali_routes = st.sidebar.multiselect(
    "Select Main Kigali Routes", 
    ['Kigali-Rubavu', 'Kigali-Muhanga', 'Kigali-Huye', 'Kigali-Nyagatare',
     'Kigali-Kayonza', 'Kigali-Musanze', 'Kigali-Gatuna', 'Kigali-Bugesera'],
    default=['Kigali-Rubavu', 'Kigali-Huye']
)
min_kigali_vehicle_count = st.sidebar.slider("Minimum Vehicle Count", 0, 150, 20)
max_kigali_travel_time = st.sidebar.slider("Maximum Travel Time (minutes)", 10, 60, 30)

refresh_rate_kigali = st.sidebar.slider("Kigali Data Refresh Rate (seconds)", 1, 10, 5, step=1)
show_kigali_table = st.sidebar.checkbox("Show Raw Data for Kigali")

# --- Bokeh Plot for Kigali Routes ---
def bokeh_kigali_chart(data):
    """Generate a Bokeh chart for live traffic data in Kigali."""
    source = ColumnDataSource(data)
    p = figure(
        title="Live Traffic Data for Kigali City Routes",
        x_axis_type="datetime",
        height=400,
        x_axis_label="Timestamp",
        y_axis_label="Vehicle Count",
    )
    p.line(x="timestamp", y="vehicle_count", line_width=2, source=source, color="blue", legend_label="Vehicles")
    p.circle(x="timestamp", y="travel_time", size=8, source=source, color="green", legend_label="Travel Time (min)")
    p.legend.title = "Metrics"
    return p

# --- Route Optimization Section in Main App ---
st.title("Kigali City Route Optimization")

st.markdown("### Live Monitoring of Main Routes in Kigali")
kigali_chart_placeholder = st.empty()  # Placeholder for the chart

while True:
    # Generate and append new simulated data
    new_kigali_data = generate_live_data_kigali()
    st.session_state.kigali_data = pd.concat([st.session_state.kigali_data, pd.DataFrame([new_kigali_data])], ignore_index=True)

    # Keep the DataFrame size manageable
    st.session_state.kigali_data = st.session_state.kigali_data.tail(50)

    # Filter the data based on user input
    filtered_kigali_data = st.session_state.kigali_data[
        (st.session_state.kigali_data['route'].isin(selected_kigali_routes)) &
        (st.session_state.kigali_data['vehicle_count'] >= min_kigali_vehicle_count) &
        (st.session_state.kigali_data['travel_time'] <= max_kigali_travel_time)
    ]

    # Display the chart
    kigali_chart = bokeh_kigali_chart(filtered_kigali_data)
    kigali_chart_placeholder.bokeh_chart(kigali_chart, use_container_width=True)

    if show_kigali_table:
        st.dataframe(filtered_kigali_data)

    # Refresh the dashboard at the user-defined interval
    time.sleep(refresh_rate_kigali)
    st.experimental_rerun()
