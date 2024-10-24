import streamlit as st
import pandas as pd
from io import StringIO  # Use the correct module for StringIO
import altair as alt
import plotly.express as px
import bokeh.plotting as bk
from bokeh.models import HoverTool
import geopandas as gpd
import numpy as np

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
102,1,102,KBS - Zone I - 102,3,Kabuga-Mulindi-Remera-Sonatubes-Rwandex Nyabugogo Taxi Park
103,1,103,KBS - Zone I - 103,3,Rubilizi-Kabeza-Remera-Sonatubes-Rwandex-CBD
104,1,104,KBS - Zone I - 104,3,Kibaya-Kanombe MH-Airport-Remera-Sonatubes-Rwandex-CBD
105,1,105,KBS - Zone I - 105,3,Remera Taxi Park-Chez Lando-Kacyiru-NyabugogoTaxi Park
"""

@st.cache_data  # Use Streamlit's cache for performance
def load_route_data():
    return pd.read_csv(StringIO(route_data))
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from datetime import datetime, timedelta
import time

# --- Data Simulation Setup ---
def generate_live_data():
    """Simulate live data for traffic stats."""
    np.random.seed(int(datetime.now().timestamp()))
    vehicle_count = np.random.randint(20, 100)
    travel_time = np.random.uniform(5, 25)
    route = np.random.choice(['Route A', 'Route B', 'Route C', 'Route D'])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {"timestamp": timestamp, "vehicle_count": vehicle_count, "travel_time": travel_time, "route": route}

# --- Initialize DataFrame for Live Data ---
if 'traffic_data' not in st.session_state:
    st.session_state.traffic_data = pd.DataFrame([generate_live_data() for _ in range(10)])

# --- Widgets for Interactivity and Optimization ---
st.sidebar.header("Control Panel")

# Filters for route optimization
st.sidebar.subheader("Optimization Filters")
selected_routes = st.sidebar.multiselect("Select Routes", ['Route A', 'Route B', 'Route C', 'Route D'], default=['Route A', 'Route B'])
min_vehicle_count = st.sidebar.slider("Minimum Vehicle Count", min_value=0, max_value=100, value=20)
max_travel_time = st.sidebar.slider("Maximum Travel Time (minutes)", min_value=5, max_value=30, value=20)

# Refresh and view options
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", min_value=1, max_value=10, value=5, step=1)
view_option = st.sidebar.selectbox("Select View", ["Traffic Chart", "Geo Map"])
show_table = st.sidebar.checkbox("Show Raw Data")

# --- Bokeh Plot Setup ---
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
    p.legend.title = "Metrics"
    return p

# --- Geopandas Map Setup ---
@st.cache_data
def load_geodata():
    """Load and cache geographic data."""
    return gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

geo_data = load_geodata()

# --- Main App Logic ---
if view_option == "Traffic Chart":
    st.title("Live Traffic Monitoring")

    # Filter data based on user selections
    filtered_data = st.session_state.traffic_data[
        (st.session_state.traffic_data['route'].isin(selected_routes)) &
        (st.session_state.traffic_data['vehicle_count'] >= min_vehicle_count) &
        (st.session_state.traffic_data['travel_time'] <= max_travel_time)
    ]

    # Placeholder for dynamic updates
    chart_placeholder = st.empty()

    while True:
        # Append new data to the traffic DataFrame
        new_data = generate_live_data()
        st.session_state.traffic_data = pd.concat([st.session_state.traffic_data, pd.DataFrame([new_data])], ignore_index=True)

        # Keep the DataFrame size manageable
        st.session_state.traffic_data = st.session_state.traffic_data.tail(50)

        # Apply the filters again to the latest data
        filtered_data = st.session_state.traffic_data[
            (st.session_state.traffic_data['route'].isin(selected_routes)) &
            (st.session_state.traffic_data['vehicle_count'] >= min_vehicle_count) &
            (st.session_state.traffic_data['travel_time'] <= max_travel_time)
        ]

        # Plot the updated data
        chart = bokeh_live_chart(filtered_data)
        chart_placeholder.bokeh_chart(chart, use_container_width=True)

        if show_table:
            st.dataframe(filtered_data)

        # Refresh the dashboard at the user-defined interval
        time.sleep(refresh_rate)
        st.experimental_rerun()

elif view_option == "Geo Map":
    st.title("Global Transportation Map")
    st.map(geo_data)

# --- LOAD ROUTES DATA ---
routes_df = load_route_data()

# --- DISPLAY ROUTE TABLE ---
st.title("Kigali Traffic Optimization System")
st.subheader("Available Routes")
st.dataframe(routes_df, use_container_width=True)

# --- WAZE LIVE MAP EMBED ---
st.markdown("""
## Live Traffic Map
<iframe src="https://embed.waze.com/iframe?zoom=12&lat=-1.934712&lon=29.974184&ct=livemap" 
width="100%" height="450" allowfullscreen></iframe>
""", unsafe_allow_html=True)

# --- GENERATE SIMULATION DATA ---
np.random.seed(42)
time_range = pd.date_range(start='2024-10-24', periods=100, freq='T')
traffic_data = pd.DataFrame({
    'timestamp': time_range,
    'vehicle_count': np.random.randint(20, 100, size=100),
    'travel_time': np.random.uniform(5, 25, size=100)
})

# --- ALTAR CHART FOR TIME SERIES VEHICLE COUNTS ---
def altair_vehicle_chart(data):
    chart = alt.Chart(data).mark_line().encode(
        x='timestamp:T',
        y='vehicle_count:Q',
        tooltip=['timestamp:T', 'vehicle_count:Q']
    ).properties(title='Vehicle Count Over Time')
    return chart

st.altair_chart(altair_vehicle_chart(traffic_data), use_container_width=True)

# --- PLOTLY REAL-TIME TRAVEL TIME ---
fig = px.line(traffic_data, x='timestamp', y='travel_time', title="Travel Time Trend")
st.plotly_chart(fig, use_container_width=True)

# --- BOKEH CONGESTION CHART ---
import streamlit as st
import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.models import HoverTool

# --- Dummy Traffic Data ---
np.random.seed(42)
time_range = pd.date_range(start='2024-10-24', periods=100, freq='T')
traffic_data = pd.DataFrame({
    'timestamp': time_range,
    'vehicle_count': np.random.randint(20, 100, size=100),
    'travel_time': np.random.uniform(5, 25, size=100)
})

# --- Bokeh Chart Function ---
def bokeh_congestion_chart(data):
    # Create the Bokeh figure
    p = figure(
        x_axis_type="datetime", 
        title="Congestion Analysis", 
        width=800, 
        height=400
    )

    # Add line plot for vehicle count
    p.line(data['timestamp'], data['vehicle_count'], line_width=2, legend_label="Vehicle Count", color="navy")

    # Configure Hover Tool for interactivity
    hover = HoverTool(
        tooltips=[("Time", "@x{%F %T}"), ("Vehicles", "$y")],
        formatters={"@x": "datetime"}
    )
    p.add_tools(hover)

    # Style the plot
    p.xaxis.axis_label = "Timestamp"
    p.yaxis.axis_label = "Vehicle Count"
    p.legend.location = "top_left"
    
    return p

# --- Streamlit App ---
st.title("Kigali Traffic Congestion Dashboard")
st.write("Real-time congestion data and analysis for Kigali City.")

# Render the Bokeh chart in Streamlit
try:
    st.bokeh_chart(bokeh_congestion_chart(traffic_data), use_container_width=True)
except Exception as e:
    st.error(f"Error rendering Bokeh chart: {e}")



# --- SIMULATION RESULTS AND OPTIMIZATION ---
st.sidebar.header("Optimization Options")
route_filter = st.sidebar.selectbox("Select Route", routes_df['route_short_name'])
traffic_threshold = st.sidebar.slider("Set Congestion Threshold", 0, 100, 50)

# --- SUGGEST ROUTES BASED ON THRESHOLD ---
def suggest_alternate_routes(selected_route, threshold):
    if traffic_data['vehicle_count'].mean() > threshold:
        st.warning(f"High congestion detected on Route {selected_route}. Suggesting alternate routes...")
        suggestions = routes_df[routes_df['route_short_name'] != selected_route].head(3)
        st.table(suggestions)
    else:
        st.success(f"Traffic on Route {selected_route} is under control.")

suggest_alternate_routes(route_filter, traffic_threshold)

# --- DISPLAY CONGESTION HEATMAP USING GEOPANDAS ---
st.header("Kigali City Traffic Heatmap (Demo)")
gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
ax = gdf[gdf['continent'] == 'Africa'].plot(figsize=(10, 5), edgecolor='black')
st.pyplot(ax.figure)

# --- FOOTER ---
st.markdown("""
---
**Developed by Kigali City Transport Team**  
*For optimal travel routes and live traffic updates*
""")
