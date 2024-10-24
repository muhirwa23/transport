import streamlit as st
import pandas as pd
from io import StringIO
import altair as alt
import plotly.express as px
import bokeh.plotting as bk
import geopandas as gpd
from bokeh.models import HoverTool
import numpy as np
import requests

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

@st.cache_data
def load_route_data():
    return pd.read_csv(StringIO(route_data))

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

# --- DUMMY SIMULATION DATA ---
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
def bokeh_congestion_chart(data):
    p = bk.figure(
        x_axis_type="datetime", title="Congestion Analysis",
        plot_width=800, plot_height=400
    )
    p.line(data['timestamp'], data['vehicle_count'], line_width=2)
    hover = HoverTool(tooltips=[("Time", "@x{%F %T}"), ("Vehicles", "@y")], formatters={"@x": "datetime"})
    p.add_tools(hover)
    return p

bk.show(bokeh_congestion_chart(traffic_data))

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

# --- DISPLAY CONGESTION HEATMAP USING GEOPANDAS (DEMO) ---
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
