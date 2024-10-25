import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import plotly.express as px
from datetime import datetime
from io import StringIO

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Kigali Traffic Optimization", layout="wide")

# --- LOAD FULL ROUTE DATA ---
@st.cache_data
def load_route_data():
    """Load the full route data."""
    data = """route_id,agency_id,route_short_name,route_long_name,route_type,route_desc
    101,1,101,KBS - Zone I - 101,3,Remera Taxi Park-Sonatubes-Rwandex-CBD
    ...  # Trimmed for brevity
    412,3,412,RFTC - Zone III and IV - 412,3,Nyabugogo Taxi Park-Giticyinyoni
    """
    return pd.read_csv(StringIO(data))

routes_df = load_route_data()

# --- LIVE DATA GENERATION FUNCTION ---
def generate_live_data():
    """Generates one row of live traffic data with consistent keys."""
    try:
        vehicle_count = np.random.randint(20, 100)
        travel_time = np.random.uniform(5, 25)
        route = np.random.choice(routes_df['route_short_name'])
        congestion = np.random.choice(["Low", "Moderate", "High"])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {
            "timestamp": timestamp,
            "vehicle_count": vehicle_count,
            "travel_time": travel_time,
            "route": route,
            "congestion": congestion,
        }
    except Exception as e:
        st.error(f"Error generating live data: {str(e)}")
        return None

# --- Initialize Traffic Data in Session State ---
if 'traffic_data' not in st.session_state:
    def generate_initial_data():
        np.random.seed(42)
        data = []
        for _ in range(100):
            live_data = generate_live_data()  # Generate one row of live data
            if live_data:  # Check if live_data is not None
                data.append(live_data)
        # Convert list of dictionaries to DataFrame with consistent columns
        return pd.DataFrame(data, columns=["timestamp", "vehicle_count", "travel_time", "route", "congestion"])

    st.session_state.traffic_data = generate_initial_data()

# Now you can proceed with the rest of the code here...
