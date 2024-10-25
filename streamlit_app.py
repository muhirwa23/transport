import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import plotly.express as px
from datetime import datetime, timedelta
from io import StringIO

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Kigali Traffic Optimization", layout="wide")

# --- LOAD FULL ROUTE DATA ---
@st.cache_data
def load_route_data():
    """Load the full route data."""
    data = """route_id,agency_id,route_short_name,route_long_name,route_type,route_desc
    101,1,101,KBS - Zone I - 101,3,Remera Taxi Park-Sonatubes-Rwandex-CBD
    102,1,102,KBS - Zone I - 102,3,Kabuga-Mulindi-Remera-Sonatubes-Rwandex Nyabugogo Taxi Park
    103,1,103,KBS - Zone I - 103,3,Rubilizi-Kabeza-Remera-Sonatubes-Rwandex-CBD
    104,1,104,KBS - Zone I - 104,3,Kibaya-Kanombe MH-Airport-Remera-Sonatubes-Rwandex-CBD
    105,1,105,KBS - Zone I - 105,3,Remera Taxi Park-Chez Lando-Kacyiru-Nyabugogo Taxi Park
    106,1,106,KBS - Zone I - 106,3,Remera Taxi Park-15-Ndera-Musave
    107,1,107,KBS - Zone I - 107,3,Remera Taxi Park-Mulindi-Masaka
    108,1,108,KBS - Zone I - 108,3,Remera Taxi Park-Sonatubes-Nyanza Taxi Park
    109,1,109,KBS - Zone I - 109,3,Remera Taxi Park-Sonatubes-Rwandex-Gikondo-Bwerankoli
    111,1,111,KBS - Zone I - 111,3,Kabuga-Mulindi-Remera Taxi Park
    112,1,112,KBS - Zone I - 112,3,Remera Taxi Park-Sonatubes-Rwandex-Nyabugogo Taxi Park
    113,1,113,KBS - Zone I - 113,3,Busanza-Rubilizi-Kabeza-Remera Taxi Park
    114,1,114,KBS - Zone I - 114,3,Kibaya-Kanombe MH-Airport-Remera Taxi Park
    115,1,115,KBS - Zone I - 115,3,Remera Taxi Park-Airport-Busanza
    301,3,301,RFTC - Zone III and IV - 301,3,Kinyinya-Nyarutarama-RDB-Kimihurura-Down Town Taxi Park
    302,3,302,RFTC - Zone III and IV - 302,3,Kimironko-Stadium-Chez Lando-Kimihurura-CBD
    303,3,303,RFTC - Zone III and IV - 303,3,Batsinda-Kagugu-Gakiriro-Kinamba-Down Town Taxi Park
    304,3,304,RFTC - Zone III and IV - 304,3,Kacyiru-Kimihurura-Down Town Taxi Park
    305,3,305,RFTC - Zone III and IV - 305,3,Kimironko Taxi Park-Stadium-Chez Lando-Kacyiru-Nyabugogo Taxi Park
    306,3,306,RFTC - Zone III and IV - 306,3,Kimironko Taxi Park-Zindiro-Masizi-Birembo
    308,3,308,RFTC - Zone III and IV - 308,3,AZAM Roundabout-Chez Lando-Kimihurura-CBD
    309,3,309,RFTC - Zone III and IV - 309,3,Kimironko Taxi Park-Kibagabaga-Kinyinya
    310,3,310,RFTC - Zone III and IV - 310,3,Batsinda-Kagugu-Gakiriro-Kinamba-Nyabugogo Taxi Park
    311,3,311,RFTC - Zone III and IV - 311,3,Kagugu-Bel Etoile-ULK-Kinamba-Nyabugogo Taxi Park
    313,3,313,RFTC - Zone III and IV - 313,3,Kagugu-Bel Etoile-ULK-Kinamba-Down Town Taxi Park
    314,3,314,RFTC - Zone III and IV - 314,3,Nyabugogo Taxi Park-Kinamba-UTEXRWA-Kibagabaga-Kimironko
    315,3,315,RFTC - Zone III and IV - 315,3,Kinyinya-UTEXRWA-Kinamba-Nyabugogo Taxi Park
    316,3,316,RFTC - Zone III and IV - 316,3,AZAM Roundabout-Kimironko Taxi Park
    317,3,317,RFTC - Zone III and IV - 317,3,Kinyinya-UTEXRWA-Kinamba-Down Town Taxi Park
    318,3,318,RFTC - Zone III and IV - 318,3,Batsinda-Kagugu-Kibagabaga-Kimironko Taxi Park
    321,3,321,RFTC - Zone III and IV - 321,3,Nyabugogo Taxi Park-Batsinda-Gasanze
    322,3,322,RFTC - Zone III and IV - 322,3,Kimironko Taxi Park-Mulindi-Masaka
    325,3,325,RFTC - Zone III and IV - 325,3,Kabuga-Kigali Parent School-Kimironko Taxi Park
    401,3,401,RFTC - Zone III and IV - 401,3,Ryanyuma-Rafiki-Camp Kigali-CBD
    402,3,402,RFTC - Zone III and IV - 402,3,Ryanyuma-Kimisagara-Nyabugogo-Down Town Taxi Park
    403,3,403,RFTC - Zone III and IV - 403,3,Nyacyonga-Karuruma-Muhima-Down Town Taxi Park
    404,3,404,RFTC - Zone III and IV - 404,3,Bishenyi-Ruyenzi-Giticyinyoni-Nyabugogo
    406,3,406,RFTC - Zone III and IV - 406,3,Rwarutabura-Mageragere
    411,3,411,RFTC - Zone III and IV - 411,3,Nyabugogo Taxi Park-Giticyinyoni-Nzove-Rutonde
    412,3,412,RFTC - Zone III and IV - 412,3,Nyabugogo Taxi Park-Giticyinyoni
    414,3,414,RFTC - Zone III and IV - 414,3,Nyabugogo Taxi Park-Ruliba-Karama Complex School
    201,2,201,ROYAL - Zone II - 201,3,St. Joseph-Kicukiro Centre-Sonatubes-Rwandex-CBD
    202,2,202,ROYAL - Zone II - 202,3,Nyanza Taxi Park-Gatenga-Down Town Taxi Park
    211,2,211,ROYAL - Zone II - 211,3,Nyanza Bus Park-Kicukiro Centre-Chez Lando-Kacyiru
    212,2,212,ROYAL - Zone II - 212,3,St. Joseph-Kicukiro Centre-Sonatubes-Rwandex-Nyabugogo Taxi Park
    """
    return pd.read_csv(StringIO(data))

routes_df = load_route_data()

# --- Initialize Traffic Data ---
if 'traffic_data' not in st.session_state:
    def generate_initial_data():
        np.random.seed(42)
        data = []
        for _ in range(100):
            data.append(generate_live_data())
        return pd.DataFrame(data)

    st.session_state.traffic_data = generate_initial_data()

# --- LIVE DATA GENERATION FUNCTION ---
def generate_live_data():
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

# --- Sidebar Filters ---
st.sidebar.header("Control Panel")
selected_routes = st.sidebar.multiselect("Select Routes", routes_df['route_short_name'], default=routes_df['route_short_name'].tolist())
min_vehicle_count = st.sidebar.slider("Minimum Vehicle Count", 0, 100, 20)
max_travel_time = st.sidebar.slider("Maximum Travel Time (minutes)", 5, 30, 20)

# --- Filter Traffic Data ---
filtered_data = st.session_state.traffic_data[
    (st.session_state.traffic_data['route'].isin(selected_routes)) & 
    (st.session_state.traffic_data['vehicle_count'] >= min_vehicle_count) & 
    (st.session_state.traffic_data['travel_time'] <= max_travel_time)
]

# --- Correlation Heatmap ---
st.subheader("Traffic Data Correlation Heatmap")
corr = filtered_data[['vehicle_count', 'travel_time']].corr()
corr_fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
st.plotly_chart(corr_fig, use_container_width=True)

# --- Route-Specific Statistics ---
st.subheader("Route-Specific Statistics")
route_stats = filtered_data.groupby('route').agg({
    'vehicle_count': 'mean',
    'travel_time': 'mean',
}).rename(columns={'vehicle_count': 'Avg Vehicle Count', 'travel_time': 'Avg Travel Time'}).reset_index()

route_stats_fig = px.bar(route_stats, x='route', y=['Avg Vehicle Count', 'Avg Travel Time'],
                         barmode='group', title="Average Vehicle Count & Travel Time by Route")
st.plotly_chart(route_stats_fig, use_container_width=True)

# --- Route Map Visualization ---
st.subheader("Kigali Route Map")
map_center = [-1.9579, 30.0594]
m = folium.Map(location=map_center, zoom_start=12)

# Add MarkerCluster for route points
marker_cluster = MarkerCluster().add_to(m)

# Placeholder route coordinates, to be replaced with actual lat/lon for each route
route_coords = {route: [map_center[0] + np.random.uniform(-0.02, 0.02), map_center[1] + np.random.uniform(-0.02, 0.02)] 
                for route in routes_df['route_short_name']}

for route_id, coord in route_coords.items():
    folium.Marker(
        location=coord,
        popup=f"Route: {route_id}",
        tooltip=f"Route: {route_id}"
    ).add_to(marker_cluster)

# Display the map in Streamlit
st_folium(m, width=700, height=500)

# --- Display Traffic Data ---
st.subheader("Real-time Traffic Data")
st.write(filtered_data)
