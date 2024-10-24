import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime

# Simulate Kigali public transport routes data
def get_kigali_routes():
    routes = pd.DataFrame({
        'route_id': ['Route_1', 'Route_2', 'Route_3'],
        'origin': ['Kigali Convention Centre', 'Kimironko Market', 'Nyabugogo'],
        'destination': ['Nyabugogo', 'Downtown', 'Remera'],
        'stops': [
            ['KCC', 'Kacyiru', 'Nyabugogo'],
            ['Kimironko', 'Remera', 'Downtown'],
            ['Nyabugogo', 'Gisozi', 'Remera']
        ],
        'total_distance_km': [10, 12, 8]
    })
    return routes

# Simulate traffic data for each route
def simulate_traffic_for_route(route_id):
    traffic_conditions = np.random.choice(['Heavy Traffic', 'Moderate Traffic', 'Light Traffic'], p=[0.3, 0.5, 0.2])
    delay = np.random.randint(5, 30) if traffic_conditions == 'Heavy Traffic' else np.random.randint(0, 10)
    traffic_info = {
        'route_id': route_id,
        'traffic_condition': traffic_conditions,
        'estimated_delay_minutes': delay
    }
    return traffic_info

# Suggest alternative routes based on traffic conditions
def suggest_alternative_route(routes, traffic_data):
    heavy_traffic_routes = [r['route_id'] for r in traffic_data if r['traffic_condition'] == 'Heavy Traffic']
    
    if heavy_traffic_routes:
        st.write("Heavy Traffic detected on the following routes:")
        st.write(heavy_traffic_routes)
        st.write("Suggesting alternative routes...")
        
        # Suggest other routes that do not have heavy traffic
        alternative_routes = routes[~routes['route_id'].isin(heavy_traffic_routes)]
        st.write(alternative_routes[['route_id', 'origin', 'destination']])
    else:
        st.write("No heavy traffic detected. All routes are running smoothly.")

# Main App Layout
st.set_page_config(layout="wide", page_title="Kigali Public Transport Optimization")

st.title("Kigali Public Transport Routes and Traffic")

# Load and display the routes data
routes = get_kigali_routes()
st.write("### Public Transport Routes in Kigali")
st.dataframe(routes)

# Simulate traffic for each route
st.write("### Simulated Traffic Data")
traffic_data = [simulate_traffic_for_route(route_id) for route_id in routes['route_id']]
st.json(traffic_data)

# Suggest alternative routes based on traffic conditions
suggest_alternative_route(routes, traffic_data)
