# Import necessary libraries
import pandas as pd
import streamlit as st
import plotly.express as px
import altair as alt
from bokeh.plotting import figure
from pygal.style import Style
import pygal
import geopandas as gpd
import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Set Page Configuration for Streamlit
st.set_page_config(
    page_title="Kigali Traffic Optimization Dashboard",
    layout="wide",
    page_icon="🚦",
    initial_sidebar_state="expanded",
)

# Styling the Dashboard with CSS
st.markdown("""
    <style>
        .main { background-color: #f0f2f6; }
        h1, h2, h3 { color: #013220; }
        .sidebar .sidebar-content { background-color: #003f5c; color: white; }
        .stButton button { background-color: #28a745; color: white; border-radius: 10px; }
        iframe { border-radius: 10px; border: 2px solid #013220; }
    </style>
""", unsafe_allow_html=True)

# Sidebar: Route Selection
st.sidebar.title("📍 Select Routes")
st.sidebar.markdown("Filter the routes you want to monitor:")

@st.cache_data
def load_route_data():
    """Load Kigali city route data."""
    data = """
    route_id,agency_id,route_short_name,route_long_name,route_type,route_desc
    101,1,101,KBS - Zone I - 101,3,Remera Taxi Park-Sonatubes-Rwandex-CBD
    102,1,102,KBS - Zone I - 102,3,Kabuga-Mulindi-Remera-Sonatubes-Rwandex Nyabugogo Taxi Park
    201,2,201,ROYAL - Zone II - 201,3,St. Joseph – Kikukiro Centre de Santé – Sonatubes – Rwandex - CBD
    301,3,301,RFTC - Zone III and IV - 301,3,Kinyinya - Nyarutarama - RDB - Kimihurura - Down Town Taxi Park
    401,3,401,RFTC - Zone III and IV - 401,3,Nyamirambo (Ryanyuma) - Rafiki - Camp Kigali - CBD
    """
    return pd.read_csv(pd.compat.StringIO(data))

# Load and display route data
routes_df = load_route_data()
selected_routes = st.sidebar.multiselect(
    "Choose Routes to Monitor:",
    options=routes_df['route_long_name'],
    default=routes_df['route_long_name']
)

# Header
st.title("🚦 Kigali Traffic Optimization Dashboard")
st.subheader("Monitor traffic, predict congestion, and optimize routes in real-time")

# Altair: Real-Time Congestion Chart
st.markdown("### 📈 Real-Time Congestion Monitoring (Altair)")

def generate_altair_chart():
    congestion_data = pd.DataFrame({
        'time': pd.date_range(start='2024-10-24', periods=100, freq='T'),
        'congestion_level': np.random.randint(0, 3, 100)
    })
    chart = alt.Chart(congestion_data).mark_line().encode(
        x='time:T',
        y='congestion_level:Q'
    ).properties(
        title='Congestion Levels Over Time',
        width=800,
        height=400
    )
    st.altair_chart(chart)

generate_altair_chart()

# Bokeh: Traffic Forecast Visualization
st.markdown("### 📊 Traffic Forecast Visualization (Bokeh)")

def generate_bokeh_chart():
    p = figure(
        title="Traffic Congestion Forecast",
        x_axis_label='Time (in seconds)',
        y_axis_label='Congestion Level',
        plot_width=800,
        plot_height=400
    )
    congestion_levels = np.random.randint(0, 3, 50)
    p.line(range(50), congestion_levels, legend_label='Congestion', line_width=2)
    st.bokeh_chart(p)

generate_bokeh_chart()

# Pygal: Route Speeds Chart
st.markdown("### 🚍 Route Speeds (Pygal)")

def generate_pygal_chart():
    bar_chart = pygal.Bar(style=Style(colors=('#3498db', '#e74c3c')))
    bar_chart.title = 'Average Speeds on Selected Routes'
    for route in selected_routes:
        avg_speed = np.random.randint(10, 50)
        bar_chart.add(route, avg_speed)
    st.write(bar_chart.render_swf())

generate_pygal_chart()

# Geoplotlib: Traffic Intensity Map (Static Example)
st.markdown("### 🌍 Traffic Intensity Map (Geoplotlib)")

def load_geodata():
    """Load Kigali geospatial data."""
    return gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

geodata = load_geodata()
st.map(geodata)

# Plotly: Live Traffic Data (Line Chart)
st.markdown("### 📊 Live Traffic Data Monitoring (Plotly)")

fig = px.line(
    x=pd.date_range(start='2024-10-24', periods=50, freq='T'),
    y=np.random.randint(0, 3, 50),
    labels={'x': 'Time', 'y': 'Congestion Level'},
    title='Real-Time Congestion Monitoring'
)
st.plotly_chart(fig, use_container_width=True)

# Waze Live Map Integration
st.markdown("### 📍 Live Traffic Map (Waze)")
st.markdown("""
<iframe src="https://embed.waze.com/iframe?zoom=13&lat=-1.9705786&lon=30.1044284&ct=livemap" 
        width="800" height="600" allowfullscreen></iframe>
""", unsafe_allow_html=True)

# LSTM Model for Traffic Forecasting
st.markdown("### 🧠 Traffic Forecast with LSTM")

def prepare_data(data, n_steps=3):
    """Prepare data for LSTM input."""
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

traffic_data = np.random.randint(0, 3, 100)
X, y = prepare_data(traffic_data)

model = Sequential([
    LSTM(50, activation='relu', input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=5, verbose=1)

predictions = model.predict(X)
st.line_chart(predictions.flatten())

# Footer
st.markdown("""
<hr>
<p style='text-align: center;'>
Designed for <b>Kigali City</b> | Powered by <b>Machine Learning</b> & <b>Real-Time Traffic Data</b>
</p>
""", unsafe_allow_html=True)
