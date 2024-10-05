import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="CitiBike Busyness Heatmap", layout="wide")

st.title("🗽 CitiBike Busyness Heatmap - New York City")

@st.cache(ttl=60)  # Cache the data for 60 seconds to reduce API calls
def get_citibike_data():
    # Endpoints
    station_info_url = "https://gbfs.citibikenyc.com/gbfs/en/station_information.json"
    station_status_url = "https://gbfs.citibikenyc.com/gbfs/en/station_status.json"
    
    # Fetch station information
    info_response = requests.get(station_info_url)
    info_data = info_response.json()['data']['stations']
    info_df = pd.DataFrame(info_data)
    
    # Fetch station status
    status_response = requests.get(station_status_url)
    status_data = status_response.json()['data']['stations']
    status_df = pd.DataFrame(status_data)
    
    # Merge on station_id
    merged_df = pd.merge(info_df, status_df, on='station_id')
    
    return merged_df

# Fetch data
data = get_citibike_data()

# Create columns for available bikes and docks
data['available_bikes'] = data['num_bikes_available']
data['available_docks'] = data['num_docks_available']

# Calculate Busyness Ratio
# To avoid division by zero, we'll handle cases where available_bikes is 0
data['busyness_ratio'] = data.apply(
    lambda row: row['available_docks'] / row['available_bikes'] if row['available_bikes'] > 0 else None,
    axis=1
)

# Handle stations with available_bikes = 0 by setting a high ratio (e.g., 100) to indicate maximum busyness for drop-offs
data['busyness_ratio'] = data['busyness_ratio'].fillna(100)

# Optionally, cap the busyness_ratio to a maximum value for better color scaling
data['busyness_ratio'] = data['busyness_ratio'].clip(upper=10)

# Create hover text including busyness ratio
data['hover_text'] = (
    data['name'] + "<br>" +
    "Available Bikes: " + data['available_bikes'].astype(str) + "<br>" +
    "Available Docks: " + data['available_docks'].astype(str) + "<br>" +
    "Busyness Ratio (Docks/Bikes): " + data['busyness_ratio'].round(2).astype(str)
)

# Create the map with Busyness Ratio as color and size
fig = px.scatter_mapbox(
    data,
    lat="lat",
    lon="lon",
    hover_name="name",
    hover_data={
        "available_bikes": True,
        "available_docks": True,
        "busyness_ratio": False,
        "lat": False,
        "lon": False,
        "station_id": False
    },
    color="busyness_ratio",
    size="busyness_ratio",
    color_continuous_scale="viridis",
    size_max=20,
    zoom=11,
    height=800,
    title="City congestion map based on citibike station activity"
)

fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
fig.update_coloraxes(colorbar_title="Busy index")

# Display the map
st.plotly_chart(fig, use_container_width=True)

# Optional: Add a legend explanation
st.markdown("""
**Busyness Ratio Interpretation:**
- **Higher Ratio:** More docks available relative to bikes (less busy for pickups, potentially busy for drop-offs).
- **Lower Ratio:** More bikes available relative to docks (busier for pickups, less busy for drop-offs).
""")
