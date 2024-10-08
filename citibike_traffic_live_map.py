import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import json

# Set page configuration
st.set_page_config(page_title="CitiBike & Traffic Speed Maps", layout="wide")

st.title("🗽 CitiBike Activity & Traffic Speed Maps - New York City")

# Define NYC geographic bounds
NYC_LAT_MIN = 40.4774
NYC_LAT_MAX = 40.9176
NYC_LON_MIN = -74.2591
NYC_LON_MAX = -73.7004

# HERE Maps API Key
apiKey = 'zLeOISowE3l_vz1aIVhm4wSC9U-knJ_9Cw5ZkxTDXCw'

# Function to fetch CitiBike data
@st.cache_data(ttl=60)
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

# Function to fetch Traffic Speed data from HERE Maps API
@st.cache_data(ttl=300)  # Cache for 5 minutes as traffic data updates frequently
def get_traffic_speed_data():
    streetSection = []
    sectionLen = []
    sectionSpeed = []
    sectionJam = []
    linestrings = []
    time_list = []
    
    # Traffic API endpoint link
    url = f'https://data.traffic.hereapi.com/v7/flow?in=bbox:{NYC_LON_MIN},{NYC_LAT_MIN},{NYC_LON_MAX},{NYC_LAT_MAX}&locationReferencing=shape&apiKey={apiKey}'
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status
        data = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching traffic data: {e}")
        return pd.DataFrame()
    
    # Initialize lists to store traffic data
    traffic_records = []
    
    # Iterate through each result
    for result in data.get('results', []):
        current_flow = result.get('currentFlow', {})
        speed = current_flow.get('jamFactor')
        if speed is None:
            continue  # Skip if speed data is missing
        
        location = result.get('location', {})
        shape = location.get('shape', {})
        links = shape.get('links', [])
        
        for link in links:
            points = link.get('points', [])
            if not points:
                continue  # Skip if no points
            
            # Extract latitudes and longitudes
            latitudes = [point.get('lat') for point in points if 'lat' in point]
            longitudes = [point.get('lng') for point in points if 'lng' in point]
            
            # Check if all points are within NYC bounds
            if (min(latitudes) >= NYC_LAT_MIN and max(latitudes) <= NYC_LAT_MAX and
                min(longitudes) >= NYC_LON_MIN and max(longitudes) <= NYC_LON_MAX):
                
                traffic_records.append({
                    'jamFactor': speed,
                    'latitudes': latitudes,
                    'longitudes': longitudes
                })
    
    # Convert to DataFrame
    traffic_df = pd.DataFrame(traffic_records)
    
    return traffic_df

# Function to generate CitiBike Busyness Map
def generate_citibike_map(citibike_data):
    # Create columns for available bikes and docks
    citibike_data['available_bikes'] = citibike_data['num_bikes_available']
    citibike_data['available_docks'] = citibike_data['num_docks_available']
    
    # Calculate Busyness Ratio
    # To avoid division by zero, handle cases where available_bikes is 0
    citibike_data['busyness_ratio'] = citibike_data.apply(
        lambda row: row['available_docks'] / row['available_bikes'] if row['available_bikes'] > 0 else None,
        axis=1
    )
    
    # Handle stations with available_bikes = 0 by setting a high ratio (e.g., 100)
    citibike_data['busyness_ratio'] = citibike_data['busyness_ratio'].fillna(100)
    
    # Cap the busyness_ratio to a maximum value for better color scaling
    citibike_data['busyness_ratio'] = citibike_data['busyness_ratio'].clip(upper=10)
    
    # Create hover text including busyness ratio
    citibike_data['hover_text'] = (
        citibike_data['name'] + "<br>" +
        "Available Bikes: " + citibike_data['available_bikes'].astype(str) + "<br>" +
        "Available Docks: " + citibike_data['available_docks'].astype(str) + "<br>" +
        "Busyness Ratio (Docks/Bikes): " + citibike_data['busyness_ratio'].round(2).astype(str)
    )
    
    # Initialize Plotly map with CitiBike data
    fig_citibike = px.scatter_mapbox(
        citibike_data,
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
        color_continuous_scale="Viridis",
        size_max=15,
        zoom=11,
        height=800,
        width=1000,
        title="CitiBike Stations - Busyness Heatmap (Docks/Bikes Ratio)"
    )
    
    # Update layout for mapbox
    fig_citibike.update_layout(mapbox_style="open-street-map")
    fig_citibike.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
    fig_citibike.update_layout(coloraxis_colorbar=dict(title="Busyness Ratio (Docks/Bikes)"))
    
    return fig_citibike

# Function to generate Traffic Speed Map
def generate_traffic_map(traffic_data):
    # Define speed bins and corresponding colors
    speed_bins = [1, 2, 3, 4]#[0, 10, 15, 20, 30, 40, 100]
    jam_labels = ['Min. congested', 'Mid congested', 'Max. congested']
    speed_colors = ['green', 'orange', 'red']
    
    # Assign speed categories
    traffic_data['jam_category'] = pd.cut(traffic_data['jamFactor'], bins=speed_bins, labels=jam_labels, include_lowest=True)
    
    # Create a mapping from speed category to color
    speed_color_map = dict(zip(jam_labels, speed_colors))
    
    # Initialize Plotly map
    fig_traffic = go.Figure()
    
    # Iterate through each speed category and add a separate trace
    for category in jam_labels:
        category_data = traffic_data[traffic_data['jam_category'] == category]
        if category_data.empty:
            continue
        
        # Prepare lists for latitudes and longitudes with None separators
        lat_lines = []
        lon_lines = []
        for _, row in category_data.iterrows():
            lat_lines.extend(row['latitudes'] + [None])  # None separates different lines
            lon_lines.extend(row['longitudes'] + [None])
        
        # Add Scattermapbox trace for this speed category
        fig_traffic.add_trace(
            go.Scattermapbox(
                lat=lat_lines,
                lon=lon_lines,
                mode='lines',
                line=dict(color=speed_color_map[category], width=2),
                showlegend=True,
                name=f'Traffic Speed: {category}'

            )
        )
    
    # Update layout for mapbox
    fig_traffic.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=11,
        mapbox_center={"lat": 40.7128, "lon": -74.0060},  # Centered on NYC
        margin={"r":0,"t":50,"l":0,"b":0},
        title="Traffic Speed Map - New York City"
    )
    fig_traffic.update_layout(
    height=800,  # Adjust the height as needed
    width=1000   # Adjust the width as needed
    )
    
    return fig_traffic

# Fetch data
# citibike_data = get_citibike_data()
# traffic_data = get_traffic_speed_data()

# Generate both maps
fig_citibike = generate_citibike_map(get_citibike_data())
# fig_traffic = generate_traffic_map(get_traffic_speed_data())

# User selection for map display
map_option = st.radio(
    "Select the map to display:",
    ("CitiBike Heatmap", "Traffic Speed Map")
)


# Display the selected map
if map_option == "CitiBike Heatmap":
    st.plotly_chart(fig_citibike, use_container_width=True)
elif map_option == "Traffic Speed Map":
    st.plotly_chart(generate_traffic_map(get_traffic_speed_data()), use_container_width=True)

# Optional: Add a legend explanation based on the selected map
if map_option == "CitiBike Heatmap":
    st.markdown("""
    **Busyness Ratio Interpretation:**
    - **Higher Ratio:** More docks available relative to bikes (less busy for pickups, potentially busy for drop-offs).
    - **Lower Ratio:** More bikes available relative to docks (busier for pickups, less busy for drop-offs).
    """)
# elif map_option == "Traffic Speed Map":
#     st.markdown("""
#     **Traffic Speed Interpretation:**
#     - **Red Lines (0-10 mph):** Heavily congested areas with very slow traffic.
#     - **Orange Lines (10-40 mph):** Moderately congested areas.
#     - **Yellow Lines (41-60 mph):** Light traffic.
#     - **Green Lines (61-80 mph):** Free-flowing traffic.
#     - **Blue Lines (81-100 mph):** Very fast-moving traffic, typically highways or expressways.
#     """)
