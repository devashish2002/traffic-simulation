import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import osmnx as ox
import networkx as nx
import json
#from bike_route_ import get_top_k_routes_with_penalties
from rtree import index
from shapely.geometry import LineString, Point
import folium
from folium import plugins
from streamlit_folium import st_folium

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
def get_citibike_data(manhattan=False):
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

    merged_df['available_bikes'] = merged_df['num_bikes_available']
    merged_df['available_docks'] = merged_df['num_docks_available']
    
    # Calculate Busyness Ratio
    # To avoid division by zero, handle cases where available_bikes is 0
    merged_df['busyness_ratio'] = merged_df.apply(
        lambda row: row['available_docks'] / row['available_bikes'] if row['available_bikes'] > 0 else None,
        axis=1
    )
    
    # Handle stations with available_bikes = 0 by setting a high ratio (e.g., 100)
    merged_df['busyness_ratio'] = merged_df['busyness_ratio'].fillna(100)

    if manhattan:
        # Manhattan boundaries (more precise)
        manhattan_lat_min = 40.6829
        manhattan_lat_max = 40.8820
        manhattan_lon_min = -74.0479
        manhattan_lon_max = -73.9067
                
        # Filter for Manhattan stations based on latitude and longitude
        merged_df = merged_df[
            (merged_df['lat'] >= manhattan_lat_min) & 
            (merged_df['lat'] <= manhattan_lat_max) & 
            (merged_df['lon'] >= manhattan_lon_min) & 
            (merged_df['lon'] <= manhattan_lon_max)
        ]
    
    return merged_df

# Function to fetch Traffic Speed data from HERE Maps API
@st.cache_data(ttl=600)  # Cache for 5 minutes as traffic data updates frequently
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


# def get_optimal_bike_route(G, citibike_data, traffic_data, start_station, end_station, traffic_idx, traffic_lines):
#     # Fetch street network from OpenStreetMap for NYC
#     #G = ox.graph_from_bbox(NYC_LAT_MAX, NYC_LAT_MIN, NYC_LON_MAX, NYC_LON_MIN, network_type='bike')

#     # Get lat/lon of start and end stations
#     start_lat = citibike_data.loc[citibike_data['name'] == start_station, 'lat'].values[0]
#     start_lon = citibike_data.loc[citibike_data['name'] == start_station, 'lon'].values[0]
#     end_lat = citibike_data.loc[citibike_data['name'] == end_station, 'lat'].values[0]
#     end_lon = citibike_data.loc[citibike_data['name'] == end_station, 'lon'].values[0]

#     # Find nearest nodes in the street network to the start and end stations
#     start_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
#     end_node = ox.distance.nearest_nodes(G, end_lon, end_lat)
    
#     return get_top_k_routes_with_penalties(G, start_node, end_node, traffic_data, traffic_idx, traffic_lines)

def calculate_busyness_penalty(G, route, citibike_data, station_idx):
    """
    Penalize the route based on the busyness of nearby CitiBike stations.

    Parameters:
    - route: List of nodes representing the route.
    - citibike_data: DataFrame containing CitiBike station information.
    - station_idx: R-tree spatial index for fast spatial queries of stations.

    Returns:
    - total_penalty: Total busyness penalty for the route.
    """
    total_penalty = 0
    for u, v in zip(route[:-1], route[1:]):  # Iterate through route edges
        # Get edge midpoint
        midpoint = ((G.nodes[u]['y'] + G.nodes[v]['y']) / 2, (G.nodes[u]['x'] + G.nodes[v]['x']) / 2)

        # Query nearby stations from the spatial index
        nearby_stations = list(station_idx.nearest((midpoint[1], midpoint[0], midpoint[1], midpoint[0]), 5))

        # Calculate penalty based on busyness of nearby stations
        for station_id in nearby_stations:
            station_busyness = citibike_data.loc[station_id, 'busyness_ratio']
            total_penalty += station_busyness  # Add busyness to penalty

    return total_penalty

def calculate_weighted_penalty(G, route, citibike_data, station_idx, weight_congestion):
    """
    Calculate a weighted penalty for a route that balances between distance and congestion.

    Parameters:
    - G: The street network graph.
    - route: A list of nodes representing the route.
    - citibike_data: DataFrame containing CitiBike station information.
    - station_idx: R-tree index for fast spatial queries to find nearby stations.
    - weight_congestion: Float (0 to 1), where 0 focuses entirely on shortest route and 1 focuses on least congestion.

    Returns:
    - total_penalty: The weighted penalty based on both route distance and congestion.
    """
    # Calculate the total distance of the route
    total_distance = sum(ox.distance.euclidean(G.nodes[route[i]]['y'], G.nodes[route[i]]['x'],
                                                        G.nodes[route[i+1]]['y'], G.nodes[route[i+1]]['x'])
                         for i in range(len(route) - 1))

    # Calculate the congestion penalty based on the busyness of nearby stations
    congestion_penalty = 0
    for node in route:
        node_point = Point(G.nodes[node]['x'], G.nodes[node]['y'])
        
        # Find nearby CitiBike stations using the spatial index
        nearby_stations = list(station_idx.nearest((node_point.x, node_point.y, node_point.x, node_point.y), 1))
        if nearby_stations:
            station_id = nearby_stations[0]
            #print(station_id)
            busyness_ratio = citibike_data.loc[station_id, 'busyness_ratio']
            congestion_penalty += busyness_ratio

    #print(congestion_penalty)
    # Normalize the penalties
    normalized_distance = total_distance / len(route)  # Average distance per segment
    normalized_congestion = congestion_penalty / len(route)  # Average congestion per segment

    # Compute weighted total penalty
    total_penalty = (1 - weight_congestion) * normalized_distance + weight_congestion * normalized_congestion

    #print(total_penalty)
    return total_penalty


def build_station_spatial_index(citibike_data):
    """
    Build a spatial index from the CitiBike data to efficiently find stations near route edges.

    Parameters:
    - citibike_data: DataFrame containing CitiBike station data.

    Returns:
    - idx: R-tree spatial index for fast spatial queries of stations.
    """
    idx = index.Index()
    for i, row in citibike_data.iterrows():
        # Create a point for each station's location
        point = Point(row['lon'], row['lat'])
        idx.insert(i, point.bounds)

    return idx

def get_optimal_bike_route(G, citibike_data, start_station, end_station, station_idx, weight_congestion):
    # Get lat/lon of start and end stations
    start_lat = citibike_data.loc[citibike_data['name'] == start_station, 'lat'].values[0]
    start_lon = citibike_data.loc[citibike_data['name'] == start_station, 'lon'].values[0]
    end_lat = citibike_data.loc[citibike_data['name'] == end_station, 'lat'].values[0]
    end_lon = citibike_data.loc[citibike_data['name'] == end_station, 'lon'].values[0]

    # Find nearest nodes in the street network to the start and end stations
    start_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
    end_node = ox.distance.nearest_nodes(G, end_lon, end_lat)

    # Get all possible routes between start and end nodes
    possible_routes = list(nx.all_shortest_paths(G, start_node, end_node))

    #print(possible_routes)
    # Penalize each route by the busyness of nearby stations
    best_route = None
    lowest_penalty = float('inf')
    
    for route in possible_routes[:5]:
        penalty = calculate_weighted_penalty(G, route, citibike_data, station_idx, weight_congestion)#calculate_busyness_penalty(G, route, citibike_data, station_idx)
        if penalty < lowest_penalty:
            lowest_penalty = penalty
            best_route = route

    # Extract the coordinates (lat, lon) for each node in the best route
    route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in best_route]
    #print(route_coords)

    return route_coords

# Function to calculate and overlay the optimal bike route
def generate_heatmap_with_route(G, citibike_data, start_station, end_station, station_idx, show_route=False, congestion_weight=0.5):
    """
    Generate a heatmap with an optional bike route overlay.
    
    Parameters:
    - G: The street network graph.
    - citibike_data: DataFrame containing CitiBike station data.
    - start_station: The starting station name.
    - end_station: The ending station name.
    - station_idx: R-tree index for spatial station lookups.
    - show_route: Boolean to determine if route should be displayed on the heatmap.
    - congestion_weight: Float (0 to 1) to balance between shortest route and least congested route.
    
    Returns:
    - folium_map: The heatmap with the optional route overlay.
    """
    # Get route coordinates from start to end stations
    route_coords = get_optimal_bike_route(G, citibike_data, start_station, end_station, station_idx, congestion_weight)
    
    # Calculate center point for map based on route coordinates
    center_lat = sum(lat for lat, lon in route_coords) / len(route_coords)
    center_lon = sum(lon for lat, lon in route_coords) / len(route_coords)
    
    # Initialize the map centered around the route midpoint
    folium_map = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    # Generate heatmap layer based on station busyness data
    heat_data = [
        [row['lat'], row['lon'], row['busyness_ratio']]
        for _, row in citibike_data.iterrows()
    ]
    plugins.HeatMap(heat_data, radius=10, max_zoom=13).add_to(folium_map)

    # If 'show_route' is True, calculate and add the optimal route
    if show_route:
        # Plot route on map
        folium.PolyLine(route_coords, color="blue", weight=5, opacity=0.7).add_to(folium_map)

        # Add start and end markers
        start_lat, start_lon = route_coords[0]
        end_lat, end_lon = route_coords[-1]
        folium.Marker([start_lat, start_lon], tooltip="Start", icon=folium.Icon(color="green")).add_to(folium_map)
        folium.Marker([end_lat, end_lon], tooltip="End", icon=folium.Icon(color="red")).add_to(folium_map)

    return folium_map


# Function to visualize the optimal bike route
def generate_route_map(route_coords, start_coords, end_coords):

    fig_route = go.Figure()

    latitudes, longitudes = zip(*route_coords)

    # Add the route line
    fig_route.add_trace(
        go.Scattermapbox(
            lat=latitudes,
            lon=longitudes,
            mode='lines',
            line=dict(color='blue', width=4),
            name="Optimal Bike Route"
        )
    )

    # Add a marker for the start station
    fig_route.add_trace(
        go.Scattermapbox(
            lat=[start_coords[0]],
            lon=[start_coords[1]],
            mode='markers',
            marker=dict(size=12, color='green', symbol='marker'),
            name="Start Station"
        )
    )

    # Add a marker for the end station
    fig_route.add_trace(
        go.Scattermapbox(
            lat=[end_coords[0]],
            lon=[end_coords[1]],
            mode='markers',
            marker=dict(size=12, color='red', symbol='marker'),
            name="End Station"
        )
    )

    # Update layout for mapbox
    fig_route.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=12,
        mapbox_center={"lat": latitudes[0], "lon": longitudes[0]},
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        title="Optimal Bike Route with Start and End Stations"
    )

    return fig_route


# Function to generate CitiBike Busyness Map
def generate_citibike_map(citibike_data):
    # Create columns for available bikes and docks
    # citibike_data['available_bikes'] = citibike_data['num_bikes_available']
    # citibike_data['available_docks'] = citibike_data['num_docks_available']
    
    # # Calculate Busyness Ratio
    # # To avoid division by zero, handle cases where available_bikes is 0
    # citibike_data['busyness_ratio'] = citibike_data.apply(
    #     lambda row: row['available_docks'] / row['available_bikes'] if row['available_bikes'] > 0 else None,
    #     axis=1
    # )
    
    # # Handle stations with available_bikes = 0 by setting a high ratio (e.g., 100)
    # citibike_data['busyness_ratio'] = citibike_data['busyness_ratio'].fillna(100)
    
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
citibike = get_citibike_data()
fig_citibike = generate_citibike_map(citibike)

# User selection for map display
map_option = st.radio(
    "Select the map to display:",
    ("CitiBike Heatmap", "Traffic Speed Map")
)

# Display the selected map
if map_option == "CitiBike Heatmap":
    # Display the CitiBike heatmap
    st.plotly_chart(fig_citibike, use_container_width=True)
    
    # Checkbox to toggle route display, exclusive to the CitiBike heatmap
    show_route = st.sidebar.checkbox("Show Optimal Route")
    
    if show_route:
        # Sidebar inputs for start and end stations, default to different values
        citibike = get_citibike_data(manhattan=True)
        station_names = citibike['name'].unique()
        start_station = st.sidebar.selectbox("Select Start Station", station_names, index=0)
        end_station_options = [name for name in station_names if name != start_station]
        end_station = st.sidebar.selectbox("Select End Station", end_station_options, index=1)
        congestion_weight = st.sidebar.slider("Balance Shortest vs. Least Congested Route (0:preference to shortest path, 1:preference to avoid congestion)", 0.0, 1.0, 0.5)

        # Generate heatmap with optional route overlay if valid station inputs are selected
        if start_station and end_station:
            G = ox.load_graphml('nyc_bike_network.graphml')
            station_idx = build_station_spatial_index(get_citibike_data())
            folium_map = generate_heatmap_with_route(G, citibike, start_station, end_station, station_idx, show_route, congestion_weight)
            st_folium(folium_map, width=700, height=500, key="static_route_map")

    # Add legend for CitiBike busyness ratio
    st.markdown("""
    **Busyness Ratio Interpretation:**
    - **Higher Ratio:** More docks available relative to bikes (less busy for pickups, potentially busy for drop-offs).
    - **Lower Ratio:** More bikes available relative to docks (busier for pickups, less busy for drop-offs).
    """)

elif map_option == "Traffic Speed Map":
    # Display the traffic speed map without optimal route options
    traffic = get_traffic_speed_data()
    st.plotly_chart(generate_traffic_map(traffic), use_container_width=True)
    
    # # Add legend for traffic speed interpretation
    # st.markdown("""
    # **Traffic Speed Interpretation:**
    # - **Red Lines (0-10 mph):** Heavily congested areas with very slow traffic.
    # - **Orange Lines (10-40 mph):** Moderately congested areas.
    # - **Yellow Lines (41-60 mph):** Light traffic.
    # - **Green Lines (61-80 mph):** Free-flowing traffic.
    # - **Blue Lines (81-100 mph):** Very fast-moving traffic, typically highways or expressways.
    # """)