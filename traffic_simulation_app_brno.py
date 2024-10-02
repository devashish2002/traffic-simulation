import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

# ---------------------------
# Sample Data Generation
# ---------------------------
@st.cache(allow_output_mutation=True)
def generate_sample_data(n=1000):
    """
    Generate a sample dataset mimicking TLC data with random pickup and dropoff locations,
    distance, and travel time within Brno.
    """
    np.random.seed(42)
    
    # Brno geographical boundaries
    # Approximate latitude and longitude ranges for Brno
    pickup_lat = np.random.uniform(49.14, 49.25, size=n)  # Brno latitude range
    pickup_lon = np.random.uniform(16.53, 16.72, size=n)  # Brno longitude range
    dropoff_lat = np.random.uniform(49.14, 49.25, size=n)
    dropoff_lon = np.random.uniform(16.53, 16.72, size=n)
    
    # Generate random distances (in kilometers)
    # Assuming maximum trip distance within Brno is ~15 km
    distances = np.random.uniform(1.0, 15.0, size=n)
    
    # Generate random trip durations (in minutes)
    # time = distance / average speed (30 km/h) * 60 * random factor
    travel_time = distances / 30 * 60 * np.random.uniform(0.8, 1.2, size=n)
    
    # Create a DataFrame
    data = pd.DataFrame({
        'pickup_lat': pickup_lat,
        'pickup_lon': pickup_lon,
        'dropoff_lat': dropoff_lat,
        'dropoff_lon': dropoff_lon,
        'distance_km': distances,
        'travel_time_min': travel_time
    })
    
    return data

# ---------------------------
# Road Definitions
# ---------------------------
def get_sample_roads():
    """
    Define sample roads in Brno with their names and geographic boundaries.
    For simplicity, roads are defined as rectangles with lat/lon ranges.
    """
    roads = [
        {
            'name': 'Šilingrovo nam.',
            'lat_range': (49.20, 49.21),
            'lon_range': (16.59, 16.62)
        },
        {
            'name': 'Masarykova Street',
            'lat_range': (49.19, 49.20),
            'lon_range': (16.60, 16.63)
        },
        {
            'name': 'Královo Pole',
            'lat_range': (49.21, 49.23),
            'lon_range': (16.64, 16.68)
        },
        {
            'name': 'Jílová Street',
            'lat_range': (49.18, 49.19),
            'lon_range': (16.55, 16.58)
        }
    ]
    return roads

# ---------------------------
# Simulation Function
# ---------------------------
def simulate_road_closure(data, road, delay_factor=1.5):
    """
    Simulate the effect of a road closure by adding delays to trips passing through
    the selected road's area.
    """
    lat_min, lat_max = road['lat_range']
    lon_min, lon_max = road['lon_range']
    
    # Identify trips that pass through the affected road
    affected_trips = (
        ((data['pickup_lat'] >= lat_min) & (data['pickup_lat'] <= lat_max) &
         (data['pickup_lon'] >= lon_min) & (data['pickup_lon'] <= lon_max)) |
        ((data['dropoff_lat'] >= lat_min) & (data['dropoff_lat'] <= lat_max) &
         (data['dropoff_lon'] >= lon_min) & (data['dropoff_lon'] <= lon_max))
    )
    
    # Create a copy of data to avoid modifying original
    updated_data = data.copy()
    
    # Apply delay factor to the travel time of affected trips
    updated_data.loc[affected_trips, 'travel_time_min'] *= delay_factor
    
    return updated_data, affected_trips

# ---------------------------
# Visualization Functions
# ---------------------------
def plot_travel_time_distribution(original_data, updated_data, affected_trips):
    """
    Plot the distribution of travel times before and after simulating the road closure.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot travel times for unaffected trips
    unaffected_times = original_data[~affected_trips]['travel_time_min']
    plt.hist(unaffected_times, bins=30, alpha=0.5, label='Unaffected Trips', color='green')
    
    # Plot travel times for affected trips after delay
    affected_times = updated_data[affected_trips]['travel_time_min']
    plt.hist(affected_times, bins=30, alpha=0.5, label='Affected Trips (Delayed)', color='red')
    
    plt.xlabel('Travel Time (minutes)')
    plt.ylabel('Frequency')
    plt.title('Travel Time Distribution Before and After Road Closure in Brno')
    plt.legend()
    st.pyplot(plt)

def create_folium_map(roads, selected_road=None, affected_trips=None, data=None):
    """
    Create a Folium map displaying roads and optionally affected trips.
    """
    # Initialize Folium map centered around Brno
    m = folium.Map(location=[49.1951, 16.6068], zoom_start=13)
    
    # Add roads to the map
    for road in roads:
        lat_min, lat_max = road['lat_range']
        lon_min, lon_max = road['lon_range']
        folium.Rectangle(
            bounds=[[lat_min, lon_min], [lat_max, lon_max]],
            color='blue' if road['name'] != selected_road['name'] else 'red',
            fill=True,
            fill_opacity=0.2,
            popup=road['name']
        ).add_to(m)
    
    # Optionally, add affected trips to the map
    if selected_road and affected_trips is not None and data is not None:
        affected_data = data[affected_trips]
        for _, row in affected_data.iterrows():
            folium.CircleMarker(
                location=[row['pickup_lat'], row['pickup_lon']],
                radius=3,
                color='orange',
                fill=True,
                fill_color='orange',
                popup='Affected Pickup'
            ).add_to(m)
            folium.CircleMarker(
                location=[row['dropoff_lat'], row['dropoff_lon']],
                radius=3,
                color='purple',
                fill=True,
                fill_color='purple',
                popup='Affected Dropoff'
            ).add_to(m)
    
    return m

# ---------------------------
# Streamlit App
# ---------------------------
def main():
    st.title("Brno Traffic and Congestion Simulation")
    st.write("""
    This application simulates the impact of road closures on taxi travel times within Brno, Czech Republic.
    Select a road to close, and observe how it affects overall traffic and congestion.
    """)
    
    # Load sample data
    data = generate_sample_data(n=1000)
    
    # Get sample roads
    roads = get_sample_roads()
    
    # Sidebar for road selection
    st.sidebar.header("Simulation Controls")
    road_names = [road['name'] for road in roads]
    selected_road_name = st.sidebar.selectbox("Select a road to close:", road_names)
    
    # Find the selected road's details
    selected_road = next((road for road in roads if road['name'] == selected_road_name), None)
    
    if selected_road:
        st.subheader(f"Selected Area: {selected_road['name']}")
        
        # Simulate road closure
        updated_data, affected_trips = simulate_road_closure(data, selected_road, delay_factor=1.5)
        
        # Display map
        st.write("### Road Map")
        m = create_folium_map(roads, selected_road=selected_road, affected_trips=affected_trips, data=updated_data)
        st_data = st_folium(m, width=700, height=500)
        
        # Display simulation results
        st.write("### Travel Time Distribution")
        plot_travel_time_distribution(data, updated_data, affected_trips)
        
        # Display statistics
        st.write("### Impact Statistics")
        total_trips = len(data)
        affected = affected_trips.sum()
        st.write(f"**Total Trips:** {total_trips}")
        st.write(f"**Affected Trips:** {affected} ({affected / total_trips * 100:.2f}%)")
        avg_original_time = data['travel_time_min'].mean()
        avg_updated_time = updated_data['travel_time_min'].mean()
        st.write(f"**Average Travel Time Before Closure:** {avg_original_time:.2f} minutes")
        st.write(f"**Average Travel Time After Closure:** {avg_updated_time:.2f} minutes")

if __name__ == "__main__":
    main()
