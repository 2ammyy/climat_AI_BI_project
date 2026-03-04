# backend/visualize_cross_border.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Create output directory
os.makedirs('eda_results/cross_border', exist_ok=True)

print("="*60)
print("🌍 CROSS-BORDER WEATHER PATTERN VISUALIZATIONS")
print("="*60)

# Load data
print("\n📊 Loading data...")
tunisia_df = pd.read_csv('data/merged_data_clean.csv')
neighbor_df = pd.read_csv('data/neighbor_data/neighbor_weather_latest.csv')

# Define coordinates for mapping
locations = {
    'Tunisia': {'lat': 34.0, 'lon': 9.0},
    # Neighbor countries
    'Algeria': {'lat': 28.0, 'lon': 1.0, 'color': 'red'},
    'Libya': {'lat': 27.0, 'lon': 17.0, 'color': 'orange'},
    'Italy': {'lat': 42.0, 'lon': 12.0, 'color': 'green'},
    'Malta': {'lat': 35.9, 'lon': 14.5, 'color': 'blue'},
    # Key cities
    'Tunis': {'lat': 36.8065, 'lon': 10.1815},
    'Algiers': {'lat': 36.7538, 'lon': 3.0588},
    'Tripoli': {'lat': 32.8872, 'lon': 13.1917},
    'Palermo': {'lat': 38.1157, 'lon': 13.3615},
    'Valletta': {'lat': 35.8989, 'lon': 14.5146}
}

# Influence zones mapping
influence_zones = {
    'Northwest': ['Jendouba', 'Beja', 'Kef', 'Siliana'],
    'Northeast': ['Bizerte', 'Tunis', 'Ariana', 'Ben Arous', 'Manouba', 'Nabeul', 'Zaghouan'],
    'Southwest': ['Tozeur', 'Kebili', 'Tataouine', 'Gafsa'],
    'Southeast': ['Medenine', 'Gabes', 'Sfax'],
    'Coastal': ['Sousse', 'Monastir', 'Mahdia', 'Sfax', 'Gabes'],
    'Central': ['Kairouan', 'Kasserine', 'Sidi Bouzid']
}

# 1. INTERACTIVE MAP WITH WEATHER DATA
print("\n🗺️ Creating interactive weather map...")

def create_weather_map():
    """Create an interactive Folium map with weather data"""
    
    # Center map on Mediterranean
    m = folium.Map(location=[36.0, 12.0], zoom_start=5)
    
    # Add neighbor country weather
    for _, row in neighbor_df.iterrows():
        city = row['city'].title()
        country = row['country'].title()
        
        # Color by temperature
        temp = row['temperature']
        if temp > 25:
            color = 'red'
        elif temp > 20:
            color = 'orange'
        elif temp > 15:
            color = 'green'
        else:
            color = 'blue'
        
        # Get coordinates
        coords = None
        for loc_name, loc_data in locations.items():
            if loc_name.lower() == city.lower() or loc_name.lower() == country.lower():
                coords = [loc_data['lat'], loc_data['lon']]
                break
        
        if coords:
            # Create popup with weather info
            popup_html = f"""
            <div style="font-family: Arial; width: 200px;">
                <h4>{city}, {country}</h4>
                <p>🌡️ Temperature: {temp}°C</p>
                <p>💨 Wind: {row['wind_speed']} km/h</p>
                <p>💧 Humidity: {row['humidity']}%</p>
                <p>☁️ Conditions: {row['weather_condition']}</p>
            </div>
            """
            
            folium.Marker(
                coords,
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color=color, icon='cloud')
            ).add_to(m)
    
    # Add Tunisian regions with influence zones
    for zone, governorates in influence_zones.items():
        # Get approximate center of zone
        if zone == 'Northwest':
            center = [36.5, 8.5]
            color = 'red'
        elif zone == 'Northeast':
            center = [37.0, 10.5]
            color = 'orange'
        elif zone == 'Southwest':
            center = [33.0, 8.0]
            color = 'purple'
        elif zone == 'Southeast':
            center = [33.5, 10.5]
            color = 'blue'
        elif zone == 'Coastal':
            center = [35.5, 11.0]
            color = 'green'
        else:  # Central
            center = [35.0, 9.5]
            color = 'gray'
        
        # Add zone marker
        folium.Marker(
            center,
            popup=f"{zone} Region<br>Governorates: {', '.join(governorates[:3])}...",
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(m)
    
    # Draw influence arrows (simplified)
    influences = [
        {'from': [36.75, 3.05], 'to': [36.5, 8.5], 'country': 'Algeria', 'zone': 'Northwest'},
        {'from': [32.88, 13.19], 'to': [33.5, 10.5], 'country': 'Libya', 'zone': 'Southeast'},
        {'from': [38.11, 13.36], 'to': [37.0, 10.5], 'country': 'Italy', 'zone': 'Northeast'},
        {'from': [35.89, 14.51], 'to': [35.5, 11.0], 'country': 'Malta', 'zone': 'Coastal'}
    ]
    
    for inf in influences:
        folium.PolyLine(
            [inf['from'], inf['to']],
            color='red',
            weight=2,
            opacity=0.5,
            popup=f"{inf['country']} → {inf['zone']} influence"
        ).add_to(m)
    
    # Save map
    m.save('eda_results/cross_border/weather_map.html')
    print("  ✅ Map saved: eda_results/cross_border/weather_map.html")
    
    return m

# 2. TEMPERATURE COMPARISON PLOT
print("\n🌡️ Creating temperature comparison plot...")

def plot_temperature_comparison():
    """Compare temperatures across countries"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data
    countries = []
    temps = []
    colors = []
    
    for country in ['algeria', 'libya', 'italy', 'malta']:
        country_data = neighbor_df[neighbor_df['country'] == country]
        if len(country_data) > 0:
            countries.append(country.title())
            temps.append(country_data['temperature'].mean())
            
            if country == 'algeria':
                colors.append('red')
            elif country == 'libya':
                colors.append('orange')
            elif country == 'italy':
                colors.append('green')
            elif country == 'malta':
                colors.append('blue')
    
    # Add Tunisian average (from your data)
    tunisian_temp = tunisia_df['temp_avg'].mean()
    countries.append('Tunisia')
    temps.append(tunisian_temp)
    colors.append('purple')
    
    # Create bars
    bars = ax.bar(countries, temps, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, temp in zip(bars, temps):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{temp:.1f}°C', ha='center', va='bottom')
    
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Average Temperature Comparison: Neighbors vs Tunisia')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eda_results/cross_border/temperature_comparison.png', dpi=150)
    print("  ✅ Saved: temperature_comparison.png")
    plt.show()

# 3. WIND PATTERN MAP
print("\n💨 Creating wind pattern visualization...")

def plot_wind_patterns():
    """Show wind patterns across the region"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot Mediterranean map outline (simplified)
    # Tunisia
    tunisia_coords = [(8.0, 37.0), (11.0, 37.5), (11.5, 35.0), (10.0, 32.0), (8.0, 34.0)]
    tunisia_coords = [(lon, lat) for lon, lat in tunisia_coords]  # Swap for plotting
    
    # Plot neighbor countries
    for country, locations_data in neighbor_df.groupby('country'):
        color = {'algeria': 'red', 'libya': 'orange', 'italy': 'green', 'malta': 'blue'}.get(country, 'gray')
        
        for _, row in locations_data.iterrows():
            # Find coordinates
            city = row['city'].title()
            coords = None
            for loc_name, loc_data in locations.items():
                if loc_name.lower() == city.lower():
                    coords = (loc_data['lon'], loc_data['lat'])
                    break
            
            if coords:
                # Plot wind arrow
                wind_speed = row['wind_speed']
                arrow_length = wind_speed / 10  # Scale for visibility
                
                ax.arrow(coords[0], coords[1], 0.5, 0.2,
                        head_width=0.3, head_length=0.3,
                        fc=color, ec=color, alpha=0.7,
                        label=f"{city}: {wind_speed} km/h" if _ == 0 else "")
                
                # Add city label
                ax.annotate(f"{city}\n{wind_speed} km/h", 
                           (coords[0], coords[1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Regional Wind Patterns')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 20])
    ax.set_ylim([30, 40])
    
    plt.tight_layout()
    plt.savefig('eda_results/cross_border/wind_patterns.png', dpi=150)
    print("  ✅ Saved: wind_patterns.png")
    plt.show()

# 4. INFLUENCE HEATMAP
print("\�️ Creating influence zone heatmap...")

def plot_influence_heatmap():
    """Visualize which neighbors influence which regions"""
    
    # Create influence matrix
    regions = list(influence_zones.keys())
    countries = ['Algeria', 'Libya', 'Italy', 'Malta']
    
    influence_matrix = np.zeros((len(regions), len(countries)))
    
    # Fill matrix based on influence
    for i, region in enumerate(regions):
        if region in ['Northwest', 'Central']:
            influence_matrix[i, 0] = 0.8  # Algeria
        if region in ['Southwest', 'Southeast']:
            influence_matrix[i, 1] = 0.7  # Libya
        if region in ['Northeast', 'Coastal']:
            influence_matrix[i, 2] = 0.6  # Italy
            influence_matrix[i, 3] = 0.5  # Malta
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(influence_matrix, cmap='YlOrRd', aspect='auto')
    
    # Add labels
    ax.set_xticks(np.arange(len(countries)))
    ax.set_yticks(np.arange(len(regions)))
    ax.set_xticklabels(countries)
    ax.set_yticklabels(regions)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(regions)):
        for j in range(len(countries)):
            if influence_matrix[i, j] > 0:
                text = ax.text(j, i, f"{influence_matrix[i, j]*100:.0f}%",
                              ha="center", va="center", color="black")
    
    ax.set_title("Neighbor Country Influence on Tunisian Regions")
    fig.colorbar(im, ax=ax, label='Influence Strength')
    
    plt.tight_layout()
    plt.savefig('eda_results/cross_border/influence_heatmap.png', dpi=150)
    print("  ✅ Saved: influence_heatmap.png")
    plt.show()

# 5. INTERACTIVE PLOTLY DASHBOARD
print("\n📊 Creating interactive Plotly dashboard...")

def create_plotly_dashboard():
    """Create an interactive dashboard with Plotly"""
    
    # Prepare data for Plotly
    plotly_data = []
    
    for _, row in neighbor_df.iterrows():
        plotly_data.append({
            'Country': row['country'].title(),
            'City': row['city'].title(),
            'Temperature': row['temperature'],
            'Wind Speed': row['wind_speed'],
            'Humidity': row['humidity'],
            'Conditions': row['weather_condition'],
            'Type': 'Neighbor'
        })
    
    # Add Tunisian average as reference
    plotly_data.append({
        'Country': 'Tunisia',
        'City': 'National Average',
        'Temperature': tunisia_df['temp_avg'].mean(),
        'Wind Speed': tunisia_df['wind_speed'].mean(),
        'Humidity': tunisia_df['humidity'].mean(),
        'Conditions': 'Average',
        'Type': 'Tunisia'
    })
    
    df_plotly = pd.DataFrame(plotly_data)
    
    # Create scatter plot
    fig1 = px.scatter(df_plotly, 
                     x='Temperature', 
                     y='Wind Speed',
                     size='Humidity',
                     color='Country',
                     hover_data=['City', 'Conditions'],
                     title='Weather Comparison: Neighbors vs Tunisia',
                     labels={'Temperature': 'Temperature (°C)', 
                            'Wind Speed': 'Wind Speed (km/h)'})
    
    fig1.write_html('eda_results/cross_border/interactive_scatter.html')
    print("  ✅ Saved: interactive_scatter.html")
    
    # Create bar chart
    fig2 = px.bar(df_plotly[df_plotly['Type'] == 'Neighbor'], 
                 x='City', 
                 y='Temperature',
                 color='Country',
                 title='Temperature by City in Neighboring Countries',
                 labels={'Temperature': 'Temperature (°C)'})
    
    fig2.write_html('eda_results/cross_border/temperature_bars.html')
    print("  ✅ Saved: temperature_bars.html")
    
    # Create dashboard with subplots
    from plotly.subplots import make_subplots
    
    fig3 = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temperature by Country', 'Wind Speed by Country',
                       'Humidity by Country', 'Weather Conditions'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'pie'}]]
    )
    
    # Add traces
    for i, metric in enumerate(['Temperature', 'Wind Speed', 'Humidity']):
        row = i // 2 + 1
        col = i % 2 + 1
        for country in df_plotly['Country'].unique():
            country_data = df_plotly[df_plotly['Country'] == country]
            if metric == 'Humidity' and country == 'Tunisia':
                continue
            fig3.add_trace(
                go.Bar(name=country, x=[country], y=[country_data[metric].mean()],
                      legendgroup=country, showlegend=(i==0)),
                row=row, col=col
            )
    
    # Conditions pie chart
    condition_counts = neighbor_df['weather_condition'].value_counts()
    fig3.add_trace(
        go.Pie(labels=condition_counts.index, values=condition_counts.values),
        row=2, col=2
    )
    
    fig3.update_layout(height=800, title_text="Cross-Border Weather Dashboard")
    fig3.write_html('eda_results/cross_border/dashboard.html')
    print("  ✅ Saved: dashboard.html")

# Generate all visualizations
print("\n🎨 Generating all visualizations...")

create_weather_map()
plot_temperature_comparison()
plot_wind_patterns()
plot_influence_heatmap()
create_plotly_dashboard()

print("\n" + "="*60)
print("✅ ALL VISUALIZATIONS COMPLETE!")
print("="*60)
print("\n📁 Files saved in 'eda_results/cross_border/':")
print("  1. weather_map.html - Interactive Folium map")
print("  2. temperature_comparison.png - Temperature bar chart")
print("  3. wind_patterns.png - Wind direction map")
print("  4. influence_heatmap.png - Regional influence matrix")
print("  5. interactive_scatter.html - Plotly scatter plot")
print("  6. temperature_bars.html - Interactive bar chart")
print("  7. dashboard.html - Complete Plotly dashboard")
print("\n🌐 Open the .html files in your browser to explore!")