"""
Weather Intelligence AI
Geospatial Analytics Module

Purpose:
    Provides geographic visualization and spatial analysis support
    for the Weather Intelligence and Climate Decision Support Platform.

Note:
    The original weather.csv dataset does not contain latitude/longitude
    columns. Therefore, this module uses an explicitly defined geographic
    reference point for demonstration and does not invent station
    coordinates from the dataset.
"""

from pathlib import Path

import folium
import geopandas as gpd
from shapely.geometry import Point


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "reports" / "geospatial"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# GEOGRAPHIC REFERENCE
# ============================================================

# Reference point used only for geographic visualization.
# This is not claimed to be a weather-station coordinate
# contained in the original dataset.

REFERENCE_LOCATION = {
    "name": "India Weather Intelligence Reference",
    "latitude": 20.5937,
    "longitude": 78.9629,
}


# ============================================================
# CREATE GEOSPATIAL DATA
# ============================================================

geometry = [
    Point(
        REFERENCE_LOCATION["longitude"],
        REFERENCE_LOCATION["latitude"],
    )
]

gdf = gpd.GeoDataFrame(
    {
        "location": [REFERENCE_LOCATION["name"]],
    },
    geometry=geometry,
    crs="EPSG:4326",
)


# ============================================================
# EXPORT GEOSPATIAL DATA
# ============================================================

geojson_path = OUTPUT_DIR / "weather_reference_location.geojson"

gdf.to_file(
    geojson_path,
    driver="GeoJSON",
)


# ============================================================
# CREATE INTERACTIVE MAP
# ============================================================

weather_map = folium.Map(
    location=[
        REFERENCE_LOCATION["latitude"],
        REFERENCE_LOCATION["longitude"],
    ],
    zoom_start=5,
    tiles="OpenStreetMap",
)


folium.Marker(
    location=[
        REFERENCE_LOCATION["latitude"],
        REFERENCE_LOCATION["longitude"],
    ],
    popup=(
        "<b>Weather Intelligence AI</b><br>"
        "India geographic reference point"
    ),
    tooltip="Weather Intelligence AI",
).add_to(weather_map)


# ============================================================
# ADD INFORMATION PANEL
# ============================================================

info_html = """
<div style="
    position: fixed;
    bottom: 40px;
    left: 40px;
    width: 280px;
    padding: 12px;
    background-color: white;
    border: 2px solid #444;
    z-index: 9999;
    font-size: 13px;
">
<b>Geospatial Weather Analysis</b><br><br>
CRS: EPSG:4326<br>
Reference: India<br>
Purpose: Geographic visualization<br>
Dataset limitation: Original weather dataset
does not contain latitude/longitude columns.
</div>
"""

weather_map.get_root().html.add_child(
    folium.Element(info_html)
)


# ============================================================
# SAVE MAP
# ============================================================

map_path = OUTPUT_DIR / "weather_geospatial_map.html"

weather_map.save(map_path)


# ============================================================
# SUMMARY
# ============================================================

summary_path = OUTPUT_DIR / "geospatial_analysis_summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:
    file.write("WEATHER INTELLIGENCE AI - GEOSPATIAL ANALYTICS\n")
    file.write("=" * 55 + "\n\n")

    file.write("Geospatial library status:\n")
    file.write(f"GeoPandas: {gpd.__version__}\n")
    file.write("Shapely: installed\n")
    file.write("Folium: installed\n\n")

    file.write("Coordinate Reference System: EPSG:4326\n")
    file.write("Geographic reference: India\n")
    file.write(
        "Latitude: "
        f"{REFERENCE_LOCATION['latitude']}\n"
    )
    file.write(
        "Longitude: "
        f"{REFERENCE_LOCATION['longitude']}\n\n"
    )

    file.write("Generated outputs:\n")
    file.write(f"{geojson_path}\n")
    file.write(f"{map_path}\n\n")

    file.write("Dataset limitation:\n")
    file.write(
        "The original weather dataset does not contain "
        "latitude/longitude columns. Therefore, the "
        "reference point is used only for geographic "
        "visualization and is not presented as an "
        "observed weather-station coordinate.\n"
    )


print("=" * 60)
print("GEOSPATIAL ANALYTICS COMPLETED")
print("=" * 60)
print(f"GeoPandas version : {gpd.__version__}")
print("CRS               : EPSG:4326")
print(f"GeoJSON            : {geojson_path}")
print(f"Interactive map    : {map_path}")
print(f"Summary            : {summary_path}")
print("=" * 60)