# route_visualizer.py
# Builds an interactive Folium HTML map for a list of recommended places.
#
# Features:
#   - Nearest-neighbour route ordering (start from user location)
#   - Dashed polyline connecting stops in visiting order
#   - Colour-coded markers with popups per category
#   - Legend and map title
#   - Saves as a self-contained HTML file (open in any browser)

import math
import os
from html import escape
from typing import Dict, List, Optional

import folium

# Marker colour per category
CATEGORY_COLORS = {
    "restaurant":    "red",
    "cafe":          "orange",
    "fast_food":     "orange",
    "bar":           "cadetblue",
    "attraction":    "blue",
    "historic":      "darkblue",
    "entertainment": "purple",
    "leisure":       "green",
    "shopping":      "darkgreen",
}

# Font-Awesome 4 icon per category
CATEGORY_ICONS = {
    "restaurant":    "cutlery",
    "cafe":          "coffee",
    "fast_food":     "cutlery",
    "bar":           "glass",
    "attraction":    "star",
    "historic":      "landmark",
    "entertainment": "film",
    "leisure":       "tree",
    "shopping":      "shopping-cart",
}

KARLSRUHE_CENTER = [49.0069, 8.4037]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in km between two GPS coordinates."""
    R = 6371.0  # Earth's radius in km
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def nearest_neighbour_route(
    places: List[Dict],
    start_lat: float,
    start_lon: float,
) -> List[Dict]:
    """
    Greedy nearest-neighbour sort: starting from (start_lat, start_lon),
    always pick the closest remaining place next.
    """
    remaining = places.copy()
    route: List[Dict] = []
    cur_lat, cur_lon = start_lat, start_lon

    while remaining:
        nearest = min(
            remaining,
            key=lambda p: haversine_km(cur_lat, cur_lon, p["lat"], p["lon"]),
        )
        route.append(nearest)
        remaining.remove(nearest)
        cur_lat, cur_lon = nearest["lat"], nearest["lon"]

    return route




def create_route_map(
    places: List[Dict],
    user_location: Optional[Dict] = None,
    query: str = "",
    output_file: str = "outputs/route_map.html",
) -> str:
    """
    Build and save an interactive route map as HTML.

    places:        list of dicts with name, category, address, lat, lon, score
    user_location: dict with lat, lon, name (the starting point)
    query:         shown as subtitle in the map
    output_file:   path to save the HTML
    """

    if user_location:
        center = [user_location["lat"], user_location["lon"]]
    elif places:
        center = [
            sum(p["lat"] for p in places) / len(places),
            sum(p["lon"] for p in places) / len(places),
        ]
    else:
        center = KARLSRUHE_CENTER

    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
    safe_query = escape(query[:120])

    title_html = f"""
    <div style="position:fixed; top:10px; left:60px; right:60px; z-index:1000;
                background:white; padding:8px 14px; border-radius:6px;
                box-shadow:0 2px 6px rgba(0,0,0,0.3); font-family:Arial, sans-serif;">
        <b style="font-size:15px;">GeoRAG – Karlsruhe Route Map</b><br>
        <span style="font-size:12px; color:#555;">Query: {safe_query}</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    if user_location:
        safe_user_name = escape(user_location.get("name", "Start"))
        folium.Marker(
            location=[user_location["lat"], user_location["lon"]],
            popup=folium.Popup(
                f"<b>📍 Your Location</b><br>{safe_user_name}",
                max_width=220,
            ),
            tooltip="Your Location (Start)",
            icon=folium.Icon(color="black", icon="home", prefix="fa"),
        ).add_to(m)

    # Sort places into visiting order from the user's location
    if user_location and places:
        ordered = nearest_neighbour_route(
            places, user_location["lat"], user_location["lon"]
        )
    else:
        ordered = places

    # draw the route as a dashed polyline
    if user_location and ordered:
        route_coords = [[user_location["lat"], user_location["lon"]]] + [
            [p["lat"], p["lon"]] for p in ordered
        ]
        folium.PolyLine(
            locations=route_coords,
            color="#2979FF",
            weight=3,
            opacity=0.75,
            dash_array="8 5",
            tooltip="Suggested Route",
        ).add_to(m)

    # add a marker for each place with a popup showing details
    for i, place in enumerate(ordered, 1):
        category = place.get("category", "attraction")
        colour   = CATEGORY_COLORS.get(category, "gray")
        icon     = CATEGORY_ICONS.get(category, "info-sign")
        safe_name = escape(str(place.get("name", "?")))
        safe_address = escape(str(place.get("address", "N/A")))
        safe_category = escape(category.replace("_", " ").title())

        # Distance from user
        dist_str = ""
        if user_location:
            dist = haversine_km(
                user_location["lat"], user_location["lon"],
                place["lat"], place["lon"],
            )
            dist_str = f"<br><i>{dist:.2f} km from you</i>"

        popup_html = f"""
        <div style="font-family:Arial,sans-serif; min-width:180px;">
            <h4 style="margin:0 0 4px 0; color:#222;">{i}. {safe_name}</h4>
            <b>Type:</b> {safe_category}<br>
            <b>Address:</b> {safe_address}<br>
            <b>Relevance:</b> {place.get('score', 0):.0%}{dist_str}
        </div>
        """

        folium.Marker(
            location=[place["lat"], place["lon"]],
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"Stop {i}: {safe_name}",
            icon=folium.Icon(color=colour, icon=icon, prefix="fa"),
        ).add_to(m)

        # Small numbered badge offset slightly from the main marker
        folium.Marker(
            location=[place["lat"], place["lon"]],
            icon=folium.DivIcon(
                html=(
                    f'<div style="font-size:11px;font-weight:bold;color:white;'
                    f'background:#333;border-radius:50%;width:18px;height:18px;'
                    f'text-align:center;line-height:18px;'
                    f'margin-left:12px;margin-top:-28px;">{i}</div>'
                ),
                icon_size=(18, 18),
            ),
        ).add_to(m)

    legend_html = """
    <div style="position:fixed;bottom:30px;right:15px;z-index:1000;
                background:white;padding:10px 14px;border-radius:6px;
                border:1px solid #ccc;font-family:Arial,sans-serif;font-size:12px;">
        <b>Legend</b><br>
        <span style="color:red">&#9679;</span> Restaurant<br>
        <span style="color:orange">&#9679;</span> Café / Fast Food<br>
        <span style="color:#436EEE">&#9679;</span> Attraction<br>
        <span style="color:#00008B">&#9679;</span> Historic<br>
        <span style="color:green">&#9679;</span> Leisure / Park<br>
        <span style="color:darkgreen">&#9679;</span> Shopping<br>
        <span style="color:#5F9EA0">&#9679;</span> Bar<br>
        <span style="color:black">&#9679;</span> Your Location
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    m.save(output_file)

    abs_path = os.path.abspath(output_file)
    print(f"Route map saved → {abs_path}")
    return abs_path
