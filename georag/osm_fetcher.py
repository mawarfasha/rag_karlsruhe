# osm_fetcher.py
# Fetches real place data from OpenStreetMap using the free Overpass API.
# No API key needed. Results are cached in data/places.json after the first run.

import json
import os
import time
from typing import Any, Dict, List, Optional

import requests

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]


def _build_overpass_query(bbox: str) -> str:
    # Each block fetches a specific place type.
    # We query both nodes (standalone points) and ways (buildings with outlines).
    # 'out center tags' makes ways return their center coordinate.
    return f"""
[out:json][timeout:90];
(
  // ── Restaurants & food ──────────────────────────────────────────────
  node["amenity"="restaurant"]["name"]({bbox});
  way["amenity"="restaurant"]["name"]({bbox});

  node["amenity"="cafe"]["name"]({bbox});
  way["amenity"="cafe"]["name"]({bbox});

  node["amenity"="fast_food"]["name"]({bbox});
  way["amenity"="fast_food"]["name"]({bbox});

  node["amenity"="bar"]["name"]({bbox});
  way["amenity"="bar"]["name"]({bbox});

  // ── Tourist attractions ──────────────────────────────────────────────
  node["tourism"="attraction"]["name"]({bbox});
  way["tourism"="attraction"]["name"]({bbox});

  node["tourism"="museum"]["name"]({bbox});
  way["tourism"="museum"]["name"]({bbox});

  node["tourism"="gallery"]["name"]({bbox});
  way["tourism"="gallery"]["name"]({bbox});

  node["tourism"="theme_park"]["name"]({bbox});
  way["tourism"="theme_park"]["name"]({bbox});

  node["tourism"="viewpoint"]["name"]({bbox});
  way["tourism"="viewpoint"]["name"]({bbox});

  // ── Historic sites ───────────────────────────────────────────────────
  node["historic"="monument"]["name"]({bbox});
  way["historic"="monument"]["name"]({bbox});

  node["historic"="castle"]["name"]({bbox});
  way["historic"="castle"]["name"]({bbox});

  node["historic"="memorial"]["name"]({bbox});
  way["historic"="memorial"]["name"]({bbox});

  // ── Entertainment ────────────────────────────────────────────────────
  node["amenity"="theatre"]["name"]({bbox});
  way["amenity"="theatre"]["name"]({bbox});

  node["amenity"="cinema"]["name"]({bbox});
  way["amenity"="cinema"]["name"]({bbox});

  node["leisure"="park"]["name"]({bbox});
  way["leisure"="park"]["name"]({bbox});

  node["leisure"="playground"]["name"]({bbox});
  way["leisure"="playground"]["name"]({bbox});

  // ── Shopping ─────────────────────────────────────────────────────────
  node["shop"="mall"]["name"]({bbox});
  way["shop"="mall"]["name"]({bbox});

  node["shop"="department_store"]["name"]({bbox});
  way["shop"="department_store"]["name"]({bbox});

  node["shop"="supermarket"]["name"]({bbox});
  way["shop"="supermarket"]["name"]({bbox});

  node["shop"="electronics"]["name"]({bbox});
  way["shop"="electronics"]["name"]({bbox});

  node["shop"="computer"]["name"]({bbox});
  way["shop"="computer"]["name"]({bbox});

  node["shop"="appliance"]["name"]({bbox});
  way["shop"="appliance"]["name"]({bbox});

  node["shop"="furniture"]["name"]({bbox});
  way["shop"="furniture"]["name"]({bbox});

  node["shop"="doityourself"]["name"]({bbox});
  way["shop"="doityourself"]["name"]({bbox});

  node["amenity"="marketplace"]["name"]({bbox});
  way["amenity"="marketplace"]["name"]({bbox});
);
out center tags;
"""


def _determine_category(tags: Dict[str, str]) -> Optional[str]:
    """Map raw OSM tags to a simple category name."""
    amenity = tags.get("amenity", "")
    tourism  = tags.get("tourism",  "")
    shop     = tags.get("shop",     "")
    historic = tags.get("historic", "")
    leisure  = tags.get("leisure",  "")

    if amenity == "restaurant":
        return "restaurant"
    if amenity in ("cafe",):
        return "cafe"
    if amenity == "fast_food":
        return "fast_food"
    if amenity == "bar":
        return "bar"
    if tourism in ("attraction", "museum", "gallery", "theme_park", "viewpoint"):
        return "attraction"
    if historic in ("monument", "castle", "memorial", "ruins"):
        return "historic"
    if amenity in ("theatre", "cinema"):
        return "entertainment"
    if leisure in ("park", "playground"):
        return "leisure"
    if shop or amenity == "marketplace":
        return "shopping"
    return None


def _get_subcategory(tags: Dict[str, str]) -> str:
    """Return the most specific OSM tag value we can find."""
    for key in ("amenity", "tourism", "shop", "historic", "leisure"):
        val = tags.get(key, "")
        if val:
            return val
    return ""


def _process_element(element: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Turn a raw OSM element into a clean place dict. Returns None if unusable."""
    tags = element.get("tags", {})

    name = tags.get("name", "").strip()
    if not name:
        return None

    # nodes have lat/lon directly; ways have a 'center' dict
    if element["type"] == "node":
        lat = element.get("lat")
        lon = element.get("lon")
    elif element["type"] == "way" and "center" in element:
        lat = element["center"].get("lat")
        lon = element["center"].get("lon")
    else:
        return None

    if lat is None or lon is None:
        return None

    category = _determine_category(tags)
    if not category:
        return None

    # Build a human-readable address from OSM address tags
    addr_parts = []
    street = tags.get("addr:street", "")
    number = tags.get("addr:housenumber", "")
    if street:
        addr_parts.append(f"{street} {number}".strip())
    city = tags.get("addr:city", "")
    if city:
        addr_parts.append(city)
    elif addr_parts:
        addr_parts.append("Karlsruhe")

    return {
        "id":            element.get("id"),
        "name":          name,
        "category":      category,
        "subcategory":   _get_subcategory(tags),
        "lat":           lat,
        "lon":           lon,
        "address":       ", ".join(addr_parts) if addr_parts else "Karlsruhe",
        "cuisine":       tags.get("cuisine", ""),
        "opening_hours": tags.get("opening_hours", ""),
        "website":       tags.get("website", tags.get("contact:website", "")),
        "phone":         tags.get("phone",   tags.get("contact:phone",   "")),
        "description":   tags.get("description", ""),
        "wheelchair":    tags.get("wheelchair", ""),
        "operator":      tags.get("operator", ""),
    }


def fetch_places(bbox: str) -> List[Dict[str, Any]]:
    """Query the Overpass API and return cleaned place dicts for the given bbox."""
    print("Fetching places from OpenStreetMap... (this takes ~15-30 seconds)")

    query = _build_overpass_query(bbox)

    response = None
    for attempt, mirror in enumerate(OVERPASS_MIRRORS, 1):
        try:
            print(f"  Trying mirror {attempt}/{len(OVERPASS_MIRRORS)}: {mirror}")
            response = requests.post(mirror, data={"data": query}, timeout=120)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as exc:
            print(f"  Mirror {attempt} failed: {exc}")
            if attempt < len(OVERPASS_MIRRORS):
                print("  Retrying with next mirror...")
                time.sleep(2)
            else:
                print("All Overpass mirrors failed.")
                return []

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        print(f"Could not parse Overpass API response: {exc}")
        return []

    elements = data.get("elements", [])
    print(f"  Raw elements received: {len(elements)}")

    places = []
    seen_ids = set()
    for element in elements:
        place = _process_element(element)
        if place and place["id"] not in seen_ids:
            places.append(place)
            seen_ids.add(place["id"])

    print(f"  Named places kept: {len(places)}")
    return places


def save_places(places: List[Dict[str, Any]], filepath: str) -> None:
    """Cache the place list as JSON so we don't call the API every time."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(places, fh, ensure_ascii=False, indent=2)
    print(f"Saved {len(places)} places to {filepath}")


def load_places(filepath: str) -> List[Dict[str, Any]]:
    """Load previously cached places. Returns [] when the file doesn't exist yet."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as fh:
        places = json.load(fh)
    print(f"Loaded {len(places)} cached places from {filepath}")
    return places
