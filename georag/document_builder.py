# document_builder.py
# Converts OSM place dicts into LlamaIndex Documents.
# Each Document has a natural-language description (for embedding) and
# metadata (name, lat/lon, address, etc.) that comes back with search results.

from typing import Any, Dict, List

from llama_index.core import Document



CATEGORY_LABELS = {
    "restaurant":    "restaurant",
    "cafe":          "café",
    "fast_food":     "fast-food restaurant",
    "bar":           "bar",
    "attraction":    "tourist attraction",
    "historic":      "historic site",
    "entertainment": "entertainment venue",
    "leisure":       "leisure area",
    "shopping":      "shopping venue",
}

# Extra context sentences we append based on the place subcategory
SUBCATEGORY_SENTENCES = {
    "museum":           "It is a museum – great for learning about art, history, or science.",
    "gallery":          "It is an art gallery.",
    "monument":         "It is a historic monument worth visiting.",
    "castle":           "It is a historic castle.",
    "memorial":         "It is a memorial site.",
    "theme_park":       "It is a theme park, fun for families and children.",
    "viewpoint":        "It offers a scenic viewpoint over the city.",
    "theatre":          "It is a theatre offering stage performances.",
    "cinema":           "It is a cinema showing movies.",
    "park":             "It is a public park – perfect for families, walks, and outdoor activities.",
    "playground":       "It is a playground – ideal for children.",
    "mall":             "It is a shopping mall with many stores under one roof.",
    "department_store": "It is a department store offering a wide range of goods.",
    "supermarket":      "It is a supermarket selling groceries and household items.",
    "electronics":      "It specialises in electronics and technology products.",
    "computer":         "It specialises in computers and IT equipment.",
    "appliance":        "It sells home appliances such as washing machines, fridges, and ovens.",
    "furniture":        "It sells furniture and home accessories.",
    "doityourself":     "It is a DIY / home-improvement store.",
    "marketplace":      "It is an open marketplace where various goods are sold.",
}


def _build_description(place: Dict[str, Any]) -> str:
    """Build a readable English description for a place. This is what gets embedded."""
    name        = place.get("name", "Unknown Place")
    category    = place.get("category", "place")
    subcategory = place.get("subcategory", "")
    address     = place.get("address", "Karlsruhe")
    label       = CATEGORY_LABELS.get(category, category.replace("_", " "))

    desc = f"{name} is a {label}"

    cuisine = place.get("cuisine", "").strip()
    if cuisine:
        desc += f" serving {cuisine.replace(';', ', ').replace('_', ' ')} cuisine"

    desc += f", located at {address}."

    if subcategory in SUBCATEGORY_SENTENCES:
        desc += " " + SUBCATEGORY_SENTENCES[subcategory]

    hours = place.get("opening_hours", "").strip()
    if hours:
        desc += f" Opening hours: {hours}."

    osm_desc = place.get("description", "").strip()
    if osm_desc:
        desc += f" {osm_desc}"

    wheelchair = place.get("wheelchair", "").strip()
    if wheelchair == "yes":
        desc += " It is wheelchair accessible."
    elif wheelchair == "no":
        desc += " Note: not wheelchair accessible."

    operator = place.get("operator", "").strip()
    if operator and operator.lower() != name.lower():
        desc += f" Operated by {operator}."

    return desc


def build_documents(places: List[Dict[str, Any]]) -> List[Document]:
    """Convert OSM place dicts into LlamaIndex Document objects ready for indexing."""
    documents = []

    for place in places:
        name = place.get("name", "").strip()
        if not name:
            continue  # skip unnamed places

        text = _build_description(place)

        # Milvus strings
        metadata = {
            "name":          name,
            "category":      place.get("category",      ""),
            "subcategory":   place.get("subcategory",   ""),
            "lat":           str(place.get("lat",  "")),
            "lon":           str(place.get("lon",  "")),
            "address":       place.get("address",       ""),
            "cuisine":       place.get("cuisine",       ""),
            "opening_hours": place.get("opening_hours", ""),
            "website":       place.get("website",       ""),
            "phone":         place.get("phone",         ""),
            "osm_id":        str(place.get("id",        "")),
        }

        doc = Document(
            text=text,
            metadata=metadata,
            id_=f"place_{place.get('id', len(documents))}",
        )
        documents.append(doc)

    print(f"Built {len(documents)} documents from {len(places)} OSM places.")
    return documents
