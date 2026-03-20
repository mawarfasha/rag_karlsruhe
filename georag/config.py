# config.py
# All settings for the GeoRAG system in one place.
# Change things here instead of digging through other files.

import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
INDEX_DIR   = os.path.join(BASE_DIR, "index")

# Milvus server (Docker standalone)
MILVUS_URI      = "http://localhost:19530"
COLLECTION_NAME = "karlsruhe_places"

# Embedding model – converts text into 384-dim vectors
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384

# LLM via Ollama running locally
LLM_MODEL    = "mistral"
LLM_BASE_URL = "http://localhost:11434"

# How to split place descriptions into chunks before embedding
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50

# Bounding box for Karlsruhe: "south,west,north,east"
KARLSRUHE_BBOX = "48.9762,8.3340,49.0597,8.4930"

# Default starting point when the user doesn't provide a location
DEFAULT_USER_LOCATION = {
    "lat":  49.0069,
    "lon":  8.4037,
    "name": "Marktplatz, Karlsruhe",
}

# How many places to retrieve per query
TOP_K = 5
