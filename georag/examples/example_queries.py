# examples/example_queries.py
# Runs all 5 demo queries and saves results + maps to the outputs/ folder.
#
# Run this after ingesting data:
#   python main.py ingest
#   python examples/example_queries.py

import json
import os
import sys

# Make parent directory importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from rag_pipeline import GeoRAGPipeline
from route_visualizer import create_route_map


# Starting location for all queries (change to any Karlsruhe point you like)
USER_LOCATION = config.DEFAULT_USER_LOCATION


EXAMPLE_QUERIES = [
    {
        "id":          1,
        "type":        "Restaurant – specific cuisine",
        "query":       "I am looking for the best sushi restaurant near Europaplatz.",
        "description": "Tests whether the system can find a specific cuisine type "
                       "and relate it to a named Karlsruhe landmark.",
    },
    {
        "id":          2,
        "type":        "Situation – family with children",
        "query":       "Where can I bring my 2 kids to visit in Karlsruhe? "
                       "We enjoy museums, parks, and outdoor activities.",
        "description": "Tests multi-category retrieval for a family scenario.",
    },
    {
        "id":          3,
        "type":        "Shopping – specific product",
        "query":       "I want to buy a new washing machine. "
                       "Where can I go to check in the city?",
        "description": "Tests whether appliance/electronics stores are found "
                       "for a concrete purchase intent.",
    },
    {
        "id":          4,
        "type":        "Café – remote work",
        "query":       "I need a quiet café with good coffee where I can work "
                       "on my laptop for a few hours.",
        "description": "Tests retrieval of cafés with semantic nuance (quiet, work-friendly).",
    },
    {
        "id":          5,
        "type":        "Tourism – day trip sightseeing",
        "query":       "I am visiting Karlsruhe for one day. "
                       "What are the must-see historic sites and cultural attractions?",
        "description": "Tests historic site and attraction retrieval with "
                       "an implicit route planning need.",
    },
]




def run_examples():
    """Run all 5 example queries, print results, and save maps + JSON."""

    print("=" * 65)
    print("  GeoRAG - Example Queries Demo")
    print("=" * 65)

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)

    print("\nLoading RAG pipeline...")
    pipeline = GeoRAGPipeline()
    try:
        pipeline.load_existing_index()
    except FileNotFoundError as exc:
        print(f"\nError: {exc}")
        print("Run  python main.py ingest  first.")
        sys.exit(1)

    all_results = []

    for ex in EXAMPLE_QUERIES:
        print(f"\n" + "-" * 65)
        print(f"  Query {ex['id']}  |  {ex['type']}")
        print("-" * 65)
        print(f"  Question: {ex['query']}")
        print(f"  Intent  : {ex['description']}")
        print(f"  Location: {USER_LOCATION['name']}")
        print()


        try:
            response, source_nodes = pipeline.query(ex["query"], USER_LOCATION)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        print(f"\n  Retrieved {len(source_nodes)} place(s):")
        for i, node in enumerate(source_nodes, 1):
            m = node.metadata
            cat = m.get("category", "").replace("_", " ").title()
            score = node.score if node.score is not None else 0
            print(f"    {i}. {m.get('name','?')}  [{cat}]  "
                  f"addr: {m.get('address','N/A')}  score: {score:.4f}")

        print("\n  Response:")
        for line in response.splitlines():
            print(f"    {line}")

        places = []
        for node in source_nodes:
            m = node.metadata
            try:
                lat = float(m.get("lat") or 0)
                lon = float(m.get("lon") or 0)
            except (ValueError, TypeError):
                continue
            if lat and lon:
                places.append({
                    "name":     m.get("name", "?"),
                    "category": m.get("category", ""),
                    "address":  m.get("address", ""),
                    "lat":      lat,
                    "lon":      lon,
                    "score":    node.score or 0.0,
                })

        map_file = None
        if places:
            map_file = os.path.join(config.OUTPUTS_DIR, f"example_{ex['id']}_map.html")
            create_route_map(
                places=places,
                user_location=USER_LOCATION,
                query=ex["query"],
                output_file=map_file,
            )


        result = {
            "id":          ex["id"],
            "type":        ex["type"],
            "query":       ex["query"],
            "description": ex["description"],
            "user_location": USER_LOCATION,
            "retrieved_contexts": [
                {
                    "rank":     i,
                    "name":     n.metadata.get("name", ""),
                    "category": n.metadata.get("category", ""),
                    "address":  n.metadata.get("address", ""),
                    "text":     n.text[:600],
                    "score":    n.score,
                }
                for i, n in enumerate(source_nodes, 1)
            ],
            "response":   response,
            "map_file":   map_file,
            "evaluation": {
                "places_retrieved":  len(places),
                "response_words":    len(response.split()),
                "has_llm_response":  not response.startswith("Here are the most relevant"),
            },
        }
        all_results.append(result)

    results_path = os.path.join(config.OUTPUTS_DIR, "example_results.json")
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 65}")
    print("  All 5 queries done!")
    print(f"{'=' * 65}")
    print(f"\nSaved files:")
    for r in all_results:
        if r.get("map_file"):
            print(f"  Map   → {os.path.abspath(r['map_file'])}")
    print(f"  JSON  → {os.path.abspath(results_path)}")
    print("\nOpen any .html file in a browser to see the interactive route map.")


if __name__ == "__main__":
    run_examples()
