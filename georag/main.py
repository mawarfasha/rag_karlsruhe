# main.py
# CLI for the GeoRAG system.
#
#   python main.py ingest          - fetch OSM data and build the vector index
#   python main.py ask "question"  - ask a single question about places in Karlsruhe
#   python main.py chat            - interactive mode (model stays loaded between questions)
#   python main.py check           - verify that all components are set up

import argparse
import json
import os
import sys
from typing import Optional

# Make sure Python can find sibling modules when run from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from rag_pipeline import GeoRAGPipeline
from route_visualizer import create_route_map




def _parse_location(location_str: str) -> dict:
    """
    Parse a 'lat,lon' string into a location dict.
    Raises ValueError with a helpful message on bad input.
    """
    parts = location_str.strip().split(",")
    if len(parts) != 2:
        raise ValueError("Location must be in 'lat,lon' format, e.g.  49.0069,8.4037")
    lat = float(parts[0].strip())
    lon = float(parts[1].strip())
    return {"lat": lat, "lon": lon, "name": f"Custom ({lat}, {lon})"}


def _geocode_address(address: str) -> Optional[dict]:
    """Resolve a free-text address to lat/lon using OpenStreetMap Nominatim."""
    if not address.strip():
        return None

    # Normalize a few common spelling variants so casual input still resolves.
    normalized = address.strip()
    lower = normalized.lower()
    if "strabe" in lower:
        normalized = normalized.replace("strabe", "strasse").replace("Strabe", "Strasse")
    if "straße" in lower:
        normalized = normalized.replace("straße", "strasse").replace("Straße", "Strasse")

    candidates = [normalized]
    if "karlsruhe" not in normalized.lower():
        candidates.append(f"{normalized}, Karlsruhe")
    if "germany" not in normalized.lower() and "deutschland" not in normalized.lower():
        candidates.append(f"{normalized}, Karlsruhe, Germany")

    try:
        import requests
        for query in candidates:
            resp = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": query, "format": "json", "limit": 1},
                headers={"User-Agent": "georag-cli/1.0"},
                timeout=8,
            )
            resp.raise_for_status()
            results = resp.json()
            if not results:
                continue
            top = results[0]
            return {
                "lat": float(top["lat"]),
                "lon": float(top["lon"]),
                "name": top.get("display_name", query),
            }
        return None
    except Exception:
        return None


def _resolve_location_input(raw_input: str) -> Optional[dict]:
    """Parse either 'lat,lon' or geocode a typed address."""
    text = raw_input.strip()
    if not text:
        return None
    try:
        return _parse_location(text)
    except Exception:
        return _geocode_address(text)


def _print_sources(source_nodes) -> None:
    """Print the retrieved context places to the terminal."""
    for i, node in enumerate(source_nodes, 1):
        m = node.metadata
        cat = m.get("category", "").replace("_", " ").title()
        score = node.score if node.score is not None else 0
        print(f"[{i}] {m.get('name', 'Unknown')}  ({cat})")
        print(f"    Address  : {m.get('address', 'N/A')}")
        if m.get("opening_hours"):
            print(f"    Hours    : {m['opening_hours']}")
        if m.get("website"):
            print(f"    Website  : {m['website']}")
        print(f"    Score    : {score:.4f}")
        print(f"    Snippet  : {node.text[:160]}...")
        print()
    print()


def _extract_places(source_nodes) -> list:
    """Turn source nodes into simple place dicts for the map visualizer."""
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
                "name":     m.get("name", "Unknown"),
                "category": m.get("category", ""),
                "address":  m.get("address", ""),
                "lat":      lat,
                "lon":      lon,
                "score":    node.score or 0.0,
            })
    return places




def cmd_ingest(args) -> None:
    """Fetch OpenStreetMap data, embed it, and store vectors in Milvus."""
    print("=" * 60)
    print("  GeoRAG - Data Ingestion")
    print("=" * 60)
    print("Fetching restaurants, cafes, attractions & shops in Karlsruhe...\n")

    pipeline = GeoRAGPipeline()
    pipeline.ingest_data(use_cache=not args.fresh)

    print("\nDone! You can now run:")
    print('  python main.py ask "Your question here"')


def cmd_ask(args) -> None:
    """Answer a user question using the RAG pipeline."""

    if args.location:
        try:
            user_location = _resolve_location_input(args.location)
            if not user_location:
                raise ValueError(
                    "Could not resolve location. Use 'lat,lon' or a valid address."
                )
        except ValueError as exc:
            print(f"Error: {exc}")
            sys.exit(1)
    else:
        typed = input(
            "Where are you at right now? Enter address or lat,lon "
            "(press Enter for default Marktplatz): "
        )
        resolved = _resolve_location_input(typed)
        if resolved:
            user_location = resolved
        else:
            user_location = config.DEFAULT_USER_LOCATION.copy()
            if typed.strip():
                print("Could not resolve that location. Falling back to Marktplatz, Karlsruhe.")

    question = args.question

    print("=" * 60)
    print("  GeoRAG - Query")
    print("=" * 60)
    print(f"  Question: {question}")
    print(f"  Location: {user_location['name']}")
    print("=" * 60)


    pipeline = GeoRAGPipeline()
    try:
        pipeline.load_existing_index()
    except FileNotFoundError as exc:
        print(f"\nError: {exc}")
        sys.exit(1)


    try:
        response, source_nodes = pipeline.query(question, user_location)
    except Exception as exc:
        print(f"\nQuery failed: {exc}")
        sys.exit(1)


    print("\n--- Original Query ---")
    print(question)

    if args.show_sources:
        print("\n--- Retrieved Context ---")
        _print_sources(source_nodes)

    print("\n--- Final Response ---")
    print(response)
    print()

    if not args.no_map:
        places = _extract_places(source_nodes)
        if places:
            map_path = args.map_output or os.path.join(config.OUTPUTS_DIR, "route_map.html")
            create_route_map(
                places=places,
                user_location=user_location,
                query=question,
                output_file=map_path,
            )
            print("--- Visualization of Best Routes ---")
            print(f"Route map: {os.path.abspath(map_path)}")
            print(f"Open the map in your browser:\n  {os.path.abspath(map_path)}\n")


    if args.save_output:
        output = {
            "query":    question,
            "location": user_location,
            "retrieved_contexts": [
                {
                    "name":     n.metadata.get("name", ""),
                    "category": n.metadata.get("category", ""),
                    "address":  n.metadata.get("address", ""),
                    "text":     n.text,
                    "score":    n.score,
                    "metadata": n.metadata,
                }
                for n in source_nodes
            ],
            "response": response,
        }
        with open(args.save_output, "w", encoding="utf-8") as fh:
            json.dump(output, fh, ensure_ascii=False, indent=2)
        print(f"Output saved to: {args.save_output}")



def cmd_chat(args) -> None:
    """Interactive chat mode — loads the model once, then loops for questions."""
    print("=" * 60)
    print("  GeoRAG - Interactive Chat")
    print("=" * 60)
    print("Model loading once — then ask as many questions as you like.")
    print("Type 'quit' or press Ctrl+C to exit.\n")

    if args.location:
        try:
            user_location = _parse_location(args.location)
        except ValueError as exc:
            print(f"Error: {exc}")
            sys.exit(1)
    else:
        user_location = config.DEFAULT_USER_LOCATION.copy()

    pipeline = GeoRAGPipeline()
    try:
        pipeline.load_existing_index()
    except FileNotFoundError as exc:
        print(f"\nError: {exc}")
        sys.exit(1)

    print(f"\nLocation: {user_location['name']}")
    print("Ready! Ask a question about Karlsruhe.\n")

    try:
        while True:
            try:
                question = input("You: ").strip()
            except EOFError:
                break

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break

            try:
                response, source_nodes = pipeline.query(question, user_location)
            except Exception as exc:
                print(f"Query failed: {exc}\n")
                continue

            print(f"\nGeoRAG: {response}\n")

            if not args.no_map:
                places = _extract_places(source_nodes)
                if places:
                    map_path = os.path.join(config.OUTPUTS_DIR, "route_map.html")
                    create_route_map(
                        places=places,
                        user_location=user_location,
                        query=question,
                        output_file=map_path,
                    )
                    print(f"Map saved: {os.path.abspath(map_path)}\n")

    except KeyboardInterrupt:
        print("\nBye!")


def cmd_check(args) -> None:
    """Print a checklist showing what is installed and what still needs setup."""
    print("=" * 60)
    print("  GeoRAG - Environment Check")
    print("=" * 60)

    # Python packages
    packages = {
        "llama_index.core":          "llama-index-core",
        "llama_index.embeddings.huggingface": "llama-index-embeddings-huggingface",
        "sentence_transformers":      "sentence-transformers",
        "folium":                     "folium",
        "requests":                   "requests",
        "torch":                      "torch",
    }
    print("\n[1] Python packages:")
    all_ok = True
    for module, pip_name in packages.items():
        try:
            __import__(module)
            print(f"    OK  {pip_name}")
        except ImportError:
            print(f"    MISSING  {pip_name}  ->  pip install {pip_name}")
            all_ok = False

    print("\n[2] Milvus vector database:")
    try:
        import requests as req
        r = req.get(f"{config.MILVUS_URI}/healthz", timeout=3)
        if r.status_code == 200:
            print(f"    OK  Milvus running at {config.MILVUS_URI}")
        else:
            print(f"    WARNING  Milvus responded with status {r.status_code}")
    except Exception:
        print(f"    NOT REACHABLE  ->  start Docker container: docker start milvus-standalone")

    print("\n[3] Ollama LLM:")
    try:
        import requests as req
        r = req.get(f"{config.LLM_BASE_URL}/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        if config.LLM_MODEL in " ".join(models):
            print(f"    OK  Ollama running, '{config.LLM_MODEL}' available")
        else:
            print(f"    WARNING  Ollama running, but '{config.LLM_MODEL}' not pulled")
            print(f"    Run:  ollama pull {config.LLM_MODEL}")
    except Exception:
        print(f"    NOT REACHABLE  ->  install from https://ollama.com, then:")
        print(f"    ollama serve")
        print(f"    ollama pull {config.LLM_MODEL}")

    print()
    if all_ok:
        print("All Python packages look good!")
    else:
        print("Fix the missing packages above and re-run this check.")




def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="georag",
        description="GeoRAG – Geo-aware RAG system for Karlsruhe, Germany",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1 – ingest OSM data (run once):
  python main.py ingest

  # Step 2 – ask questions:
  python main.py ask "Where can I find good sushi near Europaplatz?"
  python main.py ask "Where can I bring my 2 kids?" --show-sources
  python main.py ask "Where to buy a washing machine?" --location "49.0069,8.4037"

  # Check that everything is set up correctly:
  python main.py check
        """,
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")


    p_ingest = sub.add_parser("ingest", help="Fetch & index OSM data (run once)")
    p_ingest.add_argument(
        "--fresh", action="store_true",
        help="Re-fetch OSM data even if a cache exists",
    )


    p_ask = sub.add_parser("ask", help="Ask a question about places in Karlsruhe")
    p_ask.add_argument("question", help="Your natural-language question")
    p_ask.add_argument(
        "--location", "-l", metavar="LAT,LON",
        help="Your GPS location as 'lat,lon'  (default: Marktplatz Karlsruhe)",
    )
    p_ask.add_argument(
        "--show-sources", "-s", action="store_true",
        help="Print the retrieved context documents",
    )
    p_ask.add_argument(
        "--no-map", action="store_true",
        help="Skip map generation",
    )
    p_ask.add_argument(
        "--map-output", "-m", metavar="FILE",
        help="Where to save the HTML map  (default: outputs/route_map.html)",
    )
    p_ask.add_argument(
        "--save-output", "-o", metavar="FILE",
        help="Save full query+response as a JSON file",
    )

    sub.add_parser("check", help="Verify that all components are set up correctly")

    p_chat = sub.add_parser("chat", help="Interactive mode – ask multiple questions without reloading")
    p_chat.add_argument(
        "--location", "-l", metavar="LAT,LON",
        help="Your GPS location as 'lat,lon'  (default: Marktplatz Karlsruhe)",
    )
    p_chat.add_argument(
        "--no-map", action="store_true",
        help="Skip map generation after each answer",
    )

    return parser




def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "check":
        cmd_check(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
