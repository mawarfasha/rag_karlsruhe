# rag_pipeline.py
# Core RAG pipeline: loads OSM data, embeds it, stores in Milvus, and answers queries.
#
# Flow:
#   1. Embed each place description with all-MiniLM-L6-v2
#   2. Store vectors in Milvus (Docker standalone, no file limitations)
#   3. At query time: embed the question -> retrieve top-K places -> ask Mistral
#
# If Ollama isn't running the pipeline still works; it just returns the
# retrieved places formatted as plain text instead of an AI-generated sentence.

import os
import warnings
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

# suppress noisy startup warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from llama_index.core import (
    PromptTemplate,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore

import config
from document_builder import build_documents
from osm_fetcher import fetch_places, load_places, save_places



def _try_load_ollama_llm():
    """
    Try to connect to a locally running Ollama instance.
    Returns the LLM on success, or None if Ollama isn't available.
    """
    try:
        import requests as req
        req.get(f"{config.LLM_BASE_URL}/api/tags", timeout=3)

        from llama_index.llms.ollama import Ollama
        return Ollama(
            model=config.LLM_MODEL,
            base_url=config.LLM_BASE_URL,
            request_timeout=600.0,
            additional_kwargs={"num_predict": 256},
        )
    except Exception:
        return None




class GeoRAGPipeline:
    """
    End-to-end RAG pipeline for Karlsruhe geo queries.

    Usage (first time):
        pipeline = GeoRAGPipeline()
        pipeline.ingest_data()
        response, nodes = pipeline.query("Best sushi near Europaplatz?")

    Usage (after already ingested):
        pipeline = GeoRAGPipeline()
        pipeline.load_existing_index()
        response, nodes = pipeline.query("Where can I bring kids?")
    """

    def __init__(self):
        self.index: Optional[VectorStoreIndex] = None
        self.llm = None
        self._setup_models()

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Return great-circle distance in km between two GPS points."""
        radius = 6371.0
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        a = (
            math.sin(d_lat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(d_lon / 2) ** 2
        )
        return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def _rerank_by_location(
        self,
        nodes: List[NodeWithScore],
        user_location: Optional[Dict],
        top_k: int,
        semantic_weight: float = 0.7,
        distance_weight: float = 0.3,
    ) -> List[NodeWithScore]:
        """
        Combine semantic similarity and distance into a hybrid score.
        Higher semantic score and closer distance both improve ranking.
        """
        if not nodes:
            return []

        # If no user location is provided, keep semantic order only.
        if not user_location:
            return nodes[:top_k]

        scored_rows = []
        semantic_vals = [float(n.score or 0.0) for n in nodes]
        sem_min = min(semantic_vals)
        sem_max = max(semantic_vals)
        sem_span = (sem_max - sem_min) or 1.0

        distances = []
        for n in nodes:
            md = n.metadata or {}
            try:
                lat = float(md.get("lat") or 0.0)
                lon = float(md.get("lon") or 0.0)
            except (TypeError, ValueError):
                lat, lon = 0.0, 0.0

            if lat and lon:
                d_km = self._haversine_km(
                    float(user_location["lat"]),
                    float(user_location["lon"]),
                    lat,
                    lon,
                )
            else:
                d_km = 9999.0
            distances.append(d_km)

        max_dist = max(distances) or 1.0

        for node, dist_km in zip(nodes, distances):
            sem_raw = float(node.score or 0.0)
            sem_norm = (sem_raw - sem_min) / sem_span
            # Closer is better: 0 km -> 1.0, farthest -> 0.0
            dist_norm = max(0.0, 1.0 - (dist_km / max_dist))

            hybrid_score = (semantic_weight * sem_norm) + (distance_weight * dist_norm)

            node.metadata["distance_km"] = f"{dist_km:.3f}"
            node.metadata["semantic_score"] = f"{sem_raw:.6f}"
            node.metadata["hybrid_score"] = f"{hybrid_score:.6f}"
            scored_rows.append((hybrid_score, node))

        scored_rows.sort(key=lambda row: row[0], reverse=True)
        return [row[1] for row in scored_rows[:top_k]]

    @staticmethod
    def _build_context_from_nodes(nodes: List[NodeWithScore]) -> str:
        """Build a compact context block for LLM prompting from reranked nodes."""
        lines: List[str] = []
        for idx, node in enumerate(nodes, 1):
            md = node.metadata
            category = md.get("category", "").replace("_", " ").title()
            lines.append(f"{idx}. Name: {md.get('name', 'Unknown')}")
            lines.append(f"   Category: {category or 'N/A'}")
            lines.append(f"   Address: {md.get('address', 'N/A')}")
            if md.get("opening_hours"):
                lines.append(f"   Hours: {md.get('opening_hours')}")
            if md.get("website"):
                lines.append(f"   Website: {md.get('website')}")
            if md.get("distance_km"):
                lines.append(f"   Distance: {md.get('distance_km')} km")
            lines.append(f"   Snippet: {node.text[:280]}")
            lines.append("")
        return "\n".join(lines).strip()

    def _setup_models(self) -> None:
        """Load the embedding model and (optionally) the Mistral LLM via Ollama."""

        # ─ Embedding model ─────────────────────────────────────────────
        print("Loading embedding model (all-MiniLM-L6-v2)…")
        print("  (Downloads ~90 MB on first run – cached afterwards.)")
        embed_model = HuggingFaceEmbedding(
            model_name=config.EMBEDDING_MODEL,
            max_length=512,
        )

        # Apply settings globally so LlamaIndex uses them everywhere
        Settings.embed_model    = embed_model
        Settings.chunk_size     = config.CHUNK_SIZE
        Settings.chunk_overlap  = config.CHUNK_OVERLAP
        print("  Embedding model ready.")

        if os.getenv("GEORAG_DISABLE_OLLAMA", "0") == "1":
            print("  Ollama disabled via GEORAG_DISABLE_OLLAMA=1; using fallback mode.")
            self.llm = None
            return

        # ─ LLM (Ollama / Mistral) ───────────────────────────────────────
        print(f"Connecting to Ollama ({config.LLM_MODEL}) at {config.LLM_BASE_URL}…")
        self.llm = _try_load_ollama_llm()

        if self.llm:
            Settings.llm = self.llm
            print(f"  LLM ({config.LLM_MODEL}) ready.")
        else:
            print("  ⚠  Ollama not reachable – will return formatted context only.")
            print("     To enable AI responses: install Ollama, run 'ollama serve',")
            print(f"     then pull the model:  ollama pull {config.LLM_MODEL}")

    def ingest_data(self, use_cache: bool = True) -> None:
        """
        Fetch OSM places, embed them, and store in Milvus.
        Set use_cache=False to force a fresh fetch from OpenStreetMap.
        """
        os.makedirs(config.DATA_DIR, exist_ok=True)
        cache_file = os.path.join(config.DATA_DIR, "places.json")

        # ── Step 1: Get place data ──────────────────────────────────────
        places = load_places(cache_file) if use_cache else []

        if not places:
            places = fetch_places(config.KARLSRUHE_BBOX)
            if places:
                save_places(places, cache_file)

        if not places:
            print(
                "\nNo place data available. The Overpass API may be temporarily down.\n"
                "Wait a few minutes and try again, or check your internet connection."
            )
            return

        # ── Step 2: Build LlamaIndex documents ─────────────────────────
        documents = build_documents(places)

        # step 3: set up Milvus vector store
        print("Connecting to Milvus at", config.MILVUS_URI)
        vector_store = MilvusVectorStore(
            uri=config.MILVUS_URI,
            collection_name=config.COLLECTION_NAME,
            dim=config.EMBEDDING_DIM,
            overwrite=True,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # step 4: embed and store
        print(f"Generating embeddings for {len(documents)} places…")
        print("  (This may take a few minutes on first run.)")
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True,
        )
        print("Ingestion complete! All places stored in Milvus.")

    def load_existing_index(self) -> None:
        """Connect to existing Milvus collection without re-ingesting."""
        print("Connecting to Milvus index...")
        vector_store = MilvusVectorStore(
            uri=config.MILVUS_URI,
            collection_name=config.COLLECTION_NAME,
            dim=config.EMBEDDING_DIM,
        )
        self.index = VectorStoreIndex.from_vector_store(vector_store)
        print("Index loaded.")

    def _make_prompt_template(self, user_location: Optional[Dict]) -> PromptTemplate:
        """Build the prompt template sent to Mistral with location context."""
        if user_location:
            loc_line = (
                f"The user is currently at {user_location['name']} "
                f"(lat={user_location['lat']}, lon={user_location['lon']})."
            )
        else:
            loc_line = "The user has not specified their current location."

        template = (
            "You are a local guide for Karlsruhe, Germany.\n"
            f"{loc_line}\n\n"
            "Answer the question using ONLY the places listed below.\n"
            "Be concise. Use this format for each place:\n\n"
            "1. **Name** (Category)\n"
            "   - Address: ...\n"
            "   - Hours: ... (if available)\n"
            "   - Why: one sentence explaining why it fits the query\n\n"
            "List at most 3 places. No long paragraphs.\n\n"
            "Places:\n"
            "{context_str}\n\n"
            "Question: {query_str}\n\n"
            "Answer:"
        )
        return PromptTemplate(template)

    def query(
        self,
        question: str,
        user_location: Optional[Dict] = None,
    ) -> Tuple[str, List[NodeWithScore]]:
        """
        Run a query through the RAG pipeline.
        Returns (response_text, source_nodes) where source_nodes are the
        retrieved places with their metadata.
        """
        if self.index is None:
            raise RuntimeError(
                "No index loaded. Run ingest_data() or load_existing_index() first."
            )

        retrieval_pool = max(config.TOP_K * 4, 10)
        retriever = self.index.as_retriever(similarity_top_k=retrieval_pool)
        raw_nodes = retriever.retrieve(question)
        source_nodes = self._rerank_by_location(
            nodes=raw_nodes,
            user_location=user_location,
            top_k=config.TOP_K,
        )

        if self.llm is None:
            return self._format_fallback_response(question, source_nodes, user_location), source_nodes

        qa_template = self._make_prompt_template(user_location)
        context_str = self._build_context_from_nodes(source_nodes)
        prompt = qa_template.format(context_str=context_str, query_str=question)
        try:
            llm_response = self.llm.complete(prompt)
            response_text = getattr(llm_response, "text", str(llm_response))
            return str(response_text), source_nodes
        except Exception as exc:
            print(f"LLM generation failed, using fallback response: {exc}")
            return self._format_fallback_response(question, source_nodes, user_location), source_nodes


    def _format_fallback_response(
        self,
        question: str,
        nodes: List[NodeWithScore],
        user_location: Optional[Dict],
    ) -> str:
        """Format retrieved places as plain text when no LLM is available."""
        lines = [
            "Here are the most relevant places in Karlsruhe for your query:\n"
        ]
        for i, node in enumerate(nodes, 1):
            m = node.metadata
            cat = m.get("category", "").replace("_", " ").title()
            lines.append(f"  {i}. {m.get('name', 'Unknown')}  [{cat}]")
            lines.append(f"     Address : {m.get('address', 'N/A')}")
            if m.get("opening_hours"):
                lines.append(f"     Hours   : {m['opening_hours']}")
            if m.get("website"):
                lines.append(f"     Website : {m['website']}")
            lines.append("")

        if user_location:
            lines.append(
                f"Results are ranked by hybrid score (semantic relevance + distance) "
                f"(starting location: {user_location['name']})."
            )

        lines.extend([
            "",
            "─── AI response not available ───────────────────────────────────",
            "  Install Ollama to get natural-language answers:",
            "    1. Download from https://ollama.com",
            f"   2. Run:  ollama pull {config.LLM_MODEL}",
            "    3. Run:  ollama serve",
            "─────────────────────────────────────────────────────────────────",
        ])
        return "\n".join(lines)
