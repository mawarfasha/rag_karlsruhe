# GeoRAG Development Journal

A step-by-step log of building the GeoRAG system

---

## Step 1 — Initial Setup & Project Structure

Built the full project from scratch with the following files:

- `config.py` — all settings in one place (bbox, model names, ports, etc.)
- `osm_fetcher.py` — fetches real place data from OpenStreetMap via the free Overpass API
- `document_builder.py` — converts raw OSM place dicts into LlamaIndex `Document` objects with natural-language descriptions
- `rag_pipeline.py` — core RAG pipeline (embed → vector store → Mistral)
- `route_visualizer.py` — generates interactive HTML route maps using Folium
- `main.py` — CLI with three commands: `ingest`, `ask`, `check`
- `examples/example_queries.py` — runs 5 demo queries automatically
- `requirements.txt` and `README.md`

**Technology choices:**
- Embedding model: `all-MiniLM-L6-v2` (384-dim, ~90 MB, free via HuggingFace)
- LLM: Mistral 7B via Ollama (local, no API key needed)
- Vector database: Milvus (as specified in the task)
- RAG framework: LlamaIndex
- Map rendering: Folium (self-contained HTML output)
- OSM data: Overpass API, Karlsruhe bounding box `48.9762,8.3340,49.0597,8.4930`

---

## Step 2 — First Run Attempt: Overpass API Timeout (504)

**Problem:** Running `python main.py ingest` failed with:
```
Overpass API request failed: 504 Server Error: Gateway Timeout
RuntimeError: No place data available.
```

**Cause:** The public Overpass API (`overpass-api.de`) was temporarily overloaded — this is a free shared server with no SLA.

**Fix:**
- Added retry logic to `osm_fetcher.py` with 3 mirror servers:
  1. `https://overpass-api.de/api/interpreter` (primary)
  2. `https://overpass.kumi.systems/api/interpreter`
  3. `https://maps.mail.ru/osm/tools/overpass/api/interpreter`
- If all mirrors fail, the error is printed gracefully instead of crashing
- Also changed the `RuntimeError` crash in `rag_pipeline.py` to a soft warning + return

---

## Step 3 — Milvus Lite Not Available on Windows

**Problem:** After fixing the Overpass issue, the next crash was:
```
ModuleNotFoundError: No module named 'milvus_lite'
ConnectionConfigException: milvus-lite is required for local database connections.
Please install it with: pip install pymilvus[milvus_lite]
```

Running `pip install milvus-lite` returned:
```
ERROR: Could not find a version that satisfies the requirement milvus-lite (from versions: none)
ERROR: No matching distribution found for milvus-lite
```

**Cause:** `milvus-lite` is a lightweight embedded version of Milvus (similar to SQLite for databases). It uses native C++ binaries under the hood, and the maintainers have **never published a Windows build on PyPI**. It is only available for Linux and macOS.

**Disadvantage of this limitation:**
- Can't use the task-specified Milvus vector database without extra infrastructure on Windows
- The only Windows option is running Milvus as a full Docker container (more setup required)

---

## Step 4 — Temporary Workaround: LlamaIndex SimpleVectorStore

**To keep the system working while deciding on the Milvus approach**, switched the vector backend to LlamaIndex's built-in `SimpleVectorStore`:

- No extra packages needed — included in `llama-index-core`
- Persists vectors to disk as JSON files in the `index/` folder
- Supports the same semantic similarity search
- Works on all platforms including Windows

**Files saved in `index/`:**
- `default__vector_store.json` — all 384-dim embedding vectors for 1661 places
- `docstore.json` — original document text and metadata
- `index_store.json` — index pointers

**Result:** System fully functional — ingestion worked, queries returned results, Mistral generated responses.

**Gap:** The task explicitly requires Milvus for vector storage, so this was only a temporary stand-in.

---

## Step 5 — Moving to Milvus via Docker

**Decision:** Install Docker Desktop for Windows and run Milvus as a standalone container — the officially supported path for Windows users per the Milvus documentation.

**Steps taken:**

1. Downloaded and installed **Docker Desktop for Windows** from docker.com
2. Docker Desktop automatically enables **WSL2** (Windows Subsystem for Linux 2) during setup
3. Started the Milvus standalone container:

```powershell
docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:v2.4.0 milvus run standalone
```

Note: The first attempt used `standalone` as the Docker CMD, which failed with exit code 127 (command not found). The correct command is `milvus run standalone`.

4. Verified the container was running:
```
CONTAINER ID   IMAGE                    STATUS         PORTS
0e0fed8eb1c9   milvusdb/milvus:v2.4.0   Up 12 seconds  0.0.0.0:19530->19530/tcp
```

**Updated code for Docker Milvus:**
- `config.py`: replaced `MILVUS_DB` file path with `MILVUS_URI = "http://localhost:19530"`
- `rag_pipeline.py`: switched back to `MilvusVectorStore` using the server URI
- `main.py` check command: now pings Milvus health endpoint to confirm it's reachable

---

## Step 6 — Ollama Setup

**Problem:** `ollama` command not found in PowerShell even after installing from the script:
```powershell
irm https://ollama.com/install.ps1 | iex
```

**Cause:** Ollama installed to `%LOCALAPPDATA%\Programs\Ollama\` which wasn't in the PATH for existing terminal sessions.

**Workaround:** Used the full executable path:
```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull mistral
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" run mistral "test"
```

**Note:** When `ollama serve` failed with `bind: Only one usage of each socket address`, that actually meant Ollama was **already running** as a background service (the installer starts it automatically). No need to run `ollama serve` manually.

---

## Final Architecture

```
User query
    │
    ▼
[ Embedding model ]  ←  all-MiniLM-L6-v2 (HuggingFace)
    │
    ▼
[ Milvus (Docker) ]  ←  semantic similarity search over 1661 Karlsruhe places
    │  port 19530
    ▼
[ Top-K retrieved places ]
    │
    ▼
[ Mistral 7B (Ollama) ]  ←  local LLM, generates natural-language answer
    │  port 11434
    ▼
[ Response + Folium HTML map ]
```

**Data source:** OpenStreetMap via Overpass API — restaurants, cafes, attractions, shopping, historic sites, parks for Karlsruhe, Germany (1661 places fetched and indexed).

---

## Step 7 — Milvus Empty After Switching to Docker

**Problem:** After switching from SimpleVectorStore to Docker Milvus, all queries returned the same generic answer — just repeating the default location "Marktplatz, Karlsruhe" with no actual place recommendations.

**Cause:** The Docker Milvus container was fresh and empty. The old `index/` folder data (from SimpleVectorStore) is a completely different format — it doesn't transfer to Milvus. Ingestion had to be re-run specifically targeting the new Milvus backend.

**Fix:** Run `python main.py ingest` again after switching to Docker Milvus. This re-embeds all 1661 places and stores them in the running Milvus container.

---

## Step 8 — Ollama Connectivity Check Was Slow

**Problem:** Every `python main.py ask` call was slow even before Mistral started generating — the startup phase "Connecting to Ollama..." was taking a long time.

**Cause:** The original connectivity check called `llm.complete("Hi")` — which actually ran the full Mistral model just to verify Ollama was reachable. That's an expensive way to do a health check.

**Fix:** Replaced with a simple HTTP GET to `/api/tags` (Ollama's REST endpoint) with a 3-second timeout. Just checks if the server is up, no model inference needed. Startup is now near-instant.

---

## Step 9 — Added Interactive Chat Mode

**Problem:** Every `python main.py ask` command reloads the embedding model (~90 MB) and reconnects to everything from scratch. For multiple questions this means waiting for the full startup each time.

**Fix:** Added `python main.py chat` — an interactive loop that loads the model once and keeps it in memory. You type questions one after another and only the LLM generation time applies between them (no reload).

```powershell
python main.py chat
# Type questions, press Enter after each
# Type 'quit' to exit
```

---

## Step 10 — Mistral Too Slow on CPU

**Problem:** Mistral 7B takes **2-5 minutes per response** on CPU. With no GPU available, every query was painfully slow and sometimes appeared to hang.

**Why it's slow:** Mistral has 7 billion parameters. On CPU it processes roughly 10-20 tokens per second. A typical response is 150-300 tokens, hence 2-5 minutes.

**Trial:** Switched to `tinyllama` — a much smaller model (~600 MB vs ~4 GB):
```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull tinyllama
```
Changed `config.py`:
```python
LLM_MODEL = "tinyllama"
```

**Result:** Responses in ~15-30 seconds. However, TinyLlama hallucinates more — for example it returned "Karl-Marx-Stadt" as the city name instead of "Karlsruhe", even though the coordinates were correct.

**Trade-off:**
| Model | Speed | Quality | Use case |
|-------|-------|---------|----------|
| `mistral` | 2-5 min | High, accurate | Demo/submission |
| `tinyllama` | 15-30 sec | Lower, occasional hallucinations | Quick testing |

**Decision for submission:** Switch back to `mistral` for the final output quality. Run example queries and leave them to complete — results are saved automatically to `outputs/`.

---

## Running the System

```powershell
# 1. Start Milvus (once per machine restart)
docker start milvus-standalone

# 2. Activate the virtual environment
cd "c:\Users\boopo\OneDrive\Desktop\New folder\georag"
venv\Scripts\activate

# 3. Ingest data (only needed once, or when refreshing OSM data)
python main.py ingest

# 4. Ask questions
python main.py ask "Where can I get sushi near Europaplatz?"
python main.py ask "Where can I bring my 2 kids in Karlsruhe?"
python main.py ask "Where can I buy a washing machine?"
python main.py ask "Find me a quiet cafe to work in"
python main.py ask "Show me historic sights in Karlsruhe"

# 5. Run all 5 example queries at once
python examples/example_queries.py

# 6. Check system health
python main.py check
```

---

## Step 11 — Self-Judgement and Improvement Plan

This step records the system quality judgement after end-to-end testing.

### Overall judgement

- **Status:** System works end-to-end (ingest -> retrieve -> LLM answer -> map output)
- **Architecture:** Modular structure (`osm_fetcher`, `document_builder`, `rag_pipeline`, `route_visualizer`, `main`)
- **Main gap:** Location was not yet used in retrieval ranking, only in prompt/map context
- **Writing quality:** Mostly clear, with some inconsistent/corrupted docs

### Key findings

1. **Requirement alignment gap (high priority)**
    - Retrieval used semantic similarity only.
    - User location was injected into prompt text, but not used to re-rank candidates.
    - Route recommendations were not fully optimized by both relevance and distance.

2. **Documentation mismatch (high priority)**
    - `README.md` still stated "Milvus Lite (local file, no server needed)".
    - Actual runtime used Docker Milvus at `http://localhost:19530`.

3. **Dependency stability risk (high priority)**
    - Broad `>=` constraints existed in `requirements.txt`.
    - This created compatibility risk across `llama-index`, `sentence-transformers`, and `transformers` updates.

4. **Task description file quality issue (medium priority)**
    - Accidental command text injection/corruption existed in `Georag-system-task-description.md`.

5. **Map HTML safety hardening needed (medium priority)**
    - Query/place text was directly interpolated into map HTML without escaping.
    - Risk is low for local use, but still worth hardening.

6. **Compose warning cleanup (low priority)**
    - `docker-compose.yml` still included an obsolete `version` key.

### Runtime output judgement

- Successful final run:
  - Milvus containers running
  - Embedding model loaded
  - Ollama Mistral connected
  - Index queried successfully
  - Route map generated to `outputs/route_map.html`
- Answer quality is acceptable for demo use, but still shallow for "best" ranking without explicit multi-factor ranking.

### Improvement backlog (next iteration)

1. Implement hybrid scoring in `rag_pipeline.py`:
    - `final_score = alpha * semantic_score + beta * distance_score`
    - apply after retrieving top-K candidates
2. Update `README.md` to match actual Docker Milvus architecture.
3. Pin tested dependency versions (or add a lock file) for reproducibility.
4. Clean and restore `Georag-system-task-description.md` formatting.
5. Escape dynamic HTML in `route_visualizer.py` (query and popup fields).
6. Remove obsolete `version` in `docker-compose.yml`.

### Personal note

The core system works and demonstrates the requested stack well. The next phase should focus on **ranking correctness** and **documentation/reproducibility quality** to strengthen the submission.

---

## Step 12 — Tested Output Log (Runtime Evidence)

This step logs actual command runs and observed outputs for traceability.

### Test A — First `ask` attempt after startup (failed)

**Commands executed:**

```powershell
cd "c:\Users\boopo\OneDrive\Desktop\New folder\georag"
docker compose up -d
venv\Scripts\activate
python main.py ask "where is best arab cuisine in Karlsruhe?"
```

**Observed output summary:**

- Docker warning: compose file attribute `version` is obsolete.
- Milvus containers reported as running.
- Query started and embedding model loading began.
- Execution failed with:

```text
TypeError: SentenceTransformer.__init__() got an unexpected keyword argument 'show_progress'
```

**Interpretation:** dependency/API mismatch in the embedding stack on that run.

### Test B — Immediate retry (successful)

**Command executed:**

```powershell
python main.py ask "where is best arab cuisine in Karlsruhe?"
```

**Observed output summary:**

- Embedding model loaded successfully (`all-MiniLM-L6-v2`).
- Ollama (`mistral`) connected successfully.
- Milvus index loaded.
- Query completed without crash.

### Test C — Full repeated startup + query (successful)

**Commands executed:**

```powershell
cd "c:\Users\boopo\OneDrive\Desktop\New folder\georag"
docker compose up -d
venv\Scripts\activate
python main.py ask "where is best arab cuisine in Karlsruhe?"
```

**Observed output summary:**

- Same compose `version` warning appeared.
- Milvus containers running.
- Embedding model, Ollama, and index load all succeeded.
- Final generated answer:

```text
Arabischer Shwarma, located at Kaiserstraße 65, Karlsruhe, serves shawarma cuisine and is the best option for Arabic food in Karlsruhe. Opening hours: Mo-Fr 11:00-24:00, Sa,Su,PH 12:00-24:00.
```

- Map output generated:

```text
Route map saved -> ...\georag\outputs\route_map.html
```

### Evidence-based judgement from tests

- End-to-end path verified working (query -> retrieve -> generate -> map).
- Runtime remained sensitive to package version combinations.
- Remaining non-blocking warning at the time: obsolete compose `version` key.

---

## Step 13 — Fix Verification Checklist

This checklist was used after each code change to confirm measurable improvement.

| ID | Improvement Item | Verification Method | Status | Notes |
|----|------------------|---------------------|--------|-------|
| F1 | Add location-aware reranking in `rag_pipeline.py` | Run same query with two different user locations and confirm top results/routing change logically by distance + relevance | PASS | Verified with `--show-sources`: ordering differs by location (e.g., rank 3/4 swapped between two locations). |
| F2 | Align `README.md` with Docker Milvus architecture | Compare README setup text against actual runtime commands and ports used in `config.py` and `docker-compose.yml` | PASS | README now documents Docker Milvus (`docker compose up -d`, `http://localhost:19530`) and hybrid reranking note. |
| F3 | Pin tested dependency versions | Fresh `venv` install from `requirements.txt`, then run `python main.py check`, `python main.py ingest`, and one `ask` query successfully | PASS | Installed pinned set in `.venv`, `check` passed, `ingest` passed (1661 docs), and `ask` passed in fallback mode. |
| F4 | Clean corrupted `Georag-system-task-description.md` | Manually review markdown for accidental terminal text and verify clean render in editor preview | PASS | Removed accidental command injection text in Environment Setup section. |
| F5 | Escape dynamic HTML in `route_visualizer.py` | Insert a query containing `<script>`-like characters and verify map HTML displays escaped text (no script execution) | PASS | Verified in `outputs/route_map.html`: query appears escaped as `&lt;script&gt;...&lt;/script&gt;`. |
| F6 | Remove obsolete compose `version` field | Run `docker compose up -d` and confirm warning about obsolete `version` no longer appears | PASS | Compose starts containers without obsolete `version` warning. |

### Suggested test command set (post-fix)

```powershell
cd "c:\Users\boopo\OneDrive\Desktop\New folder\georag"
docker compose up -d
python main.py check
python main.py ask "where is best arab cuisine in Karlsruhe?"
python main.py ask "where is best arab cuisine in Karlsruhe?" --location "49.0000,8.4500"
```

### Completion rule

Mark an item as **PASS** only if the verification method is observed in terminal output or produced files, not only by code inspection.

### Verification command log (completed)

```powershell
docker compose up -d
python main.py check
python main.py ingest
python main.py ask "where is best arab cuisine in Karlsruhe?" --show-sources --no-map
python main.py ask "where is best arab cuisine in Karlsruhe?" --location "49.0000,8.4500" --show-sources --no-map
python main.py ask "test <script>alert(1)</script> arab food" --location "49.0000,8.4500"
```

Note: due local RAM limits with `mistral`, fallback mode was used for reliable verification by setting `GEORAG_DISABLE_OLLAMA=1`.

---

## Step 14 — Consolidated Findings and Problems by Step

This step summarizes **every implementation step** with findings, problems encountered, fixes applied, and final status.

### Step-by-step summary

| Step | Key findings | Problems encountered | Resolution applied | Final status |
|------|--------------|----------------------|--------------------|--------------|
| 1 | Modular project structure established across ingestion, retrieval, CLI, and visualization. | No blocking issue. | Baseline architecture and files created. | Completed |
| 2 | Overpass API integration worked, but depended on public endpoint reliability. | `504 Gateway Timeout` on the primary Overpass API. | Added retry/failover across multiple mirrors and graceful failure handling. | Completed |
| 3 | Milvus Lite was not viable on Windows. | `milvus-lite` unavailable for Windows/PyPI. | Moved to Docker-based Milvus standalone. | Completed |
| 4 | Temporary vector backend kept progress moving. | Task requirement mismatch with a non-Milvus backend. | Used `SimpleVectorStore` only as an interim workaround. | Closed (interim only) |
| 5 | Docker Milvus became the stable architecture for Windows. | First container command (`standalone`) failed. | Corrected to `milvus run standalone` and updated config/pipeline URI usage. | Completed |
| 6 | Ollama runtime was available, but shell PATH was inconsistent. | `ollama` command not found in one shell session. | Used full executable path and confirmed service already running. | Completed |
| 7 | Retrieval quality depends on Milvus collection contents. | Poor/generic answers from empty Milvus collection after backend switch. | Re-ran ingestion into Milvus. | Completed |
| 8 | Startup latency reduced from heavy LLM checks. | Expensive model inference in connectivity checks. | Switched checks to lightweight `/api/tags` HTTP ping. | Completed |
| 9 | Multi-query usability improved. | `ask` mode reloaded stack every run. | Added `chat` mode with persistent pipeline state. | Completed |
| 10 | CPU-only Mistral quality was good but slow. | 2-5 minute response latency during testing. | Evaluated TinyLlama for speed and kept Mistral for submission quality. | Completed with trade-off |
| 11 | Architecture/writing review surfaced gaps. | Missing location-aware reranking, README mismatch, weak pinning, doc corruption, missing HTML escaping, and compose warning. | Created a prioritized backlog (F1-F6). | Completed |
| 12 | Runtime evidence captured for failure and success paths. | Version/API mismatch in one run. | Logged failed and successful runs with command/output evidence. | Completed |
| 13 | All planned fixes were implemented and verified. | Verification blockers: wrong Python env, invalid pinned versions, Ollama RAM limits, and one policy-blocked cleanup command. | Re-pinned installable versions, installed in `.venv`, added fallback switch and LLM-error fallback, validated map escaping and compose warning removal, and updated checklist to PASS. | Completed |

### Additional findings from final verification cycle

- A transient code corruption artifact was detected in `rag_pipeline.py` (garbled inline comment text).
- The artifact was removed and the pipeline was re-tested successfully.
- Milvus health check may report `404` for `/healthz` in this setup even when DB ingest/query operations work.

### Remaining non-blocking risks

1. Local RAM constraints can still prevent Mistral generation on some runs.
2. Milvus health check endpoint behavior is version/deployment dependent and may show warning noise.

### Mitigations in place

1. Fallback mode added via `GEORAG_DISABLE_OLLAMA=1`.
2. Query flow catches LLM runtime failures and returns ranked fallback responses instead of crashing.
3. Verification evidence and PASS criteria documented in this journal.

---

## Final Reflection

This project runs end-to-end and covers the main task requirements: OSM ingestion, Milvus retrieval, answer generation, and route map output.

The strongest part is the integration work. The pipeline is modular, easy to run from CLI, and has fallback behavior when the LLM is not available.

There are still realistic limitations:

1. Response time can be high on CPU-only setups when using Mistral.
2. Retrieval quality is good but still sensitive to query wording and metadata completeness.
3. Environment setup can be fragile across machines without strict dependency and runtime checks.

If more time was available, the next improvements would be:

1. Add simple quantitative evaluation (for example, precision@k checks on curated queries).
2. Improve ranking calibration between semantic and distance scores for different query types.
3. Expand automated tests for CLI flows and edge-case location inputs.


---

## GitHub References

- LlamaIndex: https://github.com/run-llama/llama_index
- Milvus: https://github.com/milvus-io/milvus
- PyMilvus (Python SDK): https://github.com/milvus-io/pymilvus
- Ollama: https://github.com/ollama/ollama
- Sentence Transformers: https://github.com/UKPLab/sentence-transformers
- Hugging Face Transformers: https://github.com/huggingface/transformers
- PyTorch: https://github.com/pytorch/pytorch
- Folium: https://github.com/python-visualization/folium
- OpenStreetMap Overpass API: https://github.com/drolbr/Overpass-API
- https://github.com/olaflaitinen/citysense
