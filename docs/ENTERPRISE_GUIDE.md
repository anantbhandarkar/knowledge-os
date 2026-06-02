# Building Your Enterprise RAG with Knowledge OS

A grounded, multi-tenant retrieval engine that **verifies before it generates,
cites every claim, and abstains when the evidence is weak.** This guide takes you
from a zero-config offline trial to a production deployment with a real LLM, a
real vector store, real embeddings, and SSO-mapped access control.

---

## 1. Who this is for & the core guarantees

This guide is for platform/ML engineers standing up an internal RAG service over
sensitive corpora — policy manuals, contracts, runbooks, financial filings — where
a confidently-wrong answer is worse than no answer.

Knowledge OS makes four guarantees, enforced structurally (not by prompt etiquette):

| Guarantee | How it is enforced |
| --- | --- |
| **Verify before generate** | A verify gate (`app/verify.py:score_evidence` → `decide_gate`) scores retrieval, grounding, and citation confidence *before* the generator runs. |
| **Cite every claim** | The generator hands back `Citation` objects (`app/models.py`) that map each claim to a `chunk_id`, `section_path`, and `page_numbers`. The offline `ExtractiveGenerator` literally cannot hallucinate — it returns verbatim evidence sentences. |
| **Abstain on weak evidence** | `decide_gate` returns `PASS` / `REPAIR` / `ABSTAIN`. Insufficient evidence after repair loops yields a no-answer response, not a guess. |
| **Multi-tenant ACL in SQL, before retrieval** | Every chunk carries `tenant_id` + `acl_tags`. The store filters on these in the query (a `WHERE` clause in pgvector), so a tenant can never retrieve another tenant's vectors — the ACL is applied at the database, not in post-processing. |

The pipeline is a LangGraph of typed nodes threaded through a single
`PipelineState` (`app/models.py`), so every stage is inspectable and testable:
`plan → retrieve → fuse+rerank → verify → repair? → generate`.

---

## 2. Never feed raw documents to the pipeline

**Raw bytes never touch retrieval.** Every source file is first converted into a
**`CanonicalDoc`** — a JSON-serializable, structure-preserving intermediate
(`app/parsing/canonical.py`) — and that JSON is what gets chunked, embedded, and
stored. Persist it to `data/canonical/<doc>.json`.

```python
# app/parsing/canonical.py (the contract you build against)
class CanonicalBlock(BaseModel):
    type: BlockType                 # heading | paragraph | list | table | code | caption
    text: str = ""                  # plain-text rendering (tables: TSV-ish)
    level: int | None = None        # heading depth 1-6
    page: int | None = None         # source page, when known  -> citation provenance
    rows: list[list[str]] | None    # table cell grid, when type == "table"

class CanonicalSection(BaseModel):
    path: list[str]                 # e.g. ["Auth", "Token Management"] -> breadcrumb
    blocks: list[CanonicalBlock]

class CanonicalDoc(BaseModel):
    doc_title: str
    source_format: str              # pdf | docx | pptx | html | md | txt
    source_path: str | None
    sections: list[CanonicalSection]
    metadata: dict
```

### Why a canonical intermediate

- **Inspectable.** You can open the JSON and *see* what the parser extracted —
  bad tables and dropped headings are caught at ingest, not when an answer is wrong.
- **Version-controllable.** Diff `data/canonical/*.json` across parser upgrades to
  prove extraction quality changed, not retrieval.
- **Re-chunk without re-parsing.** Tuning chunk size or the contextual prefix
  re-reads JSON; it never re-runs Docling/OCR. Parse once, re-chunk forever.
- **Separates extraction quality from retrieval quality.** Two independent knobs:
  "did we read the PDF correctly?" vs. "did we retrieve the right chunk?"
- **Carries provenance.** `block.page` and `section.path` flow straight into
  `Evidence.page_numbers` / `section_path`, which is what makes citations real.

Tables are preserved as `rows: list[list[str]]` — **never flattened to prose** —
and heading hierarchy becomes the section `path`, so the chunker keeps a chunk
inside its heading and prefixes the breadcrumb.

### The parsing quality ladder

Use the cheapest rung that gives clean structure; climb only when you must. Do
format-aware parsing, not one-size-fits-all.

| Rung | Engine | What you get | When to use |
| --- | --- | --- | --- |
| **1 — Native** | `python-docx`, `python-pptx`, `pdfplumber`/`pypdf` | Cleanest paragraph styles & slide order (Office); cheap per-page text + ruled-line tables for clean digital PDFs | Default for `.docx` / `.pptx`; clean text-PDFs where ML is overkill |
| **2 — Pandoc** | `pandoc` universal conversion | Deterministic headings, lists, tables — no ML, fully reproducible | **Required** for the long tail: HTML, ODT, RTF, EPUB, and as the universal `PandocParser` fallback (`app/parsing/pandoc_parser.py`) |
| **3 — Docling** | `docling` (opt-in) | High-fidelity PDF: heading hierarchy, reading order, table **cell grids**, page numbers, OCR for scanned PDFs — maps ~1:1 onto `CanonicalSection`/`CanonicalBlock` | Messy/scanned/multi-column PDFs where rungs 1-2 lose structure |
| **4 — LLM refinement** | `runtime.GENERATOR.complete` (opt-in, gated) | Table-structure repair, OCR cleanup, heading re-inference on a rung-3 doc | **Only** when rung-3 confidence is low — never on the clean path |

`pandoc` is a required system dependency for the HTML/ODT/RTF/EPUB family. Docling
and the LLM pass are opt-in so the clean path stays fast and offline.

The dispatch is in `app/parsing/base.py`: `detect_format(path)` maps extension →
format, `get_parser(fmt)` returns a registered native parser or falls back to
`PandocParser`, and `parse_file(path)` is the single entry point that returns a
`CanonicalDoc`. The rung-4 refinement is import-safe: it calls
`getattr(runtime.GENERATOR, "complete", None)` and **no-ops offline** (the default
`ExtractiveGenerator` has no `.complete`), so tests stay deterministic.

```python
from app.parsing.base import parse_file

doc = parse_file("contracts/msa-2026.pdf")        # -> CanonicalDoc
Path("data/canonical/msa-2026.json").write_text(doc.model_dump_json(indent=2))
```

---

## 3. Step-by-step adoption

### 3.1 Offline trial — zero config

```bash
docker compose up            # KOS_MODE=offline, VECTOR_STORE=memory
curl localhost:8000/health   # {"status":"ok","mode":"offline","chunks":N}
```

On startup the API auto-ingests the bundled demo corpus, so a single `/chat` call
returns a real cited answer with no setup. Everything is deterministic: `HashEmbedder`,
`LexicalReranker`, `ExtractiveGenerator` — no keys, no DB, no network.

### 3.2 Persist vectors — pgvector

```bash
docker compose --profile pgvector up        # starts Postgres (pgvector/pgvector:pg16)
export VECTOR_STORE=pgvector
export DATABASE_URL=postgresql://kos:kos@localhost:5432/knowledge_os
```

`runtime.py` swaps `InMemoryVectorStore` → `PgVectorStore(DSN, dim=EMBEDDER.dim)`
with **no pipeline changes**. The store's `.setup()` runs in the API lifespan.

### 3.3 Real generation — KOS_MODE=api

```bash
export KOS_MODE=api
export KOS_PROVIDER=openrouter            # or: deepseek
export OPENROUTER_API_KEY=sk-or-...        # DEEPSEEK_API_KEY for deepseek
# optional: export KOS_MODEL=...           # provider default applied otherwise
```

This swaps `ExtractiveGenerator` → `OpenRouterGenerator` / `DeepSeekGenerator`.
The verify gate still runs first, so the LLM only ever sees evidence that already
passed grounding + citation checks.

### 3.4 Wire a real embedder — the key gap

**This is the single most important production step.** The default `HashEmbedder`
hashes tokens into a bag-of-words vector. It is stable and dependency-free — great
for demos — but **it is not semantic**: "car" and "automobile" land in different
buckets. Retrieval quality is capped until you replace it.

Provide any object with `name`, `dim`, and
`async def embed(texts, *, kind="passage") -> list[list[float]]`, then assign it in
`runtime.py` where `EMBEDDER = HashEmbedder()` is set today:

```python
# app/runtime.py
class E5Embedder:
    name = "intfloat/e5-large-v2"
    dim = 1024
    async def embed(self, texts, *, kind="passage"):
        prefix = "query: " if kind == "query" else "passage: "
        return _model.encode([prefix + t for t in texts]).tolist()

EMBEDDER = E5Embedder()        # pgvector dim is taken from EMBEDDER.dim
```

Use a managed embedding API or a local sentence-transformer. Keep the heavy import
inside the class, never at module top, so offline import stays safe. **Re-embed your
corpus from canonical JSON** after swapping (rung-2 of section 2 pays off here — no
re-parsing needed).

### 3.5 Ingest your corpus as canonical JSON

Parse each file to a `CanonicalDoc`, persist it, then feed its markdown rendering to
the ingest endpoint (which does heading-aware chunking + contextual-prefix embedding):

```python
from app.parsing.base import parse_file
from app.ingest import ingest_text

doc = parse_file("policies/travel.pdf")
await ingest_text(
    doc.to_markdown(),
    doc_title=doc.doc_title,
    tenant_id="acme",
    acl_tags=["hr", "policy"],
)
```

Or over HTTP (`POST /ingest`):

```bash
curl -s localhost:8000/ingest -H 'content-type: application/json' -d '{
  "text": "# Travel Policy\n\n## Reimbursement\nEmployees may expense economy airfare...",
  "doc_title": "Travel Policy",
  "tenant_id": "acme",
  "acl_tags": ["hr", "policy"]
}'
# -> {"ingested_chunks": 3, "doc_title": "Travel Policy"}
```

> The canonical JSON is your source of truth; `to_markdown()` is the lossless bridge
> into the chunker, preserving headings (as `section_path`) and tables.

### 3.6 Map SSO/RBAC roles to tenant + ACL tags

At your API edge, translate the identity-provider claims (Okta/Entra groups, JWT
roles) into a `tenant_id` and `acl_tags` list, then pass them on every `/chat` call.
Use `app/security/guardrails.map_roles_to_acl` as the single mapping point:

```python
from app.security.guardrails import map_roles_to_acl

tenant_id, acl_tags = map_roles_to_acl(claims["org"], claims["roles"])
# pass tenant_id + acl_tags into ChatRequest; the store filters on them in SQL
```

Never let the client supply its own `acl_tags` — derive them server-side from the
verified token. The store applies them in the retrieval query, so access control
happens *before* a single vector is compared.

### 3.7 Tune the two decision functions

Both are intentionally yours to own — they set the system's latency/cost/quality and
hallucination/refusal balance:

- **`app/routing.py:classify_lane`** — Express (fast, deterministic, ~60-70% of
  traffic) vs. Deep (constrained-agentic, decomposition + repair). Lean *safe* on
  uncertain queries; gate on `_COMPLEX_SIGNALS` and multi-clause shape, not the
  intent label alone.
- **`app/verify.py:decide_gate`** — `PASS` / `REPAIR` / `ABSTAIN` from the three
  confidence scores + `contradiction_detected`. Decide whether low
  `citation_confidence` vetoes a PASS even when grounding is high (strict = safer),
  and what to do when `repair_count` hits `MAX_REPAIR_ITERATIONS`.

### 3.8 Evaluate

```bash
python -m app.evals          # drives the compiled graph over a golden set
```

`app/evals/faithfulness.py` is a pure, offline grounding metric: the fraction of the
answer's content words present in the retrieved evidence. Run it before and after any
embedder/parser/gate change — it is your regression net.

### 3.9 Deploy

Build the image (`Dockerfile`), set the env block from the matrix below, point
`DATABASE_URL` at managed Postgres-with-pgvector, set `KOS_MODE=api` + provider key,
and ship your real embedder. Health-check `GET /health`.

---

## 4. Worked end-to-end example

```bash
# 1. Bring up offline + ask the bundled corpus a question
docker compose up -d
curl -s localhost:8000/chat -H 'content-type: application/json' -d '{
  "query": "What is our policy on economy airfare reimbursement?",
  "tenant_id": "acme",
  "acl_tags": ["hr", "policy"]
}' | jq
```

Sample cited response:

```json
{
  "answer": "Employees may expense economy airfare booked at least 14 days in advance. [Travel Policy]",
  "lane": "express",
  "verification_result": "pass",
  "retrieval_confidence": 0.82,
  "citations": [
    {
      "claim_text": "Employees may expense economy airfare booked at least 14 days in advance.",
      "source_doc_title": "Travel Policy",
      "section_path": ["Reimbursement"],
      "page_numbers": [4],
      "evidence_span": "Employees may expense economy airfare booked at least 14 days in advance.",
      "chunk_id": "a1b2c3d4-...",
      "confidence": 0.82
    }
  ]
}
```

A weak-evidence query instead returns `"verification_result": "abstain"` and a
no-answer message — the gate refused rather than guessed.

```bash
# 2. Inspect raw retrieval (pre-generation) for debugging
curl -s localhost:8000/search -H 'content-type: application/json' -d '{
  "query": "airfare reimbursement", "tenant_id": "acme", "acl_tags": ["hr","policy"]
}' | jq '.evidence[] | {doc_title, section_path, rerank_score}'
```

---

## 5. Provider / store swap matrix

Everything is env-driven via `app/runtime.py`; no pipeline code changes between modes.

| Concern | Env var | Values | Default | Notes |
| --- | --- | --- | --- | --- |
| Run mode | `KOS_MODE` | `offline` \| `api` | `offline` | `api` enables a real LLM generator |
| LLM provider | `KOS_PROVIDER` | `openrouter` \| `deepseek` | `openrouter` | OpenAI-compatible under the hood |
| Model | `KOS_MODEL` | provider model id | provider default | e.g. `deepseek-chat` |
| Provider key | `OPENROUTER_API_KEY` / `DEEPSEEK_API_KEY` | secret | — | required when `KOS_MODE=api` |
| Vector store | `VECTOR_STORE` | `memory` \| `pgvector` \| `turbovec` | `memory` | `memory` is demo-only (non-persistent) |
| Database | `DATABASE_URL` | Postgres DSN | local kos DSN | needed for pgvector/turbovec |
| Embedder | *(code)* | swap `runtime.EMBEDDER` | `HashEmbedder` | **must replace for production** (§3.4) |
| Reranker | *(code)* | swap `runtime.RERANKER` | `LexicalReranker` | cross-encoder for production |

`turbovec` is a two-step quantized scale-out store for large corpora; `pgvector` is
the production default.

---

## 6. Security

Guardrails live in `app/security/guardrails.py` — pure, offline, no LLM/network
calls, safe to import anywhere.

- **PII detection / redaction.** Detect emails, phone numbers, national IDs, and
  card numbers in both inbound queries and outbound answers; redact before logging
  and before any text leaves the trust boundary. Run redaction on canonical JSON at
  ingest so sensitive spans never reach the vector store in the clear.
- **Prompt-injection detection.** Heuristics flag instruction-override patterns
  ("ignore previous instructions", embedded system prompts, tool-call lures) in both
  the user query and in *retrieved evidence* — a poisoned document is an injection
  vector. Flagged content is stripped or routed to abstain.
- **Data-leakage prevention.**
  - ACL is enforced in the **store query** (`tenant_id` + `acl_tags` `WHERE` clause),
    so cross-tenant vectors are never even scored.
  - `acl_tags` are derived server-side from the verified SSO token
    (`map_roles_to_acl`), never trusted from the client.
  - The extractive default returns only verbatim evidence the user is already cleared
    to see — it cannot synthesize content from out-of-scope material.
  - Redact PII from request/response logs; never log raw evidence text at INFO.

---

## 7. Honest limitations & roadmap

**Today**

- `HashEmbedder` is non-semantic — a deliberate offline default, **not** a production
  embedder (§3.4). This is the highest-leverage gap to close.
- `classify_lane` and `decide_gate` ship as tunable starters; the policy is yours.
- `LexicalReranker` is lexical overlap, not a trained cross-encoder.
- Rungs 3 (Docling) and 4 (LLM refinement) of the parsing ladder are opt-in and
  require the optional dependencies installed.

**Roadmap**

- **No frontend yet** — the surface is the JSON API (`/chat`, `/search`, `/ingest`,
  `/health`). A NotebookLM-style citation UI is pending.
- **ColPali** — vision-native retrieval over document page images (figures, scanned
  layouts) without an OCR round-trip.
- **GraphRAG** — entity/relation graph retrieval for multi-hop analytical queries
  (`RetrievalMode.GRAPH` is reserved in `app/models.py`).
- **Kubernetes** — production manifests, autoscaling, and managed-Postgres wiring
  beyond the single-node `docker compose` story.
```
