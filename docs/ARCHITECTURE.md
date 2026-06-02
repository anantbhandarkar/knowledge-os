# Knowledge OS — Architecture (spec v3, June 2026)

Diagrams for the Enterprise Knowledge OS. Built against
[spec v3](../../knowledge-os-spec-2026-v3.md). v3 = v2 + a pluggable **VectorStore**
abstraction (PgVectorStore default / TurboVecStore scale-out). All diagrams are
Mermaid — they render on GitHub and in most Markdown viewers.

---

## 1. System architecture (layered)

The whole system in one view: offline corpus engineering feeds the stores; the
online query path forks into Express/Deep lanes; cross-cutting concerns wrap both.

```mermaid
flowchart TB
    subgraph CLIENT["Client"]
        UI["React UI / &lt;rag-chat&gt; Web Component<br/>clickable inline citation highlights"]
    end

    subgraph API["API — FastAPI · async · SSE / WebSocket"]
        EP["/ingest · /chat · /search · /reindex · /feedback · /health · /metrics"]
    end

    subgraph OFFLINE["Stage 0-1 · Corpus Engineering (build-time)"]
        direction TB
        PARSE["Parse<br/>Docling · Unstructured · OCR · python-docx"]
        CANON["Canonicalize + Dedup<br/>SHA-256 · MinHash/LSH · version lineage"]
        CHUNK["Structure-aware Chunking<br/>+ Contextual-Embedding prefix"]
        REP["Knowledge Representations<br/>summaries · Q&amp;A · anchors · entities · graph"]
        EMB["Embed<br/>OpenAI · Cohere · Voyage · BGE-M3 · Jina"]
        PARSE --> CANON --> CHUNK --> REP --> EMB
    end

    subgraph ONLINE["Stage 2-7 · Query Pipeline (online) — LangGraph state machine"]
        direction TB
        CLASS["Stage 2 · Classify Intent + Route"]
        CLASS -->|"Express · 60-70%"| EXP["Express Lane<br/>hybrid retrieve → rerank → light verify"]
        CLASS -->|"Deep"| PLAN["Deep Lane · Plan<br/>decompose · pick modes · tools"]
        PLAN --> DRET["Multi-mode Retrieve<br/>vector · BM25 · graph · SQL · Q&amp;A · summary"]
        DRET --> FUSE["Stage 4 · Fuse (RRF/DBSF)<br/>+ Cross-Encoder Rerank"]
        EXP --> ASM
        FUSE --> ASM
        ASM["Stage 5 · Context Assembly<br/>chunk assembly vs full-document stuffing"]
        ASM --> VER["Stage 6 · Verify<br/>grounding · contradiction · confidence"]
        VER -->|"pass"| GEN["Stage 7 · Generate + Cite<br/>closed-book examiner · span citations"]
        VER -->|"weak · Deep only · max 3"| REP2["Repair Loop<br/>reformulate · expand · escalate"]
        REP2 --> DRET
        VER -->|"insufficient"| ABS["Abstain · No-Answer"]
    end

    subgraph STORE["Storage + Retrieval — VectorStore abstraction (v3)"]
        direction TB
        VS{{"VectorStore interface<br/>add() · search() · delete()"}}
        PG[("PgVectorStore — DEFAULT<br/>pgvector HNSW + tsvector BM25<br/>+ metadata + ACL · one SQL · &lt;2M chunks")]
        TV[("TurboVecStore — SCALE-OUT<br/>TurboVec quantized index (8x less RAM)<br/>+ Postgres for ACL/BM25 · &gt;5M chunks")]
        VS --> PG
        VS --> TV
    end

    subgraph XCUT["Cross-Cutting Concerns"]
        SEC["Security<br/>tenant isolation · pre-retrieval ACL · RBAC · guardrails"]
        MEM["Memory<br/>session · durable · retrieval · semantic cache"]
        OBS["Observability<br/>OpenTelemetry · Langfuse · failure taxonomy"]
        EVAL["Evaluation<br/>RAGAS · golden set · regression CI"]
    end

    UI --> EP
    EP --> CLASS
    EP -. "ingest" .-> OFFLINE
    EMB --> VS
    EXP --> VS
    DRET --> VS
    GEN --> UI
    ABS --> UI
    SEC -. "guards" .-> ONLINE
    MEM -. "context" .-> ONLINE
    OBS -. "traces" .-> ONLINE
    EVAL -. "gates" .-> ONLINE
```

---

## 2. Query lifecycle — the LangGraph spine

This is exactly what `app/graph.py` compiles. The classifier forks; both lanes
share retrieve → rerank → assemble → verify → generate; only the Deep Lane loops.

```mermaid
flowchart LR
    START([query]) --> C["classify<br/>(routing.classify_lane)"]
    C -->|"Lane.EXPRESS"| R
    C -->|"Lane.DEEP"| P["plan"]
    P --> R["retrieve<br/>(ACL pre-filtered)"]
    R --> RR["rerank<br/>(RRF + cross-encoder)"]
    RR --> A["assemble<br/>(chunks vs full-doc)"]
    A --> V{"verify<br/>(decide_gate)"}
    V -->|"PASS"| G["generate + cite"]
    V -->|"REPAIR · Deep · &lt;3"| RP["repair"]
    RP --> R
    V -->|"ABSTAIN"| AB["no-answer"]
    V -->|"REPAIR on Express (can't loop)"| AB
    G --> END([cited answer + spans])
    AB --> END
```

---

## 3. VectorStore abstraction — the v3 delta

Why the abstraction exists and how the two implementations differ. Note ACL stays
in Postgres in BOTH paths — TurboVec is only a candidate generator.

```mermaid
flowchart TB
    PIPE["retrieval pipeline<br/>calls VectorStore.search(), never the impl"] --> CFG{{"VECTOR_STORE env<br/>config flag"}}
    CFG -->|"pgvector (default)"| PGP
    CFG -->|"turbovec"| TVP

    subgraph PGP["PgVectorStore · single SQL statement"]
        direction TB
        PG1["HNSW vector similarity<br/>+ tsvector BM25<br/>+ tenant_id / ACL / freshness filters"]
    end

    subgraph TVP["TurboVecStore · two-step"]
        direction TB
        TV1["1 · TurboVec quantized index<br/>→ top-100 candidate chunk IDs"]
        TV2["2 · Postgres filters those 100 by<br/>ACL · metadata · freshness"]
        TV1 --> TV2
    end

    PGP --> RANK["Cross-encoder reranks survivors"]
    TVP --> RANK
    RANK --> OUT([evidence set])
```

### When to use which (spec v3 decision table)

| Chunk count | pgvector RAM (d=1536) | TurboVec RAM (4-bit) | Recommendation |
|---|---|---|---|
| < 10K | < 150 MB | — | pgvector, **skip HNSW** (flat scan is faster) |
| 10K–500K | 150 MB–7 GB | — | pgvector + HNSW |
| 500K–2M | 7–30 GB | 375 MB–1.5 GB | pgvector preferred; monitor P95 |
| 2M–5M | 30–75 GB | 1.5–3.8 GB | gray zone; evaluate both |
| 5M+ | 75+ GB | 3.8+ GB | **TurboVecStore** |

**Switching triggers** (monitor in prod): HNSW build time > ingestion SLA · P95
vector query > 100ms · memory pressure evicts HNSW from shared buffers. When any
fire: run both for a week, compare P95 + recall, flip `VECTOR_STORE=turbovec`.

---

## 4. Performance budget (where latency is spent)

```mermaid
flowchart LR
    subgraph EXPRESS["Express Lane · P95 &lt; 2s · first token &lt; 1s"]
        E1["classify &lt;100ms"] --> E2["retrieve &lt;500ms"] --> E3["rerank &lt;300ms"] --> E4["verify + generate"]
    end
    subgraph DEEP["Deep Lane · P95 &lt; 5s · first token &lt; 3s"]
        D1["classify &lt;100ms"] --> D2["plan"] --> D3["retrieve &lt;1.5s"] --> D4["rerank &lt;500ms"] --> D5["verify → (repair ↺) → generate"]
    end
```

Blended P95 target < 3s, predicated on Express carrying 60-70% of traffic.
