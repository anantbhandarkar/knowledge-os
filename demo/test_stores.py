"""Live integration test: PgVectorStore + TurboVecStore + OpenRouter generation.

Run with Postgres up and OPENROUTER_API_KEY set:
    python demo/test_stores.py

Exercises, for EACH store, through the real LangGraph pipeline:
  1) a public query -> grounded, cited answer (real LLM)
  2) ACL isolation: a restricted fact is INVISIBLE without the tag, VISIBLE with it
"""

import asyncio
import os
import pathlib

from app import runtime
from app.graph import COMPILED_GRAPH
from app.ingest import ingest_text
from app.models import PipelineState
from app.providers.openrouter import OpenRouterGenerator
from app.vectorstore.pgvector import PgVectorStore
from app.vectorstore.turbovec import TurboVecStore

DSN = os.getenv("DATABASE_URL", "postgresql://kos:kos@localhost:5433/knowledge_os")
MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
CORPUS = pathlib.Path(__file__).parent / "corpus"
SECRET = "The master incident override code is OMEGA-7-FOXTROT."

_PASS, _FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"


async def ask(query: str, acl: list[str] | None = None) -> PipelineState:
    final = await COMPILED_GRAPH.ainvoke(
        PipelineState(query=query, tenant_id="acme", acl_tags=acl or [])
    )
    return PipelineState.model_validate(final)


async def truncate(store, table):
    async with store.pool.connection() as conn:
        await conn.execute(f"TRUNCATE {table}")


async def seed(store):
    runtime.STORE = store
    for md in sorted(CORPUS.glob("*.md")):
        await ingest_text(md.read_text(), doc_title=md.stem, tenant_id="acme")
    # one restricted doc, only visible to the security-team ACL
    await ingest_text(f"# Incident Override\n\n{SECRET}", doc_title="Restricted Incident Doc",
                      tenant_id="acme", acl_tags=["security-team"])


async def exercise(name: str, store, table: str):
    print(f"\n{'='*68}\n  {name}\n{'='*68}")
    await store.setup()
    await truncate(store, table)
    if hasattr(store, "_ids"):
        store._ids.clear()
        store._codes = store._codes[:0]
    await seed(store)
    print(f"  indexed chunks (tenant=acme): {await store.count(tenant_id='acme')}")

    # 1) public query -> grounded cited answer via real LLM
    s = await ask("What is the JWT expiry for contractor accounts?")
    ok = "4" in (s.answer or "")
    print(f"\n  [{_PASS if ok else _FAIL}] public query  lane={s.lane} verify={s.decision}")
    print(f"        answer: {(s.answer or '')[:130]}")
    print(f"        cites : {[c.source_doc_title for c in s.citations][:3]}")

    # 2a) ACL isolation — WITHOUT the tag, the secret must NOT be retrievable
    s_no = await ask("What is the master incident override code?", acl=[])
    leaked = "OMEGA" in (s_no.answer or "")
    print(f"\n  [{_FAIL if leaked else _PASS}] ACL deny    verify={s_no.decision} "
          f"(secret leaked: {leaked})")
    print(f"        answer: {(s_no.answer or '')[:110]}")

    # 2b) WITH the tag, the same query SHOULD surface the secret
    s_yes = await ask("What is the master incident override code?", acl=["security-team"])
    granted = "OMEGA" in (s_yes.answer or "")
    print(f"\n  [{_PASS if granted else _FAIL}] ACL allow   verify={s_yes.decision} "
          f"(secret returned: {granted})")
    print(f"        answer: {(s_yes.answer or '')[:130]}")

    await store.close()
    return ok and (not leaked) and granted


async def main():
    assert os.environ.get("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY not set"
    runtime.GENERATOR = OpenRouterGenerator(MODEL)
    print(f"generator: {runtime.GENERATOR.name}")

    dim = runtime.EMBEDDER.dim
    r1 = await exercise("PgVectorStore (Implementation A)", PgVectorStore(DSN, dim=dim), "chunks")
    r2 = await exercise("TurboVecStore (Implementation B — two-step)", TurboVecStore(DSN, dim=dim), "turbo_chunks")

    print(f"\n{'='*68}")
    print(f"  PgVectorStore : {_PASS if r1 else _FAIL}")
    print(f"  TurboVecStore : {_PASS if r2 else _FAIL}")


if __name__ == "__main__":
    asyncio.run(main())
