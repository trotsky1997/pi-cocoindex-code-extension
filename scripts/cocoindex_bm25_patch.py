from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

PATCH_MARKER = "BM25 mode patched by pi-cocoindex-code-extension"

DESIRED_SHARED = '''"""Shared context keys, embedder factory, and CodeChunk schema."""

from __future__ import annotations

import logging
import os
import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Union

import cocoindex as coco
import numpy as np
import numpy.typing as npt
from cocoindex.connectors import sqlite
from cocoindex.resources import schema as _schema

if TYPE_CHECKING:
    from cocoindex.ops.litellm import LiteLLMEmbedder
    from cocoindex.ops.sentence_transformers import SentenceTransformerEmbedder

from .settings import EmbeddingSettings

logger = logging.getLogger(__name__)

SBERT_PREFIX = "sbert/"

# Models that define a "query" prompt for asymmetric retrieval.
_QUERY_PROMPT_MODELS = {"nomic-ai/nomic-embed-code", "nomic-ai/CodeRankEmbed"}


class BM25Embedder(_schema.VectorSchemaProvider):
    """Minimal local embedder used to keep CocoIndex's vector schema valid in BM25 mode."""

    def __init__(self) -> None:
        self._vector = np.zeros((1,), dtype=np.float32)
        self._vector.setflags(write=False)
        self._batch_cache: dict[int, list[npt.NDArray[np.float32]]] = {}

    @coco.fn.as_async(  # type: ignore[arg-type]
        batching=True,
        max_batch_size=256,
        memo=True,
        version=1,
        logic_tracking="self",
    )
    async def embed(
        self,
        texts: list[str],
        input_type: str | None = None,
    ) -> list[npt.NDArray[np.float32]]:
        del input_type
        count = len(texts)
        cached = self._batch_cache.get(count)
        if cached is None:
            cached = [self._vector] * count
            self._batch_cache[count] = cached
        return list(cached)

    @coco.fn.as_async(memo=True)
    async def __coco_vector_schema__(self) -> _schema.VectorSchema:
        return _schema.VectorSchema(dtype=np.dtype(np.float32), size=1)

    def __coco_memo_key__(self) -> object:
        return ("bm25", 1)


# Type alias
Embedder = Union["SentenceTransformerEmbedder", "LiteLLMEmbedder", BM25Embedder]

# Context keys
EMBEDDER = coco.ContextKey[Embedder]("embedder")
SQLITE_DB = coco.ContextKey[sqlite.ManagedConnection]("index_db", tracked=False)
CODEBASE_DIR = coco.ContextKey[pathlib.Path]("codebase", tracked=False)

# Module-level variable - set by daemon at startup (needed for CodeChunk annotation).
embedder: Embedder | None = None

# Query prompt name - set alongside embedder by create_embedder().
query_prompt_name: str | None = None


def create_embedder(settings: EmbeddingSettings) -> Embedder:
    """Create and return an embedder instance based on settings.

    Also sets the module-level ``embedder`` and ``query_prompt_name`` variables.
    """
    global embedder, query_prompt_name

    # BM25 mode patched by pi-cocoindex-code-extension.
    if settings.provider == "bm25" or settings.model == "bm25":
        instance: Embedder = BM25Embedder()
        query_prompt_name = None
        logger.info("Embedding backend (BM25 compatibility mode): %s", settings.model)
    elif settings.provider == "sentence-transformers":
        from cocoindex.ops.sentence_transformers import SentenceTransformerEmbedder

        model_name = settings.model
        # Strip the legacy sbert/ prefix if present
        if model_name.startswith(SBERT_PREFIX):
            model_name = model_name[len(SBERT_PREFIX) :]

        query_prompt_name = "query" if model_name in _QUERY_PROMPT_MODELS else None
        instance = SentenceTransformerEmbedder(
            model_name,
            device=settings.device,
            trust_remote_code=True,
        )
        logger.info("Embedding model: %s | device: %s", settings.model, settings.device)
    else:
        from cocoindex.ops.litellm import LiteLLMEmbedder

        litellm_kwargs = {}
        # Some OpenAI-compatible embedding APIs reject LiteLLM's default
        # null encoding_format, so force the standard float response here.
        if settings.model.startswith("openai/") or os.environ.get("OPENAI_API_BASE"):
            litellm_kwargs["encoding_format"] = "float"

        instance = LiteLLMEmbedder(settings.model, **litellm_kwargs)
        query_prompt_name = None
        logger.info("Embedding model (LiteLLM): %s", settings.model)

    embedder = instance
    return instance


@dataclass
class CodeChunk:
    """Schema for storing code chunks in SQLite."""

    id: int
    file_path: str
    language: str
    content: str
    start_line: int
    end_line: int
    embedding: Annotated[npt.NDArray[np.float32], EMBEDDER]
'''

DESIRED_QUERY = '''"""Query implementation for codebase search."""

from __future__ import annotations

import heapq
import re
import sqlite3
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from rapidfuzz import fuzz as _rapidfuzz_fuzz
except Exception:  # pragma: no cover - optional acceleration dependency
    _rapidfuzz_fuzz = None

from .schema import QueryResult
from .shared import BM25Embedder, EMBEDDER, SQLITE_DB, query_prompt_name

BM25_TABLE = "code_chunks_fts"
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
RERANK_FETCH_LIMIT = 500
RERANK_MIN_FETCH = 50
BM25_WEIGHT = 0.82
FUZZY_WEIGHT = 0.18


def _l2_to_score(distance: float) -> float:
    """Convert L2 distance to cosine similarity (exact for unit vectors)."""
    return 1.0 - distance * distance / 2.0


def _knn_query(
    conn: sqlite3.Connection,
    embedding_bytes: bytes,
    k: int,
    language: str | None = None,
) -> list[tuple[Any, ...]]:
    """Run a vec0 KNN query, optionally constrained to a language partition."""
    if language is not None:
        return conn.execute(
            """
            SELECT file_path, language, content, start_line, end_line, distance
            FROM code_chunks_vec
            WHERE embedding MATCH ? AND k = ? AND language = ?
            ORDER BY distance
            """,
            (embedding_bytes, k, language),
        ).fetchall()
    return conn.execute(
        """
        SELECT file_path, language, content, start_line, end_line, distance
        FROM code_chunks_vec
        WHERE embedding MATCH ? AND k = ?
        ORDER BY distance
        """,
        (embedding_bytes, k),
    ).fetchall()


def _full_scan_query(
    conn: sqlite3.Connection,
    embedding_bytes: bytes,
    limit: int,
    offset: int,
    languages: list[str] | None = None,
    paths: list[str] | None = None,
) -> list[tuple[Any, ...]]:
    """Full scan with SQL-level distance computation and filtering."""
    conditions: list[str] = []
    params: list[Any] = [embedding_bytes]

    if languages:
        placeholders = ",".join("?" for _ in languages)
        conditions.append(f"language IN ({placeholders})")
        params.extend(languages)

    if paths:
        path_clauses = " OR ".join("file_path GLOB ?" for _ in paths)
        conditions.append(f"({path_clauses})")
        params.extend(paths)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.extend([limit, offset])

    return conn.execute(
        f"""
        SELECT file_path, language, content, start_line, end_line,
               vec_distance_L2(embedding, ?) as distance
        FROM code_chunks_vec
        {where}
        ORDER BY distance
        LIMIT ? OFFSET ?
        """,
        params,
    ).fetchall()


def _bm25_index_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (BM25_TABLE,),
    ).fetchone()
    return row is not None


def _tokenize_bm25_query(query: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()

    def _push(token: str) -> None:
        token = token.strip("_").lower()
        if len(token) < 2 or token in seen:
            return
        seen.add(token)
        tokens.append(token)

    for raw in TOKEN_RE.findall(query):
        _push(raw)
        for part in raw.split("_"):
            _push(part)
        for part in CAMEL_RE.sub(" ", raw).split():
            _push(part)

    return tokens


@lru_cache(maxsize=512)
def _build_bm25_match_query(query: str) -> str:
    tokens = _tokenize_bm25_query(query)
    if not tokens:
        raise RuntimeError("Query contains no searchable BM25 tokens.")
    return " AND ".join(f"{token}*" if len(token) >= 3 else token for token in tokens)


@lru_cache(maxsize=512)
def _normalized_query(query: str) -> str:
    return " ".join(_tokenize_bm25_query(query)) or query.lower()


def _fuzzy_score(query: str, file_path: str, content: str) -> float:
    normalized_query = _normalized_query(query)
    path_text = file_path.replace("/", " ").replace("_", " ").replace("-", " ")
    # Limit body comparisons to cap rerank CPU on large chunks.
    body = content[:6000]

    if _rapidfuzz_fuzz is not None:
        return max(
            _rapidfuzz_fuzz.WRatio(normalized_query, path_text),
            _rapidfuzz_fuzz.token_set_ratio(normalized_query, body),
            _rapidfuzz_fuzz.partial_ratio(normalized_query, body),
        ) / 100.0

    query_l = normalized_query.lower()
    path_l = path_text.lower()
    body_l = body.lower()
    return max(
        SequenceMatcher(None, query_l, path_l).ratio(),
        SequenceMatcher(None, query_l, body_l[:1200]).ratio(),
    )


def _rerank_bm25_rows(
    query: str,
    rows: list[tuple[Any, ...]],
    limit: int,
    offset: int,
) -> list[tuple[Any, ...]]:
    if not rows:
        return []

    bm25_scores = [-float(row[5]) for row in rows]
    min_bm25 = min(bm25_scores)
    max_bm25 = max(bm25_scores)
    spread = max(max_bm25 - min_bm25, 1e-9)

    scored_rows: list[tuple[float, tuple[Any, ...]]] = []
    for row, bm25_score in zip(rows, bm25_scores):
        file_path, _language, content, _start_line, _end_line, _rank = row
        bm25_norm = (bm25_score - min_bm25) / spread
        fuzzy_norm = _fuzzy_score(query, file_path, content)
        combined = BM25_WEIGHT * bm25_norm + FUZZY_WEIGHT * fuzzy_norm
        scored_rows.append((combined, row[:5] + (combined,)))

    scored_rows.sort(key=lambda item: item[0], reverse=True)
    return [row for _score, row in scored_rows[offset : offset + limit]]


def _bm25_query(
    conn: sqlite3.Connection,
    query: str,
    limit: int,
    offset: int,
    languages: list[str] | None = None,
    paths: list[str] | None = None,
) -> list[tuple[Any, ...]]:
    match_query = _build_bm25_match_query(query)
    conditions = [f"{BM25_TABLE} MATCH ?"]
    params: list[Any] = [match_query]

    if languages:
        placeholders = ",".join("?" for _ in languages)
        conditions.append(f"language IN ({placeholders})")
        params.extend(languages)

    if paths:
        path_clauses = " OR ".join("file_path GLOB ?" for _ in paths)
        conditions.append(f"({path_clauses})")
        params.extend(paths)

    fetch_k = min(
        RERANK_FETCH_LIMIT,
        max(RERANK_MIN_FETCH, (limit + offset) * 4, limit + offset),
    )
    params.append(fetch_k)
    where = " AND ".join(conditions)
    rows = conn.execute(
        f"""
        SELECT file_path, language, content, start_line, end_line, bm25({BM25_TABLE}) AS rank
        FROM {BM25_TABLE}
        WHERE {where}
        ORDER BY rank
        LIMIT ?
        """,
        params,
    ).fetchall()
    return _rerank_bm25_rows(query, rows, limit, offset)


async def query_codebase(
    query: str,
    target_sqlite_db_path: Path,
    env: Any,
    limit: int = 10,
    offset: int = 0,
    languages: list[str] | None = None,
    paths: list[str] | None = None,
) -> list[QueryResult]:
    """
    Perform code search against either BM25 FTS or vec0 KNN.

    BM25 mode patched by pi-cocoindex-code-extension prefers the cached
    SQLite FTS index when the configured embedder is ``BM25Embedder``.
    """
    if not target_sqlite_db_path.exists():
        raise RuntimeError(
            f"Index database not found at {target_sqlite_db_path}. "
            "Please run a query with refresh_index=True first."
        )

    db = env.get_context(SQLITE_DB)
    embedder = env.get_context(EMBEDDER)

    with db.readonly() as conn:
        if isinstance(embedder, BM25Embedder):
            if not _bm25_index_exists(conn):
                raise RuntimeError(
                    "BM25 index table is missing from target_sqlite.db. "
                    "Please run `ccc index` to build the cached FTS search index."
                )
            rows = _bm25_query(conn, query, limit, offset, languages, paths)
            return [
                QueryResult(
                    file_path=file_path,
                    language=language,
                    content=content,
                    start_line=start_line,
                    end_line=end_line,
                    score=score,
                )
                for file_path, language, content, start_line, end_line, score in rows
            ]

        # Generate query embedding.
        query_embedding = await embedder.embed(query, query_prompt_name)
        embedding_bytes = query_embedding.astype("float32").tobytes()

        if paths:
            rows = _full_scan_query(conn, embedding_bytes, limit, offset, languages, paths)
        elif not languages or len(languages) == 1:
            lang = languages[0] if languages else None
            rows = _knn_query(conn, embedding_bytes, limit + offset, lang)
        else:
            fetch_k = limit + offset
            rows = heapq.nsmallest(
                fetch_k,
                (
                    row
                    for lang in languages
                    for row in _knn_query(conn, embedding_bytes, fetch_k, lang)
                ),
                key=lambda r: r[5],
            )

    if not paths:
        rows = rows[offset:]

    return [
        QueryResult(
            file_path=file_path,
            language=language,
            content=content,
            start_line=start_line,
            end_line=end_line,
            score=_l2_to_score(distance),
        )
        for file_path, language, content, start_line, end_line, distance in rows
    ]
'''

INDEXER_INSERT_AFTER = "splitter = RecursiveSplitter()\n"
INDEXER_HELPERS = '''\n\nBM25_TABLE = "code_chunks_fts"\n\n\ndef has_bm25_index(db: sqlite.ManagedConnection) -> bool:\n    \"\"\"Return True when the cached BM25 FTS table exists.\"\"\"\n    with db.readonly() as conn:\n        row = conn.execute(\n            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",\n            (BM25_TABLE,),\n        ).fetchone()\n    return row is not None\n\n\ndef refresh_bm25_index(db: sqlite.ManagedConnection) -> None:\n    \"\"\"Rebuild the cached BM25 FTS table from the vec chunk table.\"\"\"\n    with db.transaction() as conn:\n        conn.execute(f'DROP TABLE IF EXISTS {BM25_TABLE}')\n        conn.execute(\n            f\"\"\"\n            CREATE VIRTUAL TABLE {BM25_TABLE} USING fts5(\n                file_path UNINDEXED,\n                language UNINDEXED,\n                content,\n                start_line UNINDEXED,\n                end_line UNINDEXED,\n                tokenize='unicode61'\n            )\n            \"\"\"\n        )\n        conn.execute(\n            f\"\"\"\n            INSERT INTO {BM25_TABLE}(file_path, language, content, start_line, end_line)\n            SELECT file_path, language, content, start_line, end_line\n            FROM code_chunks_vec\n            \"\"\"\n        )\n        conn.execute(\n            f\"INSERT INTO {BM25_TABLE}({BM25_TABLE}) VALUES ('optimize')\"\n        )\n'''

PROJECT_IMPORT_OLD = "from .indexer import indexer_main\n"
PROJECT_IMPORT_NEW = "from .indexer import has_bm25_index, indexer_main, refresh_bm25_index\n"
PROJECT_SHARED_IMPORT_OLD = '''from .shared import (\n    CODEBASE_DIR,\n    EMBEDDER,\n    SQLITE_DB,\n    Embedder,\n)\n'''
PROJECT_SHARED_IMPORT_NEW = '''from .shared import (\n    BM25Embedder,\n    CODEBASE_DIR,\n    EMBEDDER,\n    SQLITE_DB,\n    Embedder,\n)\n'''
PROJECT_RUN_INDEX_OLD = '''    async def _run_index_inner(\n        self,\n        on_progress: Callable[[IndexingProgress], None] | None = None,\n    ) -> None:\n        \"\"\"Run indexing (lock must already be held).\"\"\"\n        try:\n            handle = self._app.update()\n            async for snapshot in handle.watch():\n                file_stats = snapshot.stats.by_component.get("process_file")\n                if file_stats is not None:\n                    progress = IndexingProgress(\n                        num_execution_starts=file_stats.num_execution_starts,\n                        num_unchanged=file_stats.num_unchanged,\n                        num_adds=file_stats.num_adds,\n                        num_deletes=file_stats.num_deletes,\n                        num_reprocesses=file_stats.num_reprocesses,\n                        num_errors=file_stats.num_errors,\n                    )\n                    self._indexing_stats = progress\n                    if on_progress is not None:\n                        on_progress(progress)\n                    await asyncio.sleep(0.1)\n        finally:\n            self._initial_index_done.set()\n            self._indexing_stats = None\n'''
PROJECT_RUN_INDEX_NEW = '''    async def _run_index_inner(\n        self,\n        on_progress: Callable[[IndexingProgress], None] | None = None,\n    ) -> None:\n        \"\"\"Run indexing (lock must already be held).\"\"\"\n        final_progress: IndexingProgress | None = None\n        try:\n            handle = self._app.update()\n            async for snapshot in handle.watch():\n                file_stats = snapshot.stats.by_component.get("process_file")\n                if file_stats is not None:\n                    progress = IndexingProgress(\n                        num_execution_starts=file_stats.num_execution_starts,\n                        num_unchanged=file_stats.num_unchanged,\n                        num_adds=file_stats.num_adds,\n                        num_deletes=file_stats.num_deletes,\n                        num_reprocesses=file_stats.num_reprocesses,\n                        num_errors=file_stats.num_errors,\n                    )\n                    final_progress = progress
                    self._indexing_stats = progress
                    if on_progress is not None:
                        on_progress(progress)
                    await asyncio.sleep(0.1)

            embedder = self._env.get_context(EMBEDDER)
            if isinstance(embedder, BM25Embedder):
                db = self._env.get_context(SQLITE_DB)
                needs_bm25_refresh = not has_bm25_index(db)
                if final_progress is not None:
                    needs_bm25_refresh = needs_bm25_refresh or any(
                        value > 0
                        for value in (
                            final_progress.num_adds,
                            final_progress.num_deletes,
                            final_progress.num_reprocesses,
                        )
                    )
                if needs_bm25_refresh:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, lambda: refresh_bm25_index(db))
        finally:
            self._initial_index_done.set()
            self._indexing_stats = None
'''


def load_source(module_name: str) -> tuple[Path, str]:
    module = __import__(module_name, fromlist=["__file__"])
    path = Path(module.__file__).resolve()
    return path, path.read_text()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if old not in source:
        raise RuntimeError(f"Expected block not found for {label}.")
    return source.replace(old, new, 1)


def patch_shared(source: str) -> str:
    if PATCH_MARKER in source and "class BM25Embedder" in source and "_batch_cache" in source:
        return source
    if "def create_embedder" not in source or "class CodeChunk" not in source:
        raise RuntimeError("shared.py layout is not recognized.")
    return DESIRED_SHARED


def patch_query(source: str) -> str:
    if PATCH_MARKER in source and "BM25_TABLE = \"code_chunks_fts\"" in source and "RERANK_FETCH_LIMIT" in source:
        return source
    if "async def query_codebase" not in source or "def _knn_query" not in source:
        raise RuntimeError("query.py layout is not recognized.")
    return DESIRED_QUERY


def patch_indexer(source: str) -> str:
    if PATCH_MARKER in source and "def refresh_bm25_index" in source:
        return source
    if INDEXER_INSERT_AFTER not in source:
        raise RuntimeError("indexer.py layout is not recognized.")
    updated = source.replace(INDEXER_INSERT_AFTER, INDEXER_INSERT_AFTER + INDEXER_HELPERS, 1)
    if PATCH_MARKER not in updated:
        updated = updated.replace(
            '"""CocoIndex app for indexing codebases."""',
            '"""CocoIndex app for indexing codebases."""\n\n# BM25 mode patched by pi-cocoindex-code-extension.',
            1,
        )
    return updated


def patch_project(source: str) -> str:
    if PATCH_MARKER in source and "refresh_bm25_index" in source and "BM25Embedder" in source:
        return source
    updated = replace_once(source, PROJECT_IMPORT_OLD, PROJECT_IMPORT_NEW, "project import")
    updated = replace_once(updated, PROJECT_SHARED_IMPORT_OLD, PROJECT_SHARED_IMPORT_NEW, "project shared import")
    updated = replace_once(updated, PROJECT_RUN_INDEX_OLD, PROJECT_RUN_INDEX_NEW, "project run_index")
    if PATCH_MARKER not in updated:
        updated = updated.replace(
            '"""Project management: wraps a CocoIndex Environment + App."""',
            '"""Project management: wraps a CocoIndex Environment + App."""\n\n# BM25 mode patched by pi-cocoindex-code-extension.',
            1,
        )
    return updated


def ensure_global_settings_switched() -> bool:
    from cocoindex_code.settings import user_settings_path

    settings_path = user_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    if settings_path.exists():
        data = yaml.safe_load(settings_path.read_text()) or {}
    else:
        data = {}

    embedding = data.get("embedding")
    if not isinstance(embedding, dict):
        embedding = {}
    changed = embedding.get("provider") != "bm25" or embedding.get("model") != "bm25"
    embedding["provider"] = "bm25"
    embedding["model"] = "bm25"
    data["embedding"] = embedding

    if changed or not settings_path.exists():
        settings_path.write_text(yaml.safe_dump(data, default_flow_style=False, sort_keys=False))
    return changed or not settings_path.exists()


def main() -> None:
    targets = {
        "shared": (patch_shared, *load_source("cocoindex_code.shared")),
        "indexer": (patch_indexer, *load_source("cocoindex_code.indexer")),
        "project": (patch_project, *load_source("cocoindex_code.project")),
        "query": (patch_query, *load_source("cocoindex_code.query")),
    }

    changed_paths: list[str] = []
    scanned_paths = {name: str(path) for name, (_, path, _) in targets.items()}

    try:
        for name, (patcher, path, source) in targets.items():
            updated = patcher(source)
            if updated != source:
                path.write_text(updated)
                changed_paths.append(str(path))
        settings_changed = ensure_global_settings_switched()
        if settings_changed:
            changed_paths.append("global_settings")

        if changed_paths:
            status = "patched"
            message = "Enabled BM25 mode for CocoIndex and updated cached search support."
        else:
            status = "already_patched"
            message = "BM25 mode is already enabled for CocoIndex."

        print(
            json.dumps(
                {
                    "status": status,
                    "message": message,
                    "changed_paths": changed_paths,
                    "scanned_paths": scanned_paths,
                }
            )
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "message": str(exc),
                    "changed_paths": changed_paths,
                    "scanned_paths": scanned_paths,
                }
            )
        )
        raise


if __name__ == "__main__":
    main()
