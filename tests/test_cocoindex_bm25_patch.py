from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "cocoindex_bm25_patch.py"
)
SPEC = importlib.util.spec_from_file_location("cocoindex_bm25_patch", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load patch script from {SCRIPT_PATH}")

PATCH_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCH_MODULE)


class CocoindexBm25PatchTests(unittest.TestCase):
    def test_patch_resource_schema_promotes_numpy_import_to_runtime(self) -> None:
        source = '''"""Schema-related helper types."""

from __future__ import annotations

import typing as _typing
import dataclasses as _dataclasses
import cocoindex as coco

if _typing.TYPE_CHECKING:
    import numpy as _np


@_dataclasses.dataclass(slots=True, frozen=True)
class VectorSchema:
    dtype: _np.dtype
    size: int
'''

        patched = PATCH_MODULE.patch_resource_schema(source)

        self.assertIn("import numpy as _np", patched)
        self.assertNotIn("if _typing.TYPE_CHECKING:", patched)
        self.assertIn("dtype: _np.dtype", patched)
        self.assertEqual(PATCH_MODULE.patch_resource_schema(patched), patched)

    def test_desired_shared_preserves_current_upstream_helpers(self) -> None:
        shared = PATCH_MODULE.DESIRED_SHARED

        expected_fragments = [
            "import importlib.util",
            "from typing import TYPE_CHECKING, Annotated, NamedTuple, Union",
            "DEFAULT_LITELLM_MIN_INTERVAL_MS = 5",
            "def is_sentence_transformers_installed() -> bool:",
            "class EmbeddingCheckResult(NamedTuple):",
            "async def check_embedding(embedder: Embedder) -> EmbeddingCheckResult:",
            "from .litellm_embedder import PacedLiteLLMEmbedder",
            '"Embedding model (LiteLLM): %s | min_interval_ms: %s"',
        ]

        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, shared)

    def test_desired_shared_uses_plain_async_vector_schema_method(self) -> None:
        shared = PATCH_MODULE.DESIRED_SHARED

        self.assertIn(
            "async def __coco_vector_schema__(self) -> _schema.VectorSchema:", shared
        )
        self.assertNotIn(
            "@coco.fn.as_async(memo=True)\n    async def __coco_vector_schema__(self)",
            shared,
        )

    def test_patch_shared_is_idempotent(self) -> None:
        self.assertEqual(
            PATCH_MODULE.patch_shared(PATCH_MODULE.DESIRED_SHARED),
            PATCH_MODULE.DESIRED_SHARED,
        )


if __name__ == "__main__":
    unittest.main()
