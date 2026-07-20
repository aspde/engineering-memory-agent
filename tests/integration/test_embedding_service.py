"""Integration tests for embedding service — requires native runtime (onnxruntime/torch)."""

import os
import subprocess
import sys

import pytest

# ── Force offline BEFORE any HF imports ──────────────────────────────
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# Check native runtime in a subprocess so DLL crashes don't kill the test runner.
def _check_native_runtime() -> bool:
    """Return True if onnxruntime loads in a subprocess."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import onnxruntime"],
            capture_output=True,
            timeout=15,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


_bge_available = _check_native_runtime()
_bge_skip_reason = (
    "Native runtime DLLs (onnxruntime/torch) unavailable. "
    "Install VC++ Redistributable or run in WSL/Docker/Linux."
)


@pytest.mark.skipif(not _bge_available, reason=_bge_skip_reason)
class TestBGEEmbeddingProvider:
    """Integration tests with real BGE-M3 model."""

    @pytest.fixture(scope="class")
    def provider(self) -> "BGEEmbeddingProvider":  # noqa: F821
        from backend.service.embedding_service import BGEEmbeddingProvider

        return BGEEmbeddingProvider("BAAI/bge-m3")

    def test_dimension(self, provider: "BGEEmbeddingProvider") -> None:  # noqa: F821
        assert provider.dimension == 1024

    def test_embed_sync_single(self, provider: "BGEEmbeddingProvider") -> None:  # noqa: F821
        result = provider.embed_sync(["你好世界"])
        assert len(result) == 1
        assert len(result[0]) == 1024

    def test_embed_sync_multiple(self, provider: "BGEEmbeddingProvider") -> None:  # noqa: F821
        result = provider.embed_sync(["hello", "world", "test"])
        assert len(result) == 3
        assert all(len(v) == 1024 for v in result)

    @pytest.mark.asyncio
    async def test_embed_async(self, provider: "BGEEmbeddingProvider") -> None:  # noqa: F821
        result = await provider.embed(["async test"])
        assert len(result) == 1
        assert len(result[0]) == 1024
