"""Ollama HTTP client used for text-embedding based similarity search.

Talks to a local Ollama server (default ``http://127.0.0.1:11434``) using
``requests``. Every request uses a short *connect* timeout so GUI startup
detection never hangs when the server is absent, plus a bounded *read*
timeout for the actual work (embedding generation may need to load a model
on first use).

Ollama is strictly optional: transport problems surface either as ``False``
(:meth:`OllamaClient.is_running`) or as :class:`OllamaError` — the caller's
thread is never crashed by a missing server.
"""
from __future__ import annotations

import logging
from typing import Any

import requests

log = logging.getLogger(__name__)

__all__ = ["CONNECT_TIMEOUT", "DEFAULT_HOST", "OllamaClient", "OllamaError", "detect_ollama"]

#: Default local Ollama endpoint.
DEFAULT_HOST = "http://127.0.0.1:11434"
#: Short connect timeout (seconds) applied to every call so startup never hangs.
CONNECT_TIMEOUT = 2.0
#: Read timeout (seconds) for cheap metadata calls such as /api/tags.
_SNAPSHOT_READ_TIMEOUT = 5.0
#: Lowercased substrings that strongly suggest an embedding model. Used only
#: as a fallback when /api/tags carries no usable ``capabilities`` field
#: (very old Ollama versions).
_EMBED_NAME_HINTS: tuple[str, ...] = ("embed", "minilm", "bge", "mxbai", "arctic")


class OllamaError(RuntimeError):
    """Raised when an Ollama HTTP call fails or returns an unusable payload."""


def _snippet(response: requests.Response, limit: int = 200) -> str:
    """Short single-line body preview of an HTTP response for error messages."""
    try:
        body = (response.text or "").strip().replace("\n", " ")
    except Exception:  # pragma: no cover - defensive, body unreadable
        body = ""
    return body[:limit]


class OllamaClient:
    """Small ``requests`` wrapper around the Ollama REST API subset the app
    needs: model discovery (``/api/tags``) and text embeddings."""

    def __init__(self, host: str = DEFAULT_HOST, timeout: float = 30.0) -> None:
        self.host = (host or DEFAULT_HOST).rstrip("/")
        self.timeout = float(timeout)

    # ------------------------------------------------------------------ utils
    def _url(self, path: str) -> str:
        return f"{self.host}{path}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        read_timeout: float | None = None,
    ) -> requests.Response:
        """Perform one HTTP request; raises ``requests`` exceptions on transport errors."""
        timeout = (CONNECT_TIMEOUT, read_timeout if read_timeout is not None else self.timeout)
        return requests.request(method, self._url(path), json=json, timeout=timeout)

    # ------------------------------------------------------------- discovery
    def is_running(self) -> bool:
        """True when an Ollama server answers ``GET /api/tags``; False on any error."""
        try:
            response = self._request("GET", "/api/tags", read_timeout=_SNAPSHOT_READ_TIMEOUT)
        except Exception:  # requests.RequestException and anything unexpected
            return False
        return response.status_code == 200

    def list_models(self) -> list[dict]:
        """Raw ``/api/tags`` model entries. Raises :class:`OllamaError` when down."""
        try:
            response = self._request("GET", "/api/tags", read_timeout=_SNAPSHOT_READ_TIMEOUT)
        except requests.RequestException as exc:
            raise OllamaError(f"Cannot reach Ollama at {self.host}: {exc}") from exc
        if response.status_code != 200:
            raise OllamaError(
                f"Ollama at {self.host} answered HTTP {response.status_code} for /api/tags: "
                f"{_snippet(response)}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise OllamaError(f"Ollama at {self.host} returned invalid JSON from /api/tags") from exc
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            raise OllamaError(f"Unexpected /api/tags payload from {self.host}")
        return [entry for entry in models if isinstance(entry, dict)]

    def list_embedding_models(self) -> list[str]:
        """Names of installed models able to produce embeddings.

        Detection tiers, in order:

        1. ``/api/tags`` ``capabilities`` contains ``"embedding"`` (authoritative
           on current Ollama versions);
        2. fallback when no capabilities data at all helped: the model name
           contains one of ``embed / minilm / bge / mxbai / arctic``;
        3. last resort: probe every remaining model with ``POST /api/embed``
           and keep those that answer with an actual embedding.
        """
        entries = self.list_models()

        def _name(entry: dict) -> str:
            return str(entry.get("name") or entry.get("model") or "").strip()

        names = [n for n in (_name(entry) for entry in entries) if n]

        capable = [
            _name(entry)
            for entry in entries
            if _name(entry)
            and isinstance(entry.get("capabilities"), (list, tuple, set))
            and "embedding" in entry["capabilities"]
        ]
        if capable:
            return capable

        # Fallback 1: name heuristic (server reports no usable capabilities).
        heuristic = [n for n in names if any(hint in n.lower() for hint in _EMBED_NAME_HINTS)]
        if heuristic:
            return heuristic

        # Fallback 2: actively probe /api/embed for each remaining candidate.
        return [n for n in names if self._probe_embedding(n)]

    def _probe_embedding(self, model: str) -> bool:
        """Return True when ``model`` answers an embedding request."""
        try:
            response = self._request("POST", "/api/embed", json={"model": model, "input": "probe"})
            if response.status_code == 200:
                data = response.json()
                embeddings = data.get("embeddings") if isinstance(data, dict) else None
                if isinstance(embeddings, list) and embeddings:
                    return True
            # Very old servers only know the legacy endpoint.
            response = self._request(
                "POST", "/api/embeddings", json={"model": model, "prompt": "probe"}
            )
            if response.status_code == 200:
                data = response.json()
                embedding = data.get("embedding") if isinstance(data, dict) else None
                if isinstance(embedding, list) and embedding:
                    return True
        except (requests.RequestException, ValueError):
            return False
        return False

    # ------------------------------------------------------------- embedding
    def embed(self, text: str | list[str], model: str) -> list[list[float]]:
        """Embed one text or a list of texts; one vector per input, same order.

        Tries ``POST /api/embed`` with ``{"model", "input"}`` first and falls
        back to the legacy ``POST /api/embeddings`` with ``{"model", "prompt"}``
        (one call per item) when the server answers HTTP 404. Raises
        :class:`OllamaError` with a descriptive message on any other failure.
        """
        if not model:
            raise OllamaError("embed() requires a model name")
        inputs = [text] if isinstance(text, str) else list(text)
        if not inputs:
            return []

        try:
            response = self._request("POST", "/api/embed", json={"model": model, "input": inputs})
            if response.status_code == 200:
                data = response.json()
                embeddings = data.get("embeddings") if isinstance(data, dict) else None
                if isinstance(embeddings, list) and len(embeddings) == len(inputs):
                    return [[float(value) for value in vector] for vector in embeddings]
                got = len(embeddings) if isinstance(embeddings, list) else "no"
                raise OllamaError(
                    f"Ollama /api/embed returned {got} embeddings for {len(inputs)} input(s) "
                    f"(model {model!r})"
                )
            if response.status_code != 404:
                raise OllamaError(
                    f"Ollama /api/embed answered HTTP {response.status_code} for model "
                    f"{model!r}: {_snippet(response)}"
                )
        except requests.RequestException as exc:
            raise OllamaError(f"Cannot reach Ollama at {self.host}: {exc}") from exc
        except ValueError as exc:
            raise OllamaError(
                f"Ollama returned invalid JSON from /api/embed for model {model!r}"
            ) from exc

        # Legacy fallback (HTTP 404 from /api/embed): one call per item.
        vectors: list[list[float]] = []
        for item in inputs:
            try:
                response = self._request(
                    "POST", "/api/embeddings", json={"model": model, "prompt": item}
                )
            except requests.RequestException as exc:
                raise OllamaError(f"Cannot reach Ollama at {self.host}: {exc}") from exc
            if response.status_code != 200:
                raise OllamaError(
                    f"Ollama legacy /api/embeddings answered HTTP {response.status_code} "
                    f"for model {model!r}: {_snippet(response)}"
                )
            try:
                data = response.json()
            except ValueError as exc:
                raise OllamaError(
                    f"Ollama returned invalid JSON from /api/embeddings for model {model!r}"
                ) from exc
            embedding = data.get("embedding") if isinstance(data, dict) else None
            if not isinstance(embedding, list):
                raise OllamaError(
                    f"Ollama legacy /api/embeddings returned no embedding for model {model!r}"
                )
            vectors.append([float(value) for value in embedding])
        return vectors


def detect_ollama(host: str) -> tuple[bool, list[str]]:
    """One-shot startup probe: ``(server_running, embedding_model_names)``.

    Never raises. When the server is reachable but model detection fails, the
    pair is ``(True, [])`` so the UI can still report that Ollama is alive.
    """
    client = OllamaClient(host)
    if not client.is_running():
        return False, []
    try:
        return True, client.list_embedding_models()
    except OllamaError as exc:
        log.warning("Ollama is reachable at %s but model detection failed: %s", host, exc)
        return True, []
