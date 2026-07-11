"""Qdrant adapter — a thin, transparent client over Qdrant's REST API.

Deliberately httpx-only (no qdrant-client): the official client pulls a grpc stack
and hides the HTTP surface, while every call the lab needs is one obvious REST
request — the same trade the Ollama adapter makes. Each method maps to exactly one
Qdrant endpoint, so what happens on the wire is readable here, not in a library.

Transport failures become ``VectorStoreError`` (domain terms), so callers never see
httpx. All methods are sync — indexing is a batch CLI and the search route hops to a
threadpool for the one store call it makes.

A collection stores L2-normalised vectors with cosine distance; **aliases** are the
serving pointer: search always queries "{prefix}-live" (aliases are accepted anywhere
a collection name is), and a reindex builds a fresh collection then repoints the alias
in one atomic action — Qdrant's sanctioned blue-green swap.
"""
from __future__ import annotations

import httpx

from rag.core.errors import VectorStoreError

_TIMEOUT = 30.0


class QdrantStore:
    def __init__(self, url: str, client: httpx.Client | None = None) -> None:
        # An injected client (tests: httpx.MockTransport) must carry its own base_url.
        self._client = client or httpx.Client(base_url=url, timeout=_TIMEOUT)
        self._url = url

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "QdrantStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _request(self, method: str, path: str, json: dict | None = None) -> dict:
        try:
            resp = self._client.request(method, path, json=json)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300]
            raise VectorStoreError(f"Qdrant {method} {path} → {exc.response.status_code}: {detail}") from exc
        except httpx.HTTPError as exc:
            raise VectorStoreError(
                f"Qdrant({self._url})에 연결할 수 없습니다 ({type(exc).__name__}) — "
                "`make qdrant`로 로컬 인스턴스를 띄우거나 QDRANT_URL을 확인하세요"
            ) from exc

    # ── collections ────────────────────────────────────────────────────────────
    def ping(self) -> bool:
        """True if the Qdrant instance answers at all."""
        try:
            self._request("GET", "/collections")
            return True
        except VectorStoreError:
            return False

    def list_collections(self) -> list[str]:
        result = self._request("GET", "/collections")["result"]
        return sorted(c["name"] for c in result.get("collections", []))

    def collection_info(self, name: str) -> dict | None:
        """{"points": int, "dim": int} — or None if the collection doesn't exist.
        ``name`` may be an alias (Qdrant resolves them everywhere)."""
        try:
            result = self._request("GET", f"/collections/{name}")["result"]
        except VectorStoreError as exc:
            if "404" in str(exc):
                return None
            raise
        vectors = result["config"]["params"]["vectors"]
        return {"points": result.get("points_count") or 0, "dim": int(vectors["size"])}

    def create_collection(self, name: str, dim: int) -> None:
        self._request("PUT", f"/collections/{name}",
                      json={"vectors": {"size": dim, "distance": "Cosine"}})

    def delete_collection(self, name: str) -> None:
        self._request("DELETE", f"/collections/{name}")

    # ── points ─────────────────────────────────────────────────────────────────
    def upsert(self, name: str, points: list[dict]) -> None:
        """Upsert [{"id", "vector", "payload"}] — ?wait=true so a subsequent count
        (the idempotency check) sees these points, not an in-flight WAL."""
        self._request("PUT", f"/collections/{name}/points?wait=true", json={"points": points})

    def query(self, target: str, vector: list[float], top_k: int) -> list[dict]:
        """Nearest neighbours of ``vector`` in collection/alias ``target``:
        [{"id", "score", "payload"}], best first (cosine similarity)."""
        result = self._request(
            "POST", f"/collections/{target}/points/query",
            json={"query": vector, "limit": top_k, "with_payload": True},
        )["result"]
        return [
            {"id": p["id"], "score": p["score"], "payload": p.get("payload") or {}}
            for p in result.get("points", [])
        ]

    # ── aliases (the serving pointer) ──────────────────────────────────────────
    def alias_target(self, alias: str) -> str | None:
        """The collection an alias points at (None if the alias doesn't exist)."""
        result = self._request("GET", "/aliases")["result"]
        for entry in result.get("aliases", []):
            if entry["alias_name"] == alias:
                return entry["collection_name"]
        return None

    def swap_alias(self, alias: str, collection: str) -> None:
        """Atomically repoint ``alias`` → ``collection`` (delete+create in ONE
        request, so there is no moment where the alias resolves to nothing)."""
        actions: list[dict] = []
        if self.alias_target(alias) is not None:
            actions.append({"delete_alias": {"alias_name": alias}})
        actions.append({"create_alias": {"alias_name": alias, "collection_name": collection}})
        self._request("POST", "/collections/aliases", json={"actions": actions})
