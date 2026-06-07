"""One APIRouter per resource; assembled by rag.api.app."""
from rag.api.routes import documents, health, search

__all__ = ["health", "documents", "search"]
