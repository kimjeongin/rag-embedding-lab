"""A small, coherent toy dataset for smoke-testing the training pipeline.

Deterministic (seeded) so runs reproduce. Stdlib only — no model or network. The
CLI entrypoint (``rag-gen-data``) writes the result via rag.dataset.write_jsonl.
"""
from __future__ import annotations

import random

# (title, content, [paraphrased queries]) — tech topics, mirroring the RAG corpus.
_DOCS: list[tuple[str, str, list[str]]] = [
    ("asyncio — Asynchronous I/O",
     "asyncio runs concurrent code with async/await on a single-threaded event loop, scheduling coroutines for I/O-bound work.",
     ["how does async await work in python", "python event loop and coroutines", "concurrent io-bound code in python"]),
    ("Python type hints",
     "Type hints annotate variables and function signatures so static checkers like mypy can catch bugs before runtime.",
     ["adding type annotations in python", "static type checking with mypy", "what are python type hints"]),
    ("PostgreSQL indexes",
     "An index lets PostgreSQL find rows without scanning the whole table, greatly speeding up lookups and joins.",
     ["speed up slow postgres queries", "how do database indexes work", "make a sql query faster with an index"]),
    ("PostgreSQL full text search",
     "Full text search uses tsvector and tsquery to query natural-language documents and rank them by relevance.",
     ["search text columns in postgres", "tsvector and tsquery relevance ranking", "natural language search in a sql database"]),
    ("FastAPI async endpoints",
     "FastAPI lets you write endpoint functions with async def, improving throughput for I/O-bound APIs via the event loop.",
     ["write an async api with fastapi", "async def endpoint throughput", "concurrency in a python web framework"]),
    ("Docker images and containers",
     "A Docker image is an immutable filesystem snapshot; a container is a running instance started from that image.",
     ["difference between docker image and container", "how docker containers work", "package an app into a container"]),
    ("Git branching and merging",
     "Branches let you develop features in isolation, then merge them back, with merge commits recording the integration.",
     ["create and merge a git branch", "isolate feature work in git", "how does git merging work"]),
    ("HTTP status codes",
     "HTTP status codes signal request outcomes: 2xx success, 3xx redirect, 4xx client error, 5xx server error.",
     ["what does http 404 mean", "categories of http response codes", "client vs server error status codes"]),
    ("REST API design",
     "REST models resources as URLs and uses HTTP verbs (GET, POST, PUT, DELETE) to operate on them statelessly.",
     ["principles of rest api design", "http verbs for crud operations", "what makes an api restful"]),
    ("Redis caching",
     "Redis is an in-memory key-value store often used as a cache to serve hot data with sub-millisecond latency.",
     ["use redis as a cache", "in-memory key value store for caching", "reduce database load with caching"]),
    ("SQL joins",
     "A JOIN combines rows from two tables on a related column; INNER keeps matches, LEFT keeps all left-side rows.",
     ["difference between inner and left join", "combine two sql tables", "how do sql joins work"]),
    ("Regular expressions",
     "Regular expressions describe text patterns for searching and matching, with classes, quantifiers, and groups.",
     ["match a pattern with regex", "what are regular expressions", "search text using a pattern"]),
    ("JSON data format",
     "JSON encodes structured data as nested objects and arrays of strings, numbers, booleans, and null.",
     ["what is the json format", "represent structured data as text", "objects and arrays in json"]),
    ("Linux file permissions",
     "Linux permissions grant read, write, and execute to the owner, group, and others, shown as rwx triplets.",
     ["change file permissions in linux", "what does chmod 755 mean", "read write execute permission bits"]),
    ("SSH key authentication",
     "SSH key pairs authenticate without passwords: the public key sits on the server, the private key stays with you.",
     ["set up ssh key login", "passwordless ssh authentication", "public and private key for ssh"]),
    ("Kubernetes pods",
     "A pod is the smallest deployable unit in Kubernetes, wrapping one or more containers that share network and storage.",
     ["what is a kubernetes pod", "smallest deployable unit in k8s", "containers sharing a network namespace"]),
]


def generate_dataset(test_fraction: float = 0.25, seed: int = 13):
    """Return (train, test) lists of {query, positive} records, deterministically split."""
    pairs = [
        {"query": q, "positive": {"title": title, "content": content}}
        for title, content, queries in _DOCS
        for q in queries
    ]
    random.Random(seed).shuffle(pairs)
    n_test = max(1, round(len(pairs) * test_fraction))
    return pairs[n_test:], pairs[:n_test]
