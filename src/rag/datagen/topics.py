"""The shared topic set behind BOTH the toy training data and the sample eval set.

One source of truth — 16 short tech docs, each with two **disjoint** query lists:

  - ``train_queries`` — phrasings the toy fine-tune (``dummy.py``) learns from.
  - ``eval_queries``  — held-out phrasings the sample eval (``eval_corpus.py``) scores.

Keeping them disjoint matters: if the toy training pairs and the eval queries were the
same strings (an earlier version shared them verbatim), the sample eval would reward
*memorisation* rather than retrieval — a fine-tune would look good for the wrong reason.
With the split, the eval measures generalisation to unseen phrasings of a known doc, and
there is no duplicated doc text between the two generators. Pure stdlib — no model/network.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Topic:
    """A doc plus the queries it answers, split into a train side and an eval side."""

    title: str
    content: str
    train_queries: tuple[str, ...]   # toy training pairs are built from these
    eval_queries: tuple[str, ...]    # the sample eval scores on these (disjoint phrasings)


# (title, content, train_queries, eval_queries) — same subjects, different phrasings per side.
TOPICS: tuple[Topic, ...] = (
    Topic(
        "asyncio — Asynchronous I/O",
        "asyncio runs concurrent code with async/await on a single-threaded event loop, scheduling coroutines for I/O-bound work.",
        ("how does async await work in python", "python event loop and coroutines", "concurrent io-bound code in python"),
        ("single-threaded concurrency model in python", "schedule coroutines for network calls", "non-blocking i/o using async def"),
    ),
    Topic(
        "Python type hints",
        "Type hints annotate variables and function signatures so static checkers like mypy catch bugs before runtime.",
        ("add type annotations in python", "static type checking with mypy", "what are python type hints"),
        ("catch bugs before runtime with annotations", "annotate function signatures in python", "tool that verifies variable types"),
    ),
    Topic(
        "PostgreSQL indexes",
        "An index lets PostgreSQL find rows without scanning the whole table, speeding up lookups and joins.",
        ("speed up a slow postgres query", "how do database indexes work", "make a sql lookup faster with an index"),
        ("avoid a full table scan in postgres", "why is my postgres join slow", "find rows without reading the whole table"),
    ),
    Topic(
        "PostgreSQL full text search",
        "Full text search uses tsvector and tsquery to search natural-language documents and rank them by relevance.",
        ("search text columns in postgres", "tsvector and tsquery relevance ranking", "natural language search in a sql database"),
        ("rank documents by relevance in postgres", "query prose stored in a postgres column", "keyword search over text with tsquery"),
    ),
    Topic(
        "FastAPI async endpoints",
        "FastAPI lets you write endpoints with async def, improving throughput for I/O-bound APIs via the event loop.",
        ("write an async api with fastapi", "async def endpoint throughput", "concurrency in a python web framework"),
        ("handle many concurrent requests in fastapi", "improve io-bound api throughput", "event-loop backed web endpoints"),
    ),
    Topic(
        "Docker images and containers",
        "A Docker image is an immutable filesystem snapshot; a container is a running instance started from that image.",
        ("difference between a docker image and a container", "how do docker containers work", "package an app into a container"),
        ("immutable filesystem snapshot vs running instance", "start a container from an image", "what is a docker image made of"),
    ),
    Topic(
        "Git branching and merging",
        "Branches isolate feature work; merging integrates them back, with a merge commit recording the integration.",
        ("create and merge a git branch", "isolate feature work in git", "how does git merging work"),
        ("record an integration with a merge commit", "develop a feature without touching main", "combine two branches back together"),
    ),
    Topic(
        "HTTP status codes",
        "HTTP status codes signal request outcomes: 2xx success, 3xx redirect, 4xx client error, 5xx server error.",
        ("what does http 404 mean", "categories of http response codes", "client vs server error status codes"),
        ("what does a 5xx response indicate", "how are http response codes grouped", "signal a redirect with a status code"),
    ),
    Topic(
        "REST API design",
        "REST models resources as URLs and uses HTTP verbs (GET, POST, PUT, DELETE) to operate on them statelessly.",
        ("principles of rest api design", "http verbs for crud operations", "what makes an api restful"),
        ("model resources as urls", "stateless operations over http verbs", "use get post put delete on a resource"),
    ),
    Topic(
        "Redis caching",
        "Redis is an in-memory key-value store often used as a cache to serve hot data with sub-millisecond latency.",
        ("use redis as a cache", "serve hot data with sub-millisecond latency", "reduce database load with a cache"),
        ("in-memory key-value store for hot data", "cut read latency to under a millisecond", "cache layer in front of a database"),
    ),
    Topic(
        "SQL joins",
        "A JOIN combines rows from two tables on a related column; INNER keeps matches, LEFT keeps all left rows.",
        ("difference between inner and left join", "combine two sql tables on a key", "how do sql joins work"),
        ("keep all left-side rows in a join", "match rows across tables on a column", "inner join versus outer join behaviour"),
    ),
    Topic(
        "Regular expressions",
        "Regular expressions describe text patterns for searching and matching, with character classes, quantifiers, and groups.",
        ("match a pattern with a regular expression", "what are regular expressions", "search text using a pattern"),
        ("character classes and quantifiers in regex", "capture groups in a pattern", "validate a string against a pattern"),
    ),
    Topic(
        "JSON data format",
        "JSON encodes structured data as nested objects and arrays of strings, numbers, booleans, and null.",
        ("what is the json data format", "represent structured data as text", "nested objects and arrays in json"),
        ("encode data as objects and arrays", "text format with strings numbers and null", "serialize a nested structure to text"),
    ),
    Topic(
        "Linux file permissions",
        "Linux permissions grant read, write, and execute to owner, group, and others, shown as rwx triplets.",
        ("change file permissions in linux", "what does chmod 755 mean", "read write execute permission bits"),
        ("rwx triplets for owner group and others", "grant execute permission to a file", "who can read or write a linux file"),
    ),
    Topic(
        "SSH key authentication",
        "SSH key pairs authenticate without passwords: the public key sits on the server, the private key stays with you.",
        ("set up ssh key login", "passwordless ssh authentication", "public and private keys for ssh"),
        ("log in over ssh without a password", "where the public key lives on the server", "keep the private key on the client"),
    ),
    Topic(
        "Kubernetes pods",
        "A pod is the smallest deployable unit in Kubernetes, wrapping one or more containers that share network and storage.",
        ("what is a kubernetes pod", "smallest deployable unit in kubernetes", "containers sharing a network namespace"),
        ("containers that share storage and network", "smallest thing you can deploy in k8s", "wrap multiple containers as one unit"),
    ),
)
