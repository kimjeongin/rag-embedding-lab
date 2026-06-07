"""Generate a SAMPLE BEIR-format eval set: real gold docs + many distractors.

This is placeholder data so the harness runs end-to-end today. The point of the
distractors is to make the corpus a real "haystack" — without them every metric
saturates near 1.0 (see docs/evaluation.md). Distractor subjects are kept *disjoint*
from the gold subjects (but in the same tech register) so they're plausible noise
that nonetheless answers none of the queries.

Replace the output with your in-house data in the same layout (corpus.jsonl /
queries.jsonl / qrels/test.tsv) for a real measurement. Deterministic (seeded),
stdlib only — no model or network. Written by the ``rag-gen-eval`` CLI entrypoint.
"""
from __future__ import annotations

import random

# Gold docs — each directly answers its handful of queries (the needles to retrieve).
# (title, content, [queries])
_GOLD: list[tuple[str, str, list[str]]] = [
    ("asyncio — Asynchronous I/O",
     "asyncio runs concurrent code with async/await on a single-threaded event loop, scheduling coroutines for I/O-bound work.",
     ["how does async await work in python", "python event loop and coroutines", "run concurrent io-bound code in python"]),
    ("Python type hints",
     "Type hints annotate variables and function signatures so static checkers like mypy catch bugs before runtime.",
     ["add type annotations in python", "static type checking with mypy", "what are python type hints"]),
    ("PostgreSQL indexes",
     "An index lets PostgreSQL find rows without scanning the whole table, speeding up lookups and joins.",
     ["speed up a slow postgres query", "how do database indexes work", "make a sql lookup faster with an index"]),
    ("PostgreSQL full text search",
     "Full text search uses tsvector and tsquery to search natural-language documents and rank them by relevance.",
     ["search text columns in postgres", "tsvector and tsquery relevance ranking", "natural language search in a sql database"]),
    ("FastAPI async endpoints",
     "FastAPI lets you write endpoints with async def, improving throughput for I/O-bound APIs via the event loop.",
     ["write an async api with fastapi", "async def endpoint throughput", "concurrency in a python web framework"]),
    ("Docker images and containers",
     "A Docker image is an immutable filesystem snapshot; a container is a running instance started from that image.",
     ["difference between a docker image and a container", "how do docker containers work", "package an app into a container"]),
    ("Git branching and merging",
     "Branches isolate feature work; merging integrates them back, with a merge commit recording the integration.",
     ["create and merge a git branch", "isolate feature work in git", "how does git merging work"]),
    ("HTTP status codes",
     "HTTP status codes signal request outcomes: 2xx success, 3xx redirect, 4xx client error, 5xx server error.",
     ["what does http 404 mean", "categories of http response codes", "client vs server error status codes"]),
    ("REST API design",
     "REST models resources as URLs and uses HTTP verbs (GET, POST, PUT, DELETE) to operate on them statelessly.",
     ["principles of rest api design", "http verbs for crud operations", "what makes an api restful"]),
    ("Redis caching",
     "Redis is an in-memory key-value store often used as a cache to serve hot data with sub-millisecond latency.",
     ["use redis as a cache", "serve hot data with sub-millisecond latency", "reduce database load with a cache"]),
    ("SQL joins",
     "A JOIN combines rows from two tables on a related column; INNER keeps matches, LEFT keeps all left rows.",
     ["difference between inner and left join", "combine two sql tables on a key", "how do sql joins work"]),
    ("Regular expressions",
     "Regular expressions describe text patterns for searching and matching, with character classes, quantifiers, and groups.",
     ["match a pattern with a regular expression", "what are regular expressions", "search text using a pattern"]),
    ("JSON data format",
     "JSON encodes structured data as nested objects and arrays of strings, numbers, booleans, and null.",
     ["what is the json data format", "represent structured data as text", "nested objects and arrays in json"]),
    ("Linux file permissions",
     "Linux permissions grant read, write, and execute to owner, group, and others, shown as rwx triplets.",
     ["change file permissions in linux", "what does chmod 755 mean", "read write execute permission bits"]),
    ("SSH key authentication",
     "SSH key pairs authenticate without passwords: the public key sits on the server, the private key stays with you.",
     ["set up ssh key login", "passwordless ssh authentication", "public and private keys for ssh"]),
    ("Kubernetes pods",
     "A pod is the smallest deployable unit in Kubernetes, wrapping one or more containers that share network and storage.",
     ["what is a kubernetes pod", "smallest deployable unit in kubernetes", "containers sharing a network namespace"]),
]

# Distractor subjects — same tech register as the gold docs, but DISJOINT from their
# subjects, so a distractor never actually answers a gold query.
_DISTRACTOR_SUBJECTS: list[str] = [
    "Apache Kafka", "RabbitMQ", "Apache Cassandra", "MongoDB", "Elasticsearch",
    "MySQL", "Nginx", "HAProxy", "Terraform", "Ansible", "Prometheus", "Grafana",
    "gRPC", "GraphQL", "OAuth 2.0", "CockroachDB", "Apache Spark", "Apache Airflow",
    "HashiCorp Vault", "Consul", "etcd", "Istio", "Helm", "Jenkins", "Webpack",
    "Celery", "RocksDB", "ClickHouse",
]

# Operational aspects — each (subject, aspect) pair becomes one short distractor doc.
_DISTRACTOR_ASPECTS: list[tuple[str, str]] = [
    ("deployment", "Deploying {s} to production covers packaging, rollout strategy, and zero-downtime releases."),
    ("scaling", "Scaling {s} horizontally means sharding, partitioning, and balancing load across nodes."),
    ("configuration", "Configuring {s} involves its key settings, tuning parameters, and sensible production defaults."),
    ("monitoring", "Monitoring {s} means exporting metrics, building dashboards, and alerting on unhealthy instances."),
    ("security", "Securing {s} covers authentication, authorization, encryption in transit, and network policy."),
    ("backup and restore", "Backup and restore for {s} uses snapshots, point-in-time recovery, and disaster drills."),
    ("networking", "Networking for {s} covers ports, service discovery, and connection pooling."),
    ("troubleshooting", "Troubleshooting {s} means reading logs, recognising common failure modes, and recovery steps."),
    ("upgrades", "Upgrading {s} safely needs version compatibility checks, rolling upgrades, and a rollback plan."),
    ("high availability", "High availability for {s} relies on replication, failover, and quorum."),
    ("performance tuning", "Performance tuning {s} means profiling hotspots, caching, and reducing tail latency."),
    ("cost optimization", "Cost optimization for {s} covers right-sizing, autoscaling, and cleaning up idle resources."),
    ("logging", "Centralised logging for {s} uses structured logs, log shipping, and retention policies."),
    ("access control", "Access control in {s} is managed with roles, policies, and least-privilege grants."),
    ("data modeling", "Data modeling with {s} weighs schemas, keys, and trade-offs for the query patterns you expect."),
    ("capacity planning", "Capacity planning for {s} forecasts throughput, storage growth, and headroom."),
]


def _distractor_docs() -> list[dict]:
    """Every (subject, aspect) combination as a {_id, title, text} corpus record."""
    return [
        {
            "_id": f"distractor-{k}",
            "title": f"{subject} — {aspect}",
            "text": template.format(s=subject),
        }
        for k, (subject, (aspect, template)) in enumerate(
            (s, a) for s in _DISTRACTOR_SUBJECTS for a in _DISTRACTOR_ASPECTS
        )
    ]


def generate(
    n_distractors: int | None = None, seed: int = 13
) -> tuple[list[dict], list[dict], list[tuple[str, str, int]]]:
    """Return (corpus, queries, qrels_rows) for a BEIR-format eval set.

    `n_distractors` caps the haystack size (default: use all
    len(subjects) * len(aspects) distractors). Deterministic given `seed`.
    """
    rng = random.Random(seed)

    corpus: list[dict] = []
    queries: list[dict] = []
    qrels: list[tuple[str, str, int]] = []

    for i, (title, content, qs) in enumerate(_GOLD):
        doc_id = f"gold-{i}"
        corpus.append({"_id": doc_id, "title": title, "text": content})
        for j, query in enumerate(qs):
            query_id = f"q-{i}-{j}"
            queries.append({"_id": query_id, "text": query})
            qrels.append((query_id, doc_id, 1))

    distractors = _distractor_docs()
    if n_distractors is not None and n_distractors < len(distractors):
        distractors = rng.sample(distractors, n_distractors)
    corpus.extend(distractors)

    rng.shuffle(corpus)  # don't leave all the gold docs at the front of the file
    return corpus, queries, qrels
