"""Generate a SAMPLE BEIR-format eval set: real gold docs + many distractors.

This is placeholder data so the harness runs end-to-end today. The gold docs come from
the shared ``topics`` set (using their ``eval_queries`` — disjoint from the toy training
queries, so the measurement isn't leaked). The point of the distractors is to make the
corpus a real "haystack" — without them every metric saturates near 1.0 (see
docs/evaluation.md). Distractor subjects are kept *disjoint* from the gold subjects (but
in the same tech register) so they're plausible noise that nonetheless answers none of
the queries.

Replace the output with your in-house data in the same layout (corpus.jsonl /
queries.jsonl / qrels/test.tsv) for a real measurement. Deterministic (seeded),
stdlib only — no model or network. Written by the ``rag-gen-eval`` CLI entrypoint.
"""
from __future__ import annotations

import random

from rag.datagen.topics import TOPICS

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

    for i, topic in enumerate(TOPICS):
        doc_id = f"gold-{i}"
        corpus.append({"_id": doc_id, "title": topic.title, "text": topic.content})
        for j, query in enumerate(topic.eval_queries):
            query_id = f"q-{i}-{j}"
            queries.append({"_id": query_id, "text": query})
            qrels.append((query_id, doc_id, 1))

    distractors = _distractor_docs()
    if n_distractors is not None and n_distractors < len(distractors):
        distractors = rng.sample(distractors, n_distractors)
    corpus.extend(distractors)

    rng.shuffle(corpus)  # don't leave all the gold docs at the front of the file
    return corpus, queries, qrels
