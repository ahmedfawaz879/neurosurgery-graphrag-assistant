"""Neo4j Cypher export for the knowledge graph.

`to_neo4j_cypher()` is ported from the notebook's Section 5
(`notebooks/neurosurgery_graphrag_assistant.ipynb`, cell 19) and is a real,
complete export path. It is NOT the default graph backend -- NetworkX remains
default (`Config.GRAPH_BACKEND == "networkx"`) so the repo runs with zero
external services out of the box.

`push_to_neo4j()` actually executes the generated statements against a
running Neo4j instance, gated behind `Config.GRAPH_BACKEND == "neo4j"`. This
function has NOT been run against a live Neo4j instance in this repo -- treat
it as an untested code path until it has been (see the README's Honesty /
Limitations section).
"""

from __future__ import annotations

import json

import networkx as nx

from src.config import Config


def to_neo4j_cypher(g: nx.MultiDiGraph) -> list[str]:
    """Generates Cypher statements for Neo4j. Does NOT connect to Neo4j."""
    statements: list[str] = []

    for node, data in g.nodes(data=True):
        node_id = json.dumps(node)
        label = json.dumps(data.get("label", node))
        statements.append(f"MERGE (n:Concept {{id: {node_id}}}) SET n.label = {label}")

    for u, v, data in g.edges(data=True):
        predicate = json.dumps(data.get("predicate", ""))
        source_paper = json.dumps(data.get("source_paper", ""))
        statements.append(
            f"MATCH (a:Concept {{id: {json.dumps(u)}}}), (b:Concept {{id: {json.dumps(v)}}}) "
            f"MERGE (a)-[:RELATION {{predicate: {predicate}, source_paper: {source_paper}}}]->(b)"
        )

    return statements


def push_to_neo4j(graph: nx.MultiDiGraph, uri: str, user: str, password: str) -> int:
    """Executes the generated Cypher statements against a running Neo4j instance.

    Returns the number of statements executed. Requires the optional `neo4j`
    driver dependency and a reachable Neo4j instance -- callers should gate this
    behind `Config.GRAPH_BACKEND == "neo4j"` (see `push_to_neo4j_if_configured`).
    """
    from neo4j import GraphDatabase

    statements = to_neo4j_cypher(graph)
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            for statement in statements:
                session.run(statement)
    finally:
        driver.close()
    return len(statements)


def push_to_neo4j_if_configured(graph: nx.MultiDiGraph, config: Config) -> int | None:
    """Pushes to Neo4j only if `config.GRAPH_BACKEND == "neo4j"`; otherwise a no-op
    returning `None`, so callers can invoke this unconditionally."""
    if config.GRAPH_BACKEND != "neo4j":
        return None
    return push_to_neo4j(graph, config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD)
