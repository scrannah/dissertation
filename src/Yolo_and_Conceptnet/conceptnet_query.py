#!/usr/bin/env python3
"""
conceptnet_query.py
Query local ConceptNet SQLite database (replacement for API-based version).
"""
import sqlite3
import textwrap
from pathlib import Path

# Database location (matches setup script)
DB_FILE = Path.home() / "conceptnet_data" / "conceptnet.db"

RELATIONS = [
    "/r/IsA", "/r/UsedFor", "/r/CapableOf", "/r/AtLocation",
    "/r/HasSubevent", "/r/HasFirstSubevent", "/r/HasLastSubevent",
    "/r/HasPrerequisite", "/r/MotivatedByGoal", "/r/SimilarTo", "/r/ReceivesAction"
]

# Relation labels for display
RELATION_LABELS = {
    "/r/IsA": "is a",
    "/r/UsedFor": "used for",
    "/r/CapableOf": "capable of",
    "/r/AtLocation": "at location",
    "/r/HasSubevent": "has subevent",
    "/r/HasFirstSubevent": "has first subevent",
    "/r/HasLastSubevent": "has last subevent",
    "/r/HasPrerequisite": "has prerequisite",
    "/r/MotivatedByGoal": "motivated by goal",
    "/r/SimilarTo": "similar to",
    "/r/ReceivesAction": "receives action"
}


def concept_uri(term: str) -> str:
    """Create ConceptNet URI for an English term."""
    return f"/c/en/{term.strip().replace(' ', '_')}"


def extract_label(uri: str) -> str:
    """Extract readable label from ConceptNet URI."""
    if not uri:
        return uri
    # /c/en/knife -> knife
    parts = uri.split('/')
    if len(parts) >= 4:
        return parts[-1].replace('_', ' ')
    return uri


def pretty_edge(edge, highlight_uri):
    """Format an edge for display."""
    start_uri = edge['start']
    end_uri = edge['end']
    rel = edge['relation']
    weight = edge['weight']
    surface = edge.get('surfaceText', '')

    start_label = extract_label(start_uri)
    end_label = extract_label(end_uri)
    rel_label = RELATION_LABELS.get(rel, rel)

    # Determine direction
    if start_uri == highlight_uri:
        direction = "→"
    elif end_uri == highlight_uri:
        direction = "←"
    else:
        direction = "↔"

    weight_str = f"w={weight:.2f}" if weight is not None else "w=N/A"
    line = f"- {start_label} \"{rel_label}\" {direction} {end_label}   ({weight_str})"

    if surface:
        gloss = textwrap.shorten(surface.replace("[[", "").replace("]]", ""), width=100)
        line += f"\n    e.g., {gloss}"

    return line


def query_edges(conn, uri: str, relation: str, limit: int):
    """
    Query edges for a given URI and relation.
    First tries forward direction (uri as start), then falls back to any direction.
    """
    cursor = conn.cursor()

    # Try forward direction first
    cursor.execute("""
        SELECT relation, start, end, weight, surfaceText
        FROM edges
        WHERE start = ? AND relation = ?
        ORDER BY weight DESC
        LIMIT ?
    """, (uri, relation, limit))

    edges = cursor.fetchall()

    if len(edges) >= 1:
        return [
            {
                'relation': row[0],
                'start': row[1],
                'end': row[2],
                'weight': row[3],
                'surfaceText': row[4]
            }
            for row in edges
        ], "forward"

    # Fallback to any direction
    cursor.execute("""
        SELECT relation, start, end, weight, surfaceText
        FROM edges
        WHERE (start = ? OR end = ?) AND relation = ?
        ORDER BY weight DESC
        LIMIT ?
    """, (uri, uri, relation, limit))

    edges = cursor.fetchall()

    return [
        {
            'relation': row[0],
            'start': row[1],
            'end': row[2],
            'weight': row[3],
            'surfaceText': row[4]
        }
        for row in edges
    ], "any"


def main():
    # Check if database exists
    if not DB_FILE.exists():
        print(f"Error: Database not found at {DB_FILE}")
        print("Please run setup_conceptnet.py first to create the database.")
        return

    # Interactive prompts
    term = input("Enter a term (e.g., knife, kettle, car): ").strip()
    if not term:
        print("No term provided. Exiting.")
        return

    try:
        k = int(input("How many edges per relation? [7]: ").strip() or "7")
    except ValueError:
        k = 7

    uri = concept_uri(term)
    print(f"\n=== ConceptNet samples for '{term}' -> {uri} (English) ===\n")

    # Connect to database
    conn = sqlite3.connect(DB_FILE)

    try:
        for rel in RELATIONS:
            edges, mode = query_edges(conn, uri, rel, k)
            tag = f"(top {k} edges; {'forward' if mode == 'forward' else 'fallback: any direction'})"
            print(f"{rel} {tag}")

            if not edges:
                print("  (no edges found)")
            else:
                for edge in edges:
                    print(" ", pretty_edge(edge, uri))
            print()

    finally:
        conn.close()


if __name__ == "__main__":
    main()
