# conceptnet.py
# =============
# Extracts object information from a local ConceptNet database.
# This file provides only one function: get_info(label)
# This file is not run directly - yolo.py imports it.

import sqlite3
from pathlib import Path

# Your local ConceptNet database will be here (created by running setup_conceptnet.py)
DB_PATH = Path.home() / "conceptnet_data" / "conceptnet.db"

# We only need these relations
RELATIONS = {
    "/r/IsA"           : "is a",
    "/r/UsedFor"       : "used for",
    "/r/AtLocation"    : "found at",
    "/r/CapableOf"     : "capable of",
    "/r/ReceivesAction": "receives action",
}


def get_info(label: str) -> dict:
    """
    Give an object label, it returns its information from ConceptNet.

    Example:
        get_info("cup")
        >> { "is a": ["container", "vessel"], "used for": ["drinking"] }

    Args:
        label : YOLO object name, e.g. "cup", "laptop", "person"

    Returns:
        dict of { relation: [concepts] }
        Or empty {} if database not found or object is unknown.
    """

    # First check if the database exists
    if not DB_PATH.exists():
        print("[ConceptNet] ERROR: Database not found!")
        print(f"[ConceptNet] First run setup_conceptnet.py. Path: {DB_PATH}")
        return {}

    # Convert label into ConceptNet URI format
    # e.g.  "cell phone"  ->  "/c/en/cell_phone"
    uri = "/c/en/" + label.strip().lower().replace(" ", "_")

    found = {}

    try:
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        for rel_uri, rel_name in RELATIONS.items():

            # Search in database where our object is the START node
            cursor.execute("""
                SELECT end, weight
                FROM edges
                WHERE start = ? AND relation = ?
                ORDER BY weight DESC
                LIMIT 3
            """, (uri, rel_uri))

            rows = cursor.fetchall()

            if rows:
                # Extract readable names from URI
                # "/c/en/drinking_liquid"  ->  "drinking liquid"
                concepts = []
                for (end_uri, _) in rows:
                    parts = end_uri.split("/")
                    if len(parts) >= 4:
                        concepts.append(parts[3].replace("_", " "))
                found[rel_name] = concepts

        conn.close()

    except Exception as e:
        print(f"[ConceptNet] Query error: {e}")

    return found