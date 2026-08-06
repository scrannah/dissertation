#!/usr/bin/env python3
"""
setup_conceptnet.py
Downloads ConceptNet 5.7 assertions and builds a SQLite database with indexes.
Run this once to set up your local ConceptNet database.
"""
import os
import sqlite3
import csv
import urllib.request
import gzip
import shutil
from pathlib import Path

# Configuration
DATA_DIR = Path.home() / "conceptnet_data"
CSV_GZ_URL = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
CSV_GZ_FILE = DATA_DIR / "conceptnet-assertions-5.7.0.csv.gz"
CSV_FILE = DATA_DIR / "conceptnet-assertions-5.7.0.csv"
DB_FILE = DATA_DIR / "conceptnet.db"


def download_file(url, dest):
    """Download file with progress indicator."""
    print(f"Downloading {url}...")
    print(f"Destination: {dest}")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        print(f"\rProgress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB)",
              end="")

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print("\nDownload complete!")


def decompress_gzip(gz_file, output_file):
    """Decompress .gz file."""
    print(f"Decompressing {gz_file.name}...")
    with gzip.open(gz_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Decompressed to {output_file}")


def create_database(csv_file, db_file):
    """Create SQLite database from CSV file."""
    print(f"Creating SQLite database: {db_file}")

    # Remove existing database
    if db_file.exists():
        print("Removing existing database...")
        db_file.unlink()

    # Connect to database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Create table
    print("Creating table schema...")
    cursor.execute("""
        CREATE TABLE edges (
            relation TEXT NOT NULL,
            start TEXT NOT NULL,
            end TEXT NOT NULL,
            weight REAL,
            dataset TEXT,
            license TEXT,
            sources TEXT,
            surfaceText TEXT
        )
    """)

    # Import CSV data
    print("Importing CSV data (this may take several minutes)...")
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')

        # Skip header if present
        header = next(reader, None)
        if header and header[0].startswith('URI'):
            print("Skipping header row...")
        else:
            # If no header, process first row
            if header:
                try:
                    weight = float(header[4]) if header[4] else None
                except (ValueError, IndexError):
                    weight = None

                cursor.execute("INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                               (header[1], header[2], header[3],
                                weight,
                                header[5] if len(header) > 5 else None,
                                header[6] if len(header) > 6 else None,
                                header[7] if len(header) > 7 else None,
                                header[8] if len(header) > 8 else None))

        # Process remaining rows
        count = 0
        batch = []
        batch_size = 10000

        for row in reader:
            if len(row) < 4:
                continue

            # Parse weight safely
            try:
                weight = float(row[4]) if row[4] else None
            except (ValueError, IndexError):
                weight = None

            batch.append((
                row[1],  # relation
                row[2],  # start
                row[3],  # end
                weight,  # weight
                row[5] if len(row) > 5 else None,  # dataset
                row[6] if len(row) > 6 else None,  # license
                row[7] if len(row) > 7 else None,  # sources
                row[8] if len(row) > 8 else None  # surfaceText
            ))

            count += 1

            if len(batch) >= batch_size:
                cursor.executemany("INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
                batch = []
                if count % 100000 == 0:
                    print(f"  Imported {count:,} rows...")

        # Insert remaining rows
        if batch:
            cursor.executemany("INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)

        print(f"  Total rows imported: {count:,}")

    conn.commit()

    # Create indexes
    print("Creating indexes (this may take a few minutes)...")
    print("  Creating index on 'start' column...")
    cursor.execute("CREATE INDEX idx_start ON edges(start)")

    print("  Creating index on 'end' column...")
    cursor.execute("CREATE INDEX idx_end ON edges(end)")

    print("  Creating index on 'relation' column...")
    cursor.execute("CREATE INDEX idx_relation ON edges(relation)")

    print("  Creating composite index on (start, relation)...")
    cursor.execute("CREATE INDEX idx_start_relation ON edges(start, relation)")

    conn.commit()

    # Statistics
    cursor.execute("SELECT COUNT(*) FROM edges")
    total = cursor.fetchone()[0]
    print(f"\nDatabase created successfully!")
    print(f"  Total edges: {total:,}")
    print(f"  Database file: {db_file}")
    print(f"  Database size: {db_file.stat().st_size / 1024 / 1024:.1f} MB")

    conn.close()


def main():
    print("=== ConceptNet Local Setup ===\n")

    # Create data directory
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Data directory: {DATA_DIR}\n")

    # Step 1: Download
    if not CSV_GZ_FILE.exists():
        download_file(CSV_GZ_URL, CSV_GZ_FILE)
    else:
        print(f"Compressed file already exists: {CSV_GZ_FILE}")

    # Step 2: Decompress
    if not CSV_FILE.exists():
        decompress_gzip(CSV_GZ_FILE, CSV_FILE)
    else:
        print(f"CSV file already exists: {CSV_FILE}")

    # Step 3: Create database
    if not DB_FILE.exists():
        create_database(CSV_FILE, DB_FILE)
    else:
        print(f"\nDatabase already exists: {DB_FILE}")
        response = input("Rebuild database? (y/N): ").strip().lower()
        if response == 'y':
            create_database(CSV_FILE, DB_FILE)
        else:
            print("Keeping existing database.")

    print("\n=== Setup Complete ===")
    print(f"You can now use conceptnet_query.py to query the database.")
    print(f"Database location: {DB_FILE}")

    # Optional: Clean up CSV files to save space
    print(f"\nOptional: You can delete the CSV files to save ~2GB of disk space:")
    print(f"  rm {CSV_FILE}")
    print(f"  rm {CSV_GZ_FILE}")


if __name__ == "__main__":
    main()
