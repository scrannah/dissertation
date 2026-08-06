# ConceptNet Local Setup

This setup allows you to query ConceptNet locally using SQLite instead of the API.

## Files

1. **setup_conceptnet.py** - One-time setup script to download and build the database
2. **conceptnet_query.py** - Query script (replacement for your original API-based script)

## Setup Instructions

### Step 1: Run the Setup Script

```bash
python setup_conceptnet.py
```

This will:
- Download ConceptNet 5.7.0 assertions (~1.1 GB compressed)
- Decompress the file (~2 GB uncompressed)
- Create a SQLite database with indexes (~3 GB)
- Total time: 10-15 minutes depending on your internet and disk speed

The database will be created at: `~/conceptnet_data/conceptnet.db`

### Step 2: Query the Database

```bash
python conceptnet_query.py
```

You'll be prompted for:
- A term to search (e.g., "knife", "kettle", "car")
- Number of edges per relation (default: 7)

## What Changed from the API Version

### Same functionality:
- Same relations queried
- Same output format
- Same logic for forward vs. any-direction queries

### Key points:
- One-time setup required
- Uses ~3 GB disk space
- Data from 2019 (ConceptNet 5.7) - won't auto-update
- Some edges may have surfaceText, some may not (depends on source data)

## Disk Space Requirements

- Compressed download: ~1.1 GB
- Uncompressed CSV: ~2 GB (can be deleted after setup)
- SQLite database: ~3 GB

**Total: ~6 GB during setup, ~3 GB after cleanup**

To save space after setup, you can delete the CSV files:
```bash
rm ~/conceptnet_data/conceptnet-assertions-5.7.0.csv
rm ~/conceptnet_data/conceptnet-assertions-5.7.0.csv.gz
```

## Database Schema

```sql
CREATE TABLE edges (
    relation TEXT NOT NULL,      -- e.g., /r/IsA, /r/UsedFor
    start TEXT NOT NULL,          -- e.g., /c/en/knife
    end TEXT NOT NULL,            -- e.g., /c/en/tool
    weight REAL,                  -- confidence score
    dataset TEXT,                 -- source dataset
    license TEXT,                 -- license info
    sources TEXT,                 -- source URLs
    surfaceText TEXT              -- example sentence (may be NULL)
)

-- Indexes created for fast queries:
-- idx_start, idx_end, idx_relation, idx_start_relation
```

## Troubleshooting

**"Database not found" error:**
- Make sure you ran `setup_conceptnet.py` first
- Check that `~/conceptnet_data/conceptnet.db` exists

**Download fails:**
- Check internet connection
- Try downloading manually from: https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
- Place it in `~/conceptnet_data/`

**Setup takes too long:**
- Normal! Creating indexes on 36M rows takes time
- Expect 10-15 minutes on typical hardware
- You only need to do this once

**Out of disk space:**
- You need ~6 GB free during setup
- After setup, you can delete CSV files to reclaim ~3 GB

## Performance

Typical query times (all 11 relations):
- **API (when it worked)**: 2-8 seconds
- **Local SQLite**: 50-200 ms (10-50x faster)

## Data Version

This uses ConceptNet 5.7.0 from 2019. It's the last publicly available data dump.
