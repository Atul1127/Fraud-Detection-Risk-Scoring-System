from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pymongo import MongoClient, UpdateOne


BATCH_SIZE = 1000
HISTORICAL_CREATED_AT = datetime(1970, 1, 1, tzinfo=timezone.utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap MongoDB with historical fraud transactions")
    parser.add_argument("--source", default="data/raw/train_transaction.csv", help="Historical transaction CSV")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--skip", type=int, default=0, help="Number of source rows to skip before starting; useful for resuming an interrupted backfill")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for a dry run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Historical transaction file not found: {source}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.skip < 0:
        raise ValueError("--skip must be non-negative")

    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    database_name = os.getenv("MONGODB_DATABASE", "fraudx")
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    db = client[database_name]
    collection = db["transactions"]

    collection.create_index("transaction_id", unique=True)
    collection.create_index("data.TransactionDT")
    for field in [
        "card1", "card2", "card3", "card5", "addr1", "addr2",
        "P_emaildomain", "R_emaildomain",
    ]:
        collection.create_index(f"data.{field}")

    usecols = None  # Keep all transaction columns so future feature additions can reuse the backfill.
    processed = 0
    inserted_or_updated = 0
    skipped = 0

    for chunk in pd.read_csv(source, chunksize=args.batch_size, usecols=usecols):
        if args.skip:
            if skipped + len(chunk) <= args.skip:
                skipped += len(chunk)
                continue
            drop_count = args.skip - skipped
            chunk = chunk.iloc[drop_count:]
            skipped = args.skip

        if args.limit is not None:
            remaining = args.limit - processed
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining]

        if chunk.empty:
            break
        if "TransactionID" not in chunk.columns or "TransactionDT" not in chunk.columns:
            raise ValueError("CSV must contain TransactionID and TransactionDT")

        operations = []
        for row in chunk.to_dict(orient="records"):
            transaction_id = str(row["TransactionID"])
            # NaN is not useful to Mongo queries and can make equality matching surprising.
            data = {key: (None if pd.isna(value) else value) for key, value in row.items()}
            operations.append(
                UpdateOne(
                    {"transaction_id": transaction_id},
                    {
                        "$set": {
                            "transaction_id": transaction_id,
                            "data": data,
                            "source": "historical_backfill",
                            "updated_at": datetime.now(timezone.utc),
                        },
                        "$setOnInsert": {"created_at": HISTORICAL_CREATED_AT},
                    },
                    upsert=True,
                )
            )

        result = collection.bulk_write(operations, ordered=False)
        inserted_or_updated += result.upserted_count + result.modified_count
        processed += len(chunk)
        print(f"Processed {processed:,} rows; upserts/updates this run: {inserted_or_updated:,}")

        if args.limit is not None and processed >= args.limit:
            break

    client.close()
    total_processed = args.skip + processed
    print(f"\nBackfill complete: {processed:,} source rows processed after skipping {args.skip:,}.")
    print(f"Source position reached: {total_processed:,} rows.")
    print("The backfilled records are timestamped outside the serving-monitoring window and are available to causal online feature queries.")


if __name__ == "__main__":
    main()
