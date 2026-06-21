from __future__ import annotations

import argparse

from .config import Settings
from .data import chunk_records, dataset_fingerprint, load_records
from .factory import build_components


def main() -> None:
    parser = argparse.ArgumentParser(description="Index the Egyptian private-university dataset")
    parser.add_argument("--force", action="store_true", help="reserved for explicit rebuild workflows")
    args = parser.parse_args()
    settings = Settings.from_env()
    settings.validate()
    records = load_records(settings.data_path, settings.links_path)
    chunks = chunk_records(records, settings.chunk_size, settings.chunk_overlap)
    embedder, store = build_components(settings)
    fingerprint = dataset_fingerprint(chunks, embedder.name)
    if args.force and store.collection.count():
        store.collection.modify(metadata={"hnsw:space": "cosine"})
    changed = store.index(chunks, fingerprint)
    action = "Indexed" if changed else "Index unchanged; reused"
    print(f"{action} {len(chunks)} chunks from {len(records)} records in {settings.collection_name}.")


if __name__ == "__main__":
    main()
