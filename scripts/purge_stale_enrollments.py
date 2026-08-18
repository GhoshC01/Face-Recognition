"""Data-retention maintenance tool: minimizing biometric data retention
requires actively removing enrollments that are no longer needed. This
service has no automatic expiry on its own -- HRMS decides who should still
be enrolled -- so this script is the operational mechanism for enforcing a
retention window against the FAISS index directly.

Lists (and optionally deletes) enrolled identities whose most recent
embedding is older than a configurable number of days.

Usage:
    python scripts/purge_stale_enrollments.py --older-than-days 365
    python scripts/purge_stale_enrollments.py --older-than-days 365 --delete
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.settings import get_settings  # noqa: E402
from app.core.vector_store import VectorStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--older-than-days", type=int, required=True)
    parser.add_argument(
        "--delete", action="store_true", help="Actually remove stale identities (default: list only, no changes)"
    )
    args = parser.parse_args()

    settings = get_settings()
    store = VectorStore(
        dimension=settings.embedding_dimension,
        index_dir=settings.faiss_index_dir,
        metadata_dir=settings.metadata_dir,
        index_filename=settings.faiss_index_filename,
        metadata_filename=settings.metadata_filename,
    )

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.older_than_days)
    stale: list[tuple[str, str]] = []

    for external_id in store.list_identities():
        last_enrolled_raw = store.get_last_enrolled_at(external_id)
        if last_enrolled_raw is None:
            continue
        if datetime.fromisoformat(last_enrolled_raw) < cutoff:
            stale.append((external_id, last_enrolled_raw))

    if not stale:
        print(f"No identities with enrollments older than {args.older_than_days} days.")
        return

    for external_id, last_enrolled_raw in stale:
        print(f"{external_id}\tlast enrolled: {last_enrolled_raw}")

    if args.delete:
        for external_id, _ in stale:
            removed = store.remove_embedding(external_id)
            print(f"Removed {removed} embedding(s) for {external_id}")
            if settings.enrollment_images_dir:
                photo_dir = Path(settings.enrollment_images_dir) / external_id
                if photo_dir.is_dir():
                    shutil.rmtree(photo_dir)
                    print(f"Removed saved photos for {external_id}")
    else:
        print(f"\n{len(stale)} stale identities found (listed above). Re-run with --delete to remove them.")


if __name__ == "__main__":
    main()
