#!/usr/bin/env python3
"""
Backfill residual metadata from parent experiments to snapshots.

This migration script updates existing ModelSnapshot records that are missing
residual-specific metadata (residual_connector_id, residual_base_model_id, training_mode).

The metadata is backfilled from the parent Experiment record.

Usage:
    python scripts/backfill_snapshot_metadata.py
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.database import SessionLocal
from app.models import ModelSnapshot, Experiment


def backfill_residual_metadata(db: Session, dry_run: bool = False) -> int:
    """
    Backfill residual metadata from parent experiments to snapshots.

    Args:
        db: Database session
        dry_run: If True, don't commit changes, just print what would be updated

    Returns:
        Number of snapshots updated
    """
    updated_count = 0

    # Find all residual experiments
    residual_experiments = db.query(Experiment).filter(
        Experiment.training_mode == "residual"
    ).all()

    print(f"Found {len(residual_experiments)} residual experiment(s)")

    for exp in residual_experiments:
        # Get all snapshots from this experiment
        snapshots = db.query(ModelSnapshot).filter(
            ModelSnapshot.experiment_id == exp.id
        ).all()

        print(f"\nExperiment '{exp.name}' (ID: {exp.id}):")
        print(f"  - training_mode: {exp.training_mode}")
        print(f"  - residual_connector_id: {exp.residual_connector_id}")
        print(f"  - residual_base_model_id: {exp.residual_base_model_id}")
        print(f"  - Found {len(snapshots)} snapshot(s)")

        for snap in snapshots:
            if snap.metrics_at_save is None:
                snap.metrics_at_save = {}

            # Track if we made any updates
            updated = False
            updates = []

            # Backfill missing fields from parent experiment
            if 'residual_connector_id' not in snap.metrics_at_save and exp.residual_connector_id:
                snap.metrics_at_save['residual_connector_id'] = exp.residual_connector_id
                updates.append(f"residual_connector_id={exp.residual_connector_id}")
                updated = True

            if 'residual_base_model_id' not in snap.metrics_at_save and exp.residual_base_model_id:
                snap.metrics_at_save['residual_base_model_id'] = exp.residual_base_model_id
                updates.append(f"residual_base_model_id={exp.residual_base_model_id}")
                updated = True

            if 'training_mode' not in snap.metrics_at_save:
                snap.metrics_at_save['training_mode'] = exp.training_mode
                updates.append(f"training_mode={exp.training_mode}")
                updated = True

            if updated:
                # Force SQLAlchemy to detect JSON change
                flag_modified(snap, 'metrics_at_save')
                updated_count += 1
                action = "Would update" if dry_run else "Updated"
                print(f"    {action} snapshot {snap.id} (iter {snap.iteration}): {', '.join(updates)}")
            else:
                print(f"    Snapshot {snap.id} (iter {snap.iteration}): already has metadata")

    if not dry_run:
        db.commit()
        print(f"\nCommitted {updated_count} update(s) to database")
    else:
        print(f"\n[DRY RUN] Would update {updated_count} snapshot(s)")

    return updated_count


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Backfill residual metadata from experiments to snapshots"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't commit changes, just show what would be updated"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Backfill Snapshot Metadata Migration")
    print("=" * 60)

    if args.dry_run:
        print("[DRY RUN MODE - No changes will be committed]\n")

    db = SessionLocal()
    try:
        updated = backfill_residual_metadata(db, dry_run=args.dry_run)
        print(f"\nTotal snapshots {'that would be ' if args.dry_run else ''}updated: {updated}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
