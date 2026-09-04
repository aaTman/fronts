"""Roll back the fronts icechunk store's "main" branch past the corrupted 2025 append.

Root cause (see docs/rse/specs/implement-front-xml-to-netcdf-virtualizarr.md, Issue 7): the
2025 valid times were parsed from the wrong filename field (issuance time, not the synoptic
cycle hour), producing spurious sub-hour timestamps; appending those to the store's existing
whole-hour, int64 "hours since ..." time array triggered a virtualizarr writer bug that
silently re-scaled the new values without updating the array's stored units, corrupting how
the newly-appended (but not the pre-existing) times decode.

This does NOT delete any data: icechunk snapshots are immutable, so `reset_branch` only moves
the "main" branch pointer back to the snapshot before the corrupted commit. The corrupted
snapshot remains reachable by its own ID if ever needed. `from_snapshot_id` guards against
running this against a branch that has moved on since this script was written.

Run on the HPC system:

    pixi run -e data python scripts/rollback_fronts_time_corruption.py
"""

import icechunk as ic

STORE_PATH = "/ourdisk/hpc/ai2es/tman/restructured_front_data/icechunk"
CORRUPTED_SNAPSHOT_ID = "BS4CC6J9B39KVHHSVR20"  # "Add 2025 front netCDFs from MPC/OPC surface analyses"
TARGET_SNAPSHOT_ID = "FEKZ9VYQXA15A4GVG730"  # "Write fronts to icechunk store, first attempt"


def main() -> None:
    """Reset the fronts store's main branch back to the pre-corruption snapshot."""
    storage = ic.local_filesystem_storage(STORE_PATH)
    repo = ic.Repository.open(storage)

    current = repo.lookup_branch("main")
    print(f"main currently points to: {current}")
    if current != CORRUPTED_SNAPSHOT_ID:
        raise RuntimeError(
            f"main is at {current!r}, not the expected corrupted snapshot {CORRUPTED_SNAPSHOT_ID!r}. "
            "Refusing to reset automatically -- inspect the ancestry manually first "
            "(scripts/diagnose_fronts_time_encoding.py)."
        )

    repo.reset_branch("main", TARGET_SNAPSHOT_ID, from_snapshot_id=CORRUPTED_SNAPSHOT_ID)
    print(f"main reset to: {repo.lookup_branch('main')}")


if __name__ == "__main__":
    main()
