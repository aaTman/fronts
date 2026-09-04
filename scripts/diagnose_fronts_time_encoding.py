"""READ-ONLY diagnostic for the fronts store's time-encoding warning. DELETE AFTER USE.

Investigates a SerializationWarning ("dates out of range... falling back to cftime") seen
after appending 2025 data to the fronts icechunk store, to determine whether the append
disturbed the shared ``time`` array's CF ``units``/``calendar`` attributes (which would
corrupt how previously-committed, pre-2025 timesteps decode) or whether this predates the
append. Performs no writes: only ``Repository.open`` and ``readonly_session`` calls.

Run on the HPC system:

    pixi run -e data python scripts/diagnose_fronts_time_encoding.py
"""

import icechunk as ic
import xarray as xr

STORE_PATH = "/ourdisk/hpc/ai2es/tman/restructured_front_data/icechunk"
VIRTUAL_CHUNK_LOCAL_PATH = "/ourdisk/hpc/ai2es/tman/restructured_front_data/netcdf/"


def _open_raw(repo: ic.Repository, *, branch: str | None = None, snapshot_id: str | None = None) -> xr.Dataset:
    """Open the store read-only with ``decode_times=False``, exposing the raw stored values/attrs."""
    session = repo.readonly_session(branch, snapshot_id=snapshot_id)
    return xr.open_zarr(session.store, consolidated=False, decode_times=False, chunks=None)


def _print_time_state(ds: xr.Dataset, label: str) -> None:
    time_var = ds["time"]
    print(f"\n=== {label}: raw (undecoded) time array ===")
    print("attrs:", dict(time_var.attrs))
    print("dtype:", time_var.dtype, " shape:", time_var.shape)
    print("first 5 raw values:", time_var.values[:5])
    print("last 5 raw values:", time_var.values[-5:])


def main() -> None:
    """Print snapshot history and the raw time array's stored attrs at HEAD and one snapshot back."""
    storage = ic.local_filesystem_storage(STORE_PATH)
    repo_config = ic.RepositoryConfig.default()
    url_prefix = f"file://{VIRTUAL_CHUNK_LOCAL_PATH}"
    repo_config.set_virtual_chunk_container(
        ic.VirtualChunkContainer(url_prefix=url_prefix, store=ic.local_filesystem_store(VIRTUAL_CHUNK_LOCAL_PATH))
    )
    creds = ic.containers_credentials({url_prefix: None})
    repo = ic.Repository.open(storage, config=repo_config, authorize_virtual_chunk_access=creds)

    print("=== Snapshot ancestry (main branch, newest first) ===")
    snapshots = list(repo.ancestry(branch="main"))
    for snap in snapshots[:10]:
        print(f"{snap.id}  {snap.written_at}  {snap.message!r}")
    print(f"\nTotal snapshots on main: {len(snapshots)}")

    ds_head = _open_raw(repo, branch="main")
    _print_time_state(ds_head, "Current HEAD (main)")

    if len(snapshots) > 1:
        prev_snapshot_id = snapshots[1].id
        ds_prev = _open_raw(repo, snapshot_id=prev_snapshot_id)
        _print_time_state(ds_prev, f"Previous snapshot ({prev_snapshot_id})")


if __name__ == "__main__":
    main()
