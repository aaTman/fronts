import icechunk
import xarray as xr
import dataclasses


@dataclasses.dataclasses
class TargetData:
    """Dataclass to hold the information about the target data and build it.

    Attributes:
        store_path: The path to the top level of icechunk store where the data is
            located.
    """

    store_path: str

    def build(self) -> xr.Dataset:
        """Build the target dataset from the icechunk store.

        Returns a virtual xarray Dataset of front objects with all timesteps.
        """

        # Open the local store as a read-only session
        local_storage = icechunk.local_filesystem_storage(self.store_path)
        repo = icechunk.Repository.open(local_storage)
        session = repo.readonly_session("main")

        # Open the icechunk store with xarray's open_zarr. Chunks are an empty dict but
        # are (1, 360, 920) which represents one netcdf file of front objects.
        ds = xr.open_zarr(
            session.store,
            zarr_format=3,
            consolidated=False,
            chunks={},
        )
        return ds
