import icechunk
import xarray as xr
import dataclasses
import logging
from fronts.utils import data_utils

log = logging.getLogger("fronts.data.targets")


@dataclasses.dataclass
class TargetDataConfig:
    """Dataclass to hold the information about the target data and build it.

    Attributes:
        store_path: The path to the top level of icechunk store where the data is
            located.
    """

    store_path: str
    front_types: str | list[str]
    front_dilation: int

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

        # Reformat the fronts if front_types is specified.
        if self.front_types is not None:
            log.debug("Reformatting fronts with front_types=%s...", self.front_types)
            ds = data_utils.reformat_fronts(ds, self.front_types)
            log.debug("reformat_fronts complete.")

        # Dilate the fronts if set to > 0
        if self.front_dilation > 0:
            log.debug(
                "Expanding fronts with %d dilation iteration(s)...", self.front_dilation
            )
            ds = data_utils.expand_fronts(ds, iterations=self.front_dilation)
            log.debug("expand_fronts complete.")

        return ds
