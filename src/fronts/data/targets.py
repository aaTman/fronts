import icechunk
import xarray as xr
import dataclasses
import logging
from fronts.utils import data_utils, calc

log = logging.getLogger("fronts.data.targets")

def _add_slash_to_end_if_not_present(path: str) -> str:
    """Add a slash to the end of a path if it is not present.

    Args:
        path: The path to add a slash to.

    Returns:
        The path with a slash added to the end, or the original path if it 
            already ends with a slash.
    """
    if not path.endswith("/"):
        return path + "/"
    return path

@dataclasses.dataclass
class TargetDataConfig:
    """Dataclass to hold the information about the target data and build it.

    Attributes:
        icechunk_store_path: The path to the top level of icechunk store where the data is
            located.
        file_path: The path to the netcdf files containing the target data.
        front_types: The types of fronts to include in the dataset.
        front_dilation: The number of dilation iterations to apply to the fronts.
    """

    icechunk_store_path: str
    file_path: str
    front_types: str | list[str]
    front_dilation: int

    def build(self) -> xr.Dataset:
        """Build the target dataset from the icechunk store.

        Returns a virtual xarray Dataset of front objects with all timesteps.
        """

        # Set up config and credentials to access virtual store
        config = icechunk.RepositoryConfig.default()

        # Set the virtual chunk container to the fronts netcdf directory
        config.set_virtual_chunk_container(
            icechunk.VirtualChunkContainer(
                url_prefix=_add_slash_to_end_if_not_present(f"file://{self.file_path}"),
                store=icechunk.local_filesystem_store(_add_slash_to_end_if_not_present(self.file_path)),
            ),
        )
        # Use None for credentials since the local filesystem store does not require
        # authentication
        credentials = icechunk.containers_credentials(
            {f"file://{self.file_path}": None}
        )

        # Open the local store as a read-only session
        local_storage = icechunk.local_filesystem_storage(self.icechunk_store_path)
        repo = icechunk.Repository.open(
            local_storage,
            config=config,
            authorize_virtual_chunk_access=credentials,
        )
        session = repo.readonly_session("main")

        # Open the icechunk store with xarray's open_zarr. Chunks are an empty dict but
        # are (1, 360, 920) which represents one netcdf file of front objects.
        ds = xr.open_zarr(
            session.store,
            zarr_version=3,
            consolidated=False,
            chunks="auto",
        )
        log.info("Opened fronts icechunk store.")
        log.info("Reformatting fronts with front_types=%s...", self.front_types)
        ds = data_utils.reformat_fronts(ds, self.front_types)
        log.info("reformat_fronts complete.")

        # Dilate the fronts if set to > 0
        if self.front_dilation > 0:
            log.info(
                "Expanding fronts with %d dilation iteration(s)...", self.front_dilation
            )
            ds = calc.maybe_expand_fronts_parallelized(
                ds, iterations=self.front_dilation
            )
            log.info("expand_fronts complete.")

        # Drop duplicates in time dimension if they exist
        ds = ds.drop_duplicates(dim="time")
        return ds
