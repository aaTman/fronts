import dataclasses
from typing import Any

import tensorflow as tf
import xarray as xr
import xbatcher as xb
import xbatcher.loaders.keras


def create_dataloader(
    inputs: xr.Dataset | xr.DataArray,
    targets: xr.Dataset | xr.DataArray,
    input_sizes: dict[str, int] | None = None,
    target_sizes: dict[str, int] | None = None,
    prefetch_number: int = 3,
    preload_batch: bool = False,
    input_dtype: Any = tf.float32,
    target_dtype: Any = tf.float32,
) -> tf.data.Dataset:
    """Create a tf.data.Dataset DataLoader from xarray data for inputs and targets.

    References https://xbatcher.readthedocs.io/en/latest/user-guide/training-a-neural-network-with-keras-and-xbatcher.html
    for BatchGenerator usage. This function is used primarily to take cloud-based data
    and create a DataLoader that can be used for training a model in TensorFlow/Keras.

    Args:
        inputs: An xarray Dataset or DataArray containing the input features.
            Datasets are converted to DataArrays (required by xbatcher's keras loader).
        targets: An xarray Dataset or DataArray containing the target labels.
            Datasets are converted to DataArrays (required by xbatcher's keras loader).
        input_sizes: Optional dict specifying the dims and sizes of the input batches.
            If not provided, it will be inferred from the inputs dataset.
        target_sizes: Optional dict specifying the dims and sizes of the target batches.
            If not provided, it will be inferred from the targets dataset.
        prefetch_number: The number of batches to prefetch for. Defaults to 3 for
            what should be optimal performance.
        preload_batch: Whether to preload batches into memory. Defaults to False.
        input_dtype: The data type for the input batches. Defaults to tf.float32.
        target_dtype: The data type for the target batches. Defaults to tf.float32.

    Returns a tf.data.Dataset that yields batches of (inputs, targets) for training a
        model. Each batch will have the specified input_shape and target_shape.
    """
    # xbatcher's keras CustomTFDataset calls .data on batches, which requires
    # DataArrays. Convert single-variable Datasets to DataArrays.
    if isinstance(inputs, xr.Dataset):
        if len(inputs.data_vars) == 1:
            inputs = inputs[list(inputs.data_vars)[0]]
        else:
            inputs = inputs.to_array(dim="variable")
    if isinstance(targets, xr.Dataset):
        if len(targets.data_vars) == 1:
            targets = targets[list(targets.data_vars)[0]]
        else:
            targets = targets.to_array(dim="variable")

    if input_sizes is None:
        input_sizes = dict(inputs.sizes)  # ty: ignore[no-matching-overload]
    if target_sizes is None:
        target_sizes = dict(targets.sizes)  # ty: ignore[no-matching-overload]
    # Define batch generators for features (X) and labels (y)
    X_bgen = xb.BatchGenerator(
        inputs,
        input_dims=input_sizes,  # type: ignore[arg-type]
        preload_batch=preload_batch,  # Load each batch dynamically
    )
    y_bgen = xb.BatchGenerator(
        targets, input_dims=target_sizes, preload_batch=preload_batch  # type: ignore[arg-type]
    )

    # Use xbatcher's MapDataset to wrap the generators
    batch_dataset = xbatcher.loaders.keras.CustomTFDataset(X_bgen, y_bgen)

    # Create a DataLoader using tf.data.Dataset
    train_dataloader = tf.data.Dataset.from_generator(
        lambda: iter(batch_dataset),
        output_signature=(
            tf.TensorSpec(
                shape=tuple(input_sizes.values()), dtype=input_dtype, name="inputs"
            ),  # inputs
            tf.TensorSpec(
                shape=tuple(target_sizes.values()), dtype=target_dtype, name="targets"
            ),  # targets
        ),
    ).prefetch(prefetch_number)  # Prefetch 3 batches to improve performance

    return train_dataloader


@dataclasses.dataclass
class BatchGeneratorConfig:
    """A dataclass for configuring the creation of a batch generator DataLoader.

    Attributes:
        input_sizes: Optional dict specifying the dims and sizes of the input batches.
            If not provided, it will be inferred from the inputs dataset.
        target_sizes: Optional dict specifying the dims and sizes of the target batches.
            If not provided, it will be inferred from the targets dataset.
        prefetch_number: The number of batches to prefetch for. Defaults to 3 for
            what should be optimal performance.
        preload_batch: Whether to preload batches into memory. Defaults to False.
        input_dtype: The data type for the input batches. Defaults to tf.float32.
        target_dtype: The data type for the target batches. Defaults to tf.float32.
    """

    inputs: xr.Dataset
    targets: xr.Dataset
    input_sizes: dict[str, int] | None = None
    target_sizes: dict[str, int] | None = None
    prefetch_number: int = 3
    preload_batch: bool = False
    input_dtype: Any = dataclasses.field(default_factory=lambda: tf.float32)
    target_dtype: Any = dataclasses.field(default_factory=lambda: tf.float32)

    def build(self) -> tf.data.Dataset:
        """Builds a tf.data.Dataset DataLoader using the provided configuration
        parameters.

        Returns a tf.data.Dataset that yields batches of (inputs, targets) for training
            a model. Each batch will have the specified input_shape and target_shape.
        """
        return create_dataloader(
            inputs=self.inputs,
            targets=self.targets,
            input_sizes=self.input_sizes,
            target_sizes=self.target_sizes,
            prefetch_number=self.prefetch_number,
            preload_batch=self.preload_batch,
            input_dtype=self.input_dtype,
            target_dtype=self.target_dtype,
        )
