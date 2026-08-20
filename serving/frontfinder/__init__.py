"""frontfinder: batch inference + zarr serving pipeline for the ESPR "Fronts" product.

Runs two Keras front-detection models (best_loss, model_1702) against streamed
ECMWF IFS open-data global 0.25 deg fields and writes results to a zarr
multiscale pyramid for the fronts.espr.ai maplibre/topozarr viewer.
"""

__version__ = "0.1.0"
