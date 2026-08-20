# fronts.espr.ai webapp

Single self-contained `index.html` -- maplibre-gl + `@carbonplan/zarr-layer`,
loaded via CDN import maps (no build step). Served as static files by Caddy
on the mandelhub Proxmox VM (see `deploy/Caddyfile`).

## How it finds data

`ZARR_ROOT` (`/data` by default) is expected to be reverse-proxied by Caddy
to the same directory the scheduler (`frontfinder.scheduler.run_cycle`)
writes into: `<output_root>/<model>/<cycle>.zarr` plus a
`<output_root>/<model>/latest.json` pointer file written after each
successful run. The frontend fetches `latest.json` for whichever model is
selected, rather than trying to list the zarr directory or guess the most
recent cycle -- this also means a failed model run for one cycle just leaves
the previous `latest.json` in place instead of showing a broken/partial map.

## Known API-version caveat (read before touching colors)

topozarr's docs site describes an upcoming `layer_hints` API for embedding
colormap/clim styling directly in the zarr store as `ZarrLayerVarConfig`
objects (colormap given as a *name*, e.g. `"blues"`). The pinned/installed
topozarr version (0.0.4, see `requirements.txt`) does **not** have that
API yet -- `frontfinder/zarrio/pyramid.py` instead writes `colormap`/`clim`
as plain informational zarr attrs.

Separately, `@carbonplan/zarr-layer`'s own README example takes `colormap`
as an **array of hex/rgb color stops**, not a name string. So this
frontend does not try to read the colormap name out of the zarr attrs at
all -- `CLASS_STYLE` in `index.html` hardcodes an explicit
transparent-to-class-color hex gradient per front class. If topozarr's
`layer_hints` API lands and gets adopted here, revisit this: either keep
the hardcoded gradients (recommended, since they're also used for the panel
swatch legend) or wire them from the zarr attrs and drop the duplication.

## Before deploying

Everything above the "Known API-version caveat" section was written against
real, fetched documentation and a real installed topozarr package version.
This file itself, however, has **not** been opened in an actual browser
against a real zarr-layer store -- `@carbonplan/zarr-layer`'s constructor
options were taken from its README, not exercised end-to-end. Before this
goes live: run it against one real pyramid written by `run_cycle.py` and
confirm the layer actually renders, the CDN import map resolves cleanly,
and `raster-opacity` toggling (or the correct zarr-layer equivalent) works
as expected -- adjust `index.html` if the real API differs.
