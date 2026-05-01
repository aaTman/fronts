"""
Plot performance diagrams from pre-aggregated stats files produced by
compute_stats_from_shards.py (stats_aggregate_{mask}.nc + stats_spatial_{mask}.nc).

No confidence intervals — aggregate files have no time dimension for bootstrapping.

Usage:
    pixi run python src/fronts/plot/plot_performance_diagrams_from_stats.py \\
        --stats_dir ~/models/fronts/stats --mask land

    pixi run python src/fronts/plot/plot_performance_diagrams_from_stats.py \\
        --stats_dir ~/models/fronts/stats --mask ocean --front_types CF WF SF
"""

import argparse
import cartopy.crs as ccrs
from matplotlib import colors
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np
import xarray as xr

from fronts.utils import plotting
from fronts.utils.constants import FRONT_NAMES, DOMAIN_EXTENTS


def _parse_front_types(aggregate_ds: xr.Dataset) -> list[str]:
    """Infer front type labels from variable names (tp_{FT})."""
    return [v[3:] for v in aggregate_ds.data_vars if v.startswith("tp_") and not v.startswith("tp_spatial_")]


def plot_front(
    front_type: str,
    aggregate_ds: xr.Dataset,
    spatial_ds: xr.Dataset,
    mask: str | None,
    domain: str,
    map_neighborhood: int,
    output_type: str,
    outdir: str,
) -> None:
    thresholds = aggregate_ds["threshold"].values  # (100,)

    a = aggregate_ds[f"tp_{front_type}"].values  # (neighborhood, threshold)
    b = aggregate_ds[f"fp_{front_type}"].values
    c = aggregate_ds[f"fn_{front_type}"].values
    d = aggregate_ds[f"tn_{front_type}"].values

    pod = np.divide(a, a + c, out=np.zeros_like(a), where=(a + c) > 0)
    sr = np.divide(a, a + b, out=np.zeros_like(a), where=(a + b) > 0)

    sr_matrix, pod_matrix = np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 101))
    csi_matrix = 1 / ((1 / sr_matrix) + (1 / pod_matrix) - 1)
    fb_matrix = pod_matrix * (sr_matrix ** -1)
    CSI_LEVELS = np.linspace(0, 1, 11)
    FB_LEVELS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3]
    axis_ticks = np.arange(0, 1.01, 0.1)
    axis_ticklabels = np.arange(0, 100.1, 10).astype(int)

    # Reliability diagram inputs
    num_forecasts = (a + b)[0, :]  # use 50 km neighbourhood row for fraction
    total_pixels_approx = (a + b + c + d)[0, 0]  # rough total (sum at threshold=0.01)
    relative_forecast_fraction = 100 * num_forecasts / total_pixels_approx if total_pixels_approx > 0 else num_forecasts * 0

    tp_diff = np.abs(np.diff(a, axis=-1))   # (neighborhood, 99)
    fp_diff = np.abs(np.diff(b, axis=-1))
    denom = tp_diff + fp_diff
    observed_relative_frequency = np.divide(tp_diff, denom, out=np.zeros_like(tp_diff), where=denom > 0)

    # Spatial CSI map
    spatial_csi_ds = (
        spatial_ds[f"tp_spatial_{front_type}"].sum("latitude").sum("longitude")
        / (
            spatial_ds[f"tp_spatial_{front_type}"].sum("latitude").sum("longitude")
            + spatial_ds[f"fp_spatial_{front_type}"].sum("latitude").sum("longitude")
            + spatial_ds[f"fn_spatial_{front_type}"].sum("latitude").sum("longitude")
        )
    )
    spatial_csi_map = (
        spatial_ds[f"tp_spatial_{front_type}"]
        / (
            spatial_ds[f"tp_spatial_{front_type}"]
            + spatial_ds[f"fp_spatial_{front_type}"]
            + spatial_ds[f"fn_spatial_{front_type}"]
        )
    ).max("threshold")

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    ax0, ax1 = axs

    # --- CSI diagram (panel a) ---
    cs = ax0.contour(sr_matrix, pod_matrix, fb_matrix, FB_LEVELS, colors="black", linewidths=0.5, linestyles="--")
    ax0.clabel(cs, FB_LEVELS, fontsize=8)
    csi_contour = ax0.contourf(sr_matrix, pod_matrix, csi_matrix, CSI_LEVELS, cmap="Blues")
    cbar = fig.colorbar(csi_contour, ax=ax0, pad=0.02, label="Critical Success Index (CSI)")
    cbar.set_ticks(axis_ticks)

    # --- Reliability diagram (panel b) ---
    ax1_bar = ax1.twinx()
    ax1_bar.set_ylabel("Percentage of Grid Points with Forecasts [bars]")
    ax1_bar.yaxis.set_major_locator(plt.LinearLocator(11))
    ax1_bar.bar(thresholds[:-1], relative_forecast_fraction[1:], color="blue", width=0.005, alpha=0.25)
    ax1.plot(thresholds, thresholds, color="black", linestyle="--", linewidth=0.5, label="Perfect Reliability")

    boundary_colors = ["red", "purple", "brown", "darkorange", "darkgreen"]
    cell_text = []

    for boundary, color in enumerate(boundary_colors):
        csi = np.power((1 / sr[boundary]) + (1 / pod[boundary]) - 1, -1)
        hss = 2 * ((a[boundary] * d[boundary]) - (b[boundary] * c[boundary])) / (
            ((a[boundary] + c[boundary]) * (c[boundary] + d[boundary]))
            + ((a[boundary] + b[boundary]) * (b[boundary] + d[boundary]))
        )

        max_csi = np.nanmax(csi)
        max_idx = np.where(csi == max_csi)[0]
        max_csi_threshold = thresholds[max_idx][0]
        max_csi_pod = pod[boundary][max_idx][0]
        max_csi_sr = sr[boundary][max_idx][0]
        max_csi_fb = max_csi_pod / max_csi_sr if max_csi_sr > 0 else float("nan")
        max_hss = hss[max_idx][0]
        far = 1 - max_csi_sr

        cell_text.append([
            r"$\bf{%.3f}$" % max_csi,
            r"$\bf{%.3f}$" % max_hss,
            r"$\bf{%.1f}$" % (max_csi_pod * 100),
            r"$\bf{%.1f}$" % (far * 100),
            r"$\bf{%.3f}$" % max_csi_fb,
        ])

        ax0.plot(max_csi_sr, max_csi_pod, color=color, marker="*", markersize=10)
        ax0.plot(sr[boundary], pod[boundary], color=color, linewidth=1)
        ax1.plot(thresholds[:-1], observed_relative_frequency[boundary], color=color, linewidth=1)

    ax0.set_xticklabels(axis_ticklabels[::-1])
    ax0.set_xlabel("False Alarm Rate (FAR; %)")
    ax0.set_ylabel("Probability of Detection (POD; %)")
    ax0.set_title(r"$\bf{a)}$ $\bf{CSI}$ $\bf{diagram}$")
    ax1.set_xticklabels(axis_ticklabels)
    ax1.set_xlabel("Forecast Probability (uncalibrated; %)")
    ax1.set_ylabel("Observed Relative Frequency (%) [lines]")
    ax1.set_title(r"$\bf{b)}$ $\bf{Reliability}$ $\bf{diagram}$")

    for ax in axs:
        ax.set_xticks(axis_ticks)
        ax.set_yticks(axis_ticks)
        ax.set_yticklabels(axis_ticklabels)
        ax.grid(color="black", alpha=0.1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # --- Data table (panel c) ---
    if domain == "conus":
        table_axis_extent = [0.063, -0.038, 0.39, 0.239]
        table_scale = (1, 3.3)
        table_title_kwargs = dict(x=0.5, y=0.098, pad=-4)
        spatial_axis_extent = [0.5, -0.582, 0.512, 0.544]
        cbar_kwargs = {"label": "CSI", "pad": 0, "shrink": 1}
        spatial_plot_xlabels = [-140, -105, -70]
        spatial_plot_ylabels = [30, 40, 50]
    elif domain == "full":
        table_axis_extent = [0.063, -0.038, 0.39, 0.229]
        table_scale = (1, 2.8)
        table_title_kwargs = dict(x=0.5, y=0.096, pad=-4)
        spatial_axis_extent = [0.523, -0.5915, 0.48, 0.66]
        cbar_kwargs = {"label": "CSI", "pad": 0, "shrink": 0.675}
        spatial_plot_xlabels = [-150, -120, -90, -60, -30, 0, 120, 150, 180]
        spatial_plot_ylabels = [0, 20, 40, 60, 80]
    else:
        raise ValueError("Domain '%s' is not supported. Use 'conus' or 'full'." % domain)

    table_axis = plt.axes(table_axis_extent)
    table_axis.set_title(r"$\bf{c)}$ $\bf{Data}$ $\bf{table}$", **table_title_kwargs)
    table_axis.axis("off")
    stats_table = table_axis.table(
        cellText=cell_text,
        rowLabels=["50 km", "100 km", "150 km", "200 km", "250 km"],
        rowColours=boundary_colors,
        colLabels=["CSI", "HSS", "POD %", "FAR %", "FB"],
        cellLoc="center",
    )
    stats_table.scale(*table_scale)
    for cell in stats_table._cells:
        stats_table._cells[cell].set_alpha(0.7)
        stats_table._cells[cell].set_text_props(
            fontproperties=FontProperties(size="x-large", stretch="expanded")
        )

    # --- Spatial CSI map (panel d) ---
    csi_cmap = plotting.truncated_colormap("gnuplot2", maxval=0.9, n=10)
    spatial_axis = plt.axes(spatial_axis_extent, projection=ccrs.Miller(central_longitude=250))
    plotting.plot_background(extent=DOMAIN_EXTENTS[domain], ax=spatial_axis)

    norm_probs = colors.Normalize(vmin=0.1, vmax=1)
    spatial_csi_map_plot = xr.where(spatial_csi_map >= 0.1, spatial_csi_map, float("NaN"))
    spatial_csi_map_plot.sel(neighborhood=map_neighborhood).plot(
        ax=spatial_axis,
        x="longitude",
        y="latitude",
        norm=norm_probs,
        cmap=csi_cmap,
        transform=ccrs.PlateCarree(),
        alpha=0.6,
        cbar_kwargs=cbar_kwargs,
    )
    spatial_axis.set_title(r"$\bf{d)}$ $\bf{%d}$ $\bf{km}$ $\bf{CSI}$ $\bf{map}$" % map_neighborhood)
    gl = spatial_axis.gridlines(draw_labels=True, zorder=0, dms=True, x_inline=False, y_inline=False)
    gl.right_labels = False
    gl.top_labels = False
    gl.left_labels = True
    gl.bottom_labels = True
    gl.xlocator = FixedLocator(spatial_plot_xlabels)
    gl.ylocator = FixedLocator(spatial_plot_ylabels)
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 8}

    mask_label = f" ({mask})" if mask else ""
    plt.suptitle(
        "%ss%s" % (FRONT_NAMES.get(front_type, front_type), mask_label),
        fontsize=20,
    )

    import os
    os.makedirs(outdir, exist_ok=True)
    mask_suffix = f"_{mask}" if mask else ""
    filename = os.path.join(outdir, f"performance_{front_type}{mask_suffix}.{output_type}")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=500)
    plt.close()
    print(f"Saved: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats_dir", type=str, required=True, help="Directory containing stats_aggregate_{mask}.nc and stats_spatial_{mask}.nc files.")
    parser.add_argument("--mask", type=str, default=None, choices=["land", "ocean"], help="'land' or 'ocean' — selects which stats files to load.")
    parser.add_argument("--front_types", type=str, nargs="+", default=None, help="Front types to plot (default: all in file).")
    parser.add_argument("--domain", type=str, default="conus", choices=["conus", "full"], help="Domain for spatial plot layout.")
    parser.add_argument("--map_neighborhood", type=int, default=250, help="Neighbourhood (km) for spatial CSI map.")
    parser.add_argument("--output_type", type=str, default="png", help="Image format (png, pdf, etc.).")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory (defaults to --stats_dir).")
    args = parser.parse_args()

    import os
    mask_suffix = f"_{args.mask}" if args.mask else ""
    aggregate_path = os.path.join(args.stats_dir, f"stats_aggregate{mask_suffix}.nc")
    spatial_path = os.path.join(args.stats_dir, f"stats_spatial{mask_suffix}.nc")

    aggregate_ds = xr.open_dataset(aggregate_path)
    spatial_ds = xr.open_dataset(spatial_path)

    front_types = args.front_types or _parse_front_types(aggregate_ds)
    outdir = args.outdir or args.stats_dir

    for ft in front_types:
        print(f"Plotting {ft} …")
        plot_front(
            front_type=ft,
            aggregate_ds=aggregate_ds,
            spatial_ds=spatial_ds,
            mask=args.mask,
            domain=args.domain,
            map_neighborhood=args.map_neighborhood,
            output_type=args.output_type,
            outdir=outdir,
        )
