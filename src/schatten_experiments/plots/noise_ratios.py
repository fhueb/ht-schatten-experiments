"""Normalised noise-ratio plots."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from schatten_experiments.config import load_config


class RatioRow(NamedTuple):
    """One layer point in the normalised ratio plot."""

    name: str
    family: str
    value: float


class RatioSpec(NamedTuple):
    """One normalised noise-ratio figure to generate."""

    numerator: str
    denominator: str
    output_stem: str
    ylabel: str


class NoiseSpec(NamedTuple):
    """One raw noise-norm figure to generate."""

    norm_key: str
    output_stem: str
    ylabel: str


FAMILY_ORDER: tuple[str, ...] = (
    "attention.qkv",
    "attention.out",
    "attention.other",
    "mlp.up",
    "mlp.down",
    "mlp.other",
    "other",
    "embedding",
)

FAMILY_COLORS = {"embedding": "#7f7f7f"}

DEFAULT_RATIO_SPECS: tuple[RatioSpec, ...] = (
    RatioSpec("1", "2", "noise_ratio_1_over_2_normalized", r"Normalized $\sigma_{\ell_1} / \sigma_{\ell_2}$"),
    RatioSpec("S1", "S2", "noise_ratio_S1_over_S2_normalized", r"Normalized $\sigma_{S_1} / \sigma_{S_2}$"),
)

DEFAULT_NOISE_SPECS: tuple[NoiseSpec, ...] = (
    NoiseSpec("S1", "noise_S1", r"$\sigma_{S_1}$"),
    NoiseSpec("1", "noise_1", r"$\sigma_{\ell_1}$"),
)

FAMILY_RANK = {family: idx for idx, family in enumerate(FAMILY_ORDER)}


def _apply_panel_style(ax: plt.Axes) -> None:
    """Match the paper figure panel style."""
    ax.set_facecolor("#fafafa")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#333333")


def _load_stats_entry(path: str | Path) -> dict[str, Any]:
    """Load one structure-examiner stats entry."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"Empty stats list: {path}")
        payload = payload[0]
    if not isinstance(payload, dict) or "stats" not in payload:
        raise ValueError(f"Invalid stats payload: {path}")
    return payload


def _shape_to_matrix_dims(shape: object) -> tuple[int, int] | None:
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        return None
    try:
        rows, cols = int(shape[0]), int(shape[1])
    except (TypeError, ValueError):
        return None
    if rows <= 0 or cols <= 0:
        return None
    return rows, cols


def _ratio_bound_for_shape(shape: object, numerator: str, denominator: str) -> float | None:
    """Return the sharp finite-dimensional upper bound for the supported ratios."""
    dims = _shape_to_matrix_dims(shape)
    if dims is None:
        return None
    rows, cols = dims
    if (numerator, denominator) == ("1", "2"):
        return float(np.sqrt(rows * cols))
    if (numerator, denominator) == ("S1", "S2"):
        return float(np.sqrt(min(rows, cols)))
    raise ValueError(f"Unsupported ratio: {numerator} / {denominator}.")


def _normalize_ratio(ratio: float, bound: float) -> float:
    """Map the feasible interval [1, bound] to [0, 1]."""
    return float(np.clip((ratio - 1.0) / (bound - 1.0), 0.0, 1.0))


def _is_embedding_name(name: str) -> bool:
    low = name.lower()
    return "embed" in low or "lm_head" in low


def _layer_index(name: str) -> int | None:
    match = re.search(r"\.layers\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def _layer_family(name: str) -> str:
    low = name.lower()
    if _is_embedding_name(name):
        return "embedding"
    if "attention.query_key_value" in low or "attn.w_qkv" in low:
        return "attention.qkv"
    if "attention.dense" in low or "attn.w_out" in low:
        return "attention.out"
    if "mlp.dense_h_to_4h" in low or "mlp.w_up" in low:
        return "mlp.up"
    if "mlp.dense_4h_to_h" in low or "mlp.w_down" in low:
        return "mlp.down"
    if "attention" in low or "attn" in low:
        return "attention.other"
    if "mlp" in low:
        return "mlp.other"
    return "other"


def _sort_key(name: str) -> tuple[int, int, int, str]:
    layer_index = _layer_index(name)
    return (
        FAMILY_RANK.get(_layer_family(name), 99),
        int(layer_index is None),
        layer_index if layer_index is not None else 10**9,
        name,
    )


def _series_moment(values: object, r: float) -> float | None:
    """Collapse a minibatch series to its r-th power mean."""
    if isinstance(values, list):
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        return float(np.power(np.mean(np.power(np.abs(arr), r)), 1.0 / r))
    try:
        value = float(values)
    except (TypeError, ValueError):
        return None
    return abs(value) if np.isfinite(value) else None


def _extract_layer_values(
    stats: dict[str, Any],
    norm_key: str,
    r: float,
    include_embeddings: bool,
) -> dict[str, float]:
    """Extract one componentwise noise norm for each plotted layer."""
    componentwise = stats.get("componentwise_noise", {})
    out: dict[str, float] = {}
    for name, payload in componentwise.items():
        if not isinstance(name, str) or not isinstance(payload, dict):
            continue
        if not include_embeddings and _is_embedding_name(name):
            continue
        value = _series_moment(payload.get(norm_key), r)
        if value is not None:
            out[name] = value
    return out


def _shape_by_layer(stats: dict[str, Any]) -> dict[str, list[int]]:
    """Map parameter names to 2D tensor shapes, with and without Accelerate's prefix."""
    shapes = {}
    singular = stats.get("full_singular_values_2d", {})
    if isinstance(singular, dict):
        for name, payload in singular.items():
            if isinstance(payload, dict) and isinstance(payload.get("shape"), list):
                shapes[name] = payload["shape"]
                if name.startswith("module."):
                    shapes[name[len("module.") :]] = payload["shape"]
                else:
                    shapes[f"module.{name}"] = payload["shape"]
    return shapes


def ratio_rows(
    stats: dict[str, Any],
    *,
    numerator_norm_key: str,
    denominator_norm_key: str,
    series_moment_r: float,
    include_embeddings: bool = True,
) -> list[RatioRow]:
    """Compute paper-normalised layerwise noise ratios."""
    if not np.isfinite(series_moment_r) or series_moment_r <= 0.0:
        raise ValueError(f"series_moment_r must be positive and finite, got {series_moment_r}.")
    numerator = _extract_layer_values(stats, numerator_norm_key, series_moment_r, include_embeddings)
    denominator = _extract_layer_values(stats, denominator_norm_key, series_moment_r, include_embeddings)
    shapes = _shape_by_layer(stats)
    rows: list[RatioRow] = []
    for name in sorted(set(numerator) & set(denominator), key=_sort_key):
        den = denominator[name]
        if den <= 0:
            continue
        ratio = numerator[name] / den
        bound = _ratio_bound_for_shape(shapes.get(name), numerator_norm_key, denominator_norm_key)
        if bound is None or bound <= 1.0:
            continue
        rows.append(RatioRow(name, _layer_family(name), _normalize_ratio(ratio, bound)))
    return rows


def noise_rows(
    stats: dict[str, Any],
    *,
    norm_key: str,
    series_moment_r: float,
    include_embeddings: bool = True,
) -> list[RatioRow]:
    """Compute unnormalised layerwise noise norms for matrix weights only."""
    if not np.isfinite(series_moment_r) or series_moment_r <= 0.0:
        raise ValueError(f"series_moment_r must be positive and finite, got {series_moment_r}.")
    values = _extract_layer_values(stats, norm_key, series_moment_r, include_embeddings)
    matrix_names = set(_shape_by_layer(stats))
    return [
        RatioRow(name, _layer_family(name), values[name])
        for name in sorted(set(values) & matrix_names, key=_sort_key)
    ]


def plot_ratio_rows(
    rows: list[RatioRow],
    *,
    output_path: str | Path,
    ylabel: str,
    ylim: tuple[float, float] | None = (0.0, 1.0),
    mean_label: str | None = "Mean Ratio",
    family_legend_loc: str = "upper left",
    family_legend_anchor: tuple[float, float] = (0.0, 1.0),
    label_map: dict[str, str] | None = None,
    show_x_labels: bool = True,
) -> Path:
    """Render the normalised noise-ratio scatter plot."""
    if not rows:
        raise ValueError("No ratio rows to plot.")
    label_map = label_map or {}
    palette = plt.colormaps["tab10"]
    color_by_family = {family: palette(idx % 10) for idx, family in enumerate(FAMILY_ORDER)}
    color_by_family.update(FAMILY_COLORS)
    xs = np.arange(len(rows))
    ys = np.asarray([row.value for row in rows], dtype=np.float64)
    families = [row.family for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    _apply_panel_style(ax)
    for x, row in zip(xs, rows):
        ax.scatter(
            x,
            row.value,
            s=180,
            color=color_by_family.get(row.family, "#555555"),
            alpha=0.92,
            edgecolor="black",
            linewidths=1.25,
            zorder=3,
    )
    if mean_label is not None:
        ax.axhline(float(np.mean(ys)), color="black", linewidth=3.2, linestyle="-", label=mean_label, zorder=0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=20, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.28)
    ax.grid(False, axis="x")
    ax.xaxis.grid(False)
    ax.set_xlim(-0.8, len(rows) - 0.2)

    centers, labels = [], []
    start = 0
    prev = families[0]
    for idx, family in enumerate(families[1:], start=1):
        if family != prev:
            centers.append((start + idx - 1) / 2.0)
            labels.append(prev)
            ax.axvline(idx - 0.5, color="#888888", linestyle=":", linewidth=1.2, alpha=0.42)
            start = idx
            prev = family
    centers.append((start + len(families) - 1) / 2.0)
    labels.append(prev)
    ax.set_xticks(centers)
    if show_x_labels:
        ax.set_xticklabels(
            [label_map.get(label, label) for label in labels],
            rotation=20,
            ha="right",
            fontsize=15,
        )
    else:
        ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=15)

    family_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color_by_family[fam],
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=12,
            label=label_map.get(fam, fam),
        )
        for fam in FAMILY_ORDER
        if fam in families
    ]
    if family_handles:
        legend = ax.legend(
            handles=family_handles,
            title="Weight Family",
            loc=family_legend_loc,
            bbox_to_anchor=family_legend_anchor,
            framealpha=0.93,
            fontsize=15,
            title_fontsize=15,
        )
        ax.add_artist(legend)
    ref_handles, ref_labels = ax.get_legend_handles_labels()
    if ref_handles:
        ax.legend(
            ref_handles,
            ref_labels,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            framealpha=0.95,
            fontsize=15,
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_snapshot(
    stats_file: str | Path,
    *,
    output_path: str | Path,
    ratio_spec: RatioSpec,
    series_moment_r: float,
    include_embeddings: bool = True,
    family_legend_loc: str = "upper left",
    family_legend_anchor: tuple[float, float] = (0.0, 1.0),
    label_map: dict[str, str] | None = None,
    show_x_labels: bool = True,
) -> Path:
    """Plot one checkpoint's normalised noise-ratio snapshot."""
    entry = _load_stats_entry(stats_file)
    rows = ratio_rows(
        entry["stats"],
        numerator_norm_key=ratio_spec.numerator,
        denominator_norm_key=ratio_spec.denominator,
        series_moment_r=series_moment_r,
        include_embeddings=include_embeddings,
    )
    return plot_ratio_rows(
        rows,
        output_path=output_path,
        ylabel=ratio_spec.ylabel,
        ylim=(0.0, 1.0),
        mean_label="Mean Ratio",
        family_legend_loc=family_legend_loc,
        family_legend_anchor=family_legend_anchor,
        label_map=label_map,
        show_x_labels=show_x_labels,
    )


def plot_noise_snapshot(
    stats_file: str | Path,
    *,
    output_path: str | Path,
    noise_spec: NoiseSpec,
    series_moment_r: float,
    include_embeddings: bool = True,
    ylim: tuple[float, float] | None = None,
    family_legend_loc: str = "upper left",
    family_legend_anchor: tuple[float, float] = (0.0, 1.0),
    label_map: dict[str, str] | None = None,
    show_x_labels: bool = True,
) -> Path:
    """Plot one checkpoint's raw noise-norm snapshot."""
    entry = _load_stats_entry(stats_file)
    rows = noise_rows(
        entry["stats"],
        norm_key=noise_spec.norm_key,
        series_moment_r=series_moment_r,
        include_embeddings=include_embeddings,
    )
    return plot_ratio_rows(
        rows,
        output_path=output_path,
        ylabel=noise_spec.ylabel,
        ylim=ylim,
        mean_label=None,
        family_legend_loc=family_legend_loc,
        family_legend_anchor=family_legend_anchor,
        label_map=label_map,
        show_x_labels=show_x_labels,
    )


def _shared_ylim(row_sets: list[list[RatioRow]]) -> tuple[float, float]:
    values = np.asarray([row.value for rows in row_sets for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0.0, 1.0)
    ymin = min(0.0, float(np.min(values)))
    ymax = float(np.max(values))
    if ymax <= ymin:
        pad = abs(ymax) * 0.05 or 1.0
        return (ymin, ymax + pad)
    return (ymin, ymax + 0.05 * (ymax - ymin))


def plot_configured_ratios(
    config: dict[str, Any],
    run_dir: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Generate all normalised noise-ratio snapshots requested by the config."""
    plot_config = config.get("plots", {})
    if not isinstance(plot_config, dict):
        raise ValueError("Config field 'plots' must be a mapping.")
    checkpoints = plot_config.get("ratio_snapshot_checkpoints", ["initialization", "final"])
    if isinstance(checkpoints, str):
        checkpoints = [checkpoints]
    elif not isinstance(checkpoints, (list, tuple)):
        raise ValueError("Config field 'plots.ratio_snapshot_checkpoints' must be a string or list.")
    r = float(plot_config.get("ratio_snapshot_r", 1.5))
    include_embeddings = bool(plot_config.get("include_embeddings", True))
    show_x_labels = not bool(plot_config.get("hide_x_labels", False))
    label_map = dict(plot_config.get("label_map") or {})

    out_paths: list[Path] = []
    stats_files = {
        str(checkpoint): Path(run_dir) / "stats" / ("final" if checkpoint in {"", "final"} else str(checkpoint)) / "opt_stats.json"
        for checkpoint in checkpoints
    }
    for checkpoint in checkpoints:
        stats_name = "final" if checkpoint in {"", "final"} else checkpoint
        stats_file = stats_files[str(checkpoint)]
        for ratio_spec in DEFAULT_RATIO_SPECS:
            move_family_legend = stats_name == "final" and ratio_spec.numerator == "1" and ratio_spec.denominator == "2"
            out_paths.append(
                plot_snapshot(
                    stats_file,
                    output_path=Path(output_dir) / f"{ratio_spec.output_stem}_{stats_name}.pdf",
                    ratio_spec=ratio_spec,
                    series_moment_r=r,
                    include_embeddings=include_embeddings,
                    family_legend_loc="lower left" if move_family_legend else "upper left",
                    family_legend_anchor=(0.0, 0.0) if move_family_legend else (0.0, 1.0),
                    label_map=label_map,
                    show_x_labels=show_x_labels,
                )
            )
    for noise_spec in DEFAULT_NOISE_SPECS:
        rows_by_checkpoint = {
            str(checkpoint): noise_rows(
                _load_stats_entry(stats_files[str(checkpoint)])["stats"],
                norm_key=noise_spec.norm_key,
                series_moment_r=r,
                include_embeddings=include_embeddings,
            )
            for checkpoint in checkpoints
        }
        shared_ylim = _shared_ylim(list(rows_by_checkpoint.values()))
        for checkpoint in checkpoints:
            stats_name = "final" if checkpoint in {"", "final"} else checkpoint
            out_paths.append(
                plot_ratio_rows(
                    rows_by_checkpoint[str(checkpoint)],
                    output_path=Path(output_dir) / f"{noise_spec.output_stem}_{stats_name}.pdf",
                    ylabel=noise_spec.ylabel,
                    ylim=shared_ylim,
                    mean_label=None,
                    label_map=label_map,
                    show_x_labels=show_x_labels,
                )
            )
    return out_paths


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Plot normalised noise-ratio snapshots from structure-examiner stats.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing stats/<checkpoint>/opt_stats.json.",
    )
    parser.add_argument("--output-dir", required=True, help="Figure output directory.")
    parser.add_argument("--config", default="configs/noise_ratios.yaml", help="YAML plot config.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    for path in plot_configured_ratios(config, args.run_dir, args.output_dir):
        print(f"Saved plot to: {path}")


if __name__ == "__main__":
    main()
