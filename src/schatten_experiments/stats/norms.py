from __future__ import annotations

import math
from typing import Any

import torch


def norm_p_to_key(p: float) -> str:
    p_float = float(p)
    if math.isinf(p_float):
        return "inf"
    return str(int(p_float)) if p_float.is_integer() else str(p_float)


def schatten_p_to_key(p: float) -> str:
    return f"S{norm_p_to_key(p)}"


def resolve_norm_ps(values: list[float] | None, with_schatten: bool = False) -> list[float]:
    norm_ps = [1.0, 2.0] if not values else [float(value) for value in values]
    out: list[float] = []
    for value in norm_ps:
        if value <= 0 or math.isnan(value):
            raise ValueError("Norm orders must be positive.")
        if value not in out:
            out.append(value)
    if with_schatten and 1.0 not in out:
        out.insert(0, 1.0)
    if with_schatten and 2.0 not in out:
        out.append(2.0)
    return out


def resolve_schatten_ps(norm_ps: list[float]) -> list[float]:
    """Schatten noise-ratio plots require only S1 and S2."""
    out: list[float] = []
    for value in norm_ps:
        p = float(value)
        if p in (1.0, 2.0) and p not in out:
            out.append(p)
    return out


def tensor_entrywise_norm(tensor: torch.Tensor, p: float) -> float:
    p_float = float(p)
    detached = tensor.detach()
    if math.isinf(p_float):
        return detached.abs().max().item() if detached.numel() else 0.0
    return detached.norm(p=p_float).item()


def aggregate_norm_values(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    p_float = float(p)
    if math.isinf(p_float):
        return max(values)
    return math.pow(sum(float(value) ** p_float for value in values), 1.0 / p_float)


def schatten_1_norm_via_gram(tensor: torch.Tensor) -> float:
    matrix = tensor.detach().float()
    if matrix.numel() == 0:
        return 0.0
    gram = matrix.T @ matrix if matrix.shape[0] >= matrix.shape[1] else matrix @ matrix.T
    return torch.linalg.eigvalsh(gram).clamp_min_(0).sqrt_().sum().item()


def tensor_singular_values(tensor: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(tensor.detach().float())


def summarize_tensor_dict(
    tensors: dict[int, torch.Tensor],
    norm_ps: list[float],
    param_id_to_name: dict[int, str],
    *,
    with_schatten: bool = False,
    include_singular_values: bool = False,
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, dict[str, Any]]]:
    """Summarize tensor norms globally and per parameter."""
    schatten_ps = resolve_schatten_ps(norm_ps) if with_schatten else []
    aggregate_inputs = {norm_p_to_key(p): [] for p in norm_ps}
    for p in schatten_ps:
        aggregate_inputs[schatten_p_to_key(p)] = []

    componentwise: dict[str, dict[str, float]] = {}
    singular_values: dict[str, dict[str, Any]] = {}

    for param_id, tensor in tensors.items():
        name = param_id_to_name.get(param_id, str(param_id))
        entry: dict[str, float] = {}
        for p in norm_ps:
            key = norm_p_to_key(p)
            value = tensor_entrywise_norm(tensor, p)
            entry[key] = value
            aggregate_inputs[key].append(value)

        if tensor.ndim == 2 and (schatten_ps or include_singular_values):
            if 1.0 in schatten_ps:
                value = schatten_1_norm_via_gram(tensor)
                entry["S1"] = value
                aggregate_inputs["S1"].append(value)
            if 2.0 in schatten_ps:
                value = entry.get("2", tensor_entrywise_norm(tensor, 2.0))
                entry["S2"] = value
                aggregate_inputs["S2"].append(value)
            if include_singular_values:
                svals = tensor_singular_values(tensor)
                singular_values[name] = {
                    "shape": [int(tensor.shape[0]), int(tensor.shape[1])],
                    "singular_values": svals.cpu().tolist(),
                }

        componentwise[name] = entry

    aggregate = {
        norm_p_to_key(p): aggregate_norm_values(aggregate_inputs[norm_p_to_key(p)], p)
        for p in norm_ps
    }
    for p in schatten_ps:
        aggregate[schatten_p_to_key(p)] = aggregate_norm_values(aggregate_inputs[schatten_p_to_key(p)], p)
    return aggregate, componentwise, singular_values


def dgrad_difference(left: dict[int, torch.Tensor], right: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    return {param_id: tensor - right[param_id] for param_id, tensor in left.items() if param_id in right}


def dgrad_inner(left: dict[int, torch.Tensor], right: dict[int, torch.Tensor]) -> float:
    total = 0.0
    for param_id, tensor in left.items():
        if param_id in right:
            total += torch.sum(tensor * right[param_id]).item()
    return float(total)


def safe_cos(inner: float, left_norm: float, right_norm: float) -> float | None:
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return inner / left_norm / right_norm


def init_series_stats(
    param_name_to_ndim: dict[str, int],
    norm_ps: list[float],
    *,
    with_schatten: bool,
) -> dict[str, Any]:
    norm_keys = [norm_p_to_key(p) for p in norm_ps]
    schatten_keys = [schatten_p_to_key(p) for p in resolve_schatten_ps(norm_ps)] if with_schatten else []
    return {
        "batch_norm": {key: [] for key in norm_keys + schatten_keys},
        "noise": {key: [] for key in norm_keys + schatten_keys},
        "componentwise_batch_norm": {
            name: {key: [] for key in norm_keys + (schatten_keys if ndim == 2 else [])}
            for name, ndim in param_name_to_ndim.items()
        },
        "componentwise_noise": {
            name: {key: [] for key in norm_keys + (schatten_keys if ndim == 2 else [])}
            for name, ndim in param_name_to_ndim.items()
        },
    }


def append_summary_series(
    stats: dict[str, Any],
    stat_name: str,
    aggregate: dict[str, float],
    componentwise: dict[str, dict[str, float]],
) -> None:
    component_key = f"componentwise_{stat_name}"
    for key, value in aggregate.items():
        stats[stat_name][key].append(value)
    for name, values in componentwise.items():
        name_stats = stats[component_key].setdefault(name, {})
        for key, value in values.items():
            name_stats.setdefault(key, []).append(value)


def get_minibatch_stats(
    batch_grad: dict[int, torch.Tensor],
    full_grad: dict[int, torch.Tensor],
    norm_ps: list[float],
    stats: dict[str, Any],
    param_id_to_name: dict[int, str],
    *,
    with_schatten: bool,
) -> None:
    batch_summary, batch_componentwise, _ = summarize_tensor_dict(
        batch_grad,
        norm_ps,
        param_id_to_name,
        with_schatten=with_schatten,
    )
    append_summary_series(stats, "batch_norm", batch_summary, batch_componentwise)

    noise_summary, noise_componentwise, _ = summarize_tensor_dict(
        dgrad_difference(batch_grad, full_grad),
        norm_ps,
        param_id_to_name,
        with_schatten=with_schatten,
    )
    append_summary_series(stats, "noise", noise_summary, noise_componentwise)

    stats.setdefault("inner", []).append(dgrad_inner(batch_grad, full_grad))
    l2_full = stats["full_norm"]["2"]
    l2_batch = stats["batch_norm"]["2"][-1]
    stats.setdefault("cos", []).append(safe_cos(stats["inner"][-1], l2_full, l2_batch))
