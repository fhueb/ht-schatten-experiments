from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


def zeropower_via_svd(g: Tensor, steps: Optional[int] = None) -> Tensor:
    """Exact Schatten-infinity LMO via SVD."""
    u, _, vh = torch.linalg.svd(g, full_matrices=False)
    return u @ vh


def zeropower_via_newtonschulz5(g: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    """Approximate Schatten-infinity LMO used by standard Muon."""
    if g.ndim != 2:
        raise ValueError("Newton-Schulz Muon backend requires a 2D tensor.")
    a, b, c = (3.4445, -4.7750, 2.0315)

    x = g.bfloat16() / (g.norm() + eps)
    transposed = g.size(0) > g.size(1)
    if transposed:
        x = x.T

    for _ in range(steps):
        aa = x @ x.T
        bb = aa @ x
        x = a * x + b * bb + c * aa @ bb

    if transposed:
        x = x.T
    return x.to(g.dtype)


ZEROPOWER_BACKENDS = {
    "svd": zeropower_via_svd,
    "newtonschulz5": zeropower_via_newtonschulz5,
}


def _canonicalise_schatten_p(p) -> float:
    if p is None:
        return math.inf
    if isinstance(p, str):
        if p.lower() == "inf":
            return math.inf
        p = float(p)
    p = float(p)
    if p < 1.0:
        raise ValueError(f"Invalid schatten_p={p}; must satisfy p >= 1.")
    return p


def _dual_q_from_p(p: float) -> float:
    if math.isinf(p):
        return 1.0
    if p == 1.0:
        return math.inf
    return p / (p - 1.0)


def _even_integer_dual_q_from_p(p: float, tol: float = 1e-6) -> int | None:
    if not math.isfinite(p) or p <= 1.0:
        return None
    q = _dual_q_from_p(p)
    q_round = int(round(q))
    if abs(q - q_round) <= tol and q_round >= 2 and q_round % 2 == 0:
        return q_round
    return None


def schatten_lmo_via_svd(g: Tensor, p: float, eps: float = 1e-12) -> Tensor:
    """Exact LMO for the unit Schatten-p ball."""
    if g.ndim != 2:
        raise ValueError("Schatten LMO requires a 2D tensor.")

    work_dtype = torch.float32 if g.dtype in (torch.float16, torch.bfloat16) else g.dtype
    x = g.to(work_dtype)
    u, s, vh = torch.linalg.svd(x, full_matrices=False)

    if math.isinf(p):
        return (u @ vh).to(g.dtype)

    if p == 1.0:
        if s.numel() == 0 or s[0] <= eps:
            return torch.zeros_like(g)
        return (u[:, :1] @ vh[:1, :]).to(g.dtype)

    q = _dual_q_from_p(p)
    coeff = s.pow(q - 1.0)
    denom = s.pow(q).sum().clamp_min(eps).pow((q - 1.0) / q)
    return ((u * (coeff / denom).unsqueeze(0)) @ vh).to(g.dtype)


def schatten_lmo_even_q(g: Tensor, q: int, eps: float = 1e-12) -> Tensor:
    """Fast exact LMO for finite p when the dual exponent q is even."""
    if g.ndim != 2:
        raise ValueError("Schatten LMO requires a 2D tensor.")
    if q < 2 or q % 2:
        raise ValueError(f"q must be an even integer >= 2, got {q}.")

    work_dtype = torch.float32 if g.dtype in (torch.float16, torch.bfloat16) else g.dtype
    x = g.to(work_dtype)
    x = x / (x.norm() + eps)

    k = q // 2
    y = x
    if k > 1:
        if x.size(0) >= x.size(1):
            gram = x.T @ x
            for _ in range(k - 1):
                y = y @ gram
        else:
            gram = x @ x.T
            for _ in range(k - 1):
                y = gram @ y

    norm_q_power_q = (x * y).sum().abs().clamp_min(eps)
    denom = norm_q_power_q.pow((q - 1.0) / q)
    return (y / denom).to(g.dtype)


def schatten_lmo(
    g: Tensor,
    p: float,
    zeropower_fn=zeropower_via_newtonschulz5,
    ns_steps: int = 5,
    eps: float = 1e-7,
) -> Tensor:
    """Dispatch the Schatten-p LMO used by Muon."""
    p = _canonicalise_schatten_p(p)
    if math.isinf(p):
        return zeropower_fn(g, steps=ns_steps)

    even_q = _even_integer_dual_q_from_p(p)
    if even_q is not None:
        return schatten_lmo_even_q(g, q=even_q, eps=eps)

    return schatten_lmo_via_svd(g, p=p, eps=eps)


class Muon(Optimizer):
    """Muon with a Schatten-p LMO on 2D parameter groups and AdamW fallback."""

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        adam_eps: float = 1e-8,
        ns_steps: int = 5,
        backend: str = "newtonschulz5",
        adjust_lr_fn: Optional[str] = "match_rms_adamw",
        schatten_p: float | str = math.inf,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if adjust_lr_fn is not None and adjust_lr_fn not in ("original", "match_rms_adamw"):
            raise ValueError(f"Invalid adjust_lr_fn: {adjust_lr_fn}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            betas=betas,
            weight_decay=weight_decay,
            adam_eps=adam_eps,
            ns_steps=ns_steps,
            backend=backend,
            adjust_lr_fn=adjust_lr_fn,
            muon=True,
            schatten_p=_canonicalise_schatten_p(schatten_p),
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            group_backend = group.get("backend", backend)
            if group_backend not in ZEROPOWER_BACKENDS:
                raise ValueError(f"Invalid backend: {group_backend}.")
            group["schatten_p"] = _canonicalise_schatten_p(group.get("schatten_p", math.inf))

    @staticmethod
    def _init_state_2d(param: Tensor, state: dict) -> None:
        state["momentum_buffer"] = torch.zeros_like(param.data)

    @staticmethod
    def _init_state_adam(param: Tensor, state: dict) -> None:
        state["m"] = torch.zeros_like(param.data)
        state["v"] = torch.zeros_like(param.data)
        state["step"] = 0

    @staticmethod
    def _adjust_lr(lr: float, adjust_lr_fn: Optional[str], param_shape: torch.Size) -> float:
        rows, cols = param_shape[:2]
        if adjust_lr_fn == "match_rms_adamw":
            return lr * 0.2 * (max(rows, cols) ** 0.5)
        if adjust_lr_fn == "original":
            return lr * (max(1.0, rows / cols) ** 0.5)
        return lr

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            adam_eps = group["adam_eps"]
            ns_steps = group["ns_steps"]
            zeropower_fn = ZEROPOWER_BACKENDS[group["backend"]]
            adjust_lr_fn = group["adjust_lr_fn"]
            use_muon = group.get("muon", True)
            schatten_p = _canonicalise_schatten_p(group.get("schatten_p", math.inf))

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]

                if use_muon:
                    if grad.ndim != 2:
                        raise ValueError("Muon/Schatten updates require 2D parameters.")
                    if not state:
                        self._init_state_2d(param, state)

                    buf = state["momentum_buffer"]
                    buf.lerp_(grad, 1 - momentum)
                    update = grad.lerp(buf, momentum) if nesterov else buf.clone()
                    update = schatten_lmo(update, p=schatten_p, zeropower_fn=zeropower_fn, ns_steps=ns_steps)

                    if weight_decay != 0.0:
                        param.data.mul_(1 - lr * weight_decay)
                    param.data.add_(update, alpha=-self._adjust_lr(lr, adjust_lr_fn, param.shape))
                    continue

                if not state:
                    self._init_state_adam(param, state)

                state["step"] += 1
                step = state["step"]
                m = state["m"]
                v = state["v"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                m_hat = m / (1 - beta1**step)
                v_hat = v / (1 - beta2**step)
                param.data.addcdiv_(m_hat, v_hat.sqrt().add_(adam_eps), value=-lr)
                if weight_decay != 0.0:
                    param.data.mul_(1 - lr * weight_decay)

        return loss
