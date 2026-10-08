"""Standalone generalized regression neural network for SCM features."""

from __future__ import annotations

import math

import torch


def kernel_scale(spread: float) -> float:
    """Reject Gaussian denominators outside the model's finite float32 range."""
    bounds = torch.finfo(torch.float32)
    minimum = math.sqrt(bounds.tiny / 2.0)
    maximum = math.sqrt(bounds.max / 2.0)
    if (isinstance(spread, bool) or not math.isfinite(spread)
            or spread < minimum or spread > maximum):
        raise ValueError("GRNN spread must give a finite positive normal float32 value for 2*sigma^2.")
    return 2.0 * spread**2


class ScmGrnnRegressor:
    """Gaussian-kernel GRNN regressor backed by memorized SCM reference samples."""

    def __init__(
        self,
        *,
        spread: float = 0.01,
        standardize_input: bool = False,
        eps: float = 1.0e-12,
        device: str | torch.device = "cpu",
    ) -> None:
        self.kernel_scale = kernel_scale(spread)
        if isinstance(eps, bool) or not math.isfinite(eps) or not 0.0 < eps <= 1.0:
            raise ValueError("GRNN eps must be finite and in (0,1]; larger floors bias normalized kernel weights.")
        if type(standardize_input) is not bool:
            raise ValueError("GRNN standardize_input must be a boolean.")
        self.spread = float(spread)
        self.standardize_input = bool(standardize_input)
        self.eps = float(eps)
        self.device = torch.device(device)
        self.feature_mean: torch.Tensor | None = None
        self.feature_std: torch.Tensor | None = None
        self.x_reference: torch.Tensor | None = None
        self.y_reference_km: torch.Tensor | None = None

    def to(self, device: str | torch.device) -> "ScmGrnnRegressor":
        self.device = torch.device(device)
        for name in ("feature_mean", "feature_std", "x_reference", "y_reference_km"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.to(self.device))
        return self

    @torch.no_grad()
    def fit_reference(self, x_reference: torch.Tensor, y_reference_km: torch.Tensor) -> None:
        x = x_reference.to(self.device, dtype=torch.float32)
        if y_reference_km.ndim not in (1, 2) or (y_reference_km.ndim == 2 and y_reference_km.size(1) != 1):
            raise ValueError("GRNN reference labels must have shape [N] or [N,1].")
        y = y_reference_km.to(self.device, dtype=torch.float32).reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError(f"GRNN reference features must be 2-D, got {tuple(x.shape)}.")
        if x.size(0) < 1 or x.size(1) < 1 or not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("GRNN reference features/labels must be nonempty and finite.")
        if x.size(0) != y.size(0):
            raise ValueError(f"Feature/label count mismatch: {x.size(0)} vs {y.size(0)}.")

        if self.standardize_input:
            mean = x.mean(dim=0)
            std = x.std(dim=0, unbiased=False).clamp_min(self.eps)
        else:
            mean = torch.zeros(x.size(1), dtype=x.dtype, device=x.device)
            std = torch.ones(x.size(1), dtype=x.dtype, device=x.device)

        normalized = (x - mean) / std.clamp_min(self.eps)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not torch.isfinite(normalized).all():
            raise ValueError("GRNN reference normalization produced nonfinite values.")
        self.feature_mean = mean
        self.feature_std = std
        self.x_reference = normalized
        self.y_reference_km = y

    def _require_reference(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self.x_reference is None
            or self.y_reference_km is None
            or self.feature_mean is None
            or self.feature_std is None
        ):
            raise RuntimeError("GRNN reference has not been built or loaded.")
        return self.x_reference, self.y_reference_km, self.feature_mean, self.feature_std

    @torch.no_grad()
    def predict(self, x: torch.Tensor, *, batch_size: int = 256) -> torch.Tensor:
        x_ref, y_ref, mean, std = self._require_reference()
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("GRNN batch_size must be a positive integer.")
        x_all = x.to(self.device, dtype=torch.float32)
        if x_all.ndim != 2:
            raise ValueError(f"GRNN prediction features must be 2-D, got {tuple(x_all.shape)}.")
        if x_all.size(0) < 1 or not torch.isfinite(x_all).all():
            raise ValueError("GRNN query features must be nonempty and finite.")
        if x_all.size(1) != x_ref.size(1):
            raise ValueError(
                f"Feature dimension mismatch: input has {x_all.size(1)}, "
                f"reference has {x_ref.size(1)}."
            )

        outputs: list[torch.Tensor] = []
        for start in range(0, x_all.size(0), batch_size):
            batch = x_all[start : start + batch_size]
            batch = (batch - mean) / std.clamp_min(self.eps)
            dist2 = torch.cdist(batch, x_ref, p=2).square()
            if not torch.isfinite(dist2).all():
                raise ValueError("GRNN distances are nonfinite; check input scale and reference.")

            # Row-wise offset preserves normalized weights and prevents all-zero
            # underflow when the spread is small in a high-dimensional feature space.
            dist2 = dist2 - dist2.min(dim=1, keepdim=True).values
            weights = torch.exp(-dist2 / self.kernel_scale)
            denominator = weights.sum(dim=1, keepdim=True).clamp_min(self.eps)
            outputs.append((weights @ y_ref) / denominator)

        return torch.cat(outputs, dim=0).cpu()

    def to_artifact(self) -> dict[str, object]:
        x_ref, y_ref, mean, std = self._require_reference()
        return {
            "spread": self.spread,
            "standardize_input": self.standardize_input,
            "eps": self.eps,
            "feature_mean": mean.detach().cpu(),
            "feature_std": std.detach().cpu(),
            "x_reference": x_ref.detach().cpu(),
            "y_reference_km": y_ref.detach().cpu(),
        }

    @classmethod
    def from_artifact(
        cls,
        artifact: dict[str, object],
        *,
        device: str | torch.device = "cpu",
    ) -> "ScmGrnnRegressor":
        model = cls(
            spread=artifact["spread"],
            standardize_input=artifact["standardize_input"],
            eps=artifact["eps"],
            device=device,
        )
        model.feature_mean = torch.as_tensor(artifact["feature_mean"], dtype=torch.float32).to(
            model.device
        )
        model.feature_std = torch.as_tensor(artifact["feature_std"], dtype=torch.float32).to(
            model.device
        )
        model.x_reference = torch.as_tensor(artifact["x_reference"], dtype=torch.float32).to(
            model.device
        )
        model.y_reference_km = torch.as_tensor(
            artifact["y_reference_km"], dtype=torch.float32
        ).to(model.device)
        x, y, mean, std = model._require_reference()
        if x.ndim != 2 or x.size(0) < 1 or x.size(1) < 1:
            raise ValueError("GRNN artifact x_reference must be a nonempty [N,D] tensor.")
        if tuple(y.shape) != (x.size(0), 1):
            raise ValueError("GRNN artifact y_reference_km must have shape [N,1].")
        if tuple(mean.shape) != (x.size(1),) or tuple(std.shape) != (x.size(1),):
            raise ValueError("GRNN artifact feature statistics must have shape [D].")
        if any(not torch.isfinite(value).all() for value in (x, y, mean, std)) or not (std > 0).all():
            raise ValueError("GRNN artifact tensors must be finite, with positive feature_std.")
        return model
