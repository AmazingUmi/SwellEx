"""Standalone generalized regression neural network for SCM features."""

from __future__ import annotations

import torch


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
        if spread <= 0.0:
            raise ValueError("GRNN spread must be positive.")
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
        y = y_reference_km.to(self.device, dtype=torch.float32).reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError(f"GRNN reference features must be 2-D, got {tuple(x.shape)}.")
        if x.size(0) != y.size(0):
            raise ValueError(f"Feature/label count mismatch: {x.size(0)} vs {y.size(0)}.")

        if self.standardize_input:
            mean = x.mean(dim=0)
            std = x.std(dim=0, unbiased=False).clamp_min(self.eps)
        else:
            mean = torch.zeros(x.size(1), dtype=x.dtype, device=x.device)
            std = torch.ones(x.size(1), dtype=x.dtype, device=x.device)

        self.feature_mean = mean
        self.feature_std = std
        self.x_reference = (x - mean) / std.clamp_min(self.eps)
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
        x_all = x.to(self.device, dtype=torch.float32)
        if x_all.ndim != 2:
            raise ValueError(f"GRNN prediction features must be 2-D, got {tuple(x_all.shape)}.")
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

            # Row-wise offset preserves normalized weights and prevents all-zero
            # underflow when the spread is small in a high-dimensional feature space.
            dist2 = dist2 - dist2.min(dim=1, keepdim=True).values
            weights = torch.exp(-dist2 / (2.0 * self.spread**2))
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
            spread=float(artifact["spread"]),
            standardize_input=bool(artifact.get("standardize_input", False)),
            eps=float(artifact.get("eps", 1.0e-12)),
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
        return model
