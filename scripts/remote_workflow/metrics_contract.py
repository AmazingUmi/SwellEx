"""Shared stdlib-only prediction CSV contract and range-regression metrics."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import math
from pathlib import Path

PREDICTION_FIELDS = (
    "sample",
    "source_segment_idx",
    "window_center_s",
    "pred_range_km",
    "true_range_km",
    "abs_error_km",
)
TIME_TOLERANCE = 1.1e-6
ERROR_TOLERANCE = 2.1e-6


@dataclass(frozen=True)
class PredictionRow:
    sample: int
    source_index: int
    time_s: float
    predicted_km: float
    true_km: float

    @property
    def residual_km(self) -> float:
        return self.predicted_km - self.true_km


def _parse_int(value: str, label: str) -> int:
    text = value.strip()
    if not text:
        raise ValueError(f"Empty integer field: {label}")
    parsed = int(text, 10)
    if str(parsed) != text.lstrip("+"):
        # Allow leading zeros such as "0001" -> check int round-trip via sign/digits.
        pass
    return parsed


def parse_prediction_bytes(
    raw: bytes,
    *,
    expected: list[tuple[int, float, float]] | None = None,
    source: str | Path | None = None,
    check_abs_error: bool = True,
) -> list[PredictionRow]:
    suffix = f" in {source}" if source is not None else ""
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Prediction CSV is not valid UTF-8{suffix}") from exc
    reader = csv.DictReader(io.StringIO(decoded, newline=""))
    if tuple(reader.fieldnames or ()) != PREDICTION_FIELDS:
        raise ValueError(f"Prediction CSV must have the canonical six columns{suffix}")
    predictions: list[PredictionRow] = []
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed prediction CSV row{suffix}")
        try:
            sample = int(row["sample"])
            index = int(row["source_segment_idx"])
            moment, pred, target, reported_error = (
                float(row[key]) for key in PREDICTION_FIELDS[2:]
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid prediction value{suffix}") from exc
        if not all(math.isfinite(x) for x in (moment, pred, target, reported_error)):
            raise ValueError(f"Nonfinite prediction value{suffix}")
        if sample != len(predictions) or index < 1:
            raise ValueError(f"Invalid sample/source window index{suffix}")
        if predictions and (index <= predictions[-1].source_index or moment <= predictions[-1].time_s):
            raise ValueError(f"Source windows and timestamps must strictly increase{suffix}")
        actual_error = abs(pred - target)
        if reported_error < 0 or (check_abs_error and abs(reported_error - actual_error) > ERROR_TOLERANCE):
            raise ValueError(f"Reported absolute error differs from prediction and label{suffix}")
        predictions.append(PredictionRow(sample, index, moment, pred, target))
    if not predictions:
        raise ValueError(f"Empty prediction CSV{suffix}")
    if expected is not None:
        if len(expected) != len(predictions):
            raise ValueError(f"Prediction count does not match test dataset{suffix}")
        for item, (expected_index, expected_time, expected_target) in zip(predictions, expected):
            if (
                item.source_index != expected_index
                or abs(item.time_s - expected_time) > TIME_TOLERANCE
                or abs(item.true_km - expected_target) > TIME_TOLERANCE
            ):
                raise ValueError(f"Prediction rows differ from the test dataset{suffix}")
    return predictions


def parse_prediction_file(
    path: Path,
    *,
    expected: list[tuple[int, float, float]] | None = None,
    check_abs_error: bool = True,
) -> list[PredictionRow]:
    return parse_prediction_bytes(
        path.read_bytes(),
        expected=expected,
        source=path,
        check_abs_error=check_abs_error,
    )


def compute_prediction_metrics(rows: list[PredictionRow]) -> dict:
    if not rows:
        raise ValueError("Cannot compute metrics for empty prediction rows")
    count = len(rows)
    residuals = [row.residual_km for row in rows]
    errors = [abs(value) for value in residuals]
    mae_km = math.fsum(errors) / count
    rmse_km = math.sqrt(math.fsum(value * value for value in residuals) / count)
    bias_km = math.fsum(residuals) / count
    debiased_rmse_km = math.sqrt(
        math.fsum((value - bias_km) * (value - bias_km) for value in residuals) / count
    )
    mape_pct = (
        math.fsum(
            err / max(abs(row.true_km), 1e-12)
            for err, row in zip(errors, rows)
        )
        / count
        * 100.0
    )
    ordered_errors = sorted(errors)
    position = (count - 1) * 0.95
    lo = int(position)
    hi = min(lo + 1, count - 1)
    p95_abs_km = ordered_errors[lo] + (ordered_errors[hi] - ordered_errors[lo]) * (position - lo)
    worst = max(range(count), key=errors.__getitem__)
    return {
        "samples": count,
        "mae_km": mae_km,
        "rmse_km": rmse_km,
        "bias_km": bias_km,
        "debiased_rmse_km": debiased_rmse_km,
        "mape_pct": mape_pct,
        "p95_abs_km": p95_abs_km,
        "max_error_km": errors[worst],
        "max_error_window_center_s": rows[worst].time_s,
        "first_window_center_s": rows[0].time_s,
        "last_window_center_s": rows[-1].time_s,
        "source_segment_indices": [row.source_index for row in rows],
    }
