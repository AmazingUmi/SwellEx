"""Shared prediction export and plotting helpers."""

from __future__ import annotations

from pathlib import Path


PredictionRow = tuple[int, int, float, float, float, float]


def save_prediction_plot(
    rows: list[PredictionRow], output_path: Path, title: str = "Source Range Prediction"
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing Python dependency for plotting: {exc.name}. Install it with:\n"
            "  python -m pip install matplotlib"
        ) from exc

    if not rows:
        raise ValueError("Cannot plot predictions because no prediction rows were produced.")

    sorted_rows = sorted(rows, key=lambda row: row[2])
    time_s = [row[2] for row in sorted_rows]
    pred_km = [row[3] for row in sorted_rows]
    true_km = [row[4] for row in sorted_rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(time_s, true_km, color="#1f77b4", linewidth=1.8, label="Label")
    ax.scatter(
        time_s,
        pred_km,
        color="#d62728",
        s=24,
        alpha=0.8,
        edgecolors="none",
        label="Prediction",
    )
    ax.set_xlabel("Window center time (s)")
    ax.set_ylabel("Range (km)")
    ax.set_title(title)
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_predictions_csv(rows: list[PredictionRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(
            "sample,source_segment_idx,window_center_s,"
            "pred_range_km,true_range_km,abs_error_km\n"
        )
        for row in rows:
            f.write(
                f"{row[0]},{row[1]},{row[2]:.6f},"
                f"{row[3]:.6f},{row[4]:.6f},{row[5]:.6f}\n"
            )
