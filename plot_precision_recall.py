from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

curve_path = Path("evaluation_results/precision_recall_curve.csv")
output_path = Path("evaluation_results/precision_recall_curve.png")

if not curve_path.exists():
    raise FileNotFoundError(
        f"Cannot find {curve_path}. Run evaluate_retrieval.py first."
    )

print("Loading precision-recall data...")
df = pd.read_csv(curve_path)

recall_levels = np.linspace(0, 1, 101)
plt.figure(figsize=(9, 6))

for system, system_df in df.groupby("system"):
    interpolated_curves = []

    for _, query_df in system_df.groupby("query_id"):
        query_df = query_df.sort_values("recall")

        recall = query_df["recall"].to_numpy(dtype=float)
        precision = query_df["precision"].to_numpy(dtype=float)

        # Precision envelope: best precision at or above each recall level.
        precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]

        # Include the origin and interpolate onto a common recall grid.
        recall = np.concatenate(([0.0], recall))
        precision_envelope = np.concatenate(
            ([precision_envelope[0]], precision_envelope)
        )

        interpolated = np.interp(
            recall_levels, recall, precision_envelope,
            left=precision_envelope[0], right=0.0
        )
        interpolated_curves.append(interpolated)

    if interpolated_curves:
        mean_precision = np.mean(interpolated_curves, axis=0)
        plt.plot(recall_levels, mean_precision, label=system)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curves on the Cranfield Dataset")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved graph to: {output_path.resolve()}")
plt.show()