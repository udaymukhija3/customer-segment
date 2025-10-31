import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np
from joblib import dump
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


REQUIRED_COLUMNS: List[str] = ["age", "annual_income", "spending_score"]


def read_dataset(csv_path: Path) -> np.ndarray:
	with csv_path.open("r", newline="") as f:
		reader = csv.DictReader(f)
		missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
		if missing:
			raise ValueError(f"CSV is missing required columns: {missing}. Expected columns: {REQUIRED_COLUMNS}")
		rows: List[List[float]] = []
		for row in reader:
			try:
				rows.append([
					float(row["age"]),
					float(row["annual_income"]),
					float(row["spending_score"]),
				])
			except (TypeError, ValueError) as e:
				raise ValueError(f"Invalid numeric value in row: {row}") from e
	return np.asarray(rows, dtype=np.float64)


def main() -> None:
	parser = argparse.ArgumentParser(description="Train KMeans (k=5) for customer segmentation.")
	parser.add_argument("--input", type=str, required=True, help="Path to CSV with columns: age, annual_income, spending_score")
	parser.add_argument("--artifacts_dir", type=str, default=str(Path("artifacts")), help="Directory to save artifacts")
	args = parser.parse_args()

	csv_path = Path(args.input)
	if not csv_path.exists():
		raise FileNotFoundError(f"Input CSV not found: {csv_path}")

	X = read_dataset(csv_path)

	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
	kmeans.fit(X_scaled)

	artifacts_dir = Path(args.artifacts_dir)
	artifacts_dir.mkdir(parents=True, exist_ok=True)

	dump(kmeans, artifacts_dir / "kmeans_model.pkl")
	dump(scaler, artifacts_dir / "scaler.pkl")

	print(f"Saved artifacts to {artifacts_dir}/kmeans_model.pkl and {artifacts_dir}/scaler.pkl")


if __name__ == "__main__":
	main()


