"""Model evaluation and segment business profiling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.metrics.cluster import adjusted_rand_score


@dataclass
class ModelCandidate:
    """Comparable clustering candidate produced during model selection."""

    n_clusters: int
    silhouette_score: float
    davies_bouldin_score: float
    calinski_harabasz_score: float
    inertia: float
    cluster_balance: float
    average_confidence: float
    problematic_share: float
    stability_ari: float
    composite_score: float
    cluster_sizes: Dict[int, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def collect_clustering_metrics(model: KMeans, features_scaled: np.ndarray) -> Dict[str, Any]:
    """Compute clustering quality metrics for a fitted KMeans model."""
    labels = model.labels_
    distances = model.transform(features_scaled)
    assigned_distances = distances[np.arange(len(features_scaled)), labels]
    confidence_scores = 1.0 / (1.0 + assigned_distances)
    silhouette_values = silhouette_samples(features_scaled, labels)

    cluster_sizes = {
        int(cluster_id): int(size)
        for cluster_id, size in zip(*np.unique(labels, return_counts=True))
    }
    size_values = np.array(list(cluster_sizes.values()), dtype=float)
    balance = float(max(0.0, 1.0 - (size_values.std(ddof=0) / size_values.mean())))

    return {
        "inertia": float(model.inertia_),
        "silhouette_score": float(silhouette_score(features_scaled, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(features_scaled, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(features_scaled, labels)),
        "cluster_sizes": cluster_sizes,
        "cluster_balance": balance,
        "average_confidence": float(confidence_scores.mean()),
        "problematic_share": float((silhouette_values < 0).mean()),
    }


def estimate_kmeans_stability(
    features_scaled: np.ndarray,
    n_clusters: int,
    baseline_labels: np.ndarray,
    random_state: int,
    runs: int,
) -> float:
    """Estimate cluster stability across repeated fits."""
    stability_scores: List[float] = []

    for offset in range(1, runs + 1):
        candidate = KMeans(
            n_clusters=n_clusters,
            n_init=25,
            random_state=random_state + offset,
        )
        candidate.fit(features_scaled)
        stability_scores.append(adjusted_rand_score(baseline_labels, candidate.labels_))

    if not stability_scores:
        return 0.0
    return float(np.mean(stability_scores))


def score_candidate(
    model: KMeans,
    features_scaled: np.ndarray,
    random_state: int,
    stability_runs: int,
) -> ModelCandidate:
    """Evaluate a fitted KMeans model and turn it into a ranked candidate."""
    metrics = collect_clustering_metrics(model, features_scaled)
    stability = estimate_kmeans_stability(
        features_scaled=features_scaled,
        n_clusters=model.n_clusters,
        baseline_labels=model.labels_,
        random_state=random_state,
        runs=stability_runs,
    )

    silhouette_component = (metrics["silhouette_score"] + 1.0) / 2.0
    davies_component = 1.0 / (1.0 + metrics["davies_bouldin_score"])
    balance_component = metrics["cluster_balance"]
    confidence_component = metrics["average_confidence"]
    cleanliness_component = 1.0 - metrics["problematic_share"]

    composite_score = (
        0.35 * silhouette_component
        + 0.20 * davies_component
        + 0.20 * stability
        + 0.15 * balance_component
        + 0.05 * confidence_component
        + 0.05 * cleanliness_component
    )

    return ModelCandidate(
        n_clusters=model.n_clusters,
        silhouette_score=metrics["silhouette_score"],
        davies_bouldin_score=metrics["davies_bouldin_score"],
        calinski_harabasz_score=metrics["calinski_harabasz_score"],
        inertia=metrics["inertia"],
        cluster_balance=metrics["cluster_balance"],
        average_confidence=metrics["average_confidence"],
        problematic_share=metrics["problematic_share"],
        stability_ari=stability,
        composite_score=float(composite_score),
        cluster_sizes=metrics["cluster_sizes"],
    )


def rank_candidates(candidates: List[ModelCandidate]) -> List[Dict[str, Any]]:
    """Sort model candidates from best to worst."""
    return [
        candidate.to_dict()
        for candidate in sorted(candidates, key=lambda item: item.composite_score, reverse=True)
    ]


def _level(value: float, lower: float, upper: float) -> str:
    if value <= lower:
        return "low"
    if value >= upper:
        return "high"
    return "mid"


def _segment_story(age_level: str, income_level: str, spending_level: str) -> Dict[str, Any]:
    if income_level == "high" and spending_level == "high":
        if age_level == "low":
            return {
                "name": "Rising VIPs",
                "description": "Young, affluent customers with strong purchase intent.",
                "actions": [
                    "Target with premium launches and early-access drops.",
                    "Prioritize loyalty retention with personalized perks.",
                ],
            }
        return {
            "name": "Premium Loyalists",
            "description": "High-value customers with sustained spending power.",
            "actions": [
                "Invest in concierge experiences and cross-sell bundles.",
                "Use them as a priority audience for high-margin campaigns.",
            ],
        }

    if income_level == "high" and spending_level == "low":
        return {
            "name": "Affluent Conservatives",
            "description": "Financially strong customers who spend selectively.",
            "actions": [
                "Lead with trust, proof points, and premium utility messaging.",
                "Test upgrade nudges tied to long-term value rather than urgency.",
            ],
        }

    if income_level == "low" and spending_level == "high":
        return {
            "name": "Aspirational Spenders",
            "description": "Budget-constrained customers who still show strong demand.",
            "actions": [
                "Offer financing, bundles, or limited-time offers to capture intent.",
                "Monitor discount sensitivity to protect margin.",
            ],
        }

    if income_level == "low" and spending_level == "low":
        return {
            "name": "Value Seekers",
            "description": "Price-sensitive customers with restrained spending behavior.",
            "actions": [
                "Emphasize value packs, price anchors, and win-back campaigns.",
                "Avoid over-investing in high-touch premium experiences.",
            ],
        }

    if age_level == "high":
        return {
            "name": "Established Regulars",
            "description": "Mature, steady customers with balanced spending habits.",
            "actions": [
                "Lean into reliability, service quality, and retention programs.",
                "Use lifecycle nudges to increase order frequency.",
            ],
        }

    return {
        "name": "Mainstream Momentum",
        "description": "Core customers sitting near the center of the portfolio.",
        "actions": [
            "Use them to validate broad campaigns before scaling.",
            "Personalize offers based on incremental spend potential.",
        ],
    }


def build_segment_catalog(raw_customers: pd.DataFrame, labels: np.ndarray) -> List[Dict[str, Any]]:
    """Create business-readable segment summaries."""
    working = raw_customers.copy()
    working["segment_id"] = labels

    age_bounds = working["age"].quantile([0.33, 0.67]).tolist()
    income_bounds = working["annual_income"].quantile([0.33, 0.67]).tolist()
    spending_bounds = working["spending_score"].quantile([0.33, 0.67]).tolist()

    catalog: List[Dict[str, Any]] = []
    total_customers = len(working)

    for segment_id in sorted(working["segment_id"].unique()):
        segment_frame = working[working["segment_id"] == segment_id]
        avg_age = float(segment_frame["age"].mean())
        avg_income = float(segment_frame["annual_income"].mean())
        avg_spending = float(segment_frame["spending_score"].mean())

        age_level = _level(avg_age, age_bounds[0], age_bounds[1])
        income_level = _level(avg_income, income_bounds[0], income_bounds[1])
        spending_level = _level(avg_spending, spending_bounds[0], spending_bounds[1])
        story = _segment_story(age_level, income_level, spending_level)

        catalog.append(
            {
                "segment_id": int(segment_id),
                "segment_name": story["name"],
                "segment_description": story["description"],
                "recommended_actions": story["actions"],
                "size": int(len(segment_frame)),
                "customer_share": round(float(len(segment_frame) / total_customers), 4),
                "avg_age": round(avg_age, 2),
                "avg_annual_income": round(avg_income, 2),
                "avg_spending_score": round(avg_spending, 2),
                "age_level": age_level,
                "income_level": income_level,
                "spending_level": spending_level,
            }
        )

    return catalog
