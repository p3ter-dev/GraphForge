"""
Model Registry and Validation-Based Promotion Gate
==================================================
GraphForge Assignment 4: Model Governance & Promotion Logic

Features:
  - Strict promotion gate comparing Candidate against current Champion
  - Configurable comparison metric and improvement threshold delta
  - Explicit tie, missing-metric, and first-model behavior
  - Blocked regression protection
  - Rollback capability to restore previous champion
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class ModelRegistrationRecord:
    version: str
    model_name: str
    architecture: str
    package_dir: str
    metrics: Dict[str, float]
    status: str  # "champion", "candidate", "rejected", "archived"
    registered_at: str
    git_commit: str = "unknown"
    promotion_notes: str = ""


class ModelRegistry:
    """
    Manages model registration, promotion gating, and rollback.
    """

    def __init__(self, registry_file: str | Path = "artifacts/model_registry.json"):
        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                return json.load(f)
        return {
            "current_champion": None,
            "previous_champion": None,
            "models": {},
            "history": [],
        }

    def _save(self) -> None:
        with open(self.registry_file, "w") as f:
            json.dump(self.data, f, indent=2)

    def get_champion(self) -> Optional[Dict[str, Any]]:
        champ_ver = self.data["current_champion"]
        if champ_ver and champ_ver in self.data["models"]:
            return self.data["models"][champ_ver]
        return None

    def evaluate_and_promote(
        self,
        candidate_record: ModelRegistrationRecord,
        metric_name: str = "val_acc",
        threshold_delta: float = 0.005,
    ) -> Tuple[bool, str]:
        """
        Evaluate candidate against current champion.

        Rules:
        1. First-Model Behavior: If no champion exists, candidate is automatically promoted.
        2. Missing-Metric Behavior: If candidate lacks metric_name, promotion is rejected.
        3. Tie Behavior: Candidate must STRICTLY exceed champion by >= threshold_delta.
           Ties leave current champion in place.
        4. Blocked Regression: If candidate score <= champion score, rejected.
        """
        import datetime
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        candidate_record.registered_at = now_str
        ver = candidate_record.version

        # Store candidate in registry
        self.data["models"][ver] = asdict(candidate_record)

        # 1. Missing metric check
        if metric_name not in candidate_record.metrics:
            candidate_record.status = "rejected"
            candidate_record.promotion_notes = f"Rejected: Metric '{metric_name}' missing from candidate metrics."
            self.data["models"][ver] = asdict(candidate_record)
            self._save()
            return False, candidate_record.promotion_notes

        candidate_score = candidate_record.metrics[metric_name]
        current_champion = self.get_champion()

        # 2. First model behavior
        if current_champion is None:
            candidate_record.status = "champion"
            candidate_record.promotion_notes = (
                f"Promoted: Initial champion registered with {metric_name}={candidate_score:.4f}."
            )
            self.data["current_champion"] = ver
            self.data["models"][ver] = asdict(candidate_record)
            self.data["history"].append({
                "timestamp": now_str,
                "action": "initial_promotion",
                "version": ver,
                "score": candidate_score,
            })
            self._save()
            return True, candidate_record.promotion_notes

        # 3. Champion comparison
        champion_score = current_champion["metrics"].get(metric_name, 0.0)
        score_diff = candidate_score - champion_score

        if score_diff >= threshold_delta:
            # Successful promotion!
            old_champ_ver = self.data["current_champion"]
            self.data["previous_champion"] = old_champ_ver
            self.data["models"][old_champ_ver]["status"] = "archived"

            candidate_record.status = "champion"
            candidate_record.promotion_notes = (
                f"Promoted: {metric_name} improved from {champion_score:.4f} to {candidate_score:.4f} "
                f"(+{score_diff:.4f} >= threshold {threshold_delta:.4f})."
            )
            self.data["current_champion"] = ver
            self.data["models"][ver] = asdict(candidate_record)
            self.data["history"].append({
                "timestamp": now_str,
                "action": "promoted",
                "new_version": ver,
                "previous_version": old_champ_ver,
                "score": candidate_score,
            })
            self._save()
            return True, candidate_record.promotion_notes
        else:
            # Blocked regression or insufficient improvement
            candidate_record.status = "rejected"
            candidate_record.promotion_notes = (
                f"Blocked Regression: Candidate {metric_name}={candidate_score:.4f} did not exceed "
                f"champion {champion_score:.4f} by required threshold +{threshold_delta:.4f} (diff={score_diff:+.4f})."
            )
            self.data["models"][ver] = asdict(candidate_record)
            self.data["history"].append({
                "timestamp": now_str,
                "action": "blocked_regression",
                "version": ver,
                "score": candidate_score,
                "champion_score": champion_score,
            })
            self._save()
            return False, candidate_record.promotion_notes

    def rollback(self) -> Tuple[bool, str]:
        """
        Restore the previous champion model.
        """
        import datetime
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        prev_ver = self.data.get("previous_champion")

        if not prev_ver or prev_ver not in self.data["models"]:
            return False, "Rollback failed: No previous approved champion available."

        demoted_ver = self.data["current_champion"]
        if demoted_ver:
            self.data["models"][demoted_ver]["status"] = "archived"

        self.data["current_champion"] = prev_ver
        self.data["models"][prev_ver]["status"] = "champion"
        self.data["previous_champion"] = None

        msg = f"Rollback successful: Restored version '{prev_ver}' as active champion."
        self.data["history"].append({
            "timestamp": now_str,
            "action": "rollback",
            "restored_version": prev_ver,
            "demoted_version": demoted_ver,
        })
        self._save()
        return True, msg
