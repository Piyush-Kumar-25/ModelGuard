from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)


class ConfigureRequest(BaseModel):
    dataset_id: str
    target_column: str
    task: Literal["auto", "classification", "regression"] = "auto"
    time_column: str | None = None
    primary_metric: str | None = None
    random_seed: int = Field(default=42, ge=0, le=2**31 - 1)
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)


class ShieldRequest(BaseModel):
    min_samples: int = Field(default=50, ge=10)
    min_support: float = Field(default=0.05, ge=0.01, le=0.5)
    max_depth: int = Field(default=3, ge=1, le=5)
    bootstrap_iterations: int = Field(default=200, ge=0, le=2000)


class Scenario(BaseModel):
    type: Literal["numeric_shift", "categorical_shift", "missingness", "noise", "extreme_values", "class_prevalence", "combined"]
    feature: str | None = None
    magnitude: float = Field(default=0.1, ge=0, le=10)
    parameters: dict[str, Any] = Field(default_factory=dict)


class DataTwinRequest(BaseModel):
    scenarios: list[Scenario] = Field(min_length=1, max_length=20)
    include_labels: bool = True


class ExperimentRequest(BaseModel):
    experiment_type: Literal["feature_removal", "increased_missingness", "added_noise", "reduced_training_size", "increased_class_imbalance", "distribution_shift", "subgroup_reweighting", "subgroup_oversampling", "threshold_adjustment", "model_retraining"]
    parameters: dict[str, Any] = Field(default_factory=dict)
    random_seed: int = Field(default=42, ge=0, le=2**31 - 1)


class DriftRequest(BaseModel):
    dataset_id: str | None = None
    psi_threshold: float = Field(default=0.2, ge=0.01, le=2)
    js_threshold: float = Field(default=0.1, ge=0.001, le=1)
    include_labels: bool = False
