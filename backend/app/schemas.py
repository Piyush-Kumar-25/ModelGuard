from typing import Any, Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator

class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid', allow_inf_nan=False)

class ProjectCreate(Strict):
    name: str=Field(min_length=1,max_length=120)
    description: str|None=Field(default=None,max_length=1000)

class ConfigureRequest(Strict):
    dataset_id: str
    target_column: str
    task: Literal['auto','classification','regression']='auto'
    positive_class: str|int|float|None=None
    time_column: str|None=None
    group_column: str|None=None
    record_id_column: str|None=None
    primary_metric: str|None=None
    random_seed: int=Field(default=42,ge=0,le=2**31-1)
    test_size: float=Field(default=.2,ge=.1,le=.4)
    validation_size: float=Field(default=.2,ge=.1,le=.4)
    excluded_features: list[str]=Field(default_factory=list)
    confirmed_leakage: list[str]=Field(default_factory=list)
    optional_features: list[str]=Field(default_factory=list)
    feature_availability_confirmed: bool=False
    max_trials: int=Field(default=16,ge=1,le=64)
    time_budget_seconds: int=Field(default=300,ge=5,le=1800)
    fit_timeout_seconds: int=Field(default=60,ge=1,le=600)
    weight_trials: int=Field(default=16,ge=1,le=100)
    meaningful_gain: float=Field(default=.01,ge=0,le=.5)
    calibration_gain: float=Field(default=.01,ge=0,le=.5)
    selection_tolerance: float=Field(default=.02,ge=0,le=.2)
    acceptance_drop: float=Field(default=.05,gt=0,le=.5)
    critical_drop: float=Field(default=.2,gt=0,le=1)
    psi_threshold: float=Field(default=.2,gt=0,le=5)
    js_threshold: float=Field(default=.1,gt=0,le=1)
    missingness_drift_threshold: float=Field(default=.1,gt=0,le=1)
    wasserstein_threshold: float=Field(default=.5,gt=0,le=10)
    drift_fraction_threshold: float=Field(default=.2,gt=0,le=1)
    calibration_drift_threshold: float=Field(default=.05,gt=0,le=1)
    minimum_monitoring_labels: int=Field(default=30,ge=10,le=10000)
    persistent_batches: int=Field(default=2,ge=2,le=20)
    @model_validator(mode='after')
    def coherent(self):
        if self.test_size+self.validation_size>.7: raise ValueError('At least 30% training data required')
        if self.critical_drop<self.acceptance_drop: raise ValueError('critical_drop must be >= acceptance_drop')
        return self

class ShieldRequest(Strict):
    min_samples: int=Field(default=50,ge=10,le=100000)
    min_support: float=Field(default=.05,ge=.01,le=.5)
    max_depth: int=Field(default=3,ge=1,le=5)
    bootstrap_iterations: int=Field(default=200,ge=0,le=2000)

class Scenario(Strict):
    type: Literal['numeric_shift','numeric_scaling','categorical_shift','missingness','noise','extreme_values','input_corruption','class_prevalence','combined']
    feature: str|None=None
    magnitude: float=Field(default=.1,ge=0,le=10)
    parameters: dict[str,Any]=Field(default_factory=dict)

class DataTwinRequest(Strict):
    scenarios: list[Scenario]=Field(min_length=1,max_length=20)
    include_labels: bool=True
    intensities: list[float]=Field(default_factory=lambda:[1.],min_length=1,max_length=20)
    random_seed: int=Field(default=42,ge=0,le=2**31-1)
    @model_validator(mode='after')
    def bounded(self):
        if any(not 0<=v<=10 for v in self.intensities):raise ValueError('Intensity must be between 0 and 10')
        if len(self.intensities)*len(self.scenarios)>100:raise ValueError('At most 100 scenario evaluations')
        return self

class ExperimentRequest(Strict):
    experiment_type: Literal['feature_removal','increased_missingness','added_noise','distribution_shift','extreme_values','input_corruption','unavailable_features','reduced_training_size','increased_class_imbalance','subgroup_reweighting','subgroup_oversampling','class_weighting','threshold_adjustment','probability_recalibration','alternative_model','alternative_ensemble','additional_labelled_data','model_retraining']
    parameters: dict[str,Any]=Field(default_factory=dict)
    random_seed: int=Field(default=42,ge=0,le=2**31-1)

class DriftRequest(Strict):
    dataset_id: str|None=None
    include_labels: bool=False

class MappingRequest(Strict):
    feature_mapping: dict[str,str]=Field(default_factory=dict,description='Source batch column to expected model column')
    target_column: str|None=None
    confirm_labels: bool=False

class PredictRequest(Strict):
    records: list[dict[str,Any]]=Field(min_length=1,max_length=10000)

class LifecycleRequest(Strict):
    action: Literal['finalize','reject','approve','promote']
