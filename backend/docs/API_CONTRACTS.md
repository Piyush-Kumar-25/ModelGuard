# ModelGuard API contracts

`openapi.json` is the complete generated schema. Unknown request fields are rejected. Structured failures have `error.code`, `error.message` and optional bounded `error.details`; internal exceptions do not expose stack traces or filesystem paths. IDs are scoped to their project. There is no built-in user authentication in this local backend.

## Jobs

Analysis/training POSTs return HTTP 202 with `job_id`, `kind` and `status`. `GET /jobs/{job_id}` returns `queued`, `running`, `completed`, `failed`, `timed_out`, `cancelled` or `interrupted`, plus percentage/stage and result/error. Poll until terminal. `POST /jobs/{job_id}/cancel` requests cancellation. A conflicting active project job returns 409.

## Reference and configuration

`POST /projects`: name and optional description. Upload multipart `file` to `/projects/{pid}/datasets` (reference role). The response contains ID, role, immutable checksum, dimensions and schema, never local storage paths. Summary: `GET /projects/{pid}/datasets/{did}/summary`. Target suggestions: `GET .../{did}/target-suggestions`; `confirmation_required` is true and target scores include name/type/cardinality/missingness/position evidence.

`POST /projects/{pid}/configure`: `dataset_id`, `target_column`, optional `task` (auto/classification/regression), positive class, time/group/record-ID columns, metric, seed, train/validation/test proportions, exclusions, confirmed leakage, optional feature fallbacks, availability confirmation, resource and evidence thresholds. Training fraction is the remainder of validation/test fractions. See OpenAPI for defaults/ranges. Configuration freezes once a current model exists for the dataset; legacy unsafe models do not prevent safe retraining.

Audit: `POST /projects/{pid}/audit`. Training: `POST .../models/train`. Results: `GET .../models/results`, including each model version/status, validation candidate reasons, requested/pruned ensemble subsets and weights, selected reason and final metrics. An unavailable metric has `value: null` and `reason` instead of NaN or omission.

## Frozen analyses

`POST /projects/{pid}/modelshield/analyze`: `min_samples`, `min_support`, `max_depth`, `bootstrap_iterations`. Discovery and confirmation partitions, structured reusable clauses, local metrics/intervals, confirmation status and support are explicit. Repeated calls return the first frozen analysis for that model.

`POST /projects/{pid}/datatwin/simulate`:

```json
{
  "include_labels": true,
  "random_seed": 42,
  "intensities": [0, 0.5, 1],
  "scenarios": [{
    "type": "numeric_shift",
    "feature": "income",
    "magnitude": 0.5,
    "parameters": {"direction": "decrease", "units": "standard_deviations"}
  }]
}
```

Types: numeric_shift, numeric_scaling, noise, missingness, extreme_values, categorical_shift, class_prevalence, input_corruption, combined. Scaling accepts `parameters.factor`; categorical/prevalence shifts accept `parameters.distribution`; combinations accept `parameters.scenarios`. `parameters.features` selects several inputs for a compatible intervention. Probability-like magnitudes must remain in [0,1]; directions/units and requested distributions are validated. Up to 100 total scenario/intensity evaluations. Results include resolved definition, affected features, distributions, model outputs, optional label metrics, measured change, sensitivity ordering and practical tipping points.

## ExperimentLab and lifecycle

`POST /projects/{pid}/experiments`: `experiment_type`, bounded `parameters`, seed.

| Category | Types / parameters |
|---|---|
| Frozen stress | increased_missingness, added_noise, distribution_shift, extreme_values, input_corruption, unavailable_features; feature/features, magnitude and type-specific parameters |
| Feature removal | feature_removal: features |
| Region training intervention | subgroup_reweighting / subgroup_oversampling: region_id, factor (1–10) |
| Class weights | class_weighting: optional string-keyed weights, otherwise inverse frequency |
| Reduced data | reduced_training_size: fraction; increased_class_imbalance: class and fraction |
| Threshold | threshold_adjustment: threshold (binary only) |
| Calibration | probability_recalibration: sigmoid or isotonic method |
| Candidate sets | alternative_model: one models entry; alternative_ensemble: 2–4 names; model_retraining: optional models list |
| New labelled rows | additional_labelled_data: dataset_id; production labels must be confirmed first |

Registry names: `linear`, `decision_tree`, `random_forest`, `extra_trees`, `hist_gradient_boosting`, `xgboost`, `lightgbm`, `catboost`. Alternative ensemble sets can still select a simpler winner if the ensemble gives no qualifying benefit. Unknown parameters/names are errors.

Challengers have their own ID/version and private training reference. Validation comparisons do not consume final-test evidence. `POST /projects/{pid}/models/{mid}/lifecycle` takes `{"action":"finalize|reject|approve|promote"}`. Finalize locks/reserves one challenger final-test comparison per reference dataset. Approve requires finalized primary/calibration/supported-subgroup non-inferiority. Promote requires approval and the unchanged parent incumbent. The promoted model becomes `incumbent`, its lifecycle metadata records `promoted`, and the former incumbent becomes `superseded`. No automatic promotion occurs.

## Monitoring

- `POST /projects/{pid}/monitoring/batches`: multipart file, optional model_id, ISO batch_timestamp, record_id_column. Returns compatibility, extra/missing/type/category/range warnings and label detection. A new upload returns 201. A byte-identical upload for the same project and model returns 200 with the original batch ID, timestamp and mapping, and `duplicate_upload: true`. An upload retry does not create new monitoring evidence.
- `GET .../monitoring/batches`: explicit stored batches and mapping/availability metadata.
- `POST .../monitoring/batches/{bid}/mapping`: source-to-expected `feature_mapping`, optional ground-truth `target_column`, `confirm_labels`.
- `POST .../monitoring/batches/{bid}/labels`: actual-label CSV, optional target_column; unique stable IDs must match that batch. Partial files add non-null values, including when inline labels are present. The response distinguishes `matched_labels`, `new_labels` and `total_labelled_records`. Identical resubmissions are no-ops. Contradictions return 409 `label_conflict` without changing confirmed labels. Reanalyze the batch to recompute drift and performance after changes.
- `POST .../monitoring/batches/{bid}/analyze`: batch-specific background job. All confirmed labels are used; unconfirmed labels are not.
- `GET .../monitoring/history?model_id=...`: latest analysis per distinct batch, ordered by batch timestamp, with legacy byte-identical duplicates excluded. Historical model links remain fixed after promotion. `analysis_stale` means the last analysis predates a label or mapping change and must not be used for current decisions.

The legacy `/drift/analyze` requires an explicit production `dataset_id`; it cannot choose a latest dataset implicitly. Its legacy `include_labels` field does not bypass explicit label confirmation.

Schema statuses are `Compatible`, `Compatible with warnings`, `Incompatible`. Missing/fully null critical inputs block prediction. Optional fallback is allowed only after validation, including simultaneous omissions. Extra features remain available for profiling but never enter the frozen model. Renames require confirmation. Target presence does not repair missing predictive inputs. Prediction/probability columns cannot be confirmed as ground truth.

Drift outputs separate raw changes, practical flags, statistical evidence, predictions/confidence, components, subgroups, confirmed labels, calibration, persistence and action rationale. Strong recommendations require supported degradation and persistent important-feature drift. A recommendation does not execute retraining or replacement.

## Reports, downloads and prediction

Project report/reliability endpoints and model-specific downloads use the same evidence rules. Model-specific card/report support JSON and Markdown with attachment headers; the card indicates `deployment_export_allowed`. Export returns an `application/zip` inference-safe Python bundle only for a finalized incumbent or a finalized, approved challenger of the active incumbent. Experimental, rejected, stale-approved and superseded models cannot be exported (409 for ineligible current-format models); evaluation cards and reports remain available. POST model prediction accepts `records` (1–10,000 objects) and returns predictions, class order/probabilities where applicable, model/version and schema warnings. Incompatible records return structured HTTP 422.

Reports separate pre-deployment readiness, production stability and overall lifecycle status with coverage and explicit rules. No-production status is Not Evaluated. Numerical indicators are disabled. Reproducibility metadata includes dataset ID/checksum, model version, schema version/fingerprint and recorded training-time Python/library versions.
