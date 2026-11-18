# ModelGuard backend

ModelGuard is an evidence-first FastAPI service for labelled tabular data. It audits an immutable reference CSV, selects a frozen individual model or weighted ensemble, evaluates failure regions and controlled interventions, exports a Python deployment candidate, and monitors explicitly uploaded production batches. This repository contains no frontend and does not deploy or replace models automatically.

## Setup

Python 3.12 is the verified runtime. Python 3.11–3.12 is the intended compatibility range; other versions are not verified. Use scikit-learn 1.7 or newer within the declared `<2` range. The included test evidence records exact installed versions.

```sh
cd backend
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
# Optional native candidates; Auto also works without these packages:
pip install -r requirements-optional.txt
export MODELGUARD_DATA_DIR="$PWD/localdata"
export MODELGUARD_DB_PATH="$PWD/localdata/modelguard.db"
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Use one application worker for this local SQLite release. Put authenticated access, authorization, TLS, quotas and backups around the service before shared deployment. There is no built-in multi-user identity layer. Do not expose the unauthenticated development server to an untrusted network.

Interactive contracts are available at `/docs`; `docs/openapi.json` is a checked-in snapshot. `docs/API_CONTRACTS.md` explains payloads and lifecycle rules. `../REQUIREMENTS_COVERAGE.md` maps the supplied specification to implementation and tests.

## Architecture and persistence

- `app/main.py`, `schemas.py`, `errors.py`: typed API, bounded requests, structured failures and downloads.
- `database.py`, `migrations.py`, `jobs.py`: SQLite records, versioned migrations and supervised local process jobs.
- `services/data.py`, `splitting.py`, `audit.py`: immutable datasets, explicit target confirmation, split manifests and DataDoctor.
- `registry.py`, `training.py`, `modeling.py`, `metrics.py`, `inference.py`: Auto selection, calibration, resource limits, metrics and inference-only wrappers.
- `shield.py`, `twin.py`, `experiments.py`, `drift.py`: frozen-model reliability workflows and separately versioned challengers.
- `reliability.py`, `exporting.py`: measured evidence, reports, cards and Python bundles.

Dataset roles are `reference`, `production_batch` and `production_labels`. The database stores role, project, immutable checksum and path, schema profile, upload/batch timestamps, model version link, confirmed target/label availability, record ID, mappings, compatibility and delayed-label parentage. Public responses omit local paths.

Model storage separates `pipeline.joblib` (prediction state only), private `evaluation.json` (configuration and split indices), the immutable source CSV, and generated reports. Challenger training modifications are recorded in a separate checksummed private `training_reference.csv`. That snapshot is used as the challenger's actual monitoring reference and is never exported. Source rows and split DataFrames are not embedded in inference artifacts.

### Migrations and old installations

Startup runs transactional numbered migrations in `schema_migrations`; it does not require deleting a database. Back up the database and its artifact directory before upgrading. Existing original-release models are marked `legacy`: their unsafe artifacts are not loaded or exported. Retrain a baseline against the existing reference to create a safe current model; history remains stored. New code does not deserialize arbitrary user-uploaded models.

Jobs persist status, stage, progress, result, bounded failure details and timestamps. Restart marks unfinished jobs `interrupted`; it does not silently resume them. Candidate fits have independent subprocess deadlines. Overall job timeout or cancellation terminates the process group on POSIX, including nested fit workers. Only one active job per project is accepted. Model training reports candidate, ensemble and final-test stages. A distributed durable queue and multi-worker coordination remain future infrastructure.

## Reference workflow

1. `POST /projects` with `{"name":"My project"}`.
2. `POST /projects/{project_id}/datasets` with multipart `file=reference.csv`.
3. `GET /projects/{project_id}/datasets/{dataset_id}/target-suggestions`.
4. `POST /projects/{project_id}/configure` to explicitly confirm a target. Suggestions never finalize it.
5. `POST /projects/{project_id}/audit`, then poll `/jobs/{job_id}`.
6. Review leakage warnings and, before training, configure exclusions and confirmed leakage decisions.
7. `POST /projects/{project_id}/models/train`; poll its job, then fetch `/models/results`.
8. Run ModelShield, DataTwin and ExperimentLab against the frozen selected model.
9. Fetch `/reliability` and `/report`; download model-specific cards, reports and deployment candidates.

Example configuration:

```json
{
  "dataset_id": "DATASET_ID",
  "target_column": "churn",
  "task": "classification",
  "positive_class": "yes",
  "record_id_column": "customer_event_id",
  "group_column": "customer_id",
  "primary_metric": "pr_auc",
  "random_seed": 42,
  "validation_size": 0.2,
  "test_size": 0.2,
  "excluded_features": ["post_outcome_balance"],
  "confirmed_leakage": ["post_outcome_balance"],
  "feature_availability_confirmed": true
}
```

Time/group/record-ID fields are removed from predictive inputs. Feature availability confirmation is a domain attestation, not automated proof. Configuration and baseline selection freeze once a current model exists for that reference. To change the reference contract, upload a new reference version or create a challenger.

### Splitting and metrics

Persisted training/validation/final-test indices are disjoint. Random classification splits are stratified; regression splits are reproducible. Unsupported class counts produce explicit errors. Temporal splits preserve strict timestamp ordering, including tied timestamps. Entity splits keep groups disjoint. Combined group/chronological splits exclude boundary-spanning entities and report exclusions. Exact duplicate feature/target rows are excluded before splitting; originals remain unchanged. Semantic near duplicates still require domain review.

Training fits, calibration and supported early stopping use training partitions only. Validation chooses trials, ensemble members and weights. The final test is read only after the configuration is locked. Binary labels may be numeric or strings. The configured positive class is honored; otherwise the last sorted class is the documented default. Multiclass PR-AUC is macro one-vs-rest; weighted PR-AUC is also reported. Metrics include the requested precision/recall/F1 variants, ROC-AUC, log loss, Brier score and ECE. Regression includes MAE, RMSE, median absolute error, R² and MAPE when valid. Undefined metrics are `{ "value": null, "reason": "..." }`.

### Auto and ensembles

There is one Auto mode. Candidates are linear Logistic/Ridge, Decision Tree, Random Forest, Extra Trees, HistGradientBoosting, XGBoost, LightGBM and CatBoost. Optional dependencies are dynamically loaded; unavailable or unsuitable candidates have explicit skip/failure reasons and do not fail otherwise successful selection.

Linear/tree candidates use bounded sparse categorical encoding; HistGradientBoosting uses bounded ordinal encoding; CatBoost retains native categorical columns. Numeric preprocessing is fitted on training data. Default limits are 16 candidate trials (two bounded configurations per family), a 300-second candidate-search budget, 60 seconds per fit, one native thread per model, 100,000 training rows, 500 columns and 5 million input cells. Large classification probability matrices are also bounded. Advanced estimators select iteration counts through early stopping on an internal training holdout and refit on their allowed base-training partition. Classification probabilities use sigmoid calibration on a separate training holdout; recalibration challengers can use isotonic. Group/time constraints and oversampled source-row lineage also apply to calibration. Unsupported calibration partitions cause explicit candidate failures, not claims of calibration.

Failed, clearly dominated and effectively duplicate candidates are excluded from a pool of up to five. Every pool single and subset of sizes two through four is compared. Nonnegative weights are optimized from seeded validation predictions, normalized to one and pruned below 5%. Requested subsets and their pruned configurations are retained in results. Weights are dataset-specific. Selection requires a meaningful predictive, calibration, subgroup or robustness gain under non-inferiority and model-size/latency guards. A four-component ensemble is not preferred merely for being larger. The bounded robustness comparison uses controlled validation missingness; it is not a claim of robustness to all possible shifts.

## Reliability modules

DataDoctor reports measured values, methods, thresholds, affected features, severity, evidence, recommendations and limitations. It includes nulls, duplicates, variance, mixed types, class balance, categorical support, outliers, pairwise correlations, bounded SVD multicollinearity, identifiers, target quality and heuristic leakage/time/group/availability warnings. Semantic warnings are not proof.

ModelShield discovers shallow error-tree rules on validation data and confirms fixed rules on final test data. Rules use original feature names/categories, with sample/support requirements and bootstrap local-metric/drop intervals. Unconfirmed rules cannot become Critical. Insufficient support and no error variation are explicit. Repeated requests return the same frozen analysis, preventing repeated rule tuning against the test. Dense discovery is capped at two million cells. Intervals are exploratory and not multiple-testing-adjusted.

DataTwin applies numeric location shifts with declared units/direction, scaling, noise, missingness, extremes, category distributions, valid class-prevalence resampling, input corruption, combinations and intensity sweeps to validation records. Zero intensity is an identity transformation. The pipeline/preprocessing stay frozen. Labelled analyses report practical metric changes; unlabelled analyses never invoke ground-truth metrics. Regression outputs use moments, quantiles, Wasserstein distance and paired change. Every retained-label scenario is a controlled robustness stress test, not a forecast of future ground truth.

ExperimentLab distinguishes frozen stress tests from challenger training. Remediation supports feature removal, region-ID oversampling, actual estimator/calibrator sample weights, class weights, reduced training data, class-imbalance testing, binary threshold adjustment, recalibration, alternative model/member sets, independent additional labelled data and full Auto reselection. Added rows matching original feature rows or configured identities are excluded; temporal additions must precede the frozen validation boundary. Truly later temporal data needs a new forward reference evaluation.

Challengers remain `experimental`. Validation comparisons include global/subgroup metrics, calibration, inference time, size and trade-offs on identical records. Explicit `finalize` reserves the only finalized challenger test comparison for that reference dataset, even after promotion. `approve` requires a passing finalized primary/calibration/supported-subgroup comparison. `promote` requires prior approval and the same active parent. Rejection is explicit. The selected pipeline never changes automatically.

## Production monitoring

Upload batches through `/projects/{project_id}/monitoring/batches`, never through an ambiguous latest-dataset reference. Each batch remains linked to its selected model version. Extra features are profiled and ignored by that model. Missing/fully null critical inputs and type mismatches block predictions before preprocessing. Renames require explicit mappings. Optional missing inputs are imputed only after validated fallback checks; simultaneous optional omissions also require joint fallback validation. Unseen categories, ranges and unit-like magnitude shifts are reported even when prediction is technically possible.

The project retains its original target. Exact target columns are detected but require confirmation. Renamed target candidates require mapping confirmation; prediction/probability columns are never accepted as ground truth. An unlabelled batch still supports raw/prediction/confidence monitoring. Attach actual labels later by unique stable record ID. Partial labels are reported as partial; originals are never overwritten.

Drift compares raw production features with actual saved training data before preprocessing, then uses the original fitted transforms for model-aware outputs. Results include missingness, PSI, KS statistics/p-values, Wasserstein distance, categorical Jensen–Shannon distance/proportions, raw and transformed summaries, prediction/probability/confidence changes, ensemble disagreement, fixed-region prevalence, label-based performance and calibration. Practical thresholds and statistical p-values are separate. Native categorical transformations can report raw drift with an explicit transformed-summary limitation.

History retains distinct batches ordered by batch timestamp; reanalyzing a batch does not create artificial persistence. Strong retraining advice needs persistent important-feature drift and supported performance degradation with a bootstrap lower confidence bound above the acceptance threshold. Minimum monitoring-label support defaults to 30. Small samples and unavailable primary metrics cannot establish stability. Unlabelled recommendations remain cautious. Recalibration, threshold adjustment, retraining evaluation and schema redevelopment are separate actions. None promotes a model.

### Evidence and thresholds

Seven dimensions are reported: Data Quality, Leakage Safety, Evaluation Integrity, Subgroup Robustness, Noise and Missingness Robustness, Distribution-Shift Robustness, and Production Stability. Statuses are Pass, Monitor, Needs Attention, Critical, Not Evaluated and Insufficient Evidence. Scenarios accumulate by model rather than replacing earlier adverse evidence. Evaluation Integrity requires a disjoint manifest and locked final evaluation; model existence is insufficient. Production Stability is Not Evaluated until a batch exists.

Reports separate `pre_deployment_readiness`, `production_stability` and `overall_lifecycle_status`, including coverage and explicit precedence. Critical findings are never averaged away. The numerical indicator remains disabled, even if the legacy `include_indicator` flag is requested. Recommendations are grouped as Immediate, Before deployment, Monitoring and Future improvement; evidence taxonomy distinguishes facts, statistics, heuristics, simulations, predictions, recommendations and unavailable evidence.

Configuration exposes practical acceptance drop (default 0.05), critical drop (0.20), meaningful selection/calibration gains (0.01), selection tolerance (0.02), PSI (0.20), JS (0.10), missingness change (0.10), normalized Wasserstein (0.50), affected drift fraction (0.20), calibration drift (0.05), persistence count (2) and monitoring-label minimum (30). These are disclosed heuristic decision thresholds, not universal clinical/financial safety guarantees.

## Downloads and inference

- `GET /projects/{pid}/models/{mid}/card?format=json|markdown`
- `GET /projects/{pid}/models/{mid}/report?format=json|markdown`
- `GET /projects/{pid}/models/{mid}/export`
- `POST /projects/{pid}/models/{mid}/predict` with `{"records":[{...}]}`

The deployment ZIP contains the inference pipeline, ensemble configuration, schema, reversible target mapping, threshold, metadata, pinned inference dependencies, an inference example, JSON/Markdown Model Card, JSON reliability report, custom inference-class source and checksums. No private evaluation snapshots or raw training/test/production rows are included. Source package ZIPs and trained deployment bundles are different artifacts.

Extract a trusted bundle, create an environment matching its Python/library metadata, install its `requirements.txt`, then run `python inference_example.py input.csv`. The example verifies checksums and schema before inference. Checksums detect accidental corruption; they are not a signature authenticating an untrusted publisher. **Joblib may execute code: load only trusted artifacts.** Python bundles are not universally portable. PDF and ONNX are optional future formats; neither is advertised as implemented.

## Verification and limitations

```sh
python -m compileall -q app tests
pytest -q --cov=app.services --cov=app.jobs --cov=app.migrations --cov-report=term-missing
```

All fixtures are synthetic and are test evidence only, not real-world product performance. See `../IMPLEMENTATION_STATUS.md` for the final measured test count, versions and coverage. Finite samples, one held-out reference, bounded search, multiple exploratory slices, retained-label simulation assumptions and delayed-label selection bias constrain conclusions. Candidate timeouts bound individual fits; overall jobs have a separate deadline. Small or incompatible group/time/class partitions may be rejected. Graceful restart does not resume unfinished work. A timed-out job may leave a completed immutable artifact that must be inspected before resubmission; no partial artifact is automatically promoted.

Future work includes distributed workers, multi-user authorization, forward/rolling validation and calibrated multiple-testing procedures, optional PDF/validated ONNX, and external deployment orchestration. LLM explanations, image models, bring-your-own-pickle audit, stacking, synthetic production labels, arbitrary authoritative scores and automatic deployment/replacement are intentionally excluded from this release.
