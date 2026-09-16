# ModelGuard backend

ModelGuard is an evidence-first FastAPI backend for dataset auditing, leakage-safe baseline comparison, automatic failure-region discovery, controlled robustness simulations, remediation experiments, reliability reporting, and batch drift monitoring. The core contains no LLM dependency.

## Run locally

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # export variables manually or use your process manager
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the generated OpenAPI UI. State defaults to `./data`; uploaded originals are stored once as `projects/<id>/datasets/<id>/original.csv` and are never rewritten. Model bundles are local, application-created joblib files; do not replace them with untrusted serialized files.

## Workflow

1. `POST /projects` with `{"name":"Churn reliability"}`.
2. `POST /projects/{id}/datasets` as multipart form field `file` containing a CSV.
3. `POST /projects/{id}/configure` with dataset, target, task (`auto` is supported), optional time column, selection metric, split, and seed.
4. Submit `audit`, `models/train`, `modelshield/analyze`, `datatwin/simulate`, `experiments`, or `drift/analyze` requests.
5. These compute-heavy endpoints return HTTP 202 and a job ID. Poll `GET /jobs/{job_id}` until `completed` or `failed`.
6. Read `/reliability` or `/report`. The optional indicator is disabled by default; add `?include_indicator=true` to expose its fully disclosed provisional calculation.

Production batches use the same upload endpoint and are passed to drift analysis as `dataset_id`. Ground-truth metrics are emitted only when labels are explicitly included and available.

## Reproducibility and evidence

- Preprocessing is fitted inside each candidate pipeline and only on training folds/data.
- Classification uses stratification when feasible; a configured time column selects a chronological holdout.
- Model selection uses the configured cross-validation metric, never accuracy alone.
- ModelShield uses a shallow error-prediction tree, configurable minimum support, and bootstrap uncertainty. Small regions are suppressed.
- DataTwin outputs are explicitly labelled controlled simulations.
- Drift reports practical magnitude separately from statistical significance and do not trigger automatic retraining from a p-value.
- Overall reliability uses transparent precedence rules; critical findings cannot be averaged away, and missing dimensions are not passed.

## Tests

```bash
pytest -q
```

## Current statistical limitations

The audit’s leakage, semantic type, post-outcome, and near-duplicate screens are documented heuristics requiring domain review. Failure regions are exploratory associations, not causal explanations. Confidence intervals condition on the discovered rule and current held-out sample. Drift persistence requires multiple production batches. Optional XGBoost and SHAP are deliberately not required in this initial deterministic implementation.

