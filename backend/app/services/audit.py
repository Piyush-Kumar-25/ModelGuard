from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr

from ..database import dump, insert, now, uid
from .data import configured


def finding(name: str, measured: Any, method: str, features: list[str], threshold: Any, status: str, severity: str, evidence: Any, recommendation: str, limitations: str, heuristic: bool) -> dict[str, Any]:
    return {"check_name": name, "measured_value": measured, "method": method, "affected_features": features, "threshold_used": threshold, "status": status, "severity": severity, "supporting_evidence": evidence, "recommendation": recommendation, "limitations": limitations, "evidence_type": "heuristic" if heuristic else "statistically_measured"}


def audit(project_id: str) -> dict[str, Any]:
    cfg, df, meta = configured(project_id)
    target = cfg["target_column"]
    x, y = df.drop(columns=[target]), df[target]
    out: list[dict[str, Any]] = []
    miss = x.isna().mean().sort_values(ascending=False)
    affected = list(miss[miss > 0].index.astype(str))
    out.append(finding("Missing values", {str(k): round(float(v), 4) for k, v in miss[miss > 0].items()}, "per-column null fraction", affected, {"attention": 0.2, "critical": 0.5}, "Needs Attention" if any(miss > .2) else ("Monitor" if affected else "Pass"), "high" if any(miss > .5) else "medium" if affected else "none", None, "Use training-fitted imputation and investigate missingness mechanism.", "Missingness mechanism is not identifiable from this dataset alone.", False))
    dup = int(df.duplicated().sum())
    out.append(finding("Exact duplicate contamination", {"rows": dup, "fraction": round(dup/len(df), 4)}, "full-row duplicated comparison", list(df.columns) if dup else [], 0, "Needs Attention" if dup else "Pass", "high" if dup/len(df) > .05 else "medium" if dup else "none", None, "Deduplicate before splitting or use grouped splitting.", "Identical legitimate repeated events may exist.", False))
    constants = [str(c) for c in x if x[c].nunique(dropna=True) <= 1]
    lowvar = [str(c) for c in x.select_dtypes(include=np.number) if c not in constants and x[c].var(skipna=True) <= 1e-8]
    out.append(finding("Constant and low-variance columns", {"constant": constants, "near_zero_variance": lowvar}, "unique counts and numeric variance", constants + lowvar, 1e-8, "Needs Attention" if constants else "Monitor" if lowvar else "Pass", "low", None, "Remove non-informative features after domain review.", "Variance is scale-dependent.", False))
    invalid = []
    for c in x.select_dtypes(include="object"):
        vals = x[c].dropna().astype(str)
        numeric_rate = pd.to_numeric(vals, errors="coerce").notna().mean() if len(vals) else 0
        if .2 < numeric_rate < .95: invalid.append(str(c))
    out.append(finding("Inconsistent data types", invalid, "mixed numeric parseability in text columns", invalid, {"mixed_rate": [0.2, 0.95]}, "Needs Attention" if invalid else "Pass", "medium" if invalid else "none", None, "Normalize types using documented parsing rules.", "Semantic types require domain knowledge.", True))
    if cfg["task"] == "classification":
        shares = y.value_counts(normalize=True, dropna=False)
        ratio = float(shares.min()/shares.max())
        out.append(finding("Class imbalance", {str(k): round(float(v), 4) for k, v in shares.items()}, "class prevalence ratio", [target], 0.2, "Needs Attention" if ratio < .1 else "Monitor" if ratio < .2 else "Pass", "high" if ratio < .1 else "medium" if ratio < .2 else "none", {"minority_to_majority_ratio": round(ratio, 4)}, "Use stratified evaluation and macro-F1/PR-AUC.", "Thresholds are configurable heuristics.", True))
    cats = x.select_dtypes(exclude=np.number)
    high_card = [str(c) for c in cats if x[c].nunique(dropna=True) > min(100, .2*len(x))]
    sparse = [str(c) for c in cats if x[c].value_counts(normalize=True).lt(.01).any()]
    out.append(finding("Categorical cardinality", {"high_cardinality": high_card, "rare_levels": sparse}, "unique counts and category prevalence", high_card+sparse, {"high_cardinality": "min(100,20% rows)", "rare": .01}, "Needs Attention" if high_card else "Monitor" if sparse else "Pass", "medium" if high_card else "low" if sparse else "none", None, "Consolidate rare categories and validate unseen-category handling.", "Business-important rare categories may merit retention.", True))
    nums = x.select_dtypes(include=np.number)
    pairs = []
    if nums.shape[1] > 1:
        corr = nums.corr().abs()
        for i, a in enumerate(corr.columns):
            for b in corr.columns[i+1:]:
                if corr.loc[a,b] >= .9: pairs.append({"features": [str(a),str(b)], "correlation": round(float(corr.loc[a,b]),4)})
    out.append(finding("High correlations and multicollinearity", pairs, "absolute Pearson correlation", sorted({z for p in pairs for z in p["features"]}), .9, "Needs Attention" if pairs else "Pass", "medium" if pairs else "none", pairs[:20], "Review redundancy; consider regularization or feature removal.", "Pearson correlation detects linear pairwise relationships only.", False))
    outlier = {}
    for c in nums:
        q1,q3=nums[c].quantile([.25,.75]); iqr=q3-q1
        if iqr > 0:
            frac=((nums[c]<q1-1.5*iqr)|(nums[c]>q3+1.5*iqr)).mean()
            if frac > .01: outlier[str(c)] = round(float(frac),4)
    out.append(finding("Outliers and unusual ranges", outlier, "Tukey 1.5 IQR rule", list(outlier), .01, "Monitor" if outlier else "Pass", "medium" if any(v>.1 for v in outlier.values()) else "low" if outlier else "none", None, "Validate units and use robust transformations where justified.", "The IQR rule may flag valid heavy-tailed values.", False))
    # Near duplicates: hash rows after normalizing strings and rounding numerics.
    norm=x.copy()
    for c in norm:
        norm[c]=norm[c].round(3) if pd.api.types.is_numeric_dtype(norm[c]) else norm[c].astype(str).str.strip().str.lower()
    near=int(norm.duplicated(keep=False).sum()-df.duplicated(keep=False).sum()); near=max(0,near)
    out.append(finding("Near-duplicate contamination", {"candidate_rows": near}, "normalized strings and rounded numerics row hash", list(x.columns) if near else [], .01, "Monitor" if near/len(df)>.01 else "Pass", "medium" if near/len(df)>.05 else "low" if near else "none", None, "Inspect candidates and use grouped splitting if records share an entity.", "Approximate rule can miss or over-count semantic duplicates.", True))
    leak_names=re.compile(r"(^target$|label|outcome|result|post_|after_|final_|approved|discharged)",re.I)
    leak=[]
    for c in x:
        score=None
        try:
            if pd.api.types.is_numeric_dtype(x[c]) and pd.api.types.is_numeric_dtype(y): score=abs(float(x[c].corr(y)))
            elif x[c].nunique(dropna=True) < 100: score=float(pd.crosstab(x[c],y,normalize="index").max(axis=1).mean())
        except Exception: pass
        if leak_names.search(str(c)) or (score is not None and score>.98): leak.append({"feature":str(c),"association":None if score is None else round(score,4),"name_warning":bool(leak_names.search(str(c)))})
    status="Critical" if any((z["association"] or 0)>.999 for z in leak) else "Needs Attention" if leak else "Pass"
    out.append(finding("Target leakage and post-outcome candidates", leak, "feature-name pattern plus near-perfect univariate association", [z["feature"] for z in leak], {"association":.98}, status, "critical" if status=="Critical" else "high" if leak else "none", leak, "Confirm feature availability at prediction time; exclude post-outcome fields.", "Association and names cannot prove temporal leakage; domain review is required.", True))
    result={"project_id":project_id,"dataset_id":meta["id"],"findings":out,"summary":{"checks":len(out),"critical":sum(f["status"]=="Critical" for f in out),"needs_attention":sum(f["status"]=="Needs Attention" for f in out)},"generated_at":now()}
    insert("analyses",{"id":uid(),"project_id":project_id,"kind":"audit","result_json":dump(result),"created_at":now()})
    return result
