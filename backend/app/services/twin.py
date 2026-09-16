from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from ..database import dump, insert, now, uid
from ..errors import ModelGuardError
from .modeling import evaluate, latest_model

LABEL="Controlled scenario simulation—not a prediction of the actual future."


def _apply(df: pd.DataFrame, spec: dict, rng) -> tuple[pd.DataFrame,list[str]]:
    out=df.copy(); typ=spec["type"]; mag=spec["magnitude"]; feature=spec.get("feature"); changed=[]
    targets=[feature] if feature else list(out.select_dtypes(include=np.number).columns)
    if typ=="numeric_shift":
        for c in targets:
            if c not in out or not pd.api.types.is_numeric_dtype(out[c]): continue
            out[c]=out[c]+mag*out[c].std(skipna=True); changed.append(c)
    elif typ=="noise":
        for c in targets:
            if c not in out or not pd.api.types.is_numeric_dtype(out[c]): continue
            out[c]=out[c]+rng.normal(0,mag*(out[c].std(skipna=True) or 1),len(out)); changed.append(c)
    elif typ=="missingness":
        targets=[feature] if feature else list(out.columns)
        for c in targets:
            if c in out: out.loc[rng.random(len(out))<min(mag,1),c]=np.nan; changed.append(c)
    elif typ=="extreme_values":
        for c in targets:
            if c not in out or not pd.api.types.is_numeric_dtype(out[c]): continue
            mask=rng.random(len(out))<min(mag,1); out.loc[mask,c]=out[c].median()+rng.choice([-1,1],mask.sum())*5*(out[c].std(skipna=True) or 1); changed.append(c)
    elif typ=="categorical_shift":
        if not feature or feature not in out: raise ModelGuardError("categorical_shift requires a valid feature","invalid_scenario")
        values=out[feature].dropna().value_counts();
        if values.empty: raise ModelGuardError("Categorical feature has no observed values","invalid_scenario")
        mask=rng.random(len(out))<min(mag,1); out.loc[mask,feature]=values.index[0]; changed=[feature]
    elif typ=="combined":
        for nested in spec.get("parameters",{}).get("scenarios",[]):
            out,ch=_apply(out,nested,rng); changed.extend(ch)
    return out,sorted(set(changed))


def _distribution_change(a,b):
    result={}
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]):
            scale=float(a[c].std(skipna=True) or 1); result[str(c)]={"standardized_mean_change":float(abs(b[c].mean()-a[c].mean())/scale),"missingness_change":float(b[c].isna().mean()-a[c].isna().mean())}
        else:
            keys=a[c].astype(str).value_counts(normalize=True).index.union(b[c].astype(str).value_counts(normalize=True).index)
            p=a[c].astype(str).value_counts(normalize=True).reindex(keys,fill_value=0); q=b[c].astype(str).value_counts(normalize=True).reindex(keys,fill_value=0)
            result[str(c)]={"jensen_shannon_distance":float(jensenshannon(p,q,base=2)),"missingness_change":float(b[c].isna().mean()-a[c].isna().mean())}
    return result


def simulate(project_id: str, payload: dict) -> dict:
    model,art=latest_model(project_id); rng=np.random.default_rng(art["config"]["random_seed"]); x=art["x_test"]; y=art["y_test"]; pipe=art["pipeline"]; baseline=model["metrics"]
    results=[]
    for spec in payload["scenarios"]:
        scenario_y=y
        prevalence=None
        if spec["type"]=="class_prevalence":
            if not payload["include_labels"] or model["task"]!="classification": raise ModelGuardError("class_prevalence requires classification labels","invalid_scenario")
            cls=spec.get("parameters",{}).get("target_class"); desired=float(spec.get("parameters",{}).get("proportion",spec["magnitude"]))
            if not 0.01<=desired<=0.99: raise ModelGuardError("Class prevalence proportion must be in [0.01, 0.99]","invalid_scenario")
            values=np.asarray(y); available=np.unique(values)
            match=[v for v in available if str(v)==str(cls)]
            if not match: raise ModelGuardError("target_class is not present in evaluation labels","invalid_scenario")
            cls=match[0]; positive=np.flatnonzero(values==cls); other=np.flatnonzero(values!=cls)
            if not len(positive) or not len(other): raise ModelGuardError("Class prevalence simulation needs target and non-target samples","invalid_scenario")
            npos=int(round(len(x)*desired)); idx=np.concatenate([rng.choice(positive,npos,replace=npos>len(positive)),rng.choice(other,len(x)-npos,replace=(len(x)-npos)>len(other))]); rng.shuffle(idx)
            shifted=x.iloc[idx].copy(); scenario_y=y.iloc[idx].copy(); changed=[]; prevalence={"target_class":str(cls),"baseline":float(np.mean(values==cls)),"scenario":float(np.mean(np.asarray(scenario_y)==cls))}
        else: shifted,changed=_apply(x,spec,rng)
        pred=pipe.predict(shifted); proba=pipe.predict_proba(shifted) if hasattr(pipe,"predict_proba") else None
        base_pred=np.asarray(art["predictions"]); keys=np.union1d(base_pred,pred); bp=np.array([np.mean(base_pred==k) for k in keys]); sp=np.array([np.mean(pred==k) for k in keys])
        item={"scenario":spec,"label":LABEL,"changed_features":changed,"class_prevalence_change":prevalence,"feature_distribution_changes":_distribution_change(x,shifted),"prediction_distribution_change":{"total_variation_distance":float(.5*np.abs(bp-sp).sum()),"paired_changed_prediction_fraction":None if spec["type"]=="class_prevalence" else float(np.mean(pred!=base_pred))}}
        if proba is not None: item["confidence_change"]={"baseline_mean_max_probability":float(np.mean(np.max(art["probabilities"],axis=1))),"scenario_mean_max_probability":float(np.mean(np.max(proba,axis=1)))}
        if payload["include_labels"]:
            metrics,_,_=evaluate(pipe,shifted,scenario_y,model["task"]); item["performance_metrics"]=metrics; item["metric_differences"]={k:(metrics[k]-v if isinstance(v,(int,float)) and isinstance(metrics.get(k),(int,float)) else None) for k,v in baseline.items()}
        else: item["performance_metrics"]=None; item["ground_truth_notice"]="Labels excluded: no accuracy, F1, or other ground-truth metric was calculated."
        results.append(item)
    result={"project_id":project_id,"model_id":model["id"],"label":LABEL,"results":results,"generated_at":now()}
    insert("analyses",{"id":uid(),"project_id":project_id,"kind":"datatwin","result_json":dump(result),"created_at":now()}); return result
