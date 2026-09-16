from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from ..database import dump, insert, now, uid
from ..errors import ModelGuardError
from .modeling import build_preprocessor, evaluate, latest_model


def _subgroup(pipe,x,y,task,p):
    feature=p.get("subgroup_feature"); value=p.get("subgroup_value")
    if feature not in x or value is None: return None
    mask=x[feature].astype(str)==str(value)
    if mask.sum()<10: return {"status":"Insufficient Evidence","sample_count":int(mask.sum())}
    metrics,_,_=evaluate(pipe,x.loc[mask],y.loc[mask],task)
    return {"status":"Evaluated","sample_count":int(mask.sum()),"feature":feature,"value":value,"metrics":metrics}


def run(project_id: str, request: dict) -> dict:
    model,art=latest_model(project_id); typ=request["experiment_type"]; p=request["parameters"]; seed=request["random_seed"]; rng=np.random.default_rng(seed)
    xtr=art["x_train"].copy(); ytr=art["y_train"].copy(); xte=art["x_test"].copy(); yte=art["y_test"].copy(); base_pipe=art["pipeline"]; task=model["task"]
    before_sub=_subgroup(base_pipe,xte,yte,task,p); resulting_pipe=base_pipe; notes=[]
    if typ=="feature_removal":
        features=p.get("features",[])
        if not features or any(f not in xtr for f in features): raise ModelGuardError("Provide valid features for removal","invalid_experiment")
        xtr=xtr.drop(columns=features); xte=xte.drop(columns=features)
    elif typ in {"increased_missingness","added_noise","distribution_shift"}:
        feature=p.get("feature"); magnitude=float(p.get("magnitude",.1))
        if feature not in xte or not 0<=magnitude<=5: raise ModelGuardError("Provide a valid feature and magnitude","invalid_experiment")
        if typ=="increased_missingness": xte.loc[rng.random(len(xte))<min(magnitude,1),feature]=np.nan
        elif not pd.api.types.is_numeric_dtype(xte[feature]): raise ModelGuardError("Noise/shift requires a numeric feature","invalid_experiment")
        elif typ=="added_noise": xte[feature]+=rng.normal(0,magnitude*(xte[feature].std() or 1),len(xte))
        else: xte[feature]+=magnitude*(xte[feature].std() or 1)
    elif typ=="reduced_training_size":
        fraction=float(p.get("fraction",.5))
        if not .1<=fraction<1: raise ModelGuardError("fraction must be in [0.1, 1)","invalid_experiment")
        idx=rng.choice(len(xtr),max(10,int(len(xtr)*fraction)),replace=False); xtr=xtr.iloc[idx]; ytr=ytr.iloc[idx]
    elif typ in {"subgroup_oversampling","subgroup_reweighting"}:
        f=p.get("subgroup_feature"); v=p.get("subgroup_value"); factor=int(p.get("factor",2))
        if f not in xtr or not 2<=factor<=10: raise ModelGuardError("Provide subgroup_feature, subgroup_value and factor 2-10","invalid_experiment")
        mask=xtr[f].astype(str)==str(v); extra=xtr.loc[mask]; ey=ytr.loc[mask]
        if len(extra)<2: raise ModelGuardError("Subgroup has insufficient training rows","insufficient_evidence")
        xtr=pd.concat([xtr]+[extra]*(factor-1)); ytr=pd.concat([ytr]+[ey]*(factor-1)); notes.append("Reweighting implemented as deterministic subgroup oversampling.")
    elif typ=="increased_class_imbalance":
        if task!="classification": raise ModelGuardError("Class imbalance experiment requires classification","invalid_experiment")
        minority=ytr.value_counts().idxmin(); fraction=float(p.get("minority_retention",.5)); keep=(ytr!=minority)|pd.Series(rng.random(len(ytr))<fraction,index=ytr.index); xtr,ytr=xtr.loc[keep],ytr.loc[keep]
    elif typ=="threshold_adjustment":
        if task!="classification" or art["probabilities"] is None or len(base_pipe.classes_)!=2: raise ModelGuardError("Threshold adjustment requires probabilistic binary classification","invalid_experiment")
        threshold=float(p.get("threshold",.5))
        if not .01<=threshold<=.99: raise ModelGuardError("threshold must be in [0.01, 0.99]","invalid_experiment")
        pred=np.where(art["probabilities"][:,1]>=threshold,base_pipe.classes_[1],base_pipe.classes_[0])
        from .modeling import classification_metrics
        after=classification_metrics(yte,pred,art["probabilities"],base_pipe.classes_); after_sub=None
        result=_result(project_id,model,request,model["metrics"],after,before_sub,after_sub,notes); return result
    elif typ not in {"model_retraining"}:
        raise ModelGuardError("Experiment type is not implemented","invalid_experiment")
    needs_retrain=typ not in {"increased_missingness","added_noise","distribution_shift"}
    if needs_retrain:
        estimator=clone(base_pipe.named_steps["model"]); resulting_pipe=Pipeline([("preprocessor",build_preprocessor(xtr)),("model",estimator)]).fit(xtr,ytr)
    after,_,_=evaluate(resulting_pipe,xte,yte,task); after_sub=_subgroup(resulting_pipe,xte,yte,task,p)
    return _result(project_id,model,request,model["metrics"],after,before_sub,after_sub,notes)


def _result(project_id,model,request,before,after,before_sub,after_sub,notes):
    diffs={k:(after[k]-v if isinstance(v,(int,float)) and isinstance(after.get(k),(int,float)) else None) for k,v in before.items()}
    primary=model["details"]["evaluation"]["selection_metric"]; d=diffs.get(primary); recommendation="Review the global and subgroup trade-off before adopting this intervention."
    result={"experiment_id":uid(),"experiment_type":request["experiment_type"],"parameters":request["parameters"],"dataset_version":model["dataset_id"],"model_version":model["version"],"baseline_metrics":before,"resulting_metrics":after,"metric_differences":diffs,"subgroup_performance":{"before":before_sub,"after":after_sub},"global_performance":{"before":before,"after":after},"random_seed":request["random_seed"],"timestamp":now(),"evidence_based_recommendation":recommendation,"notes":notes}
    insert("experiments",{"id":result["experiment_id"],"project_id":project_id,"experiment_type":request["experiment_type"],"params_json":dump(request["parameters"]),"result_json":dump(result),"random_seed":request["random_seed"],"created_at":result["timestamp"]}); return result
