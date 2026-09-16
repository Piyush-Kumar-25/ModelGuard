from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier

from ..database import dump, insert, now, uid
from ..errors import ModelGuardError
from .modeling import latest_model


def _leaf_rules(tree, names):
    t=tree.tree_; rules={}
    def walk(node, parts):
        if t.children_left[node]==t.children_right[node]: rules[node]=" AND ".join(parts) if parts else "all samples"; return
        n=names[t.feature[node]]; val=t.threshold[node]
        walk(t.children_left[node],parts+[f"{n} <= {val:.4g}"])
        walk(t.children_right[node],parts+[f"{n} > {val:.4g}"])
    walk(0,[]); return rules


def analyze(project_id: str, params: dict) -> dict:
    model,art=latest_model(project_id); pipe=art["pipeline"]; x=art["x_test"]; y=np.asarray(art["y_test"]); pred=np.asarray(art["predictions"]); task=model["task"]
    if len(x)<20: raise ModelGuardError("Evaluation set is too small for failure-region discovery","insufficient_evidence")
    if task=="classification": errors=(pred!=y).astype(int); metric_name="macro_f1"; global_metric=float(f1_score(y,pred,average="macro",zero_division=0)); higher=True
    else:
        residual=np.abs(pred-y); errors=(residual>np.quantile(residual,.75)).astype(int); metric_name="rmse"; global_metric=float(np.sqrt(mean_squared_error(y,pred))); higher=False
    transformed=pipe.named_steps["preprocessor"].transform(x)
    if hasattr(transformed,"toarray"): transformed=transformed.toarray()
    names=pipe.named_steps["preprocessor"].get_feature_names_out()
    required=max(params["min_samples"],int(np.ceil(params["min_support"]*len(x))))
    if required>=len(x): raise ModelGuardError("Minimum region support is too high for this evaluation set","insufficient_evidence")
    tree=DecisionTreeClassifier(max_depth=params["max_depth"],min_samples_leaf=required,class_weight="balanced",random_state=art["config"]["random_seed"]).fit(transformed,errors)
    leaves=tree.apply(transformed); rules=_leaf_rules(tree,names); regions=[]; rng=np.random.default_rng(art["config"]["random_seed"])
    for leaf in np.unique(leaves):
        mask=leaves==leaf; n=int(mask.sum()); support=n/len(x)
        if n<required: continue
        if task=="classification": local=float(f1_score(y[mask],pred[mask],average="macro",zero_division=0)); drop=global_metric-local
        else: local=float(np.sqrt(mean_squared_error(y[mask],pred[mask]))); drop=local-global_metric
        if drop<=0: continue
        boots=[]
        if params["bootstrap_iterations"] and n>=20:
            indices=np.flatnonzero(mask)
            for _ in range(params["bootstrap_iterations"]):
                b=rng.choice(indices,n,replace=True)
                try: boots.append(float(f1_score(y[b],pred[b],average="macro",zero_division=0)) if task=="classification" else float(np.sqrt(mean_squared_error(y[b],pred[b]))))
                except ValueError: pass
        ci=[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))] if boots else None
        vals,counts=np.unique(y[mask],return_counts=True)
        evidence="Strong" if n>=max(100,2*required) and ci else "Moderate" if n>=required and ci else "Insufficient Evidence"
        relative=drop/max(abs(global_metric),1e-12)
        status="Critical" if relative>=.3 and evidence!="Insufficient Evidence" else "Needs Attention" if relative>=.15 and evidence!="Insufficient Evidence" else "Monitor" if evidence!="Insufficient Evidence" else "Insufficient Evidence"
        regions.append({"rule":rules[int(leaf)],"sample_count":n,"support_percentage":round(100*support,2),"metric":metric_name,"global_metric":global_metric,"local_metric":local,"absolute_performance_drop":drop,"relative_performance_drop":relative,"local_error_rate":float(errors[mask].mean()),"class_distribution":{str(v):int(c) for v,c in zip(vals,counts)},"metric_95pct_bootstrap_ci":ci,"reliability_status":status,"evidence_strength":evidence})
    regions.sort(key=lambda r:(r["absolute_performance_drop"]*np.sqrt(r["sample_count"]),r["support_percentage"]),reverse=True)
    result={"project_id":project_id,"model_id":model["id"],"method":"shallow decision tree predicting held-out errors","minimum_samples":required,"global_metric":{"name":metric_name,"value":global_metric},"failure_regions":regions[:20],"limitations":["Tree rules are exploratory candidates, not causal explanations.","Bootstrap intervals condition on this held-out sample and discovered rule."],"generated_at":now()}
    insert("analyses",{"id":uid(),"project_id":project_id,"kind":"modelshield","result_json":dump(result),"created_at":now()}); return result

