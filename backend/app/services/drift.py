from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance

from ..database import dump, insert, now, uid
from ..errors import ModelGuardError
from .data import configured, load_frame
from .modeling import evaluate, latest_model


def _psi(a,b,bins=10):
    edges=np.unique(np.nanquantile(a,np.linspace(0,1,bins+1)))
    if len(edges)<3:return 0.0
    edges[0],edges[-1]=-np.inf,np.inf; p=np.histogram(a.dropna(),edges)[0]/max(a.notna().sum(),1); q=np.histogram(b.dropna(),edges)[0]/max(b.notna().sum(),1)
    p=np.clip(p,1e-6,None); q=np.clip(q,1e-6,None); return float(np.sum((q-p)*np.log(q/p)))


def analyze(project_id: str, request: dict) -> dict:
    cfg,ref,_=configured(project_id); prod,prod_meta=load_frame(project_id,request.get("dataset_id")); target=cfg["target_column"]
    common=[c for c in ref.columns if c in prod.columns and c!=target]
    if not common: raise ModelGuardError("Reference and production data have no common feature columns","incompatible_dataset")
    findings=[]
    for c in common:
        r,p=ref[c],prod[c]; warning="Sample sizes below 30 reduce test reliability." if min(len(r),len(p))<30 else None
        if pd.api.types.is_numeric_dtype(r) and pd.api.types.is_numeric_dtype(p):
            ks=ks_2samp(r.dropna(),p.dropna()) if r.notna().any() and p.notna().any() else None; psi=_psi(r,p); wd=float(wasserstein_distance(r.dropna(),p.dropna())) if r.notna().any() and p.notna().any() else None; scale=float(r.std() or 1); practical=max(psi,0 if wd is None else wd/scale)
            stats={"psi":psi,"ks_statistic":None if ks is None else float(ks.statistic),"ks_p_value":None if ks is None else float(ks.pvalue),"standardized_wasserstein":None if wd is None else wd/scale,"mean_change":float(p.mean()-r.mean()),"variance_change":float(p.var()-r.var())}
            threshold=request["psi_threshold"]
        else:
            keys=r.astype(str).value_counts().index.union(p.astype(str).value_counts().index); rc=r.astype(str).value_counts().reindex(keys,fill_value=0); pc=p.astype(str).value_counts().reindex(keys,fill_value=0); rv=rc/rc.sum(); pv=pc/pc.sum(); js=float(jensenshannon(rv,pv,base=2)); chi=chi2_contingency(np.vstack([rc,pc])) if len(keys)>1 else None; practical=js; stats={"jensen_shannon_distance":js,"chi_square_p_value":None if chi is None else float(chi.pvalue),"max_category_proportion_change":float(np.max(np.abs(rv-pv)))}; threshold=request["js_threshold"]
        miss=float(p.isna().mean()-r.isna().mean()); practical=max(practical,abs(miss)); status="Needs Attention" if practical>=threshold else "Monitor" if practical>=threshold/2 else "Pass"
        findings.append({"feature":str(c),"reference_statistics":{"missing_fraction":float(r.isna().mean()),"sample_count":len(r)},"production_statistics":{"missing_fraction":float(p.isna().mean()),"sample_count":len(p)},"drift_magnitude":practical,"tests":stats,"statistical_significance":"Significant" if any(k.endswith("p_value") and v is not None and v<.05 for k,v in stats.items()) else "Not significant or unavailable","practical_significance":status,"configured_threshold":threshold,"sample_size_warning":warning,"model_sensitivity":"Not Evaluated","monitoring_recommendation":"Investigate persistence and performance impact." if status!="Pass" else "Continue monitoring."})
    findings.sort(key=lambda z:z["drift_magnitude"],reverse=True)
    model_perf=None; pred_change=None
    try:
        model,art=latest_model(project_id); xprod=prod.reindex(columns=art["feature_columns"]); pp=art["pipeline"].predict(xprod); base=np.asarray(art["predictions"]); pred_change={"reference_unique_predictions":len(np.unique(base)),"production_unique_predictions":len(np.unique(pp))}
        if request["include_labels"] and target in prod and prod[target].notna().all(): model_perf=evaluate(art["pipeline"],xprod,prod[target],model["task"])[0]
    except Exception: pass
    severe=sum(f["practical_significance"]=="Needs Attention" for f in findings); recommendation="Investigate and collect another batch; retraining is not justified from drift tests alone."
    if severe>=2 and model_perf is not None: recommendation="Compare observed performance to acceptance thresholds; retrain only if degradation is material and drift persists."
    result={"project_id":project_id,"production_dataset_id":prod_meta["id"],"features":findings,"prediction_distribution_changes":pred_change,"observed_performance":model_perf,"retraining_recommendation":recommendation,"decision_factors":{"practically_drifted_features":severe,"performance_labels_available":model_perf is not None,"persistence_across_batches":"Not Evaluated"},"generated_at":now()}
    insert("analyses",{"id":uid(),"project_id":project_id,"kind":"drift","result_json":dump(result),"created_at":now()}); return result
