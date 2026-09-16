from __future__ import annotations

from typing import Any

from ..database import all_rows, load, one
from .data import get_project, summarize
from .modeling import model_results

STATUSES=["Not Evaluated","Insufficient Evidence","Pass","Monitor","Needs Attention","Critical"]


def _latest(project_id,kind):
    row=one("SELECT result_json FROM analyses WHERE project_id=? AND kind=? ORDER BY created_at DESC LIMIT 1",(project_id,kind))
    return load(row["result_json"]) if row else None


def _dim(status,evidence,thresholds,findings,coverage,limitations,recommendation):
    return {"status":status,"measured_evidence":evidence,"thresholds_used":thresholds,"important_findings":findings,"evidence_coverage":coverage,"uncertainty_or_sample_limitations":limitations,"recommendation":recommendation}


def assessment(project_id: str, include_indicator=False) -> dict[str,Any]:
    get_project(project_id); audit=_latest(project_id,"audit"); shield=_latest(project_id,"modelshield"); twin=_latest(project_id,"datatwin"); drift=_latest(project_id,"drift"); models=model_results(project_id)["models"]; model=models[0] if models else None
    dims={}
    if audit:
        dataf=[f for f in audit["findings"] if "leakage" not in f["check_name"].lower() and "post-outcome" not in f["check_name"].lower()]
        st=max((f["status"] for f in dataf),key=lambda s:STATUSES.index(s)); dims["Data Quality"]=_dim(st,dataf,{},[f["check_name"] for f in dataf if f["status"]!="Pass"],"Core automated checks completed",[f["limitations"] for f in dataf],"Address high-severity findings before relying on the model.")
        leak=next((f for f in audit["findings"] if "leakage" in f["check_name"].lower()),None); dims["Leakage Safety"]=_dim(leak["status"],leak,leak["threshold_used"],leak["affected_features"],"Heuristic screen completed",[leak["limitations"]],leak["recommendation"])
    else:
        dims["Data Quality"]=dims["Leakage Safety"]=_dim("Not Evaluated",None,{},[],"None",["Run DataDoctor audit."],"Run the audit.")
    if model: dims["Evaluation Integrity"]=_dim("Pass",model["details"]["evaluation"],{},[model["selected_model"]],"Held-out test and cross-validation completed",model["details"]["limitations"],"Revalidate on external or later data before production use.")
    else: dims["Evaluation Integrity"]=_dim("Not Evaluated",None,{},[],"None",["No model trained."],"Train baseline candidates.")
    if shield:
        regs=shield["failure_regions"]; st=max((r["reliability_status"] for r in regs),key=lambda s:STATUSES.index(s)) if regs else "Pass"; dims["Subgroup Robustness"]=_dim(st,{"regions":regs[:5]},{"minimum_samples":shield["minimum_samples"]},[r["rule"] for r in regs[:5]],"Held-out failure-tree analysis",shield["limitations"],"Investigate supported regions and test remediation without hiding global trade-offs.")
    else: dims["Subgroup Robustness"]=_dim("Not Evaluated",None,{},[],"None",["ModelShield not run."],"Run failure-region discovery.")
    if twin:
        missing=[r for r in twin["results"] if r["scenario"]["type"] in {"missingness","noise","extreme_values","combined"}]; shifts=[r for r in twin["results"] if r["scenario"]["type"] in {"numeric_shift","categorical_shift","combined"}]
        dims["Noise and Missingness Robustness"]=_dim("Monitor" if missing else "Not Evaluated",missing,{},[r["scenario"] for r in missing],f"{len(missing)} controlled scenarios",[twin["label"]],"Compare metric changes to domain acceptance thresholds.")
        dims["Distribution-Shift Robustness"]=_dim("Monitor" if shifts else "Not Evaluated",shifts,{},[r["scenario"] for r in shifts],f"{len(shifts)} controlled scenarios",[twin["label"]],"Expand scenarios using plausible domain shifts.")
    else:
        dims["Noise and Missingness Robustness"]=_dim("Not Evaluated",None,{},[],"None",["No scenarios run."],"Run missingness/noise scenarios.")
        dims["Distribution-Shift Robustness"]=_dim("Not Evaluated",None,{},[],"None",["No scenarios run."],"Run distribution-shift scenarios.")
    if drift:
        statuses=[f["practical_significance"] for f in drift["features"]]; st="Needs Attention" if "Needs Attention" in statuses else "Monitor" if "Monitor" in statuses else "Pass"; dims["Production Stability"]=_dim(st,drift,{"per_feature":"configured in finding"},[f["feature"] for f in drift["features"][:5]],"One production batch",["Persistence requires multiple batches."],drift["retraining_recommendation"])
    else: dims["Production Stability"]=_dim("Not Evaluated",None,{},[],"None",["No production batch analyzed."],"Analyze a production batch.")
    evaluated=[d for d in dims.values() if d["status"] not in {"Not Evaluated","Insufficient Evidence"}]
    if any(d["status"]=="Critical" for d in dims.values()): overall="Critical"
    elif any(d["status"]=="Needs Attention" for d in dims.values()): overall="Needs Attention"
    elif len(evaluated)<len(dims): overall="Insufficient Evidence"
    elif any(d["status"]=="Monitor" for d in dims.values()): overall="Monitor"
    else: overall="Pass"
    result={"project_id":project_id,"dimensions":dims,"evidence_coverage":{"evaluated":len(evaluated),"total":len(dims)},"overall_reliability_status":overall,"decision_rule":"Critical evidence dominates; then Needs Attention; missing dimensions produce Insufficient Evidence; no averaging is used."}
    if include_indicator:
        weights={k:1/7 for k in dims}; scores={"Pass":100,"Monitor":70,"Needs Attention":40,"Critical":0}; included=[k for k,v in dims.items() if v["status"] in scores]; value=round(sum(scores[dims[k]["status"]]*weights[k] for k in included)/sum(weights[k] for k in included)) if included else None
        result["provisional_indicator"]={"value":value,"scale":100,"weights":weights,"included_dimensions":included,"calculation":"weighted mean over evaluated dimensions only; unevaluated dimensions receive no favourable value","evidence_coverage":f"{len(included)} of 7 dimensions","confidence":"Moderate" if len(included)>=5 else "Low","status":overall,"note":"Provisional heuristic; not a validated probability that the model is reliable."}
    return result


def report(project_id: str, include_indicator=False):
    project=get_project(project_id); audit=_latest(project_id,"audit"); shield=_latest(project_id,"modelshield"); twin=_latest(project_id,"datatwin"); drift=_latest(project_id,"drift"); models=model_results(project_id)["models"]; experiments=[load(r["result_json"]) for r in all_rows("SELECT result_json FROM experiments WHERE project_id=? ORDER BY created_at DESC",(project_id,))]
    reliability=assessment(project_id,include_indicator)
    profile=summarize(project_id,audit["dataset_id"]) if audit else None
    return {"project":{"id":project["id"],"name":project["name"]},"dataset_profile":profile,"data_doctor":audit,"preprocessing_and_evaluation":None if not models else models[0]["details"]["evaluation"],"candidate_model_results":None if not models else models[0]["details"]["candidates"],"selected_baseline":None if not models else models[0],"model_limitations":[] if not models else models[0]["details"]["limitations"],"failure_regions":None if not shield else shield["failure_regions"],"controlled_scenarios":twin,"remediation_and_stress_experiments":experiments,"drift_watch":drift,"reliability_assessment":reliability,"recommendations":[v["recommendation"] for v in reliability["dimensions"].values()],"known_limitations":["Automated checks cannot establish causality or feature availability at prediction time.","Results depend on representativeness of uploaded data and configured thresholds."],"evidence_labels":{"observed_facts":"dataset profile and counts","statistical_findings":"tests and measured metrics","heuristic_warnings":"explicitly marked in DataDoctor","controlled_simulations":"DataTwin only","recommendations":"decision support, not automatic production action","unavailable_evidence":"Not Evaluated dimensions"}}
