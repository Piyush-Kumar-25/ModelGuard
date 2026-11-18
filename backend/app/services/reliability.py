from ..database import all_rows,load,now
from .data import configured,get_project,load_frame,summarize
from .modeling import load_model,model_results
from .shield import analyses
from .drift import history

DIMENSIONS=['Data Quality','Leakage Safety','Evaluation Integrity','Subgroup Robustness','Noise and Missingness Robustness','Distribution-Shift Robustness','Production Stability']
RANK={'Critical':6,'Needs Attention':5,'Insufficient Evidence':4,'Not Evaluated':3,'Monitor':2,'Pass':1}

def worst(statuses):return max(statuses,key=lambda s:RANK[s]) if statuses else 'Not Evaluated'
def summarize_dimensions(dimensions):
    evaluated=sum(d['status'] not in {'Not Evaluated','Insufficient Evidence'} for d in dimensions);status=worst([d['status'] for d in dimensions]);return {'status':status,'evidence_coverage':{'evaluated':evaluated,'applicable':len(dimensions),'fraction':evaluated/len(dimensions) if dimensions else 0},'decision_rule':'Critical dominates; then Needs Attention, Insufficient Evidence, Not Evaluated, Monitor, Pass. No averaging of critical evidence.'}

def assessment(pid,include_indicator=False,model_id=None):
    cfg,_,meta=configured(pid);model=None;art=None
    try:model,art=load_model(pid,model_id);cfg=art['config']
    except Exception as exc:
        from ..errors import ModelGuardError
        if not isinstance(exc,ModelGuardError) or exc.code!='not_found':raise
    mid=model['id'] if model else None;audits=[a for a in analyses(pid,'audit') if a.get('dataset_id')==cfg['dataset_id']];audit=audits[-1] if audits else None;dimensions=[]
    add=lambda name,status,evidence,reason:dimensions.append({'dimension':name,'status':status,'evidence':evidence,'decision_rule':reason})
    quality=[f for f in audit['findings'] if f['check_name'] not in {'Target leakage and post-outcome candidates','Prediction-time availability','Group contamination','Chronological leakage'}] if audit else []
    add(DIMENSIONS[0],worst([f['status'] for f in quality]),quality,'Measured DataDoctor thresholds; heuristic findings are explicitly marked.')
    leaks=[f for f in audit['findings'] if f['check_name']=='Target leakage and post-outcome candidates'] if audit else [];excluded=set(cfg['excluded_features']+cfg['confirmed_leakage']);unresolved=[c for f in leaks for c in f['affected_features'] if c not in excluded]
    leakstatus='Needs Attention' if unresolved else 'Pass' if audit and cfg.get('feature_availability_confirmed') else 'Insufficient Evidence' if audit else 'Not Evaluated'
    add(DIMENSIONS[1],leakstatus,{'unresolved_candidates':unresolved,'excluded_features':sorted(excluded),'prediction_time_availability_confirmed':cfg.get('feature_availability_confirmed',False)},'Heuristic screening cannot prove leakage safety; availability confirmation and resolved exclusions are required.')
    evaluation=model['details'].get('evaluation',{}) if model else {};manifest=evaluation.get('manifest',{});sets=[set(manifest.get(k,[])) for k in ['train','validation','test']];disjoint=all(sets) and not any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3));valid=bool(disjoint and evaluation.get('final_test_evaluated_after_lock') and evaluation.get('test_used_for_selection') is False)
    add(DIMENSIONS[2],'Pass' if valid else 'Insufficient Evidence' if model else 'Not Evaluated',evaluation,'Requires persisted disjoint three-way manifest and locked test evaluation, not mere model existence.')
    shields=analyses(pid,'modelshield',mid) if mid else [];regions=[r for a in shields for r in a['failure_regions']];add(DIMENSIONS[3],worst([r['status'] for r in regions]) if regions else 'Insufficient Evidence' if shields else 'Not Evaluated',regions,'Only independently confirmed supported regions may be Critical; discovery alone is insufficient.')
    scenarios=[r for a in analyses(pid,'datatwin',mid) for r in a['results']] if mid else []
    for name,kinds in [(DIMENSIONS[4],{'missingness','noise','input_corruption','extreme_values'}),(DIMENSIONS[5],{'numeric_shift','numeric_scaling','categorical_shift','class_prevalence','combined'})]:
        selected=[r for r in scenarios if r['scenario_definition']['type'] in kinds];add(name,worst([r['status'] for r in selected]),selected,'Retained-label controlled effects versus configured acceptance/critical thresholds; unlabelled sensitivity is insufficient performance evidence.')
    production=history(pid,mid)['batches'] if mid else []
    latest=production[-1] if production else None
    production_status=('Insufficient Evidence' if latest.get('analysis_stale') else latest['status']) if latest else 'Not Evaluated'
    add(DIMENSIONS[6],production_status,latest,'Latest distinct production batch. Reanalyze after label or mapping changes; no batch means Not Evaluated.')
    pre=summarize_dimensions(dimensions[:6]);overall=summarize_dimensions(dimensions);result={'project_id':pid,'model_id':mid,'dimensions':dimensions,'pre_deployment_readiness':pre,'production_stability':dimensions[-1],'overall_lifecycle_status':overall,'evidence_coverage':overall['evidence_coverage'],'numerical_indicator_enabled':False}
    if include_indicator:result['indicator_note']='Disabled: no authoritative numerical reliability score is provided.'
    return result

def report(pid,include_indicator=False,model_id=None):
    rel=assessment(pid,include_indicator,model_id);model,art=load_model(pid,rel['model_id']) if rel['model_id'] else (None,None);cfg=art['config'] if art else configured(pid)[0];_,meta=load_frame(pid,cfg['dataset_id']);mid=rel['model_id'];audit=[a for a in analyses(pid,'audit') if a.get('dataset_id')==cfg['dataset_id']];recommendations={'Immediate':[],'Before deployment':[],'Monitoring':[],'Future improvement':[]}
    experiments=[]
    if mid:
        for row in all_rows('SELECT * FROM experiments WHERE project_id=? ORDER BY created_at',(pid,)):
            result=load(row['result_json'])
            if mid in (result.get('model_id'),result.get('id'),result.get('incumbent_id')):
                experiments.append({'id':row['id'],'type':row['experiment_type'],'result':result})
    for d in rel['dimensions']:
        if d['status'] in {'Critical','Needs Attention'}:recommendations['Immediate'].append({'dimension':d['dimension'],'recommendation':'Investigate and address the measured findings before approval.','evidence_type':'recommendation'})
        elif d['status'] in {'Not Evaluated','Insufficient Evidence'}:recommendations['Before deployment' if d['dimension']!='Production Stability' else 'Monitoring'].append({'dimension':d['dimension'],'recommendation':'Collect and monitor future production batches.' if d['dimension']=='Production Stability' else 'Collect sufficient evidence for '+d['dimension']+'.','evidence_type':'recommendation'})
    recommendations['Monitoring'].append({'recommendation':'Review confirmed labels and persistent drift; challenger evaluation never automatically replaces the incumbent.','evidence_type':'recommendation'})
    recommendations['Future improvement'].append({'recommendation':'Validate on independent future data and domain-specific costs.','evidence_type':'recommendation'})
    return {'project_id':pid,'model_id':mid,'executive_summary':{'pre_deployment_status':rel['pre_deployment_readiness']['status'],'overall_status':rel['overall_lifecycle_status']['status'],'evidence_coverage':rel['evidence_coverage']},'dataset_profile':summarize(pid,cfg['dataset_id']),'target_and_task':{k:cfg.get(k) for k in ['target_column','task','positive_class','positive_class_source','primary_metric']},'data_doctor':audit,'exclusions_and_leakage_decisions':{k:cfg.get(k) for k in ['excluded_features','confirmed_leakage','feature_availability_confirmed']},'split_evaluation_strategy':model['details'].get('evaluation') if model else None,'model_selection':model['details'] if model else None,'selected_incumbent':{'id':mid,'version':model['version'],'status':model['status'],'name':model['selected_name']} if model else None,'global_metrics':model['metrics'] if model else None,'modelshield':analyses(pid,'modelshield',mid) if mid else [],'datatwin':analyses(pid,'datatwin',mid) if mid else [],'experimentlab':experiments,'driftwatch':history(pid,mid) if mid else {'batches':[]},'reliability':rel,'prioritized_recommendations':recommendations,'evidence_taxonomy':{'observed_fact':'Counts, schemas and immutable hashes','statistical_finding':'Metrics and bootstrap intervals','heuristic_warning':'Semantic leakage and unit-like warnings','controlled_simulation':'DataTwin and stress interventions','prediction':'Frozen-model outputs, not actual outcomes','recommendation':'Human-review action derived from evidence','unavailable_evidence':'Explicit Not Evaluated or Insufficient Evidence'},'reproducibility':{'random_seed':cfg['random_seed'],'dataset_id':cfg['dataset_id'],'dataset_sha256':meta['sha256'],'model_version':model['version'] if model else None,'schema_version':1,'schema_fingerprint':cfg.get('schema_fingerprint')},'limitations':['No causal or universal deployment-readiness guarantees.','Unlabelled drift cannot confirm accuracy degradation.','One finalized challenger comparison per reference dataset protects against repeated test-set optimization.'],'generated_at':now()}
