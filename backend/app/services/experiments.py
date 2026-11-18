import copy,time
import numpy as np
import pandas as pd
from ..database import connect,dump,insert,load,now,one,uid,update
from ..errors import ModelGuardError
from .data import load_frame,profile,schema_fingerprint
from .modeling import load_model,fit_search,save_model,model_public,size_bytes
from .metrics import evaluate,relative_drop,value
from .shield import analyses,region_mask,save_analysis
from . import twin

STRESS={'increased_missingness':'missingness','added_noise':'noise','distribution_shift':'numeric_shift','extreme_values':'extreme_values','input_corruption':'input_corruption','unavailable_features':'missingness'}

def parameter_check(kind,p):
    allowed={'feature_removal':{'features'},'subgroup_reweighting':{'region_id','factor'},'subgroup_oversampling':{'region_id','factor'},'class_weighting':{'weights'},'reduced_training_size':{'fraction'},'increased_class_imbalance':{'class','fraction'},'threshold_adjustment':{'threshold'},'probability_recalibration':{'method'},'alternative_model':{'models'},'alternative_ensemble':{'models'},'model_retraining':{'models'},'additional_labelled_data':{'dataset_id'},**{k:{'feature','features','magnitude','direction','units','distribution','fraction'} for k in STRESS}}
    if set(p)-allowed[kind]:raise ModelGuardError('Unsupported experiment parameters',details={'keys':sorted(set(p)-allowed[kind])})
    for key,low,high in [('factor',1,10),('fraction',.1,1),('threshold',.01,.99),('magnitude',0,10)]:
        if key in p and (not isinstance(p[key],(int,float)) or not low<=p[key]<=high):raise ModelGuardError('Invalid experiment '+key)

def subgroup_comparison(pid,parent,pipe,art,x,y):
    out=[]
    for a in analyses(pid,'modelshield',parent['id']):
        for r in a['failure_regions']:
            # Feature-removal challengers retain original inputs for interpreting fixed rules.
            mask=region_mask(x,r);n=int(mask.sum())
            if n<10:out.append({'region_id':r['region_id'],'sample_count':n,'status':'Insufficient Evidence'});continue
            ix=np.flatnonzero(mask);cfg=art['config'];before=evaluate(art['pipeline'],x.iloc[ix],np.asarray(y)[mask],cfg['task'],cfg.get('positive_class'))[0];after=evaluate(pipe,x.iloc[ix],np.asarray(y)[mask],cfg['task'],cfg.get('positive_class'))[0]
            out.append({'region_id':r['region_id'],'sample_count':n,'before':before,'after':after,'relative_drop':relative_drop(before,after,cfg['primary_metric'])})
    return out

def compare(pid,parent,art,pipe,partition):
    cfg=art['config'];x=art['x_'+partition];y=art['y_'+partition];st=time.perf_counter();before=evaluate(art['pipeline'],x,y,cfg['task'],cfg.get('positive_class'))[0];bt=time.perf_counter()-st;st=time.perf_counter();after=evaluate(pipe,x,y,cfg['task'],cfg.get('positive_class'))[0];at=time.perf_counter()-st;drop=relative_drop(before,after,cfg['primary_metric']);subgroups=subgroup_comparison(pid,parent,pipe,art,x,y);b=value(before.get('brier_score'));a=value(after.get('brier_score'));calibration_ok=b is None or a is None or a-b<=cfg['selection_tolerance'];subgroup_ok=all(r.get('relative_drop') is None or r['relative_drop']<=cfg['acceptance_drop'] for r in subgroups);passed=drop is not None and drop<=cfg['acceptance_drop'] and calibration_ok and subgroup_ok
    return {'partition':partition,'same_evaluation_records':True,'before':before,'after':after,'relative_primary_drop':drop,'subgroups':subgroups,'calibration':{'before':b,'after':a},'inference_seconds':{'before':bt,'after':at},'size_bytes':{'before':size_bytes(art['pipeline']),'after':size_bytes(pipe)},'evaluation_passed':bool(passed),'acceptance_criteria':{'maximum_relative_primary_drop':cfg['acceptance_drop'],'maximum_brier_increase':cfg['selection_tolerance'],'supported_subgroups_noninferior':subgroup_ok},'evidence_strength':'Finite-sample paired-record comparison; no causal guarantee','adoption_recommendation':'Eligible for explicit review after finalization' if passed else 'Reject or investigate the measured trade-offs'}

def run(pid,params):
    kind=params['experiment_type'];p=params.get('parameters',{});parameter_check(kind,p);parent,art=load_model(pid);cfg=copy.deepcopy(art['config']);cfg['random_seed']=params.get('random_seed',cfg['random_seed']);rng=np.random.default_rng(cfg['random_seed']);experiment_id=uid()
    if kind in STRESS:
        typ=STRESS[kind]
        if kind=='distribution_shift' and p.get('distribution'):typ='categorical_shift'
        scenario={'type':typ,'feature':p.get('feature'),'magnitude':1. if kind=='unavailable_features' else p.get('magnitude',.1),'parameters':{k:v for k,v in p.items() if k not in {'feature','magnitude'}}}
        result=twin.simulate(pid,{'scenarios':[scenario],'include_labels':True,'random_seed':cfg['random_seed'],'intensities':[1.]});result={**result,'category':'stress','intervention':kind,'incumbent_unchanged':True};save_analysis(pid,'experiment_stress',result)
    else:
        x=art['x_train'].copy().reset_index(drop=True);y=art['y_train'].copy().reset_index(drop=True);context=art['context'].copy().reset_index(drop=True);weights=None;allowed=None;method='sigmoid';pipe=None;notes=[]
        if kind=='feature_removal':
            features=p.get('features',[])
            if not features or not set(features)<=set(x) or len(set(features))==len(x.columns):raise ModelGuardError('Remove a nonempty proper subset of features')
            x=x.drop(columns=features);cfg['feature_columns']=list(x);cfg['optional_features']=[c for c in cfg['optional_features'] if c in x];cfg['validated_optional_features']=[]
        elif kind in {'subgroup_oversampling','subgroup_reweighting'}:
            regions=[r for a in analyses(pid,'modelshield',parent['id']) for r in a['failure_regions']];region=next((r for r in regions if r['region_id']==p.get('region_id')),None)
            if not region:raise ModelGuardError('Known ModelShield region_id required')
            mask=region_mask(x,region);factor=float(p.get('factor',2))
            if not mask.any():raise ModelGuardError('Region has no training support')
            if kind=='subgroup_reweighting':weights=np.where(mask,factor,1.);notes.append('Actual estimator and calibrator sample weights; no duplicated rows.')
            else:
                extra=rng.choice(np.flatnonzero(mask),int(mask.sum()*(factor-1)),replace=True);idx=np.r_[np.arange(len(x)),extra];x=x.iloc[idx].reset_index(drop=True);y=y.iloc[idx].reset_index(drop=True);context=context.iloc[idx].reset_index(drop=True);notes.append('Training replicas retain source-row lineage for calibration separation.')
        elif kind=='class_weighting':
            if cfg['task']!='classification':raise ModelGuardError('Class weighting requires classification')
            requested=p.get('weights');weights=np.asarray([float(requested.get(str(v),1)) for v in y]) if requested else np.asarray([len(y)/(y.nunique()*y.value_counts()[v]) for v in y])
            if not np.isfinite(weights).all() or np.any(weights<=0) or np.any(weights>100):raise ModelGuardError('Class weights must be finite in (0,100]')
        elif kind in {'reduced_training_size','increased_class_imbalance'}:
            frac=float(p.get('fraction',.7))
            if kind=='increased_class_imbalance':
                if cfg['task']!='classification' or p.get('class') not in set(y):raise ModelGuardError('Existing classification class required')
                ix=np.flatnonzero(y.to_numpy()==p['class']);keep=np.r_[np.flatnonzero(y.to_numpy()!=p['class']),rng.choice(ix,max(1,int(len(ix)*frac)),replace=False)]
            else:keep=np.sort(rng.choice(len(x),max(16,int(len(x)*frac)),replace=False))
            x=x.iloc[keep].reset_index(drop=True);y=y.iloc[keep].reset_index(drop=True);context=context.iloc[keep].reset_index(drop=True)
        elif kind=='threshold_adjustment':
            if cfg['task']!='classification' or len(art['pipeline'].classes_)!=2:raise ModelGuardError('Threshold requires binary probabilities')
            pipe=copy.deepcopy(art['pipeline']);pipe.threshold=float(p.get('threshold',.5))
        elif kind=='probability_recalibration':
            if cfg['task']!='classification':raise ModelGuardError('Recalibration requires classification')
            method=p.get('method','isotonic')
            if method not in {'sigmoid','isotonic'}:raise ModelGuardError('Unknown calibration method')
            allowed=[n for n,_ in art['pipeline'].components]
        elif kind in {'alternative_model','alternative_ensemble','model_retraining'}:
            allowed=p.get('models')
            if kind=='alternative_model' and (not isinstance(allowed,list) or len(allowed)!=1):raise ModelGuardError('Alternative model requires one model name')
            if kind=='alternative_ensemble' and (not isinstance(allowed,list) or not 2<=len(allowed)<=4):raise ModelGuardError('Alternative ensemble requires 2–4 allowed names')
        elif kind=='additional_labelled_data':
            if not p.get('dataset_id'):raise ModelGuardError('Additional dataset_id required')
            extra,meta=load_frame(pid,p['dataset_id'])
            if meta['role']=='production_labels':raise ModelGuardError('Use the corresponding production batch with attached labels')
            if meta['role']=='production_batch':
                from .drift import batch_frame
                extra,_,labels=batch_frame(pid,meta['id'])
                if labels is None:raise ModelGuardError('Confirm actual production labels first')
                extra[cfg['target_column']]=np.asarray(labels)
            if not set(cfg['feature_columns']+[cfg['target_column']])<=set(extra):raise ModelGuardError('Additional data lacks required labelled schema')
            original,_=load_frame(pid,cfg['dataset_id']);extra=extra.dropna(subset=[cfg['target_column']]);hashes=pd.util.hash_pandas_object(original[cfg['feature_columns']],index=False);extra=extra.loc[~pd.util.hash_pandas_object(extra[cfg['feature_columns']],index=False).isin(set(hashes))]
            for field in ['record_id_column','group_column']:
                c=cfg.get(field)
                if c:
                    if c not in extra:raise ModelGuardError('Additional data needs configured identity fields')
                    extra=extra.loc[~extra[c].isin(set(original[c]))]
            if cfg.get('time_column'):
                c=cfg['time_column']
                if c not in extra:raise ModelGuardError('Additional data needs configured time')
                boundary=pd.to_datetime(original.loc[art['manifest']['validation']+art['manifest']['test'],c],utc=True).min();timestamps=pd.to_datetime(extra[c],utc=True,errors='coerce');extra=extra.loc[timestamps<boundary];notes.append('Recent rows after the frozen validation boundary are excluded to preserve temporal evaluation; use a new reference version for forward validation.')
            extra=extra.drop_duplicates(subset=cfg['feature_columns']+[cfg['target_column']])
            if extra.empty:raise ModelGuardError('No independent additional labelled rows remain')
            from .drift import validate_labels,schema_check
            validate_labels(extra[cfg['target_column']],art)
            if not schema_check(extra,cfg)['prediction_allowed']:raise ModelGuardError('Additional input schema incompatible')
            cx=pd.DataFrame({'source_row':['additional:'+uid() for _ in range(len(extra))]})
            for field,key in [('time_column','time'),('group_column','group')]:
                if cfg.get(field):cx[key]=extra[cfg[field]].to_numpy()
            x=pd.concat([x,extra[cfg['feature_columns']]],ignore_index=True);y=pd.concat([y,extra[cfg['target_column']]],ignore_index=True);context=pd.concat([context,cx],ignore_index=True)
        if pipe is None:pipe,search=fit_search(x,y,art['x_validation'][cfg['feature_columns']],art['y_validation'],cfg,weights,context,allowed,method)
        else:search={'selected':{'members':[n for n,_ in pipe.components],'weights':pipe.weights},'selection_reason':'Explicit binary threshold evaluated on validation only'}
        cfg['schema']=profile(x);cfg['schema_fingerprint']=schema_fingerprint(cfg['feature_columns'],cfg['schema']);comparison=compare(pid,parent,art,pipe,'validation');details={'intervention':kind,'parameters':p,'notes':notes,'search':search,'validation_comparison':comparison,'finalized':False,'automatic_promotion':False,'evaluation':{'manifest':art['manifest'],'selection_partition':'validation','test_used_for_selection':False,'final_test_evaluated_after_lock':False}}
        result=save_model(pid,pipe,cfg,art['manifest'],details,{},'experimental',parent['id'],(x,y,context));result.update(category='challenger',incumbent_id=parent['id'],comparison=comparison)
    insert('experiments',dict(id=experiment_id,project_id=pid,experiment_type=kind,params_json=dump(p),result_json=dump(result),random_seed=cfg['random_seed'],created_at=now()));return {**result,'experiment_id':experiment_id}

def lifecycle(pid,mid,action):
    model,art=load_model(pid,mid)
    if not model['parent_id']:raise ModelGuardError('Only challengers have this lifecycle')
    if action=='finalize':
        if model['status']!='experimental':raise ModelGuardError('Only experimental challengers may be finalized','invalid_transition',409)
        # Reserve the one final-test comparison for a reference dataset before reading test metrics.
        with connect() as con:
            con.execute('BEGIN IMMEDIATE');siblings=con.execute('SELECT id,details_json FROM models WHERE project_id=? AND dataset_id=? AND parent_id IS NOT NULL',(pid,model['dataset_id'])).fetchall()
            if any(load(r['details_json']).get('final_test_reserved') for r in siblings):raise ModelGuardError('Final-test comparison already consumed for this reference; use a new reference dataset','test_budget_consumed',409)
            details=load(con.execute('SELECT details_json FROM models WHERE id=?',(mid,)).fetchone()[0]);details['final_test_reserved']=True;con.execute('UPDATE models SET details_json=? WHERE id=?',(dump(details),mid))
        parent,pa=load_model(pid,model['parent_id']);comparison=compare(pid,parent,pa,art['pipeline'],'test');details.update(finalized=True,final_comparison=comparison);details['evaluation']['final_test_evaluated_after_lock']=True;update('UPDATE models SET details_json=?,metrics_json=? WHERE id=?',(dump(details),dump(comparison['after']),mid))
    else:
        with connect() as con:
            con.execute('BEGIN IMMEDIATE');current=dict(con.execute('SELECT * FROM models WHERE project_id=? AND id=?',(pid,mid)).fetchone());details=load(current['details_json'])
            if action=='reject':
                if current['status'] not in {'experimental','approved_candidate'}:raise ModelGuardError('Cannot reject an active or terminal model','invalid_transition',409)
                status='rejected'
            elif action=='approve':
                if current['status']!='experimental' or not details.get('finalized') or not details['final_comparison']['evaluation_passed']:raise ModelGuardError('Approval requires a passing finalized comparison','evaluation_required',409)
                status='approved_candidate'
            elif action=='promote':
                if current['status']!='approved_candidate':raise ModelGuardError('Explicit approval required before promotion','approval_required',409)
                active=con.execute("SELECT id FROM models WHERE project_id=? AND status='incumbent'",(pid,)).fetchone()
                if not active or active[0]!=current['parent_id']:raise ModelGuardError('Incumbent changed; evaluate against the active version','stale_challenger',409)
                con.execute("UPDATE models SET status='superseded' WHERE project_id=? AND status='incumbent'",(pid,));status='incumbent';details['lifecycle']='promoted'
            else:raise ModelGuardError('Unknown lifecycle action')
            con.execute('UPDATE models SET status=?,details_json=? WHERE id=?',(status,dump(details),mid))
    return model_public(one('SELECT * FROM models WHERE id=?',(mid,)))
