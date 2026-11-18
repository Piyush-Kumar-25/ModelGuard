from __future__ import annotations
import copy, io, itertools, json, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,mean_squared_error,mean_absolute_error,average_precision_score,roc_auc_score,log_loss,r2_score,median_absolute_error
from threadpoolctl import threadpool_limits
from ..database import all_rows,connect,dump,insert,load,now,one,project_dir,uid
from ..errors import ModelGuardError,require
from .data import configured,load_frame,profile,sha,schema_fingerprint
from .inference import WeightedEnsemble
from .metrics import evaluate,classification_metrics,regression_metrics,utility,value,relative_drop
from .registry import registry,NAMES
from .training import fit_with_deadline
from .splitting import split_indices


def size_bytes(pipe):
    stream=io.BytesIO();joblib.dump(pipe,stream);return stream.tell()
def prediction_utility(y,p,task,key,classes,positive):
    y=np.asarray(y)
    try:
        if task=='regression':
            return {'rmse':lambda:-np.sqrt(mean_squared_error(y,p)),'mae':lambda:-mean_absolute_error(y,p),'r2':lambda:r2_score(y,p),'median_absolute_error':lambda:-median_absolute_error(y,p)}[key]()
        pred=classes[np.argmax(p,axis=1)]
        if key in {'macro_f1','weighted_f1','f1'}:return f1_score(y,pred,labels=classes,average='binary',pos_label=positive,zero_division=0) if key=='f1' and len(classes)==2 else f1_score(y,pred,labels=classes,average='weighted' if key=='weighted_f1' else 'macro',zero_division=0)
        if key=='accuracy':return float(np.mean(y==pred))
        if key=='log_loss':return -log_loss(y,p,labels=classes)
        if len(np.unique(y))!=len(classes):return -np.inf
        if key=='roc_auc':return roc_auc_score(y==positive,p[:,int(np.flatnonzero(classes==positive)[0])]) if len(classes)==2 else roc_auc_score(y,p,labels=classes,multi_class='ovr',average='macro')
        if key=='pr_auc':return average_precision_score(y==positive,p[:,int(np.flatnonzero(classes==positive)[0])]) if len(classes)==2 else average_precision_score((y[:,None]==classes).astype(int),p,average='macro')
    except ValueError:return -np.inf
    return -np.inf

def subset_search(predictions,y,cfg,classes,ranked):
    rng=np.random.default_rng(cfg['random_seed']);out=[];seen=set();pool=ranked[:5]
    for count in range(1,min(4,len(pool))+1):
        for subset in itertools.combinations(pool,count):
            stack=np.asarray([predictions[n] for n in subset]);proposals=[np.ones(count)/count]
            if count>1:proposals.extend(rng.dirichlet(np.ones(count),size=cfg['weight_trials']))
            best=None
            for weights in proposals:
                keep=weights>=.05;w=weights[keep];w=w/w.sum();names=tuple(np.asarray(subset)[keep]);p=np.tensordot(w,stack[keep],axes=(0,0));score=prediction_utility(y,p,cfg['task'],cfg['primary_metric'],classes,cfg.get('positive_class'))
                if best is None or score>best[0]:best=(score,names,w,p)
            score,names,w,p=best
            key=(names,tuple(np.round(w,6)))
            # Retain every requested subset even if pruning yields a duplicate configuration.
            out.append({'requested_members':list(subset),'members':list(names),'weights':w.tolist(),'utility':float(score) if np.isfinite(score) else None,'prediction':p,'duplicate_after_pruning':key in seen})
            seen.add(key)
    return out

def supported_masks(x):
    masks=[]
    for c in x.columns[:20]:
        if pd.api.types.is_numeric_dtype(x[c]):
            for op,q in [('low',.25),('high',.75)]:
                v=x[c].quantile(q);m=(x[c]<=v if op=='low' else x[c]>=v).to_numpy()
                if m.sum()>=15:masks.append((f'{c}:{op}',m))
        else:
            for cat,n in x[c].value_counts().head(5).items():
                if n>=15:masks.append((f'{c}={cat}',(x[c]==cat).to_numpy()))
    return masks

def meaningful_improvement(candidate,current,cfg):
    gain=candidate['utility']-current['utility'];tolerance=cfg['selection_tolerance']
    safe=(gain>=-tolerance and candidate['worst_subgroup_utility']>=current['worst_subgroup_utility']-tolerance and candidate['robustness_utility']>=current['robustness_utility']-tolerance)
    cal_ok=candidate['calibration'] is None or current['calibration'] is None or candidate['calibration']<=current['calibration']+tolerance
    cost_ok=candidate['size_bytes']<=max(20_000_000,4*current['size_bytes']) and candidate['inference_seconds']<=max(.25,4*current['inference_seconds'])
    extra=(candidate['calibration'] is not None and current['calibration'] is not None and current['calibration']-candidate['calibration']>=cfg['calibration_gain']) or candidate['worst_subgroup_utility']-current['worst_subgroup_utility']>=cfg['meaningful_gain'] or candidate['robustness_utility']-current['robustness_utility']>=cfg['meaningful_gain']
    return bool(safe and cal_ok and cost_ok and (gain>=cfg['meaningful_gain'] or extra))

def fit_search(x,y,xv,yv,cfg,weights=None,context=None,allowed=None,calibration_method='sigmoid'):
    if len(x)>100000 or x.shape[1]>500 or len(x)*x.shape[1]>5_000_000:raise ModelGuardError('Training exceeds bounded CPU/memory budget; use a reviewed smaller reference','resource_limit',413)
    from ..jobs import progress
    started=time.monotonic();models,records=registry(cfg['task'],cfg['random_seed'],x);fitted={};predictions={};meta={};latency={};sizes={};trials=0
    if allowed and not set(allowed)<=set(NAMES):raise ModelGuardError('Unknown candidate names')
    classes=np.asarray(sorted(pd.Series(y).unique())) if cfg['task']=='classification' else None
    if classes is not None and len(xv)*len(classes)*5>10_000_000:raise ModelGuardError('Validation probability matrix exceeds memory budget','resource_limit',413)
    with threadpool_limits(limits=1):
        for round_no in range(2):
            for name,original in models.items():
                if allowed and name not in allowed:
                    if round_no==0:records.append({'model':name,'status':'skipped','reason':'Excluded by explicit challenger candidate set'})
                    continue
                remaining=cfg['time_budget_seconds']-(time.monotonic()-started)
                if trials>=cfg['max_trials'] or remaining<=1:
                    if round_no==0:records.append({'model':name,'status':'skipped','reason':'Trial or search time budget exhausted'})
                    continue
                est=copy.deepcopy(original)
                if round_no:
                    params=est.get_params()
                    if 'max_depth' in params:est.set_params(max_depth=3)
                    elif 'alpha' in params:est.set_params(alpha=10.)
                    elif 'C' in params:est.set_params(C=.1)
                    elif 'max_leaf_nodes' in params:est.set_params(max_leaf_nodes=7)
                    elif 'depth' in params:est.set_params(depth=3)
                trials+=1
                progress("training:"+name,min(75,15+int(60*trials/cfg['max_trials'])))
                try:
                    fitted_pipe,info=fit_with_deadline((name,est,x,y,cfg,weights,context,calibration_method),min(cfg['fit_timeout_seconds'],remaining))
                    st=time.perf_counter();metrics,_,p=evaluate(fitted_pipe,xv,yv,cfg['task'],cfg.get('positive_class'));elapsed=time.perf_counter()-st;score=utility(metrics,cfg['primary_metric'])
                    if score is None:raise ModelGuardError('Primary validation metric unavailable','metric_unavailable')
                    pred=p if p is not None else fitted_pipe.predict(xv)
                    records.append({'model':name,'trial':round_no,'status':'completed','validation_metrics':metrics,'utility':score,'fit':info})
                    if name not in fitted or score>meta[name]['utility']:
                        fitted[name]=fitted_pipe;predictions[name]=pred;meta[name]={'utility':score,'fit':info};latency[name]=elapsed;sizes[name]=size_bytes(fitted_pipe)
                except Exception as exc:
                    records.append({'model':name,'trial':round_no,'status':'failed','reason':exc.message if isinstance(exc,ModelGuardError) else 'Candidate failed safely','code':getattr(exc,'code','candidate_failed')})
        if not fitted:raise ModelGuardError('No candidate completed; see structured candidate reasons','training_failed',422,records)
        ranked=sorted(fitted,key=lambda n:meta[n]['utility'],reverse=True);best_score=meta[ranked[0]]['utility'];pool=[];dominated=[]
        for name in ranked:
            redundant=any(np.mean(np.abs(predictions[name]-predictions[n]))<1e-6 for n in pool)
            if meta[name]['utility']<best_score-.15 or redundant:dominated.append({'model':name,'reason':'Clearly lower validation utility or redundant predictions'})
            elif len(pool)<5:pool.append(name)
        progress("ensemble_selection",80)
        candidates=subset_search(predictions,yv,cfg,classes,pool);masks=supported_masks(xv);stress=xv.copy();rng=np.random.default_rng(cfg['random_seed'])
        for c in stress:
            stress.loc[rng.random(len(stress))<.1,c]=np.nan
        stressed={n:(fitted[n].predict_proba(stress) if cfg['task']=='classification' else fitted[n].predict(stress)) for n in pool}
        detailed=[]
        for r in candidates:
            if r['utility'] is None:continue
            p=r.pop('prediction');names=r['members'];w=r['weights'];met=classification_metrics(yv,classes[p.argmax(axis=1)],p,classes,cfg.get('positive_class')) if classes is not None else regression_metrics(yv,p)
            group_scores=[prediction_utility(np.asarray(yv)[m],p[m],cfg['task'],cfg['primary_metric'],classes,cfg.get('positive_class')) for _,m in masks];group_scores=[v for v in group_scores if np.isfinite(v)]
            sp=np.tensordot(w,np.asarray([stressed[n] for n in names]),axes=(0,0));rob=prediction_utility(yv,sp,cfg['task'],cfg['primary_metric'],classes,cfg.get('positive_class'))
            r.update(validation_metrics=met,calibration=value(met.get('brier_score')),worst_subgroup_utility=min(group_scores) if group_scores else r['utility'],subgroup_evidence='supported predefined validation slices' if group_scores else 'insufficient support',robustness_utility=float(rob) if np.isfinite(rob) else -1e10,inference_seconds=sum(latency[n] for n in names),size_bytes=sum(sizes[n] for n in names),prediction_diversity=float(np.mean(np.std([predictions[n] for n in names],axis=0))))
            detailed.append(r)
        singles=[r for r in detailed if len(r['members'])==1];selected=max(singles,key=lambda r:r['utility'])
        for r in sorted(detailed,key=lambda r:(len(r['members']),-r['utility'])):
            if meaningful_improvement(r,selected,cfg):selected=r
        pipe=WeightedEnsemble([(n,fitted[n]) for n in selected['members']],selected['weights'],cfg['task'],classes.tolist() if classes is not None else None,None,cfg.get('positive_class'))
    return pipe,{'mode':'Auto','candidates':records,'dominated_candidates':dominated,'ensemble_comparisons':detailed,'selected':selected,'selection_reason':'Simplest qualifying configuration under primary, calibration, supported subgroup, robustness, size and latency guards. Weights use validation only.','weight_pruning_threshold':.05,'training_metadata':{n:meta[n] for n in selected['members']},'elapsed_seconds':time.monotonic()-started,'budget_exhausted':trials>=cfg['max_trials'] or time.monotonic()-started>=cfg['time_budget_seconds']}

def training_context(df,indices,cfg):
    c=pd.DataFrame({'source_row':indices})
    for field,key in [('time_column','time'),('group_column','group')]:
        if cfg.get(field):c[key]=df.loc[indices,cfg[field]].to_numpy()
    return c

def save_model(pid,pipe,cfg,manifest,details,metrics,status='incumbent',parent_id=None,training_snapshot=None):
    from .exporting import versions
    details['library_versions']=versions()
    mid=uid();folder=project_dir(pid)/'models'/mid;folder.mkdir(parents=True);path=folder/'pipeline.joblib';joblib.dump(pipe,path);path.chmod(0o600)
    state={'config':cfg,'manifest':manifest,'feature_columns':cfg['feature_columns']}
    if training_snapshot is not None:
        x,y,context=training_snapshot;frame=x.copy();target='__target_'+uid();frame[target]=np.asarray(y);context_cols={c:'__context_'+uid() for c in context}
        for c,d in context_cols.items():frame[d]=context[c].to_numpy()
        sp=folder/'training_reference.csv';frame.to_csv(sp,index=False);sp.chmod(0o600);state['snapshot']={'path':str(sp),'sha256':sha(sp),'target':target,'context_columns':context_cols}
    statepath=folder/'evaluation.json';statepath.write_text(dump(state));statepath.chmod(0o600)
    with connect() as con:
        con.execute('BEGIN IMMEDIATE');version=con.execute('SELECT COALESCE(MAX(version),0)+1 FROM models WHERE project_id=?',(pid,)).fetchone()[0]
        if status=='incumbent':
            if con.execute("SELECT id FROM models WHERE project_id=? AND dataset_id=? AND status!='legacy'",(pid,cfg['dataset_id'])).fetchone():raise ModelGuardError('Baseline already frozen for this dataset','baseline_frozen',409)
            con.execute("UPDATE models SET status='superseded' WHERE project_id=? AND status='incumbent'",(pid,))
        r=dict(id=mid,project_id=pid,dataset_id=cfg['dataset_id'],version=version,path=str(path),task=cfg['task'],selected_name='+'.join(n for n,_ in pipe.components),metrics_json=dump(metrics),details_json=dump(details),created_at=now(),status=status,parent_id=parent_id,checksum=sha(path),state_path=str(statepath))
        con.execute(f"INSERT INTO models ({','.join(r)}) VALUES ({','.join('?' for _ in r)})",list(r.values()))
    return model_public(r)

def train(pid):
    cfg,df,meta=configured(pid)
    if one("SELECT id FROM models WHERE project_id=? AND dataset_id=? AND status!='legacy'",(pid,meta['id'])):raise ModelGuardError('Baseline is frozen for this reference dataset','baseline_frozen',409)
    manifest=split_indices(df,cfg);x=df[cfg['feature_columns']];y=df[cfg['target_column']];a,b,c=[manifest[k] for k in ['train','validation','test']]
    context=training_context(df,a,cfg);pipe,details=fit_search(x.loc[a],y.loc[a],x.loc[b],y.loc[b],cfg,context=context)
    validation=evaluate(pipe,x.loc[b],y.loc[b],cfg['task'],cfg.get('positive_class'))[0];validated=[]
    for feature in cfg['optional_features']:
        fallback=x.loc[b].copy();fallback[feature]=np.nan;met=evaluate(pipe,fallback,y.loc[b],cfg['task'],cfg.get('positive_class'))[0];drop=relative_drop(validation,met,cfg['primary_metric'])
        if drop is not None and drop<=cfg['acceptance_drop']:validated.append(feature)
    cfg['joint_optional_fallback_validated']=False
    if validated:
        fallback=x.loc[b].copy();fallback[validated]=np.nan;met=evaluate(pipe,fallback,y.loc[b],cfg['task'],cfg.get('positive_class'))[0];drop=relative_drop(validation,met,cfg['primary_metric']);cfg['joint_optional_fallback_validated']=drop is not None and drop<=cfg['acceptance_drop']
    cfg['validated_optional_features']=validated;cfg['schema']=profile(x.loc[a]);cfg['schema_fingerprint']=schema_fingerprint(cfg['feature_columns'],cfg['schema'])
    from ..jobs import progress
    progress('locked_final_test',90)
    st=time.perf_counter();metrics=evaluate(pipe,x.loc[c],y.loc[c],cfg['task'],cfg.get('positive_class'))[0]
    details.update(evaluation={'manifest':manifest,'selection_partition':'validation','final_test_evaluated_after_lock':True,'test_used_for_selection':False},inference_seconds=time.perf_counter()-st,artifact_size_bytes=size_bytes(pipe),dataset_sha256=meta['sha256'],description='The best-performing baseline under the bounded validation selection rules.',limitations=['Finite held-out samples; no guarantee of future performance.','Selection uses a bounded search, not exhaustive optimization.'])
    r=save_model(pid,pipe,cfg,manifest,details,metrics);return {**r,'model_id':r['id'],'description':details['description']}

def model_public(r):return {'id':r['id'],'model_id':r['id'],'dataset_id':r['dataset_id'],'version':r['version'],'task':r['task'],'status':r['status'],'parent_id':r['parent_id'],'selected_model':r['selected_name'],'metrics':load(r['metrics_json']),'details':load(r['details_json']),'created_at':r['created_at']}
def get_model(pid,mid=None):
    r=one('SELECT * FROM models WHERE project_id=? AND id=?',(pid,mid)) if mid else one("SELECT * FROM models WHERE project_id=? AND status='incumbent' ORDER BY version DESC LIMIT 1",(pid,))
    r=require(r,'No compatible incumbent model found; legacy models require retraining')
    if r['status']=='legacy' or not r['checksum'] or not Path(r['path']).is_file() or sha(r['path'])!=r['checksum']:raise ModelGuardError('Model integrity check failed or legacy artifact requires retraining','artifact_invalid',409)
    return r

def load_model(pid,mid=None):
    r=get_model(pid,mid);state=load(Path(r['state_path']).read_text());cfg=state['config'];df,_=load_frame(pid,r['dataset_id']);pipe=joblib.load(r['path']);art={'pipeline':pipe,'config':cfg,'manifest':state['manifest'],'feature_columns':state['feature_columns']}
    for part in ['train','validation','test']:
        indices=state['manifest'][part];art['x_'+part]=df.loc[indices,state['feature_columns']].copy();art['y_'+part]=df.loc[indices,cfg['target_column']].copy()
    art['context']=training_context(df,state['manifest']['train'],cfg)
    if state.get('snapshot'):
        snap=state['snapshot']
        if sha(snap['path'])!=snap['sha256']:raise ModelGuardError('Private training snapshot integrity check failed','artifact_invalid',500)
        frame=pd.read_csv(snap['path']);art['x_train']=frame[state['feature_columns']];art['y_train']=frame[snap['target']];art['context']=frame[list(snap['context_columns'].values())].rename(columns={v:k for k,v in snap['context_columns'].items()})
    r['metrics']=load(r['metrics_json']);r['details']=load(r['details_json']);return r,art
latest_model=load_model

def model_results(pid):return {'models':[model_public(r) for r in all_rows('SELECT * FROM models WHERE project_id=? ORDER BY version DESC',(pid,))]}
