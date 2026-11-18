import math
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from ..database import all_rows,dump,insert,load,now,uid
from ..errors import ModelGuardError
from .modeling import load_model
from .metrics import evaluate,value


def save_analysis(pid,kind,result):
    result.update(project_id=pid,generated_at=now());insert('analyses',dict(id=uid(),project_id=pid,kind=kind,result_json=dump(result),created_at=now()));return result

def analyses(pid,kind,mid=None):
    values=[load(r['result_json']) for r in all_rows('SELECT result_json FROM analyses WHERE project_id=? AND kind=? ORDER BY created_at',(pid,kind))]
    return [r for r in values if mid is None or r.get('model_id')==mid]

def region_mask(x,region):
    mask=np.ones(len(x),dtype=bool)
    for c in region['clauses']:
        if c['kind']=='numeric':vals=pd.to_numeric(x[c['feature']],errors='coerce').fillna(c['fill']).to_numpy();test=vals<=c['value']
        else:test=(x[c['feature']].fillna('__MISSING__').astype(str)==str(c['category'])).to_numpy();test=test.astype(float)<=c['value']
        mask &= test if c['operator']=='<=' else ~test
    return mask

def design(x,reference,columns=None):
    if columns is None:
        columns=[]
        for c in reference:
            if pd.api.types.is_numeric_dtype(reference[c]):columns.append({'feature':c,'kind':'numeric','fill':float(reference[c].median()) if reference[c].notna().any() else 0.})
            else:
                for cat in reference[c].fillna('__MISSING__').astype(str).value_counts().head(10).index:columns.append({'feature':c,'kind':'categorical','category':cat})
    if len(x)*len(columns)>2_000_000:raise ModelGuardError('Discovery matrix exceeds memory guard','resource_limit',413)
    arrays=[pd.to_numeric(x[c['feature']],errors='coerce').fillna(c['fill']).to_numpy() if c['kind']=='numeric' else (x[c['feature']].fillna('__MISSING__').astype(str)==c['category']).astype(float).to_numpy() for c in columns]
    return np.column_stack(arrays),columns

def rules(tree,columns):
    out={};t=tree.tree_
    def walk(node,parts):
        if t.children_left[node]==t.children_right[node]:out[node]=parts;return
        c=columns[t.feature[node]]
        for child,op in [(t.children_left[node],'<='),(t.children_right[node],'>')]:walk(child,parts+[{**c,'operator':op,'value':float(t.threshold[node])}])
    walk(0,[]);return out

def error_values(pipe,x,y,task):
    pred=pipe.predict(x);return (pred!=np.asarray(y)).astype(float) if task=='classification' else np.abs(pred-np.asarray(y))

def local_evidence(pipe,x,y,mask,cfg,iterations,rng):
    n=int(mask.sum());errors=error_values(pipe,x,y,cfg['task']);global_error=float(errors.mean());delta=float(errors[mask].mean()-global_error) if n else None
    if not n:return {'sample_count':0,'support_percentage':0.,'metrics':None,'performance_drop':None,'drop_ci':None,'metric_intervals':{}}
    metrics=evaluate(pipe,x.iloc[np.flatnonzero(mask)],np.asarray(y)[mask],cfg['task'],cfg.get('positive_class'))[0];boots=[];local={k:[] for k in (['error_rate','precision_macro','recall_macro','macro_f1'] if cfg['task']=='classification' else ['mae','rmse'])}
    for _ in range(iterations):
        ix=rng.integers(0,len(x),len(x));selected=ix[mask[ix]]
        if len(selected)<2:continue
        boots.append(float(errors[selected].mean()-errors[ix].mean()))
        met=evaluate(pipe,x.iloc[selected],np.asarray(y)[selected],cfg['task'],cfg.get('positive_class'))[0]
        for k in local:
            v=value(met.get(k))
            if v is not None:local[k].append(v)
    interval=lambda a:[float(np.quantile(a,.025)),float(np.quantile(a,.975))] if len(a)>=10 else None
    dist={str(k):int(v) for k,v in pd.Series(np.asarray(y)[mask]).value_counts().items()} if cfg['task']=='classification' else {'mean':float(np.mean(np.asarray(y)[mask])),'variance':float(np.var(np.asarray(y)[mask])),'quantiles':np.quantile(np.asarray(y)[mask],[.1,.5,.9]).tolist()}
    return {'sample_count':n,'support_percentage':100*n/len(x),'metrics':metrics,'performance_drop':delta,'relative_performance_drop':delta/max(global_error,.01),'drop_ci':interval(boots),'metric_intervals':{k:{'interval':interval(v),'reason':None if interval(v) else 'Insufficient bootstrap support'} for k,v in local.items()},'target_distribution':dist}

def analyze(pid,params):
    model,art=load_model(pid);prior=analyses(pid,'modelshield',model['id'])
    if prior:return prior[0] # Freeze discovered rules before final-test confirmation; do not mine test via repeated queries.
    cfg=art['config'];pipe=art['pipeline'];x=art['x_validation'];y=art['y_validation'];xt=art['x_test'];yt=art['y_test'];required=max(params['min_samples'],math.ceil(params['min_support']*len(x)));errors=error_values(pipe,x,y,cfg['task']);regions=[];reason=None
    if len(x)<2*required:reason='Discovery partition too small for supported slices'
    elif np.ptp(errors)<1e-12:reason='No error variation for region discovery'
    else:
        matrix,columns=design(x,art['x_train']);tree=DecisionTreeRegressor(max_depth=params['max_depth'],min_samples_leaf=required,random_state=cfg['random_seed']).fit(matrix,errors);rulemap=rules(tree,columns);leaves=tree.apply(matrix);rng=np.random.default_rng(cfg['random_seed'])
        for leaf,clauses in rulemap.items():
            mask=leaves==leaf
            if errors[mask].mean()<=errors.mean():continue
            rule={'clauses':clauses};test_mask=region_mask(xt,rule);discovery=local_evidence(pipe,x,y,mask,cfg,0,rng);confirmation=local_evidence(pipe,xt,yt,test_mask,cfg,params['bootstrap_iterations'],rng)
            support=confirmation['sample_count']>=max(params['min_samples'],math.ceil(params['min_support']*len(xt)));ci=confirmation['drop_ci'];confirmed=bool(support and ci and ci[0]>0);drop=confirmation.get('relative_performance_drop') or 0
            status='Critical' if confirmed and drop>=cfg['critical_drop'] else 'Needs Attention' if confirmed and drop>=cfg['acceptance_drop'] else 'Monitor' if confirmed else 'Insufficient Evidence'
            text=[]
            for c in clauses:
                text.append(f"{c['feature']} {c['operator']} {c['value']:.4g}" if c['kind']=='numeric' else f"{c['feature']} {'!=' if c['operator']=='<=' else '='} {c['category']!r}")
            regions.append({'region_id':uid(),'rule':' AND '.join(text) or 'all samples',**rule,'discovery':discovery,'confirmation':confirmation,'confirmed':confirmed,'status':status,'evidence_strength':'Statistical confirmation' if confirmed else 'Unconfirmed exploratory region','limitations':['Bootstrap intervals are conditional and not adjusted for multiple rule discovery.']})
    return save_analysis(pid,'modelshield',{'model_id':model['id'],'method':'Validation shallow error tree; fixed rules confirmed on untouched final test','discovery_partition':'validation','confirmation_partition':'test','failure_regions':regions,'status':'Insufficient Evidence' if reason or not regions else max((r['status'] for r in regions),key=lambda s:{'Critical':5,'Needs Attention':4,'Monitor':2,'Insufficient Evidence':1}[s]),'unavailable_reason':reason,'limitations':['Exploratory associations are not causal explanations.','Repeated requests return the same frozen rules and confirmation.']})
