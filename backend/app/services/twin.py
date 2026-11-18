import copy
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from ..errors import ModelGuardError
from .modeling import load_model
from .data import profile
from .metrics import evaluate,relative_drop
from .shield import save_analysis


def distribution(pred,task):
    p=np.asarray(pred)
    if task=='classification':return {str(k):float(v) for k,v in pd.Series(p).value_counts(normalize=True).items()}
    return {'mean':float(p.mean()),'variance':float(p.var()),'quantiles':np.quantile(p,[.05,.25,.5,.75,.95]).tolist()}

def perturb(x,y,scenario,intensity,rng,depth=0):
    if depth>4:raise ModelGuardError('Combined scenario nesting exceeds four levels')
    x=x.copy();y=np.asarray(y).copy() if y is not None else None;s=copy.deepcopy(scenario);kind=s['type'];p=s.get('parameters',{});mag=float(s.get('magnitude',.1))*intensity;c=s.get('feature');changed=[];definition={**s,'intensity':intensity,'effective_magnitude':mag}
    if intensity==0:
        definition.update(identity=True)
        if kind=='numeric_scaling':definition['effective_factor']=1.
        return x,y,[],definition
    if kind=='combined':
        items=p.get('scenarios',[])
        if not 1<=len(items)<=10:raise ModelGuardError('Combined requires 1–10 scenarios')
        definitions=[]
        for item in items:
            x,y,features,d=perturb(x,y,item,intensity,rng,depth+1);changed+=features;definitions.append(d)
        definition['resolved_scenarios']=definitions;return x,y,sorted(set(changed)),definition
    if kind=='class_prevalence':
        if y is None:raise ModelGuardError('Class prevalence needs confirmed labels')
        desired=p.get('distribution',{})
        if not desired:raise ModelGuardError('Provide requested class distribution')
        classes=np.unique(y);desired={str(k):float(v) for k,v in desired.items()}
        if set(desired)!=set(map(str,classes)) or min(desired.values())<0 or not np.isclose(sum(desired.values()),1):raise ModelGuardError('Invalid requested class distribution')
        if mag==0:return x,y,[],{**definition,'identity':True}
        alpha=min(1,mag);observed={str(k):float(np.mean(y==k)) for k in classes};prob=np.asarray([(1-alpha)+alpha*desired[str(k)]/observed[str(k)] for k in y]);prob/=prob.sum();idx=rng.choice(len(x),len(x),replace=True,p=prob);definition.update(units='prevalence mixture fraction',requested_distribution=desired)
        return x.iloc[idx].reset_index(drop=True),y[idx],['target_prevalence'],definition
    features=p.get('features') or ([c] if c else list(x.columns))
    if any(f not in x for f in features):raise ModelGuardError('Scenario feature absent from model inputs')
    if kind in {'numeric_shift','numeric_scaling','noise','extreme_values'} and any(not pd.api.types.is_numeric_dtype(x[f]) for f in features):raise ModelGuardError('Scenario requires numeric features')
    for feature in features:
        series=x[feature];std=float(series.std()) if pd.api.types.is_numeric_dtype(series) and series.notna().sum()>1 else 1.;std=std if std>0 else 1.
        if kind=='numeric_shift':
            direction=p.get('direction','increase');units=p.get('units','standard_deviations')
            if direction not in {'increase','decrease'} or units not in {'standard_deviations','absolute'}:raise ModelGuardError('Invalid shift direction or units')
            shift=mag*(std if units=='standard_deviations' else 1)*(1 if direction=='increase' else -1);x[feature]=series+shift;definition.update(direction=direction,units=units,applied_offset=shift)
        elif kind=='numeric_scaling':
            factor=float(p.get('factor',s.get('magnitude',1.1)))
            if factor<0 or not np.isfinite(factor):raise ModelGuardError('Scale factor must be finite and nonnegative')
            effective=1+intensity*(factor-1)
            if effective<0:raise ModelGuardError('Effective scale must be nonnegative')
            x[feature]=series*effective;definition.update(units='multiplicative factor',effective_factor=effective)
        elif kind=='noise':x[feature]=series+rng.normal(0,mag*std,len(x));definition['units']='standard deviations of additive zero-mean Gaussian noise'
        elif kind=='missingness':
            if mag>1:raise ModelGuardError('Missingness probability exceeds one')
            x.loc[rng.random(len(x))<mag,feature]=np.nan;definition['units']='additional null probability'
        elif kind=='extreme_values':
            fraction=float(p.get('fraction',.05))
            if not 0<=fraction<=1:raise ModelGuardError('Invalid extreme fraction')
            direction=p.get('direction','increase')
            if direction not in {'increase','decrease'}:raise ModelGuardError('Invalid extreme direction')
            mask=rng.random(len(x))<fraction;x.loc[mask,feature]=series.median()+mag*std*(1 if direction=='increase' else -1);definition.update(units='standard deviations from median',direction=direction,affected_fraction=fraction)
        elif kind=='categorical_shift':
            if pd.api.types.is_numeric_dtype(series):raise ModelGuardError('Categorical shift requires categorical input')
            dist=p.get('distribution',{});total=sum(float(v) for v in dist.values())
            if not dist or not np.isclose(total,1) or min(map(float,dist.values()))<0:raise ModelGuardError('Requested category probabilities must sum to one')
            mask=rng.random(len(x))<min(mag,1);x.loc[mask,feature]=rng.choice(list(dist),mask.sum(),p=list(dist.values()));definition.update(units='replacement mixture fraction',requested_distribution=dist)
        elif kind=='input_corruption':
            if mag>1:raise ModelGuardError('Corruption fraction exceeds one')
            mask=rng.random(len(x))<mag;x.loc[mask,feature]=series.median()+10*std if pd.api.types.is_numeric_dtype(series) else '__CORRUPTED__';definition['units']='corrupted input fraction'
        else:raise ModelGuardError('Unknown scenario type')
        changed.append(feature)
    return x,y,changed,definition

def simulate(pid,params):
    model,art=load_model(pid);pipe=art['pipeline'];cfg=art['config'];x=art['x_validation'];y=art['y_validation'] if params.get('include_labels',True) else None;task=cfg['task'];basepred=pipe.predict(x);basep=pipe.predict_proba(x) if task=='classification' else None
    baseline=evaluate(pipe,x,y,task,cfg.get('positive_class'))[0] if y is not None else None;results=[];rng=np.random.default_rng(params.get('random_seed',cfg['random_seed']))
    for scenario in params['scenarios']:
        if scenario['type']=='class_prevalence' and task!='classification':raise ModelGuardError('Class prevalence is classification only')
        for intensity in params.get('intensities',[1.]):
            xs,ys,features,definition=perturb(x,y,scenario,intensity,rng);pred=pipe.predict(xs);p=pipe.predict_proba(xs) if task=='classification' else None;metrics=evaluate(pipe,xs,ys,task,cfg.get('positive_class'))[0] if ys is not None else None;drop=relative_drop(baseline,metrics,cfg['primary_metric']) if baseline is not None else None
            status='Insufficient Evidence' if drop is None else 'Critical' if drop>=cfg['critical_drop'] else 'Needs Attention' if drop>=cfg['acceptance_drop'] else 'Pass'
            comparison={'changed_prediction_fraction':float(np.mean(pred!=basepred))} if task=='classification' else {'wasserstein_distance':float(wasserstein_distance(basepred,pred)),'mean_shift':float(np.mean(pred)-np.mean(basepred)),'variance_shift':float(np.var(pred)-np.var(basepred)),'paired_mean_absolute_change':float(np.mean(np.abs(pred-basepred))) if scenario['type']!='class_prevalence' else None}
            results.append({'scenario_definition':definition,'changed_features':features,'random_seed':params.get('random_seed',cfg['random_seed']),'baseline_distribution':profile(x),'scenario_distribution':profile(xs),'baseline_prediction_distribution':distribution(basepred,task),'scenario_prediction_distribution':distribution(pred,task),'prediction_comparison':comparison,'confidence_change':float(p.max(axis=1).mean()-basep.max(axis=1).mean()) if p is not None else None,'baseline_performance':baseline,'performance_metrics':metrics,'metric_change':{k:float(metrics[k]-baseline[k]) for k in metrics if isinstance(metrics[k],(int,float)) and isinstance(baseline.get(k),(int,float))} if metrics else None,'relative_primary_drop':drop,'status':status,'label':'Controlled scenario: robustness stress test using retained labels; not a forecast of future ground truth.' if ys is not None else 'Controlled scenario without labels; prediction sensitivity only.','limitations':['Perturbed inputs may not be causally compatible with retained labels.','Practical effect thresholds do not imply statistical significance.']})
    ranking=sorted(range(len(results)),key=lambda i:results[i]['relative_primary_drop'] if results[i]['relative_primary_drop'] is not None else abs(results[i]['confidence_change'] or results[i]['prediction_comparison'].get('mean_shift',0)),reverse=True)
    return save_analysis(pid,'datatwin',{'model_id':model['id'],'evaluation_partition':'validation','results':results,'sensitivity_ranking':ranking,'tipping_points':[i for i,r in enumerate(results) if r['status'] in {'Critical','Needs Attention'}]})
