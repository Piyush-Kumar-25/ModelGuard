from __future__ import annotations
import difflib,re
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp,wasserstein_distance
from ..database import all_rows,dump,load,now,one,update
from ..errors import ModelGuardError
from .data import get_dataset,load_frame,profile,public,upload_dataset,read_csv
from .modeling import load_model,get_model
from .metrics import evaluate,relative_drop,value
from .shield import analyses,save_analysis,region_mask
from .twin import distribution


def schema_check(df,cfg):
    required=cfg['feature_columns'];schema=cfg['schema'];optional=set(cfg.get('validated_optional_features',[]));missing=[c for c in required if c not in df];null=[c for c in required if c in df and df[c].isna().all()];critical=[c for c in set(missing+null) if c not in optional];extra=[c for c in df if c not in required and c not in {cfg['target_column'],cfg.get('record_id_column')}];mismatch=[];unseen={};ranges={};units={}
    for c in required:
        if c not in df or df[c].isna().all():continue
        s=df[c];ref=schema[c]
        if ref['kind']=='numeric':
            if not pd.api.types.is_numeric_dtype(s) or np.isinf(pd.to_numeric(s,errors='coerce').dropna()).any():mismatch.append(c);continue
            low,high=ref['min'],ref['max']
            if low is not None and high is not None:
                fraction=float(((s<low)|(s>high)).mean())
                if fraction:ranges[c]=fraction
                denom=max(abs(ref.get('mean') or 0),ref.get('std') or 0,1e-8);ratio=abs(float(s.mean()))/denom
                if ratio>10:units[c]={'magnitude_ratio':ratio,'evidence':'heuristic unit-like shift'}
        else:
            if pd.api.types.is_numeric_dtype(s):mismatch.append(c)
            unknown=~s.dropna().astype(str).isin(ref.get('categories',[]));fraction=float(unknown.mean()) if len(unknown) else 0.
            if fraction:unseen[c]={'fraction':fraction,'categories':s.dropna().astype(str)[unknown].unique()[:20].tolist(),'reference_truncated':ref.get('categories_truncated',False)}
    rename={c:difflib.get_close_matches(c,extra,n=3,cutoff=.4) for c in missing}
    joint_unvalidated=len(set(missing+null)&optional)>1 and not cfg.get('joint_optional_fallback_validated',False)
    incompatible=bool(critical or mismatch or joint_unvalidated)
    result={'status':'Incompatible' if incompatible else 'Compatible with warnings' if missing or null or extra or unseen or ranges or units else 'Compatible','missing_required_features':[c for c in missing if c not in optional],'missing_optional_features':[c for c in missing if c in optional],'fully_null_features':null,'critical_missing_or_null_features':critical,'extra_features':extra,'possible_renames':rename,'type_mismatches':mismatch,'unseen_categories':unseen,'range_violations':ranges,'unit_like_changes':units,'joint_optional_fallback_unvalidated':joint_unvalidated,'target_present':cfg['target_column'] in df,'prediction_allowed':not incompatible}
    return result

def model_inputs(df,cfg,compatibility=None):
    compatibility=compatibility or schema_check(df,cfg)
    if not compatibility['prediction_allowed']:raise ModelGuardError('Required model inputs are incompatible','schema_incompatible',422,compatibility)
    out=df.copy()
    for c in compatibility['missing_optional_features']:out[c]=np.nan
    return out[cfg['feature_columns']]

def label_detection(df,cfg):
    target=cfg['target_column'];blocked=lambda c:bool(re.match(r'(?i)predict|probability',c))
    suggestions=[] if target in df else [c for c in df if not blocked(c) and c not in cfg['feature_columns'] and (difflib.SequenceMatcher(None,c.lower(),target.lower()).ratio()>.5 or c.lower() in {'label','outcome','actual'})]
    return {'expected_target':target,'exact_target_present':target in df,'suggested_mapping':suggestions,'confirmation_required':target in df or bool(suggestions),'availability':'detected_unconfirmed' if target in df else 'absent'}

def upload_batch(pid,filename,content,model_id=None,batch_timestamp=None,record_id_column=None):
    model,art=load_model(pid,model_id);df=read_csv(content);cfg=art['config'];rid=record_id_column or cfg.get('record_id_column')
    if rid and (rid not in df or df[rid].isna().any() or df[rid].duplicated().any()):raise ModelGuardError('Batch record ID must be present, unique and non-null')
    stamp=now() if not batch_timestamp else pd.to_datetime(batch_timestamp,utc=True,errors='coerce')
    if pd.isna(stamp):raise ModelGuardError('Invalid batch timestamp')
    stamp=stamp.isoformat() if not isinstance(stamp,str) else stamp;check=schema_check(df,cfg);detection=label_detection(df,cfg)
    row=upload_dataset(pid,filename,content,'production_batch',model_id=model['id'],batch_timestamp=stamp,record_id_column=rid,compatibility_json=dump(check),label_availability=detection['availability'],target_column=cfg['target_column'])
    return {**row,'schema_compatibility':check,'labels':detection}

def list_batches(pid):
    return {'batches':[{**public(r),'schema_compatibility':load(r['compatibility_json']),'mapping':load(r['mapping_json'])} for r in all_rows("SELECT * FROM datasets WHERE project_id=? AND role='production_batch' ORDER BY batch_timestamp,created_at",(pid,))]}

def batch_frame(pid,bid):
    df,row=load_frame(pid,bid)
    if row['role']!='production_batch':raise ModelGuardError('Dedicated production batch required')
    mapping=load(row['mapping_json'],{});df=df.rename(columns=mapping.get('features',{}));y=None
    if row['label_availability'] in {'confirmed','partial'}:
        if mapping.get('labels_dataset_id'):
            labels,_=load_frame(pid,mapping['labels_dataset_id']);rid=row['record_id_column'];series=labels.set_index(rid)[mapping['target_column']];y=df[rid].map(series)
        else:y=df[mapping['target_column']]
    return df,row,y

def confirm_mapping(pid,bid,payload):
    raw,row=load_frame(pid,bid)
    if row['role']!='production_batch':raise ModelGuardError('Dedicated production batch required')
    _,art=load_model(pid,row['model_id']);cfg=art['config'];saved=load(row['mapping_json'],{});mapping=payload.get('feature_mapping',{})
    if len(set(mapping.values()))!=len(mapping) or any(src not in raw or src==cfg['target_column'] or dst not in cfg['feature_columns'] or (dst in raw and src!=dst) for src,dst in mapping.items()):raise ModelGuardError('Invalid or colliding schema mapping')
    if mapping:saved['features']=mapping
    mapped=raw.rename(columns=saved.get('features',{}))
    if payload.get('confirm_labels'):
        target=payload.get('target_column') or cfg['target_column']
        if target not in mapped or re.match(r'(?i)predict|probability',target) or target in cfg['feature_columns']:raise ModelGuardError('Invalid ground-truth column')
        validate_labels(mapped[target],art);saved.update(target_column=target);saved.pop('labels_dataset_id',None);availability='partial' if mapped[target].isna().any() else 'confirmed'
    else:availability=row['label_availability']
    check=schema_check(mapped,cfg);update('UPDATE datasets SET mapping_json=?,compatibility_json=?,label_availability=? WHERE id=?',(dump(saved),dump(check),availability,bid));return {'batch_id':bid,'mapping':saved,'schema_compatibility':check,'label_availability':availability}

def validate_labels(y,art):
    clean=y.dropna()
    if clean.empty:raise ModelGuardError('No confirmed labels provided')
    if art['config']['task']=='classification' and not set(clean)<=set(art['pipeline'].classes_):raise ModelGuardError('Labels contain unknown classes')
    if art['config']['task']=='regression' and (not pd.api.types.is_numeric_dtype(clean) or not np.isfinite(clean).all()):raise ModelGuardError('Regression labels must be finite numeric values')

def attach_labels(pid,bid,filename,content,target_column=None):
    batch,row=load_frame(pid,bid)
    if row['role']!='production_batch':raise ModelGuardError('Dedicated production batch required')
    _,art=load_model(pid,row['model_id']);cfg=art['config'];rid=row['record_id_column'];target=target_column or cfg['target_column'];labels=read_csv(content)
    if not rid or rid not in labels or target not in labels or target==rid or re.match(r'(?i)predict|probability',target):raise ModelGuardError('Stable record ID and confirmed ground-truth column required')
    if labels[rid].isna().any() or labels[rid].duplicated().any() or not set(labels[rid])<=set(batch[rid]):raise ModelGuardError('Label IDs must uniquely match existing batch records')
    validate_labels(labels[target],art);r=upload_dataset(pid,filename,content,'production_labels',parent_batch_id=bid,model_id=row['model_id'],record_id_column=rid,target_column=target,label_availability='confirmed');saved=load(row['mapping_json'],{});saved.update(labels_dataset_id=r['id'],target_column=target);availability='confirmed' if labels[target].notna().sum()==len(batch) else 'partial';update('UPDATE datasets SET mapping_json=?,label_availability=? WHERE id=?',(dump(saved),availability,bid));return {'dataset_id':r['id'],'batch_id':bid,'matched_labels':int(labels[target].notna().sum()),'label_availability':availability}

def numeric_drift(a,b):
    a=pd.to_numeric(a,errors='coerce').dropna().to_numpy();b=pd.to_numeric(b,errors='coerce').dropna().to_numpy()
    if not len(a) or not len(b):return {'available':False,'reason':'No numeric observations'}
    bins=np.unique(np.quantile(a,np.linspace(0,1,11)));bins=np.r_[-np.inf,bins[1:-1],np.inf];pa=np.histogram(a,bins)[0]+.5;pb=np.histogram(b,bins)[0]+.5;pa=pa/pa.sum();pb=pb/pb.sum();ks=ks_2samp(a,b)
    return {'available':True,'psi':float(np.sum((pb-pa)*np.log(pb/pa))),'ks_statistic':float(ks.statistic),'ks_pvalue':float(ks.pvalue),'wasserstein_distance':float(wasserstein_distance(a,b)),'normalized_wasserstein':float(wasserstein_distance(a,b)/max(np.std(a),1e-8)),'mean_change':float(b.mean()-a.mean())}

def category_drift(a,b):
    a=a.fillna('__MISSING__').astype(str);b=b.fillna('__MISSING__').astype(str);keys=sorted(set(a)|set(b));pa=a.value_counts(normalize=True).reindex(keys,fill_value=0);pb=b.value_counts(normalize=True).reindex(keys,fill_value=0)
    return {'js_distance':float(jensenshannon(pa,pb)),'proportion_changes':{k:float(pb[k]-pa[k]) for k in keys[:100]},'unseen_frequency':float((~b.isin(set(a))).mean()),'categories_truncated':len(keys)>100}

def raw_drift(reference,batch,cfg):
    out={}
    for c in reference:
        s=batch[c] if c in batch else pd.Series(np.nan,index=batch.index);r={'missingness_change':float(s.isna().mean()-reference[c].isna().mean())}
        r.update(numeric_drift(reference[c],s) if pd.api.types.is_numeric_dtype(reference[c]) else category_drift(reference[c],s));r['practically_significant']=abs(r['missingness_change'])>=cfg['missingness_drift_threshold'] or r.get('psi',0)>=cfg['psi_threshold'] or r.get('normalized_wasserstein',0)>=cfg['wasserstein_threshold'] or r.get('js_distance',0)>=cfg['js_threshold'];r['statistically_significant']=r.get('ks_pvalue',1)<.05;out[c]=r
    return out

def performance_comparison(art,x,y):
    cfg=art['config'];mask=y.notna().to_numpy();xp=x.iloc[np.flatnonzero(mask)];yp=y.iloc[np.flatnonzero(mask)];metrics=evaluate(art['pipeline'],xp,yp,cfg['task'],cfg.get('positive_class'))[0];before=evaluate(art['pipeline'],art['x_test'],art['y_test'],cfg['task'],cfg.get('positive_class'))[0];drop=relative_drop(before,metrics,cfg['primary_metric']);boots=[];minimum=cfg['minimum_monitoring_labels'];rng=np.random.default_rng(cfg['random_seed'])
    # Predictions are cached; bootstrap never fits a model or changes the reference.
    from .metrics import classification_metrics,regression_metrics
    if len(yp)>=minimum and len(art['y_test'])>=minimum and drop is not None:
        pa=art['pipeline'].predict(art['x_test']);pb=art['pipeline'].predict(xp);qa=art['pipeline'].predict_proba(art['x_test']) if cfg['task']=='classification' else None;qb=art['pipeline'].predict_proba(xp) if cfg['task']=='classification' else None;ya=np.asarray(art['y_test']);yb=np.asarray(yp)
        for _ in range(100):
            a=rng.integers(0,len(ya),len(ya));b=rng.integers(0,len(yb),len(yb))
            ma=classification_metrics(ya[a],pa[a],qa[a],art['pipeline'].classes_,cfg.get('positive_class')) if qa is not None else regression_metrics(ya[a],pa[a]);mb=classification_metrics(yb[b],pb[b],qb[b],art['pipeline'].classes_,cfg.get('positive_class')) if qb is not None else regression_metrics(yb[b],pb[b]);d=relative_drop(ma,mb,cfg['primary_metric'])
            if d is not None:boots.append(d)
    ci=np.quantile(boots,[.025,.975]).tolist() if len(boots)>=50 else None
    return {'reference_metrics':before,'production_metrics':metrics,'label_count':len(yp),'relative_primary_drop':drop,'drop_95pct_ci':ci,'confirmed_degradation':bool(ci and ci[0]>cfg['acceptance_drop']),'limitations':['Independent finite-sample bootstrap; potential selection/label-delay bias.','Reference is the frozen final test, not a forecast of production performance.']}

def action_level(schema,material,persistent,performance,calibration,important=True,acceptance_drop=.05):
    if schema=='Incompatible':return 'Full model redevelopment recommended','Resolve the schema contract; permanent changes need a new evaluated version.'
    if performance is None:return ('Evaluate retraining','Investigate and evaluate retraining; labels are unavailable, so performance degradation is unconfirmed.') if material and persistent else ('Investigate drift','Unlabelled practical drift needs investigation.') if material else ('Continue monitoring','Collect confirmed production labels.')
    drop=performance.get('relative_primary_drop')
    if drop is None:return 'Continue monitoring','Primary performance metric is unavailable for these labels.'
    if performance['confirmed_degradation'] and persistent and material and important:return 'Retraining recommended','Supported degradation and persistent important-feature drift justify a challenger evaluation.'
    if drop>acceptance_drop:return 'Evaluate retraining','Observed degradation needs supported challenger comparison; no automatic replacement.'
    if calibration:return 'Recalibrate probabilities','Evaluate a training-only probability recalibration challenger.'
    pm=performance['production_metrics'];rm=performance['reference_metrics']
    if value(pm.get('pr_auc')) is not None and value(rm.get('pr_auc')) is not None and value(pm['pr_auc'])>=value(rm['pr_auc'])-.02 and value(pm.get('f1')) is not None and value(rm.get('f1')) is not None and value(pm['f1'])<value(rm['f1'])-.03:return 'Adjust decision threshold','Evaluate threshold alternatives on validation data.'
    if material:return 'Investigate drift','Practical drift is present; degradation is not confirmed.'
    return 'Continue using current model','No material measured degradation in available evidence.'

def history(pid,mid=None):
    rows=analyses(pid,'drift',mid);bybatch={r['batch_id']:r for r in rows};return {'batches':sorted(bybatch.values(),key=lambda r:(r['batch_timestamp'],r['generated_at']))}

def analyze(pid,params):
    bid=params.get('dataset_id')
    if not bid:raise ModelGuardError('Explicit production batch ID required')
    df,row,y=batch_frame(pid,bid);model,art=load_model(pid,row['model_id']);cfg=art['config'];check=schema_check(df,cfg);raw=raw_drift(art['x_train'],df,cfg);fraction=sum(r['practically_significant'] for r in raw.values())/len(raw);material=fraction>=cfg['drift_fraction_threshold'];result={'model_id':model['id'],'batch_id':bid,'batch_timestamp':row['batch_timestamp'],'schema_compatibility':check,'raw_feature_drift':raw,'extra_feature_profile':profile(df[check['extra_features']]) if check['extra_features'] else {},'label_availability':row['label_availability'],'performance':None,'prediction_drift':None,'confidence_drift':None,'component_disagreement':None,'subgroup_drift':[],'transformed_feature_drift':{'available':False,'reason':'Schema incompatible'},'evidence_type':'observed production batch comparison'}
    calibration=False;important=False
    if check['prediction_allowed']:
        x=model_inputs(df,cfg,check);reference=art['x_train'];pipe=art['pipeline'];pr=pipe.predict(reference);pb=pipe.predict(x);result['prediction_drift']=numeric_drift(pd.Series(pr),pd.Series(pb)) if cfg['task']=='regression' else category_drift(pd.Series(pr),pd.Series(pb));result['prediction_distributions']={'reference':distribution(pr,cfg['task']),'batch':distribution(pb,cfg['task'])}
        material|=result['prediction_drift'].get('normalized_wasserstein',0)>=cfg['wasserstein_threshold'] or result['prediction_drift'].get('js_distance',0)>=cfg['js_threshold']
        if cfg['task']=='classification':
            qr=pipe.predict_proba(reference);qb=pipe.predict_proba(x);result['confidence_drift']=numeric_drift(pd.Series(qr.max(axis=1)),pd.Series(qb.max(axis=1)));result['per_class_probability_drift']={str(c):numeric_drift(pd.Series(qr[:,i]),pd.Series(qb[:,i])) for i,c in enumerate(pipe.classes_)};material|=result['confidence_drift'].get('psi',0)>=cfg['psi_threshold']
        cr=pipe.component_predictions(reference);cb=pipe.component_predictions(x);dr=float(np.std(cr,axis=0).mean());db=float(np.std(cb,axis=0).mean());result['component_disagreement']={'reference':dr,'batch':db,'change':db-dr};material|=db-dr>=.1
        # Training-reference permutation sensitivity is descriptive, not causal importance.
        sample=reference.iloc[:500];labels=art['y_train'].iloc[:500]
        base=pipe.predict(sample);scores=[];rng=np.random.default_rng(cfg['random_seed'])
        for c in sample:
            altered=sample.copy();altered[c]=rng.permutation(altered[c].to_numpy());pred=pipe.predict(altered);s=float(np.mean(pred!=base)) if cfg['task']=='classification' else float(np.mean(np.abs(pred-base)));scores.append((c,s))
        top=[c for c,s in sorted(scores,key=lambda z:z[1],reverse=True)[:max(1,len(scores)//3)]];important=any(raw[c]['practically_significant'] for c in top);result['important_features']={'features':top,'method':'training-reference permutation prediction sensitivity','important_feature_drift':important}
        transformed=[]
        for name,component in pipe.components:
            base=component.estimator.calibrated_classifiers_[0].estimator.estimator if hasattr(component,'estimator') and hasattr(component.estimator,'calibrated_classifiers_') else component
            prep=base.named_steps.get('preprocessor') if hasattr(base,'named_steps') else None
            if prep is not None:
                try:
                    a=prep.transform(reference);b=prep.transform(x)
                    if hasattr(a,'toarray'):ma=np.asarray(a.mean(axis=0)).ravel();mb=np.asarray(b.mean(axis=0)).ravel()
                    else:ma=np.asarray(a,dtype=float).mean(axis=0);mb=np.asarray(b,dtype=float).mean(axis=0)
                    transformed.append({'component':name,'mean_absolute_feature_mean_change':float(np.mean(np.abs(mb-ma))),'dimensions':len(ma)})
                except (TypeError,ValueError):transformed.append({'component':name,'reason':'Native categorical representation; raw drift reported'})
        result['transformed_feature_drift']={'available':bool(transformed),'components':transformed}
        for analysis in analyses(pid,'modelshield',model['id']):
            for region in analysis['failure_regions']:
                a=float(region_mask(reference,region).mean());b=float(region_mask(x,region).mean());result['subgroup_drift'].append({'region_id':region['region_id'],'reference_support':a,'batch_support':b,'change':b-a});material|=region['confirmed'] and b-a>=.1
        if y is not None and y.notna().any():
            result['performance']=performance_comparison(art,x,y);pm=result['performance']['production_metrics'];rm=result['performance']['reference_metrics'];a=value(pm.get('brier_score'));b=value(rm.get('brier_score'));delta=a-b if a is not None and b is not None else None;calibration=delta is not None and delta>=cfg['calibration_drift_threshold'];result['calibration_drift']={'brier_change':delta,'material':calibration}
    result['material_drift']=bool(material);result['important_feature_drift']=bool(important)
    prior=[r for r in history(pid,model['id'])['batches'] if r['batch_id']!=bid];ordered=sorted(prior+[result],key=lambda r:r['batch_timestamp']);position=next(i for i,r in enumerate(ordered) if r['batch_id']==bid);recent=ordered[max(0,position-cfg['persistent_batches']+1):position+1];persistent=len(recent)>=cfg['persistent_batches'] and all(r['material_drift'] for r in recent);result['persistent_drift']=bool(persistent)
    action,reason=action_level(check['status'],material,persistent,result['performance'],calibration,important and all(r.get('important_feature_drift',False) for r in recent),cfg['acceptance_drop']);result['recommendation']={'action':action,'reason':reason,'automatic_replacement':False};perf=result['performance'];observed_drop=perf['relative_primary_drop'] if perf else None;result['status']='Critical' if not check['prediction_allowed'] else 'Needs Attention' if calibration or (perf and perf['confirmed_degradation']) else 'Monitor' if material or (observed_drop is not None and observed_drop>cfg['acceptance_drop']) else 'Pass' if perf and perf['label_count']>=cfg['minimum_monitoring_labels'] and observed_drop is not None else 'Insufficient Evidence';result['limitations']=['Statistical p-values and practical thresholds are reported separately.','Unlabelled stability cannot establish predictive accuracy.']
    update('UPDATE datasets SET compatibility_json=? WHERE id=?',(dump(check),bid));return save_analysis(pid,'drift',result)
