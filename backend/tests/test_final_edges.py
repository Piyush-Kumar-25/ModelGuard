import json,os,time
import numpy as np
import pandas as pd
import pytest
from app.schemas import ConfigureRequest,DataTwinRequest
from app.errors import ModelGuardError
from app.services import data,modeling,drift,twin,reliability,experiments
from app.services.splitting import split_indices
from conftest import frame

@pytest.mark.parametrize('task',['binary_numeric','multiclass_pr_auc','regression'])
def test_full_training_task_types(client,task):
    df=frame(360);task_name='regression' if task=='regression' else 'classification';metric='rmse' if task=='regression' else 'pr_auc'
    if task=='binary_numeric':df['target']=(df.target=='yes').astype(int)
    elif task=='multiclass_pr_auc':df['target']=np.where(df.signal<-.3,'low',np.where(df.signal>.3,'high','middle'))
    else:df['target']=df.signal*4+df.noise*.1
    pid=data.create_project(task)['id'];d=data.upload_dataset(pid,'x.csv',df.to_csv(index=False).encode());data.configure(pid,{'dataset_id':d['id'],'target_column':'target','task':task_name,'primary_metric':metric,'record_id_column':'record_id','max_trials':2,'weight_trials':3,'fit_timeout_seconds':30});r=modeling.train(pid);assert isinstance(r['metrics'][metric],float)
    if task=='regression':
        sim=twin.simulate(pid,DataTwinRequest(scenarios=[{'type':'numeric_shift','feature':'signal','magnitude':.5}]).model_dump());comparison=sim['results'][0]['prediction_comparison'];assert comparison['wasserstein_distance']>0;assert 'paired_mean_absolute_change' in comparison
        batch=drift.upload_batch(pid,'production.csv',df.iloc[:70].to_csv(index=False).encode());drift.confirm_mapping(pid,batch['id'],{'confirm_labels':True});monitor=drift.analyze(pid,{'dataset_id':batch['id']});assert monitor['performance']['label_count']==70;assert monitor['performance']['drop_95pct_ci'] is not None;assert 'wasserstein_distance' in monitor['prediction_drift']

def test_duplicate_rows_do_not_cross_splits():
    df=frame(100);df=pd.concat([df,df.iloc[:10]],ignore_index=True);cfg=ConfigureRequest(dataset_id='x',target_column='target',task='classification').model_dump();cfg['feature_columns']=['signal','noise','region'];m=split_indices(df,cfg);assert len(m['excluded_duplicate_indices'])==10;assert all(i<100 for k in ['train','validation','test'] for i in m[k])

@pytest.mark.parametrize('kind',['extreme_values','class_prevalence','combined'])
def test_zero_intensity_is_identity(kind):
    df=frame(50);x=df[['signal','region']];y=df.target.to_numpy();scenario={'type':kind,'feature':'signal','magnitude':1.,'parameters':{'distribution':{'yes':.5,'no':.5}}}
    if kind=='combined':scenario['parameters']={'scenarios':[{'type':'extreme_values','feature':'signal','magnitude':1}]}
    out,labels,_,_=twin.perturb(x,y,scenario,0,np.random.default_rng(42));assert out.equals(x);assert np.array_equal(y,labels)

def test_joint_optional_fallback_needs_joint_validation(trained):
    cfg=modeling.load_model(trained[0])[1]['config'];cfg['validated_optional_features']=['noise','signal'];cfg['joint_optional_fallback_validated']=False;check=drift.schema_check(trained[3].drop(columns=['noise','signal']),cfg);assert not check['prediction_allowed'];assert check['joint_optional_fallback_unvalidated']

def test_undefined_primary_never_passes_monitoring(trained):
    pid,_,mid,df=trained;model,art=modeling.load_model(pid);cfg=art['config'];cfg['primary_metric']='roc_auc'
    from pathlib import Path
    from app.database import load,dump
    state=load(Path(model['state_path']).read_text());state['config']=cfg;Path(model['state_path']).write_text(dump(state));df['target']='yes';b=drift.upload_batch(pid,'batch.csv',df.to_csv(index=False).encode());drift.confirm_mapping(pid,b['id'],{'confirm_labels':True});r=drift.analyze(pid,{'dataset_id':b['id']});assert r['performance']['relative_primary_drop'] is None;assert r['status']!='Pass';assert r['recommendation']['action']=='Continue monitoring'

def test_renamed_target_requires_confirmation(trained):
    cfg=modeling.load_model(trained[0])[1]['config'];r=drift.label_detection(trained[3].rename(columns={'target':'target_actual'}),cfg);assert 'target_actual' in r['suggested_mapping'];assert r['confirmation_required']

def test_threshold_action_level():
    perf={'relative_primary_drop':.02,'confirmed_degradation':False,'production_metrics':{'pr_auc':.9,'f1':.7},'reference_metrics':{'pr_auc':.9,'f1':.8}}
    assert drift.action_level('Compatible',False,False,perf,False)[0]=='Adjust decision threshold'

def long_job():time.sleep(20);return {'ok':True}

@pytest.mark.parametrize('mode',['cancel','timeout'])
def test_job_cancel_and_timeout(client,mode):
    from app import jobs
    pid=data.create_project('worker control')['id'];r=jobs.submit(pid,'long',long_job,timeout_seconds=1 if mode=='timeout' else 30)
    if mode=='cancel':jobs.cancel(r['job_id'])
    for _ in range(100):
        result=jobs.get(r['job_id'])
        if result['status'] in {'cancelled','timed_out','failed'}:break
        time.sleep(.05)
    assert result['status']==('cancelled' if mode=='cancel' else 'timed_out')
    if mode=='timeout':assert result['error']['code']=='job_timeout'

def test_legacy_migration_keeps_retraining_available(client):
    from app.database import insert,uid,now
    pid=data.create_project('legacy')['id'];df=frame();d=data.upload_dataset(pid,'x.csv',df.to_csv(index=False).encode());insert('models',{'id':uid(),'project_id':pid,'dataset_id':d['id'],'version':1,'path':'legacy-unloadable','task':'classification','selected_name':'old','metrics_json':'{}','details_json':'{}','created_at':now()})
    cfg=data.configure(pid,{'dataset_id':d['id'],'target_column':'target','max_trials':1,'weight_trials':2,'fit_timeout_seconds':30});r=modeling.train(pid);assert r['version']==2;assert r['status']=='incumbent'
