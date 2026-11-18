import copy,hashlib,io,json,os,pickle,subprocess,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from app.database import all_rows,connect,dump,insert,load,now,one,uid,update
from app.errors import ModelGuardError
from app.schemas import ConfigureRequest,DataTwinRequest
from app.services import data,modeling,drift,twin,shield,experiments,reliability,exporting,audit
from app.services.metrics import classification_metrics,regression_metrics
from app.services.splitting import split_indices,training_holdout
from app.services.inference import WeightedEnsemble
from app.services.registry import registry
from app.services.training import fit_with_deadline
from conftest import frame

@pytest.mark.parametrize('filename,content,code',[
 ('../bad.csv',b'a\n1\n','invalid_filename'),('bad\\name.csv',b'a\n1\n','invalid_filename'),('data.txt',b'a\n1\n','invalid_filename'),
 ('data.csv',b'a,a\n1,2\n','duplicate_columns'),('data.csv',b'a,b\n1,2,3\n','malformed_csv'),('data.csv',b'','empty_dataset'),('data.csv',b'a\n\x00','malformed_csv'),('data.csv',b'a\ninf\n','malformed_csv')])
def test_upload_validation(client,filename,content,code):
    pid=data.create_project('x')['id']
    with pytest.raises(ModelGuardError) as e:data.upload_dataset(pid,filename,content)
    assert e.value.code==code

def test_oversize_and_immutable(client,monkeypatch):
    from dataclasses import replace
    pid=data.create_project('x')['id'];monkeypatch.setattr(data,'settings',replace(data.settings,max_upload_bytes=10))
    with pytest.raises(ModelGuardError) as e:data.upload_dataset(pid,'x.csv',b'a\n'+b'1\n'*20)
    assert e.value.code=='file_too_large'
    d=data.upload_dataset(pid,'x.csv',b'a,b\n1,2\n');assert 'path' not in d
    r=data.get_dataset(pid,d['id']);Path(r['path']).write_text('a,b\n2,3\n')
    with pytest.raises(ModelGuardError):data.load_frame(pid,d['id'])

def test_target_confirmation_and_roles(client):
    pid=data.create_project('target test')['id'];df=frame();d=data.upload_dataset(pid,'x.csv',df.to_csv(index=False).encode());suggest=data.suggestions(pid,d['id']);assert suggest['suggestions'][0]['column']=='target';assert suggest['confirmation_required']
    with pytest.raises(ModelGuardError):data.configured(pid)
    c=data.configure(pid,{'dataset_id':d['id'],'target_column':'target','positive_class':'no','group_column':None});assert c['positive_class']=='no';assert c['task']=='classification';assert data.get_dataset(pid,d['id'])['target_column']=='target'
    d2=data.upload_dataset(pid,'batch.csv',df.to_csv(index=False).encode(),'production_batch')
    assert data.get_dataset(pid)['id']==d['id']
    with pytest.raises(ModelGuardError):data.configure(pid,{'dataset_id':d2['id'],'target_column':'target'})

@pytest.mark.parametrize('labels',[[0,1],['negative','positive'],[0,1,2],['low','mid','high']])
def test_label_metrics(labels):
    classes=np.asarray(labels);y=np.tile(classes,20);p=np.full((len(y),len(classes)),.05/(len(classes)-1))
    for i,v in enumerate(y):p[i,np.flatnonzero(classes==v)[0]]=.95
    r=classification_metrics(y,y,p,classes,classes[0]);assert r['pr_auc']>.99;assert r['macro_f1']==1;assert r['roc_auc']>.99;assert r['log_loss']<.1
    assert r['positive_class']==(classes[0] if len(classes)==2 else None)

def test_undefined_metrics():
    r=classification_metrics(np.array(['yes']*10),np.array(['yes']*10),np.tile([.1,.9],(10,1)),np.array(['no','yes']));assert r['roc_auc']['value'] is None;assert 'reason' in r['pr_auc'];assert regression_metrics([0,1],[0,1])['mape']['value'] is None

def test_regression_metrics():
    r=regression_metrics([1,2,4],[1.1,2.2,3.9]);assert set(['mae','rmse','median_absolute_error','r2','mape'])<=set(r);assert r['rmse']>0

@pytest.mark.parametrize('strategy',['random','group','chronological','group_chronological'])
def test_three_way_split(strategy):
    df=frame(500);cfg=ConfigureRequest(dataset_id='x',target_column='target',task='classification').model_dump()
    if 'group' in strategy:df['entity']=np.repeat(np.arange(100),5);cfg['group_column']='entity'
    if 'chronological' in strategy:df['time']=pd.date_range('2024-01-01',periods=len(df),freq='h');cfg['time_column']='time'
    m=split_indices(df,cfg);parts=[m[k] for k in ['train','validation','test']];assert not any(set(a)&set(b) for a,b in [(parts[0],parts[1]),(parts[0],parts[2]),(parts[1],parts[2])]);assert m==split_indices(df,cfg)
    if 'group' in strategy:assert not set(df.loc[parts[0],'entity'])&set(df.loc[parts[1]+parts[2],'entity'])
    if 'chronological' in strategy:assert df.loc[parts[0],'time'].max()<df.loc[parts[1],'time'].min()<df.loc[parts[2],'time'].min()

@pytest.mark.parametrize('mode',['random','group','time','both','replicas'])
def test_training_calibration_holdout(mode):
    y=np.tile([0,1],100);cfg=ConfigureRequest(dataset_id='x',target_column='target',task='classification').model_dump();context=pd.DataFrame({'source_row':np.arange(200)})
    if mode in {'group','both'}:context['group']=np.repeat(np.arange(50),4)
    if mode in {'time','both'}:context['time']=pd.date_range('2024',periods=200,freq='h')
    if mode=='replicas':context['source_row']=np.repeat(np.arange(100),2)
    a,b=training_holdout(y,cfg,context);assert not set(a)&set(b)
    if mode in {'group','both'}:assert not set(context.iloc[a]['group'])&set(context.iloc[b]['group'])
    if mode in {'time','both'}:assert context.iloc[a]['time'].max()<context.iloc[b]['time'].min()
    if mode=='replicas':assert not set(context.iloc[a]['source_row'])&set(context.iloc[b]['source_row'])

def test_optional_dependency_reasons(monkeypatch):
    import app.services.registry as reg
    real=reg.importlib.import_module
    def absent(name):
        if name in {'xgboost','lightgbm','catboost'}:raise ImportError()
        return real(name)
    monkeypatch.setattr(reg.importlib,'import_module',absent);models,skips=registry('classification',42,frame()[['signal','region']]);assert len(models)==5;assert len(skips)==3;assert all('not installed' in r['reason'] for r in skips)

class FixedProbability:
    def __init__(self,p,classes=('no','yes')):self.p=np.asarray(p);self.classes_=np.asarray(classes)
    def predict_proba(self,x):return np.tile(self.p,(len(x),1))
    def predict(self,x):return self.classes_[self.predict_proba(x).argmax(axis=1)]

def test_ensemble_class_alignment_serialization():
    pipe=WeightedEnsemble([('a',FixedProbability([.2,.8])),('b',FixedProbability([.7,.3],('yes','no')))],[.25,.75],'classification',['no','yes']);x=pd.DataFrame({'a':[1,2]});p=pipe.predict_proba(x);assert np.allclose(p,[[.275,.725]]*2);assert np.array_equal(pipe.predict(x),pickle.loads(pickle.dumps(pipe)).predict(x))
    with pytest.raises(ValueError):WeightedEnsemble(pipe.components,[-.1,1.1],'classification',['no','yes'])

def test_subset_sizes_pruning_and_dataset_specific_weights():
    rng=np.random.default_rng(1);n=100;truth=rng.normal(size=n);preds={str(i):truth+rng.normal(0,i+1,n) for i in range(5)};cfg=ConfigureRequest(dataset_id='x',target_column='y',task='regression',weight_trials=80).model_dump();cfg['primary_metric']='rmse'
    out=modeling.subset_search(preds,truth,cfg,None,list(preds));assert {len(r['requested_members']) for r in out}=={1,2,3,4};assert all(np.isclose(sum(r['weights']),1) and min(r['weights'])>=.05 for r in out)
    a=modeling.subset_search(preds,preds['0'],cfg,None,['0','1']);b=modeling.subset_search(preds,preds['1'],cfg,None,['0','1']);assert max(a,key=lambda r:r['utility'])['members']!=max(b,key=lambda r:r['utility'])['members']
    assert any(len(r['members'])<len(r['requested_members']) for r in out)

def test_fit_failure_isolation_and_validation_only(monkeypatch):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from app.services.registry import preprocessor
    df=frame(120);x=df[['signal','noise']];y=x.signal*2;cfg=ConfigureRequest(dataset_id='x',target_column='target',task='regression',max_trials=2,weight_trials=2).model_dump();cfg['primary_metric']='rmse';seen=[]
    def fit(args,seconds):
        name,est,xx,yy,*_=args;seen.extend(xx.index)
        if name=='decision_tree':raise ModelGuardError('Deliberate test fit failure')
        return Pipeline([('preprocessor',preprocessor(xx,'linear')),('model',Ridge())]).fit(xx,yy),{}
    monkeypatch.setattr(modeling,'fit_with_deadline',fit);pipe,result=modeling.fit_search(x.iloc[:60],y.iloc[:60],x.iloc[60:90],y.iloc[60:90],cfg);assert set(seen)<=set(range(60));assert not set(range(90,120))&set(seen);assert any(r['status']=='failed' for r in result['candidates']);assert pipe.predict(x.iloc[90:]).shape==(30,)

class SlowEstimator:
    def fit(self,x,y,**kw):time.sleep(10);return self

def test_fit_deadline_terminates():
    cfg=ConfigureRequest(dataset_id='x',target_column='target',task='regression').model_dump();x=frame(50)[['signal']];started=time.monotonic()
    with pytest.raises(ModelGuardError) as e:fit_with_deadline(('slow',SlowEstimator(),x,np.arange(50),cfg,None,None,'sigmoid'),.2)
    assert e.value.code=='fit_timeout';assert time.monotonic()-started<5

def test_frozen_baseline_and_safe_artifact(trained):
    pid,did,mid,df=trained;r,a=modeling.load_model(pid);assert r['id']==mid;assert not hasattr(a['pipeline'],'x_train');assert not isinstance(a['pipeline'],dict)
    with pytest.raises(ModelGuardError):modeling.train(pid)
    with pytest.raises(ModelGuardError):data.configure(pid,{'dataset_id':did,'target_column':'target'})
    assert set(a['manifest']['train']).isdisjoint(a['manifest']['test'])

@pytest.mark.parametrize('kind',['numeric_shift','numeric_scaling','noise','missingness','extreme_values','input_corruption'])
def test_twin_numeric_scenarios(trained,kind):
    pid,_,mid,_=trained;result=twin.simulate(pid,DataTwinRequest(scenarios=[{'type':kind,'feature':'signal','magnitude':.2}],intensities=[0.,1.]).model_dump());assert result['model_id']==mid;assert result['evaluation_partition']=='validation';assert result['results'][0]['relative_primary_drop']==0.;assert 'forecast' in result['results'][1]['label'];assert modeling.get_model(pid)['id']==mid

def test_unlabelled_never_computes_metrics(trained,monkeypatch):
    def forbidden(*a,**k):raise AssertionError('ground-truth metrics invoked')
    monkeypatch.setattr(twin,'evaluate',forbidden);r=twin.simulate(trained[0],DataTwinRequest(include_labels=False,scenarios=[{'type':'categorical_shift','feature':'region','magnitude':1,'parameters':{'distribution':{'north':1.}}}]).model_dump());assert r['results'][0]['performance_metrics'] is None;assert r['results'][0]['status']=='Insufficient Evidence'

def test_twin_combined_prevalence_and_scale_identity():
    x=frame(100)[['signal','region']];y=np.tile(['a','b'],50);rng=np.random.default_rng(2)
    out,_,_,definition=twin.perturb(x,y,{'type':'numeric_scaling','feature':'signal','parameters':{'factor':2.}},0,rng);assert out.equals(x);assert definition['effective_factor']==1
    out,ys,_,_=twin.perturb(x,y,{'type':'class_prevalence','magnitude':1,'parameters':{'distribution':{'a':1.,'b':0.}}},1,rng);assert set(ys)=={'a'}
    out,_,features,d=twin.perturb(x,y,{'type':'combined','parameters':{'scenarios':[{'type':'missingness','feature':'signal','magnitude':.3},{'type':'input_corruption','feature':'region','magnitude':.5}]}},1,rng);assert set(features)==set(x);assert '__CORRUPTED__' in set(out.region)
    dist=twin.distribution(np.arange(10.),'regression');assert set(dist)=={'mean','variance','quantiles'}

@pytest.mark.parametrize('case,expected',[('exact','Compatible'),('extra','Compatible with warnings'),('critical','Incompatible'),('null','Incompatible'),('optional','Compatible with warnings'),('type','Incompatible'),('unseen','Compatible with warnings')])
def test_schema_cases(trained,case,expected):
    pid,_,_,df=trained;_,art=modeling.load_model(pid);cfg=art['config'];x=df.copy()
    if case=='extra':x['new_feature']=1
    if case=='critical':x=x.drop(columns='signal')
    if case=='null':x['signal']=np.nan
    if case=='optional':cfg['validated_optional_features']=['noise'];x=x.drop(columns='noise')
    if case=='type':x['signal']='text'
    if case=='unseen':x['region']='unseen'
    r=drift.schema_check(x,cfg);assert r['status']==expected
    if expected=='Incompatible':
        with pytest.raises(ModelGuardError):drift.model_inputs(x,cfg,r)
    else:assert list(drift.model_inputs(x,cfg,r))==cfg['feature_columns']

def test_renamed_schema_mapping_and_labels(trained):
    pid,_,mid,df=trained;x=df.rename(columns={'signal':'signal_renamed','target':'actual'});batch=drift.upload_batch(pid,'batch.csv',x.to_csv(index=False).encode());assert batch['model_id']==mid;assert batch['schema_compatibility']['status']=='Incompatible';bid=batch['id'];r=drift.confirm_mapping(pid,bid,{'feature_mapping':{'signal_renamed':'signal'},'target_column':'actual','confirm_labels':True});assert r['label_availability']=='confirmed';assert r['schema_compatibility']['prediction_allowed'];mapped,meta,y=drift.batch_frame(pid,bid);assert y.equals(df.target)

def test_label_detection_and_delayed_attachment(trained):
    pid,_,_,df=trained;batch=drift.upload_batch(pid,'batch.csv',df.to_csv(index=False).encode());assert batch['labels']['exact_target_present'];_,_,y=drift.batch_frame(pid,batch['id']);assert y is None
    drift.confirm_mapping(pid,batch['id'],{'confirm_labels':True});assert drift.batch_frame(pid,batch['id'])[2] is not None
    no=df.drop(columns='target');no['Predicted_Churn']='yes';b=drift.upload_batch(pid,'unlabelled.csv',no.to_csv(index=False).encode());assert b['labels']['availability']=='absent'
    with pytest.raises(ModelGuardError):drift.confirm_mapping(pid,b['id'],{'confirm_labels':True,'target_column':'Predicted_Churn'})
    labels=df[['record_id','target']].sample(frac=.5,random_state=4);res=drift.attach_labels(pid,b['id'],'labels.csv',labels.to_csv(index=False).encode());assert res['label_availability']=='partial';_,_,y=drift.batch_frame(pid,b['id']);assert y.notna().sum()==len(labels)
    bad=pd.concat([labels.iloc[:1]]*2)
    with pytest.raises(ModelGuardError):drift.attach_labels(pid,b['id'],'labels.csv',bad.to_csv(index=False).encode())

def test_monitoring_persistence_and_prediction_drift(trained):
    pid,_,_,df=trained;x=df.drop(columns='target');x['signal']+=10
    b1=drift.upload_batch(pid,'b1.csv',x.to_csv(index=False).encode(),batch_timestamp='2025-01-01');r1=drift.analyze(pid,{'dataset_id':b1['id']});assert r1['material_drift'];assert not r1['persistent_drift'];assert r1['performance'] is None;assert r1['prediction_drift'] is not None;assert r1['confidence_drift'] is not None
    drift.analyze(pid,{'dataset_id':b1['id']});assert len(drift.history(pid)['batches'])==1
    x2=x.assign(signal=x['signal']+.1)
    b2=drift.upload_batch(pid,'b2.csv',x2.to_csv(index=False).encode(),batch_timestamp='2025-01-02');r2=drift.analyze(pid,{'dataset_id':b2['id']});assert r2['persistent_drift'];assert r2['recommendation']['action']=='Evaluate retraining';assert 'unconfirmed' in r2['recommendation']['reason']
    assert r2['transformed_feature_drift']['available']

def test_incompatible_monitoring_blocks_predict(trained,monkeypatch):
    pid,_,_,df=trained;b=drift.upload_batch(pid,'bad.csv',df.drop(columns='signal').to_csv(index=False).encode());r=drift.analyze(pid,{'dataset_id':b['id']});assert r['prediction_drift'] is None;assert r['recommendation']['action']=='Full model redevelopment recommended';assert r['raw_feature_drift']['signal']['missingness_change']>.9

@pytest.mark.parametrize('schema,material,persistent,performance,calibration,expected',[
 ('Incompatible',False,False,None,False,'Full model redevelopment recommended'),
 ('Compatible',False,False,None,False,'Continue monitoring'),
 ('Compatible',True,False,None,False,'Investigate drift'),
 ('Compatible',True,True,None,False,'Evaluate retraining'),
 ('Compatible',True,True,{'relative_primary_drop':.3,'confirmed_degradation':True,'production_metrics':{},'reference_metrics':{}},False,'Retraining recommended'),
 ('Compatible',False,False,{'relative_primary_drop':0.,'confirmed_degradation':False,'production_metrics':{},'reference_metrics':{}},True,'Recalibrate probabilities'),
 ('Compatible',False,False,{'relative_primary_drop':0.,'confirmed_degradation':False,'production_metrics':{},'reference_metrics':{}},False,'Continue using current model'),
 ('Compatible',False,False,{'relative_primary_drop':None,'confirmed_degradation':False},False,'Continue monitoring')])
def test_retraining_levels(schema,material,persistent,performance,calibration,expected):assert drift.action_level(schema,material,persistent,performance,calibration)[0]==expected

def test_actual_labelled_monitoring_and_small_batch(trained):
    pid,_,_,df=trained;b=drift.upload_batch(pid,'small.csv',df.iloc[:12].to_csv(index=False).encode());drift.confirm_mapping(pid,b['id'],{'confirm_labels':True});r=drift.analyze(pid,{'dataset_id':b['id']});assert r['performance']['label_count']==12;assert not r['performance']['confirmed_degradation'];assert r['performance']['drop_95pct_ci'] is None

def test_audit_report_and_seven_dimensions(trained):
    pid,_,mid,_=trained;a=audit.audit(pid);assert len(a['findings'])>=15;assert all({'measured_value','method','affected_features','threshold_used','status','severity','supporting_evidence','recommendation','limitations','evidence_type'}<=set(f) for f in a['findings']);r=reliability.report(pid);assert len(r['reliability']['dimensions'])==7;assert r['reliability']['production_stability']['status']=='Not Evaluated';assert r['reliability']['dimensions'][2]['status']=='Pass';assert set(r['prioritized_recommendations'])=={'Immediate','Before deployment','Monitoring','Future improvement'};assert r['reproducibility']['schema_fingerprint']==exporting.card(pid,mid)['schema_fingerprint']

def test_scenario_status_accumulates(trained):
    pid,_,_,_=trained
    twin.simulate(pid,DataTwinRequest(scenarios=[{'type':'missingness','feature':'signal','magnitude':1.}]).model_dump());twin.simulate(pid,DataTwinRequest(scenarios=[{'type':'numeric_shift','feature':'signal','magnitude':0.}]).model_dump());r=reliability.assessment(pid);dims={d['dimension']:d for d in r['dimensions']};assert len(dims['Noise and Missingness Robustness']['evidence'])==1;assert dims['Distribution-Shift Robustness']['status']=='Pass'

def test_export_checksums_no_rows_and_standalone_inference(trained,tmp_path):
    pid,_,mid,df=trained;content,name=exporting.bundle(pid,mid);folder=tmp_path/'bundle';folder.mkdir()
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        assert z.testzip() is None;names=z.namelist();assert not any('evaluation' in n or 'training_reference' in n or 'original.csv' in n for n in names);checks=json.loads(z.read('checksums.json'))
        for n,h in checks.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
        z.extractall(folder)
    assert 'app/services/inference.py' in names
    df.iloc[:10].to_csv(tmp_path/'input.csv',index=False);env=os.environ.copy();env.pop('PYTHONPATH',None);r=subprocess.run([os.sys.executable,str(folder/'inference_example.py'),str(tmp_path/'input.csv')],cwd=folder,env=env,capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stderr
    expected=modeling.load_model(pid)[1]['pipeline'].predict(df.iloc[:10]).tolist();assert json.loads(r.stdout)==expected
    import joblib
    pipe=joblib.load(folder/'model_pipeline.joblib');seen=set()
    def inspect(obj):
        if id(obj) in seen:return
        seen.add(id(obj));assert not isinstance(obj,pd.DataFrame)
        if isinstance(obj,dict):
            for k,v in obj.items():assert k not in {'x_train','x_test','y_train','y_test'};inspect(v)
        elif isinstance(obj,(list,tuple)):
            for v in obj:inspect(v)
        elif hasattr(obj,'__dict__') and not isinstance(obj,type):inspect(vars(obj))
    inspect(pipe)

def test_prediction_and_download_endpoints(trained):
    from fastapi.testclient import TestClient
    from app.main import app
    pid,_,mid,df=trained
    with TestClient(app) as client:
        path=f'/projects/{pid}/models/{mid}';r=client.post(path+'/predict',json={'records':df.iloc[:3].to_dict('records')});assert r.status_code==200;assert len(r.json()['predictions'])==3
        assert client.post(path+'/predict',json={'records':[{'region':'north'}]}).status_code==422
        for endpoint in ['card','report','export']:
            r=client.get(path+'/'+endpoint);assert r.status_code==200;assert 'attachment' in r.headers['content-disposition']
        assert client.get(path+'/card?format=markdown').status_code==200
        assert client.post(f'/projects/{pid}/configure',json={'bad':'value'}).json()['error']['code']=='validation_error'

def test_migrations_preserve_existing_rows(client):
    from app.migrations import migrate
    pid=data.create_project('kept')['id'];migrate();migrate();assert data.get_project(pid)['name']=='kept';assert one('SELECT COUNT(*) AS n FROM schema_migrations')['n']==1

def test_job_restart_recovery_and_failure(client):
    from app import jobs
    pid=data.create_project('jobs')['id'];jid=uid();insert('jobs',{'id':jid,'project_id':pid,'kind':'test','status':'running','created_at':now(),'updated_at':now()});jobs.recover();assert jobs.get(jid)['status']=='interrupted';assert jobs.get(jid)['error']['code']=='worker_restart'
    job=jobs.submit(pid,'fail',data.configured,pid)
    for _ in range(200):
        result=jobs.get(job['job_id'])
        if result['status'] in {'completed','failed'}:break
        time.sleep(.05)
    assert result['status']=='failed';assert result['error']['code']=='project_not_configured';assert '/workspace' not in json.dumps(result)
