import copy
import numpy as np
import pandas as pd
import pytest
from app.database import dump,load,one,update
from app.errors import ModelGuardError
from app.schemas import ConfigureRequest,ShieldRequest
from app.services import shield,modeling,experiments,reliability,exporting
from app.services.inference import WeightedEnsemble
from app.services.metrics import regression_metrics

class RegionPredictor:
    classes_=np.asarray(['no','yes'])
    def predict(self,x):return np.where(x['z'].to_numpy()>0,'yes','no')
    def predict_proba(self,x):
        p=np.where(self.predict(x)=='yes',.9,.1);return np.column_stack([1-p,p])

class RegressionPredictor:
    def predict(self,x):return x['z'].to_numpy()

@pytest.mark.parametrize('task,confirmation',[('classification',True),('classification',False),('regression',True)])
def test_discovery_confirmation_separation_and_intervals(monkeypatch,task,confirmation):
    n=160;z=np.tile([-1.,1.],n//2);group=np.repeat([0.,1.],n//2);x=pd.DataFrame({'z':z,'group':group});xt=x.copy();xt.index=np.arange(1000,1000+n);pipe=RegionPredictor() if task=='classification' else RegressionPredictor();y=pipe.predict(x)
    if task=='classification':y=np.where(group==1,np.where(y=='yes','no','yes'),y);yt=y.copy() if confirmation else pipe.predict(xt)
    else:y=y+group*5;yt=y.copy()
    cfg=ConfigureRequest(dataset_id='fixture',target_column='target',task=task).model_dump();cfg['positive_class']='yes' if task=='classification' else None
    art={'pipeline':pipe,'config':cfg,'x_train':x,'x_validation':x,'y_validation':pd.Series(y),'x_test':xt,'y_test':pd.Series(yt,index=xt.index)};model={'id':'fixture'}
    monkeypatch.setattr(shield,'load_model',lambda pid:(model,art));monkeypatch.setattr(shield,'analyses',lambda *args:[]);monkeypatch.setattr(shield,'save_analysis',lambda pid,kind,r:r)
    seen=[];original=shield.local_evidence
    def spy(pipe,x,y,*args):seen.append(set(x.index));return original(pipe,x,y,*args)
    monkeypatch.setattr(shield,'local_evidence',spy);r=shield.analyze('fixture',ShieldRequest(min_samples=10,bootstrap_iterations=30).model_dump());assert r['failure_regions'];assert all(s<=set(range(n)) or s<=set(xt.index) for s in seen)
    for region in r['failure_regions']:
        if confirmation:assert region['confirmed'];assert region['confirmation']['drop_ci'][0]>0;assert all(v['interval'] is not None for v in region['confirmation']['metric_intervals'].values())
        else:assert not region['confirmed'];assert region['status']=='Insufficient Evidence'
        assert 'group' in region['rule'];assert region['discovery']['sample_count']>0

def test_shield_no_variation_and_freezes_rules(trained):
    pid,_,_,_=trained;params=ShieldRequest(min_samples=10,bootstrap_iterations=20).model_dump();first=shield.analyze(pid,params);second=shield.analyze(pid,{**params,'max_depth':5});assert first==second

@pytest.mark.parametrize('kind',['increased_missingness','added_noise','distribution_shift','extreme_values','input_corruption','unavailable_features'])
def test_stress_frozen_incumbent(trained,kind):
    pid,_,mid,_=trained;r=experiments.run(pid,{'experiment_type':kind,'parameters':{'feature':'signal','magnitude':.2},'random_seed':4});assert r['category']=='stress';assert r['incumbent_unchanged'];assert modeling.get_model(pid)['id']==mid

def test_challenger_threshold_finalize_approve_promote(trained):
    pid,_,mid,_=trained;r=experiments.run(pid,{'experiment_type':'threshold_adjustment','parameters':{'threshold':.5},'random_seed':4});cid=r['id'];assert r['status']=='experimental';assert modeling.get_model(pid)['id']==mid
    with pytest.raises(ModelGuardError):experiments.lifecycle(pid,cid,'promote')
    with pytest.raises(ModelGuardError):experiments.lifecycle(pid,cid,'approve')
    done=experiments.lifecycle(pid,cid,'finalize');assert done['details']['final_comparison']['same_evaluation_records'];assert done['details']['final_comparison']['evaluation_passed'];assert modeling.get_model(pid)['id']==mid
    experiments.lifecycle(pid,cid,'approve');experiments.lifecycle(pid,cid,'promote');assert modeling.get_model(pid)['id']==cid
    challenger=experiments.run(pid,{'experiment_type':'threshold_adjustment','parameters':{'threshold':.5},'random_seed':4})
    with pytest.raises(ModelGuardError) as e:experiments.lifecycle(pid,challenger['id'],'finalize')
    assert e.value.code=='test_budget_consumed'

def test_degraded_challenger_cannot_be_approved(trained):
    pid,_,mid,_=trained;r=experiments.run(pid,{'experiment_type':'threshold_adjustment','parameters':{'threshold':.99},'random_seed':4});cid=r['id'];done=experiments.lifecycle(pid,cid,'finalize');assert not done['details']['final_comparison']['evaluation_passed']
    with pytest.raises(ModelGuardError):experiments.lifecycle(pid,cid,'approve')
    assert modeling.get_model(pid)['id']==mid;assert experiments.lifecycle(pid,cid,'reject')['status']=='rejected'

def test_actual_reselection_and_private_snapshot(trained):
    pid,_,mid,_=trained;r=experiments.run(pid,{'experiment_type':'model_retraining','parameters':{'models':['linear','decision_tree']},'random_seed':8});assert r['category']=='challenger';assert r['details']['search']['mode']=='Auto';assert r['details']['search']['ensemble_comparisons'];row,art=modeling.load_model(pid,r['id']);assert len(art['x_train'])>0;assert modeling.get_model(pid)['id']==mid
    with pytest.raises(ModelGuardError) as exc:exporting.bundle(pid,r['id'])
    assert exc.value.code=='export_not_eligible'

@pytest.mark.parametrize('kind',['subgroup_reweighting','subgroup_oversampling','class_weighting','reduced_training_size','feature_removal','probability_recalibration','alternative_model','alternative_ensemble'])
def test_training_interventions_not_aliases(trained,monkeypatch,kind):
    pid,_,mid,_=trained;parent,base=modeling.load_model(pid);captured={};region={'region_id':'fixed','clauses':[{'feature':'signal','kind':'numeric','fill':0.,'operator':'>','value':0.}]}
    realanalyses=experiments.analyses
    monkeypatch.setattr(experiments,'analyses',lambda pid,k,mid=None:[{'failure_regions':[{**region,'confirmed':True}]}] if k=='modelshield' else realanalyses(pid,k,mid))
    def fake(x,y,xv,yv,cfg,weights,context,allowed,method):
        captured.update(rows=len(x),columns=list(x),weights=weights,context=context,allowed=allowed,method=method)
        # Fit a real tiny model using the supplied intervention, with a predictable test runtime.
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from app.services.registry import preprocessor
        pipe=Pipeline([('preprocessor',preprocessor(x,'linear')),('model',LogisticRegression())]).fit(x,y,**({'model__sample_weight':weights} if weights is not None else {}));return WeightedEnsemble([('linear',pipe)],[1.],'classification',pipe.classes_.tolist(),None,cfg.get('positive_class')),{'mode':'test spy with real fit'}
    monkeypatch.setattr(experiments,'fit_search',fake)
    params={'subgroup_reweighting':{'region_id':'fixed','factor':2},'subgroup_oversampling':{'region_id':'fixed','factor':2},'class_weighting':{},'reduced_training_size':{'fraction':.5},'feature_removal':{'features':['noise']},'probability_recalibration':{'method':'isotonic'},'alternative_model':{'models':['linear']},'alternative_ensemble':{'models':['linear','decision_tree']}}[kind]
    r=experiments.run(pid,{'experiment_type':kind,'parameters':params,'random_seed':4});n=len(base['x_train'])
    if kind=='subgroup_reweighting':assert captured['rows']==n;assert set(captured['weights'])=={1.,2.}
    if kind=='subgroup_oversampling':assert captured['rows']>n;assert captured['weights'] is None;assert captured['context'].source_row.duplicated().any()
    if kind=='class_weighting':assert captured['weights'] is not None;assert captured['rows']==n
    if kind=='reduced_training_size':assert captured['rows']==int(n*.5)
    if kind=='feature_removal':assert 'noise' not in captured['columns']
    if kind=='probability_recalibration':assert captured['method']=='isotonic'
    if kind.startswith('alternative'):assert captured['allowed']==params['models']
    assert r['comparison']['same_evaluation_records'];assert r['comparison']['subgroups'];assert modeling.get_model(pid)['id']==mid


def test_additional_data_independence(trained,monkeypatch):
    from app.services import data
    pid,_,mid,df=trained;new=df.iloc[:30].copy();new['record_id']+=10000;new['signal']+=.123456;extra=pd.concat([df.iloc[:10],new]);d=data.upload_dataset(pid,'additional.csv',extra.to_csv(index=False).encode());base=modeling.load_model(pid)[1];seen={}
    def fake(x,y,xv,yv,cfg,weights,context,allowed,method):seen['rows']=len(x);return copy.deepcopy(base['pipeline']),{'mode':'test inspection'}
    monkeypatch.setattr(experiments,'fit_search',fake);r=experiments.run(pid,{'experiment_type':'additional_labelled_data','parameters':{'dataset_id':d['id']},'random_seed':4});assert seen['rows']==len(base['x_train'])+30;assert len(modeling.load_model(pid,r['id'])[1]['x_train'])==seen['rows']
