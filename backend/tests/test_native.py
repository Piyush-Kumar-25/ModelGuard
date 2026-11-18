import pickle
import numpy as np
import pandas as pd
import pytest
from app.services.registry import registry
from app.services.training import fit_candidate
from app.schemas import ConfigureRequest

@pytest.mark.parametrize('name',['xgboost','lightgbm','catboost','hist_gradient_boosting'])
@pytest.mark.parametrize('task',['binary','multiclass','regression'])
def test_native_adapters(name,task):
    rng=np.random.default_rng(20);n=200;x=pd.DataFrame({'a':rng.normal(size=n),'cat':rng.choice(['red','blue'],n)});y=pd.Series(np.where(x.a>0,'yes','no')) if task=='binary' else pd.Series(np.where(x.a<-.4,'left',np.where(x.a>.4,'right','center'))) if task=='multiclass' else x.a*2+rng.normal(0,.1,n)
    cfg=ConfigureRequest(dataset_id='fixture',target_column='target',task='regression' if task=='regression' else 'classification').model_dump();models,skipped=registry(cfg['task'],42,x)
    if name not in models:pytest.skip(next(r['reason'] for r in skipped if r['model']==name))
    pipe,meta=fit_candidate(name,models[name],x,y,cfg)
    assert meta['early_stopping']['enabled'];restored=pickle.loads(pickle.dumps(pipe));assert np.array_equal(pipe.predict(x),restored.predict(x))
    if task!='regression':assert np.allclose(pipe.predict_proba(x).sum(axis=1),1);assert set(pipe.predict(x))<=set(y)
