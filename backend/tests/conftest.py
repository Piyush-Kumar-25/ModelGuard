import os,shutil
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest


def use_storage(root):
    os.environ['MODELGUARD_DATA_DIR']=str(root)
    os.environ['MODELGUARD_DB_PATH']=str(root/'modelguard.db')
    import app.config as config
    cfg=replace(config.settings,data_dir=root,db_path=root/'modelguard.db')
    config.settings=cfg
    import app.database as database
    import app.services.data as data
    import app.main as main
    database.settings=data.settings=main.settings=cfg
    database.init_db()
    return cfg

def frame(n=400):
    rng=np.random.default_rng(12);x=rng.normal(size=n);region=rng.choice(['north','south','west'],n);y=np.where(x+(region=='north')*.8+rng.normal(0,.7,n)>0,'yes','no')
    return pd.DataFrame({'record_id':np.arange(n),'signal':x,'noise':rng.normal(size=n),'region':region,'target':y})

@pytest.fixture
def client(tmp_path):
    use_storage(tmp_path/'data')
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:yield c

@pytest.fixture(scope='session')
def trained_template(tmp_path_factory):
    root=tmp_path_factory.mktemp('template')/'data';use_storage(root)
    from app.services import data,modeling
    p=data.create_project('Synthetic test fixture');df=frame();d=data.upload_dataset(p['id'],'reference.csv',df.to_csv(index=False).encode());cfg=data.configure(p['id'],{'dataset_id':d['id'],'target_column':'target','record_id_column':'record_id','max_trials':2,'time_budget_seconds':90,'fit_timeout_seconds':30,'weight_trials':4,'optional_features':['noise'],'feature_availability_confirmed':True})
    m=modeling.train(p['id']);return root,p['id'],d['id'],m['id'],df

@pytest.fixture
def trained(tmp_path,trained_template):
    original,pid,did,mid,df=trained_template;root=tmp_path/'data';shutil.copytree(original,root);use_storage(root)
    from app.database import connect
    with connect() as con:
        for table,columns in [('datasets',['path']),('models',['path','state_path'])]:
            for c in columns:con.execute(f'UPDATE {table} SET {c}=REPLACE({c},?,?)',(str(original),str(root)))
    return pid,did,mid,df.copy()
