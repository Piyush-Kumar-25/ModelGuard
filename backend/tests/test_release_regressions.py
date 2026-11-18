"""Release-gate tests runnable with unittest as well as pytest."""
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


def use_storage(root):
    from app import config, database, main
    from app.services import data
    os.environ['MODELGUARD_DATA_DIR']=str(root)
    os.environ['MODELGUARD_DB_PATH']=str(root/'modelguard.db')
    settings=replace(config.settings,data_dir=root,db_path=root/'modelguard.db')
    config.settings=database.settings=data.settings=main.settings=settings


def reset_storage(settings,environment):
    from app import config, database, main
    from app.services import data
    config.settings=database.settings=data.settings=main.settings=settings
    for key,previous in environment.items():
        if previous is None:os.environ.pop(key,None)
        else:os.environ[key]=previous


class BackendReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.storage=tempfile.TemporaryDirectory(prefix='modelguard-release-')
        cls.addClassCleanup(cls.storage.cleanup)
        from app.config import settings
        cls.addClassCleanup(reset_storage,settings,{k:os.environ.get(k) for k in ('MODELGUARD_DATA_DIR','MODELGUARD_DB_PATH')})
        cls.template=Path(cls.storage.name)/'template'
        use_storage(cls.template)
        from app.database import init_db
        from app.services import data,modeling
        init_db()
        rng=np.random.default_rng(12);n=320
        x=rng.normal(size=n)
        cls.frame=pd.DataFrame({'record_id':np.arange(n),'signal':x,
                                'noise':rng.normal(size=n),'target':np.where(x+rng.normal(0,.4,n)>0,'yes','no')})
        cls.pid=data.create_project('Release fixture')['id']
        ds=data.upload_dataset(cls.pid,'reference.csv',cls.frame.to_csv(index=False).encode())
        data.configure(cls.pid,{'dataset_id':ds['id'],'target_column':'target','record_id_column':'record_id',
                                 'max_trials':1,'weight_trials':2,'fit_timeout_seconds':30})
        cls.mid=modeling.train(cls.pid)['id']

    def setUp(self):
        from app.database import connect
        from app.main import app
        from fastapi.testclient import TestClient
        work=tempfile.TemporaryDirectory(dir=self.storage.name)
        self.addCleanup(work.cleanup)
        self.root=Path(work.name)/'data'
        shutil.copytree(self.template,self.root)
        use_storage(self.root)
        with connect() as con:
            for table,columns in [('datasets',['path']),('models',['path','state_path'])]:
                for column in columns:
                    con.execute(f'UPDATE {table} SET {column}=REPLACE({column},?,?)',(str(self.template),str(self.root)))
        self.client=self.enterContext(TestClient(app))
        self.base=f'/projects/{self.pid}'
        self.batches=self.base+'/monitoring/batches'

    def upload(self,path,df,**fields):
        return self.client.post(path,data=fields,files={'file':('input.csv',df.to_csv(index=False),'text/csv')})

    def _challenger(self):
        from app.services import experiments
        return experiments.run(self.pid,{'experiment_type':'threshold_adjustment','parameters':{'threshold':.5}})['id']

    def test_rejected_and_experimental_models_cannot_be_exported(self):
        cid=self._challenger();path=self.base+f'/models/{cid}'
        r=self.client.get(path+'/export')
        self.assertEqual(r.status_code,409,r.text)
        self.assertEqual(r.json()['error']['code'],'export_not_eligible')
        self.assertEqual(self.client.post(path+'/lifecycle',json={'action':'reject'}).status_code,200)
        r=self.client.get(path+'/export')
        self.assertEqual(r.status_code,409,r.text)
        card=self.client.get(path+'/card').json()
        self.assertFalse(card['deployment_export_allowed'])
        self.assertIn('not eligible',card['intended_use'])

    def test_approved_model_bundle_runs_outside_backend(self):
        cid=self._challenger();path=self.base+f'/models/{cid}'
        for action in ('finalize','approve'):
            response=self.client.post(path+'/lifecycle',json={'action':action})
            self.assertEqual(response.status_code,200,response.text)
        response=self.client.get(path+'/export')
        self.assertEqual(response.status_code,200)
        folder=self.root/'bundle'
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            self.assertIsNone(z.testzip())
            self.assertNotIn('training_reference.csv',z.namelist())
            self.assertTrue(json.loads(z.read('model_card.json'))['deployment_export_allowed'])
            z.extractall(folder)
        features=self.frame.iloc[:8].drop(columns='target')
        features.to_csv(folder/'inputs.csv',index=False)
        local=subprocess.run([sys.executable,'inference_example.py','inputs.csv'],cwd=folder,
                             capture_output=True,text=True,timeout=30)
        self.assertEqual(local.returncode,0,local.stderr)
        r=self.client.post(path+'/predict',json={'records':features.to_dict('records')})
        self.assertEqual(r.status_code,200,r.text)
        self.assertEqual(json.loads(local.stdout),r.json()['predictions'])
        self.assertEqual(self.client.post(path+'/lifecycle',json={'action':'promote'}).status_code,200)
        self.assertEqual(self.client.get(path+'/export').status_code,200)
        self.assertEqual(self.client.get(self.base+f'/models/{self.mid}/export').status_code,409)

    def test_duplicate_batches_are_idempotent_and_not_independent_evidence(self):
        from app.services import data,drift
        x=self.frame.iloc[:80].drop(columns='target').assign(signal=lambda d:d.signal+10)
        r=self.upload(self.batches,x,batch_timestamp='2027-01-01')
        self.assertEqual(r.status_code,201,r.text)
        batch=r.json()
        retry=self.upload(self.batches,x,batch_timestamp='2027-02-01')
        self.assertEqual(retry.status_code,200,retry.text)
        self.assertEqual(retry.json()['id'],batch['id'])
        self.assertEqual(retry.json()['batch_timestamp'],batch['batch_timestamp'])
        self.assertTrue(retry.json()['duplicate_upload'])
        self.assertFalse(drift.analyze(self.pid,{'dataset_id':batch['id']})['persistent_drift'])
        # Simulate an older database that had stored the same CSV as another batch.
        old=data.upload_dataset(self.pid,'legacy.csv',x.to_csv(index=False).encode(),'production_batch',
                                model_id=self.mid,batch_timestamp='2027-02-01T00:00:00+00:00',
                                record_id_column='record_id',label_availability='absent')
        self.assertFalse(drift.analyze(self.pid,{'dataset_id':old['id']})['persistent_drift'])
        self.assertEqual(len(drift.history(self.pid,self.mid)['batches']),1)
        independent=self.upload(self.batches,x.assign(signal=x.signal+.2),batch_timestamp='2027-03-01')
        self.assertEqual(independent.status_code,201,independent.text)
        current=drift.analyze(self.pid,{'dataset_id':independent.json()['id']})
        self.assertTrue(current['persistent_drift'])
        self.assertEqual(len(drift.history(self.pid,self.mid)['batches']),2)
        later=data.upload_dataset(self.pid,'legacy-later.csv',x.to_csv(index=False).encode(),'production_batch',
                                  model_id=self.mid,batch_timestamp='2027-04-01T00:00:00+00:00',
                                  record_id_column='record_id',label_availability='absent')
        late=drift.analyze(self.pid,{'dataset_id':later['id']})
        self.assertEqual(late['duplicate_of_batch_id'],batch['id'])
        self.assertFalse(late['persistent_drift'])

    def test_delayed_labels_merge_without_silent_overwrite(self):
        from app.services import drift,reliability
        x=self.frame[['record_id','signal','noise']].iloc[:50]
        b=self.upload(self.batches,x)
        self.assertEqual(b.status_code,201,b.text)
        bid=b.json()['id'];labels=f'{self.batches}/{bid}/labels'
        drift.analyze(self.pid,{'dataset_id':bid})
        y=self.frame[['record_id','target']].iloc[:50]
        first=self.upload(labels,y.iloc[:25])
        self.assertEqual(first.json()['total_labelled_records'],25)
        self.assertTrue(drift.history(self.pid,self.mid)['batches'][0]['analysis_stale'])
        self.assertEqual(reliability.assessment(self.pid)['production_stability']['status'],'Insufficient Evidence')
        second=self.upload(labels,y.iloc[25:].rename(columns={'target':'actual'}),target_column='actual')
        self.assertEqual(second.status_code,201,second.text)
        self.assertEqual(second.json()['total_labelled_records'],50)
        self.assertEqual(second.json()['label_availability'],'confirmed')
        self.assertEqual(self.upload(labels,y.iloc[20:30]).json()['new_labels'],0)
        contradictory=y.iloc[:1].assign(target=lambda d:d.target.map({'yes':'no','no':'yes'}))
        conflict=self.upload(labels,contradictory)
        self.assertEqual(conflict.status_code,409,conflict.text)
        self.assertEqual(conflict.json()['error']['code'],'label_conflict')
        changed=drift.analyze(self.pid,{'dataset_id':bid})
        self.assertEqual(changed['performance']['label_count'],50)
        self.assertFalse(changed['analysis_stale'])

    def test_inline_and_attached_labels_coexist(self):
        from app.services import drift
        original=self.frame.iloc[:40].copy()
        partial=original.rename(columns={'target':'actual'})
        partial.loc[20:,'actual']=None
        bid=self.upload(self.batches,partial).json()['id'];path=f'{self.batches}/{bid}'
        r=self.client.post(path+'/mapping',json={'target_column':'actual','confirm_labels':True})
        self.assertEqual(r.status_code,200,r.text)
        r=self.upload(path+'/labels',original[['record_id','target']].iloc[20:])
        self.assertEqual(r.status_code,201,r.text)
        self.assertEqual(r.json()['new_labels'],20)
        self.assertEqual(r.json()['total_labelled_records'],40)
        self.assertEqual(drift.batch_frame(self.pid,bid)[2].tolist(),original.target.tolist())

    def test_concurrent_partial_label_uploads_do_not_lose_coverage(self):
        from concurrent.futures import ThreadPoolExecutor
        from app.services import drift
        batches=self.frame.iloc[:40]
        bid=self.upload(self.batches,batches.drop(columns='target')).json()['id']
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(drift.attach_labels,self.pid,bid,'partial.csv',
                                 batch[['record_id','target']].to_csv(index=False).encode())
                     for batch in (batches.iloc[:20],batches.iloc[20:])]
            self.assertEqual(sorted(f.result().get('total_labelled_records') for f in futures),[20,40])
        self.assertTrue(drift.batch_frame(self.pid,bid)[2].notna().all())

    def test_small_labelled_sample_does_not_recommend_using_current_model(self):
        from app.services import drift
        bid=self.upload(self.batches,self.frame.iloc[:15]).json()['id'];path=f'{self.batches}/{bid}'
        r=self.client.post(path+'/mapping',json={'confirm_labels':True})
        self.assertEqual(r.status_code,200,r.text)
        report=drift.analyze(self.pid,{'dataset_id':bid})
        self.assertEqual(report['performance']['label_count'],15)
        self.assertNotEqual(report['recommendation']['action'],'Continue using current model')
        self.assertIn('Too few confirmed labels',report['recommendation']['reason'])

    def test_report_does_not_mix_experiments_from_another_model_version(self):
        from app.database import dump,insert,now,uid
        from app.services import reliability
        cid=self._challenger()
        insert('experiments',{'id':uid(),'project_id':self.pid,'experiment_type':'added_noise','params_json':'{}',
                              'result_json':dump({'model_id':'unrelated-version','category':'stress'}),
                              'random_seed':42,'created_at':now()})
        experiments=reliability.report(self.pid,model_id=self.mid)['experimentlab']
        self.assertEqual([e['result']['id'] for e in experiments],[cid])


if __name__=='__main__':
    unittest.main()
