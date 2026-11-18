import hashlib,io,json,platform,zipfile
from importlib.metadata import version,PackageNotFoundError
from pathlib import Path
from ..database import dump
from ..errors import ModelGuardError
from .modeling import load_model,get_model
from .data import load_frame
from .reliability import report

CORE=['numpy','pandas','scikit-learn','scipy','joblib','threadpoolctl']
def versions():
    out={'python':platform.python_version()}
    for name in CORE+['xgboost','lightgbm','catboost']:
        try:out[name]=version(name)
        except PackageNotFoundError:pass
    return out

def card(pid,mid):
    m,a=load_model(pid,mid);cfg=a['config'];_,data=load_frame(pid,m['dataset_id'])
    return {'model_id':mid,'version':m['version'],'status':m['status'],'task':cfg['task'],'target':cfg['target_column'],'positive_class':cfg.get('positive_class'),'positive_class_source':cfg.get('positive_class_source'),'selected_members':[n for n,_ in a['pipeline'].components],'weights':a['pipeline'].weights,'class_order':a['pipeline'].classes_.tolist() if cfg['task']=='classification' else None,'target_mapping':{str(i):c for i,c in enumerate(a['pipeline'].classes_.tolist())} if cfg['task']=='classification' else None,'decision_threshold':a['pipeline'].threshold,'feature_columns':cfg['feature_columns'],'feature_schema':cfg['schema'],'validated_optional_features':cfg.get('validated_optional_features',[]),'schema_version':1,'schema_fingerprint':cfg['schema_fingerprint'],'dataset_id':m['dataset_id'],'dataset_sha256':data['sha256'],'metrics':m['metrics'],'evaluation':m['details'].get('evaluation'),'library_versions':m['details'].get('library_versions',versions()),'intended_use':'Human-reviewed deployment candidate for the documented tabular task. No automatic deployment.','limitations':['Joblib must only be loaded from trusted sources.','Requires a compatible Python and library environment; not universally portable.','No raw training, test or production rows belong in the inference artifact.','Finite-sample metrics do not guarantee future performance.']}

def markdown(title,data):return '# '+title+'\n\n```json\n'+json.dumps(data,indent=2,ensure_ascii=False,default=str)+'\n```\n'

def bundle(pid,mid):
    m,a=load_model(pid,mid);c=card(pid,mid);r=report(pid,model_id=mid);cfg=a['config'];selected=[n for n,_ in a['pipeline'].components];deps=CORE+[n for n in ['xgboost','lightgbm','catboost'] if n in selected];v=c['library_versions'];files={
      'model_pipeline.joblib':Path(m['path']).read_bytes(),
      'ensemble_config.json':dump({'members':selected,'weights':a['pipeline'].weights}).encode(),
      'feature_schema.json':dump({'columns':cfg['feature_columns'],'schema':cfg['schema'],'validated_optional_features':cfg.get('validated_optional_features',[]),'joint_optional_fallback_validated':cfg.get('joint_optional_fallback_validated',False),'fingerprint':cfg['schema_fingerprint']}).encode(),
      'target_mapping.json':dump(c['target_mapping']).encode(),
      'decision_threshold.json':dump({'threshold':a['pipeline'].threshold,'positive_class':cfg.get('positive_class')}).encode(),
      'model_metadata.json':dump({'model_id':mid,'version':m['version'],'task':cfg['task'],'library_versions':v,'trusted_source_only':True,'deployment_status':'candidate'}).encode(),
      'requirements.txt':('\n'.join(f'{n}=={v[n]}' for n in deps)+'\n').encode(),
      'model_card.json':dump(c).encode(),'model_card.md':markdown('ModelGuard Model Card',c).encode(),
      'reliability_report.json':dump(r).encode(),
      'app/__init__.py':b'', 'app/services/__init__.py':b'',
      'app/services/inference.py':Path(__file__).with_name('inference.py').read_bytes(),
      'inference_example.py':INFERENCE.encode()}
    # Export every file needed by custom inference classes, and no evaluation state or source rows.
    files['checksums.json']=dump({n:hashlib.sha256(v).hexdigest() for n,v in files.items()}).encode();stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
        for name,content in files.items():z.writestr(name,content)
    return stream.getvalue(),f'ModelGuard_Model_Bundle_v{m["version"]}.zip'

INFERENCE='''"""Run: python inference_example.py input.csv. Load trusted bundles only.
Install requirements.txt with the Python version documented in model_metadata.json.
Checksums detect accidental corruption, not malicious replacement of a whole bundle.
"""
import hashlib,json,sys
from pathlib import Path
import joblib,numpy as np,pandas as pd
root=Path(__file__).resolve().parent
checks=json.loads((root/'checksums.json').read_text())
for name,expected in checks.items():
    path=(root/name).resolve()
    if root not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Bundle integrity failure')
spec=json.loads((root/'feature_schema.json').read_text())
x=pd.read_csv(sys.argv[1]);optional=set(spec['validated_optional_features'])
missing=[c for c in spec['columns'] if c not in x or x[c].isna().all()]
if len(set(missing)&optional)>1 and not spec.get('joint_optional_fallback_validated',False):raise ValueError('Joint optional fallback was not validated')
for c in spec['columns']:
    if c not in x:
        if c not in optional:raise ValueError('Missing required feature: '+c)
        x[c]=np.nan
    if x[c].isna().all() and c not in optional:raise ValueError('Fully null required feature: '+c)
    if x[c].notna().any():
        numeric=pd.api.types.is_numeric_dtype(x[c])
        if (spec['schema'][c]['kind']=='numeric')!=numeric:raise ValueError('Incompatible type: '+c)
        if numeric and np.isinf(x[c].dropna()).any():raise ValueError('Nonfinite feature: '+c)
pipe=joblib.load(root/'model_pipeline.joblib')
print(json.dumps(pipe.predict(x[spec['columns']]).tolist()))
'''
