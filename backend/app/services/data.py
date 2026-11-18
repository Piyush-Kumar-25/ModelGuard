from __future__ import annotations
import csv, hashlib, io, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from ..config import settings
from ..database import dump,insert,load,now,one,project_dir,uid,update
from ..errors import ModelGuardError,require
from ..schemas import ConfigureRequest


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def public(row):return {k:v for k,v in row.items() if k not in {'path','state_path','schema_json','config_json','mapping_json','compatibility_json'}}
def get_project(pid):return require(one('SELECT * FROM projects WHERE id=?',(pid,)),'Project not found')
def get_dataset(pid,did=None):
    if did:r=one('SELECT * FROM datasets WHERE id=? AND project_id=?',(did,pid))
    else:r=one("SELECT * FROM datasets WHERE project_id=? AND role='reference' ORDER BY created_at DESC LIMIT 1",(pid,))
    return require(r,'Dataset not found')
def load_frame(pid,did=None):
    r=get_dataset(pid,did);p=Path(r['path'])
    if not p.is_file() or sha(p)!=r['sha256']:raise ModelGuardError('Dataset integrity check failed','artifact_invalid',500)
    return pd.read_csv(p),r

def infer_task(y):return 'classification' if not pd.api.types.is_numeric_dtype(y) or y.nunique()<=max(20,int(len(y)*.05)) else 'regression'
def create_project(name,description=None):
    if not name.strip():raise ModelGuardError('Project name cannot be blank')
    r=dict(id=uid(),name=name.strip(),description=description,config_json='{}',created_at=now());insert('projects',r);project_dir(r['id']);return {**public(r),'config':{}}
def _safe_filename(name):
    if Path(name).name!=name or not re.fullmatch(r'[A-Za-z0-9._ -]+',name or '') or not name.lower().endswith('.csv'):raise ModelGuardError('Only safe .csv filenames are accepted','invalid_filename')
    return name

def read_csv(content):
    if not content:raise ModelGuardError('Empty CSV','empty_dataset')
    if len(content)>settings.max_upload_bytes:raise ModelGuardError('File exceeds upload limit','file_too_large',413)
    try:
        text=content.decode('utf-8-sig')
        if '\0' in text:raise ValueError()
        rows=csv.reader(io.StringIO(text),strict=True);header=next(rows)
        if not header or any(not c.strip() for c in header):raise ValueError()
        if len(set(header))!=len(header):raise ModelGuardError('Duplicate columns','duplicate_columns')
        if len(header)>500:raise ModelGuardError('At most 500 columns supported','resource_limit',413)
        count=0
        for row in rows:
            if not row:continue
            if len(row)!=len(header):raise ValueError()
            count+=1
            if count>settings.max_csv_rows:raise ModelGuardError('Row limit exceeded','too_many_rows',413)
        df=pd.read_csv(io.StringIO(text))
        if df.empty:raise ValueError()
        nums=df.select_dtypes(include=np.number)
        if np.isinf(nums.to_numpy()).any():raise ValueError()
        return df
    except ModelGuardError:raise
    except Exception:raise ModelGuardError('Malformed or empty UTF-8 CSV','malformed_csv') from None

def profile(df):
    out={}
    for c in df:
        s=df[c];num=pd.api.types.is_numeric_dtype(s)
        r={'dtype':str(s.dtype),'kind':'numeric' if num else 'categorical','missing_fraction':float(s.isna().mean()),'unique':int(s.nunique())}
        if num:
            r.update({k:(float(v) if pd.notna(v) else None) for k,v in {'min':s.min(),'max':s.max(),'mean':s.mean(),'std':s.std()}.items()})
        else:r['categories']=s.dropna().astype(str).unique()[:1000].tolist();r['categories_truncated']=s.nunique()>1000
        out[str(c)]=r
    return out

def schema_fingerprint(columns,schema):return hashlib.sha256(json.dumps({'columns':columns,'schema':schema},sort_keys=True,allow_nan=False).encode()).hexdigest()
def upload_dataset(pid,filename,content,role='reference',*,connection=None,parsed_frame=None,**extra):
    get_project(pid);safe=_safe_filename(filename);df=read_csv(content) if parsed_frame is None else parsed_frame
    if role not in {'reference','production_batch','production_labels'}:raise ModelGuardError('Invalid dataset role')
    did=uid();folder=project_dir(pid)/'datasets'/did;folder.mkdir(parents=True);p=folder/'original.csv';p.write_bytes(content);p.chmod(0o600)
    schema=profile(df)
    r=dict(id=did,project_id=pid,original_name=safe,path=str(p),sha256=hashlib.sha256(content).hexdigest(),rows=len(df),columns=len(df.columns),schema_json=dump(schema),created_at=now(),role=role,**extra)
    if connection is None:
        insert('datasets',r)
    else:
        connection.execute(f"INSERT INTO datasets ({','.join(r)}) VALUES ({','.join('?' for _ in r)})",tuple(r.values()))
    return {**public(r),'schema':schema}

def summarize(pid,did):
    df,r=load_frame(pid,did);return {'dataset_id':did,'rows':len(df),'columns':list(df.columns),'inferred_schema':profile(df),'role':r['role'],'sha256':r['sha256']}
def suggestions(pid,did):
    df,r=load_frame(pid,did)
    if r['role']!='reference':raise ModelGuardError('Target is fixed for production batches')
    ranked=[]
    for i,c in enumerate(df):
        s=df[c];name=bool(re.search(r'(^|_)(target|label|class|outcome|churn|fraud|default|price)($|_)',c,re.I));n=int(s.nunique());miss=float(s.isna().mean())
        score=4*name+(i==len(df.columns)-1)+int(2<=n<=20)-2*miss
        if re.match(r'(?i)predict|probability',c):score=-100
        ranked.append({'column':c,'score':score,'suggested_task':infer_task(s),'unique':n,'missing_fraction':miss,'dtype':str(s.dtype),'position':i,'target_name_match':name})
    return {'suggestions':sorted(ranked,key=lambda r:r['score'],reverse=True),'confirmation_required':True}

def configure(pid,payload):
    get_project(pid)
    if one("SELECT id FROM jobs WHERE project_id=? AND status IN ('queued','running')",(pid,)):raise ModelGuardError('Wait for the active project job before changing configuration','job_conflict',409)
    cfg=ConfigureRequest(**payload).model_dump();df,r=load_frame(pid,cfg['dataset_id']);target=cfg['target_column']
    if r['role']!='reference':raise ModelGuardError('Configure a reference dataset')
    if one("SELECT id FROM models WHERE project_id=? AND dataset_id=? AND status!='legacy'",(pid,r['id'])):raise ModelGuardError('Configuration is frozen; create a new reference version or challenger','configuration_frozen',409)
    if target not in df or df[target].nunique()<2 or re.match(r'(?i)predict|probability',target):raise ModelGuardError('Invalid ground-truth target','invalid_target')
    if cfg['task']=='auto':cfg['task']=infer_task(df[target])
    if cfg['task']=='regression' and not pd.api.types.is_numeric_dtype(df[target]):raise ModelGuardError('Regression target must be numeric')
    if cfg['task']=='classification' and df[target].nunique()>100:raise ModelGuardError('At most 100 target classes supported','resource_limit')
    for key in ['time_column','group_column','record_id_column']:
        c=cfg[key]
        if c and (c not in df or c==target or df[c].isna().any()):raise ModelGuardError('Invalid '+key)
    if cfg['record_id_column'] and df[cfg['record_id_column']].duplicated().any():raise ModelGuardError('Record IDs must be unique')
    for key in ['excluded_features','confirmed_leakage','optional_features']:
        if any(c not in df or c==target for c in cfg[key]):raise ModelGuardError('Invalid '+key)
    exclusions=set(cfg['excluded_features']+cfg['confirmed_leakage']+[cfg[k] for k in ['time_column','group_column','record_id_column'] if cfg[k]])
    cfg['feature_columns']=[c for c in df if c!=target and c not in exclusions]
    if not cfg['feature_columns']:raise ModelGuardError('At least one usable feature required')
    if not set(cfg['optional_features'])<=set(cfg['feature_columns']):raise ModelGuardError('Optional features must be selected inputs')
    allowed={'classification':{'macro_f1','weighted_f1','f1','roc_auc','pr_auc','accuracy','log_loss'},'regression':{'rmse','mae','r2','median_absolute_error'}}
    cfg['primary_metric']=cfg['primary_metric'] or ('macro_f1' if cfg['task']=='classification' else 'rmse')
    if cfg['primary_metric'] not in allowed[cfg['task']]:raise ModelGuardError('Invalid primary metric','invalid_metric')
    classes=sorted(df[target].dropna().unique().tolist())
    if cfg['task']=='classification' and len(classes)==2:
        if cfg['positive_class'] is None:cfg['positive_class']=classes[-1];cfg['positive_class_source']='deterministic sorted last class'
        elif cfg['positive_class'] not in classes:raise ModelGuardError('Positive class absent from target')
        else:cfg['positive_class_source']='user configured'
    else:cfg['positive_class']=None
    update('UPDATE projects SET config_json=? WHERE id=?',(dump(cfg),pid))
    update("UPDATE datasets SET target_column=?,label_availability='confirmed',record_id_column=? WHERE id=?",(target,cfg['record_id_column'],r['id']))
    return cfg

def configured(pid):
    cfg=load(get_project(pid)['config_json'],{})
    if not cfg.get('dataset_id'):raise ModelGuardError('Confirm target and configuration first','project_not_configured')
    df,r=load_frame(pid,cfg['dataset_id']);return cfg,df,r
