import importlib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,ExtraTreesClassifier,ExtraTreesRegressor,HistGradientBoostingClassifier,HistGradientBoostingRegressor
from .inference import NativeFrame

NAMES=['linear','decision_tree','random_forest','extra_trees','hist_gradient_boosting','xgboost','lightgbm','catboost']

def registry(task,seed,x):
    classification=task=='classification';out={};skips=[]
    basic={
      'linear':LogisticRegression(max_iter=600,random_state=seed) if classification else Ridge(),
      'decision_tree':(DecisionTreeClassifier if classification else DecisionTreeRegressor)(max_depth=6,min_samples_leaf=4,random_state=seed),
      'random_forest':(RandomForestClassifier if classification else RandomForestRegressor)(n_estimators=80,max_depth=12,min_samples_leaf=2,n_jobs=1,random_state=seed),
      'extra_trees':(ExtraTreesClassifier if classification else ExtraTreesRegressor)(n_estimators=80,max_depth=12,min_samples_leaf=2,n_jobs=1,random_state=seed),
      'hist_gradient_boosting':(HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor)(max_iter=100,max_leaf_nodes=15,early_stopping=False,random_state=seed)}
    out.update(basic)
    for name,module,cls in [('xgboost','xgboost','XGBClassifier' if classification else 'XGBRegressor'),('lightgbm','lightgbm','LGBMClassifier' if classification else 'LGBMRegressor'),('catboost','catboost','CatBoostClassifier' if classification else 'CatBoostRegressor')]:
        try:
            lib=importlib.import_module(module)
            args={'n_estimators':100,'max_depth':5,'random_state':seed,'n_jobs':1}
            if name=='xgboost':args.update(tree_method='hist',verbosity=0)
            if name=='lightgbm':args.update(verbosity=-1,deterministic=True,force_col_wise=True)
            if name=='catboost':args={'iterations':100,'depth':5,'random_seed':seed,'thread_count':1,'verbose':False,'allow_writing_files':False,'cat_features':list(x.select_dtypes(exclude=np.number).columns)}
            out[name]=getattr(lib,cls)(**args)
        except ImportError:skips.append({'model':name,'status':'skipped','reason':f'Optional dependency {module} is not installed'})
        except Exception:skips.append({'model':name,'status':'skipped','reason':'Optional dependency cannot initialize on this platform'})
    return out,skips

def preprocessor(x,name):
    num=list(x.select_dtypes(include=np.number).columns);cat=[c for c in x if c not in num]
    if name=='catboost':return NativeFrame(tuple(cat))
    parts=[]
    if num:parts.append(('numeric',Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]),num))
    if cat:
        enc=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1,encoded_missing_value=-1) if name=='hist_gradient_boosting' else OneHotEncoder(handle_unknown='ignore',min_frequency=2,max_categories=100,sparse_output=True)
        parts.append(('categorical',Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value='__MISSING__',keep_empty_features=True)),('encode',enc)]),cat))
    return ColumnTransformer(parts,sparse_threshold=0 if name=='hist_gradient_boosting' else 1.)
