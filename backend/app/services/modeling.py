from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..database import all_rows, dump, insert, load, now, one, project_dir, uid
from ..errors import ModelGuardError, require
from .data import configured


def _python(v: Any) -> Any:
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.ndarray): return v.tolist()
    return v


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    nums=list(x.select_dtypes(include=np.number).columns)
    cats=[c for c in x.columns if c not in nums]
    transformers=[]
    if nums: transformers.append(("numeric",Pipeline([("imputer",SimpleImputer(strategy="median",add_indicator=True)),("scaler",StandardScaler())]),nums))
    if cats: transformers.append(("categorical",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",min_frequency=2))]),cats))
    return ColumnTransformer(transformers, remainder="drop")


def _candidates(task: str, seed: int):
    if task == "classification":
        return {
            "logistic_regression": LogisticRegression(max_iter=1500,class_weight="balanced",random_state=seed),
            "decision_tree": DecisionTreeClassifier(max_depth=8,class_weight="balanced",random_state=seed),
            "random_forest": RandomForestClassifier(n_estimators=150,max_depth=12,class_weight="balanced",random_state=seed,n_jobs=-1),
            "gradient_boosting": GradientBoostingClassifier(random_state=seed),
        }
    return {
        "ridge_regression": Ridge(alpha=1.0),
        "decision_tree": DecisionTreeRegressor(max_depth=8,random_state=seed),
        "random_forest": RandomForestRegressor(n_estimators=150,max_depth=12,random_state=seed,n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=seed),
    }


def classification_metrics(y, pred, proba=None, classes=None) -> dict[str, Any]:
    m={"accuracy":accuracy_score(y,pred),"precision_weighted":precision_score(y,pred,average="weighted",zero_division=0),"recall_weighted":recall_score(y,pred,average="weighted",zero_division=0),"f1":f1_score(y,pred,average="binary",zero_division=0) if len(np.unique(y))==2 else None,"macro_f1":f1_score(y,pred,average="macro",zero_division=0),"weighted_f1":f1_score(y,pred,average="weighted",zero_division=0),"confusion_matrix":confusion_matrix(y,pred).tolist()}
    if proba is not None:
        try:
            if proba.shape[1]==2:
                positive=(y==classes[1]).astype(int)
                m.update({"roc_auc":roc_auc_score(positive,proba[:,1]),"pr_auc":average_precision_score(positive,proba[:,1]),"brier_score":brier_score_loss(positive,proba[:,1])})
            else:
                binary=label_binarize(y,classes=classes)
                m.update({"roc_auc":roc_auc_score(binary,proba,multi_class="ovr",average="weighted"),"pr_auc":average_precision_score(binary,proba,average="weighted")})
        except ValueError: pass
    return {k:_python(v) for k,v in m.items()}


def regression_metrics(y, pred) -> dict[str, Any]:
    m={"mae":mean_absolute_error(y,pred),"rmse":math.sqrt(mean_squared_error(y,pred)),"r2":r2_score(y,pred)}
    if np.all(np.asarray(y)!=0): m["mape"]=float(np.mean(np.abs((np.asarray(y)-pred)/np.asarray(y))))
    return {k:_python(v) for k,v in m.items()}


def evaluate(pipeline, x, y, task):
    pred=pipeline.predict(x)
    if task=="classification":
        proba=pipeline.predict_proba(x) if hasattr(pipeline,"predict_proba") else None
        return classification_metrics(y,pred,proba,getattr(pipeline,"classes_",None)),pred,proba
    return regression_metrics(y,pred),pred,None


def train(project_id: str) -> dict[str, Any]:
    cfg,df,meta=configured(project_id); target=cfg["target_column"]; seed=cfg["random_seed"]
    clean=df.dropna(subset=[target]).copy()
    if len(clean)<30: raise ModelGuardError("At least 30 labelled rows are required", "insufficient_data")
    x=clean.drop(columns=[target]); y=clean[target]
    if cfg.get("time_column"):
        order=pd.to_datetime(x[cfg["time_column"]],errors="coerce").sort_values().index
        if pd.to_datetime(x.loc[order,cfg["time_column"]],errors="coerce").isna().any(): raise ModelGuardError("Time column contains invalid timestamps", "invalid_time_column")
        cut=int(len(order)*(1-cfg["test_size"])); train_idx,test_idx=order[:cut],order[cut:]
        x_train,x_test,y_train,y_test=x.loc[train_idx],x.loc[test_idx],y.loc[train_idx],y.loc[test_idx]; split="chronological"
    else:
        strat=y if cfg["task"]=="classification" and y.value_counts().min()>=2 else None
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=cfg["test_size"],random_state=seed,stratify=strat); split="stratified_random" if strat is not None else "random"
    if cfg["task"]=="classification" and y_train.value_counts().min()<2: raise ModelGuardError("Training split has a class with fewer than two samples", "insufficient_class_samples")
    prep=build_preprocessor(x_train); results=[]; fitted={}
    metric=cfg["primary_metric"]; scoring={"macro_f1":"f1_macro","weighted_f1":"f1_weighted","f1":"f1","roc_auc":"roc_auc","pr_auc":"average_precision","rmse":"neg_root_mean_squared_error","mae":"neg_mean_absolute_error","r2":"r2"}[metric]
    min_class=int(y_train.value_counts().min()) if cfg["task"]=="classification" else 5
    folds=min(5,max(2,min_class),max(2,len(x_train)//20))
    cv=(StratifiedKFold(folds,shuffle=True,random_state=seed) if cfg["task"]=="classification" else KFold(folds,shuffle=True,random_state=seed))
    for name,est in _candidates(cfg["task"],seed).items():
        pipe=Pipeline([("preprocessor",clone(prep)),("model",est)])
        try:
            scores=cross_val_score(pipe,x_train,y_train,scoring=scoring,cv=cv,n_jobs=1,error_score="raise")
            pipe.fit(x_train,y_train); metrics,pred,proba=evaluate(pipe,x_test,y_test,cfg["task"])
            cv_values=(-scores if scoring.startswith("neg_") else scores)
            results.append({"model":name,"status":"completed","cross_validation":{"folds":folds,"metric":metric,"mean":float(cv_values.mean()),"std":float(cv_values.std())},"test_metrics":metrics})
            fitted[name]=pipe
        except Exception as exc:
            results.append({"model":name,"status":"failed","reason":str(exc)[:300]})
    good=[r for r in results if r["status"]=="completed"]
    if not good: raise ModelGuardError("All candidate models failed", "training_failed", 422, results)
    reverse=metric not in {"rmse","mae"}; selected=sorted(good,key=lambda r:r["cross_validation"]["mean"],reverse=reverse)[0]
    name=selected["model"]; pipeline=fitted[name]; final_metrics,pred,proba=evaluate(pipeline,x_test,y_test,cfg["task"])
    prior=one("SELECT MAX(version) AS v FROM models WHERE project_id=?",(project_id,)); version=int(prior["v"] or 0)+1; mid=uid()
    folder=project_dir(project_id)/"models"/mid; folder.mkdir(parents=True,exist_ok=False); path=folder/"pipeline.joblib"
    artifact={"pipeline":pipeline,"x_train":x_train,"y_train":y_train,"x_test":x_test,"y_test":y_test,"predictions":pred,"probabilities":proba,"config":cfg,"feature_columns":list(x.columns),"dataset_id":meta["id"]}
    joblib.dump(artifact,path)
    details={"description":"The best-performing baseline under the current dataset, metric and evaluation setup.","evaluation":{"split":split,"test_fraction":cfg["test_size"],"random_seed":seed,"cross_validation_folds":folds,"selection_metric":metric},"candidates":results,"limitations":["Performance is estimated from one held-out split and finite cross-validation folds.","Candidate search is intentionally bounded and is not universal model optimization."]}
    insert("models",{"id":mid,"project_id":project_id,"dataset_id":meta["id"],"version":version,"path":str(path),"task":cfg["task"],"selected_name":name,"metrics_json":dump(final_metrics),"details_json":dump(details),"created_at":now()})
    return {"model_id":mid,"version":version,"selected_model":name,"description":details["description"],"metrics":final_metrics,**details}


def latest_model(project_id: str):
    row=require(one("SELECT * FROM models WHERE project_id=? ORDER BY version DESC LIMIT 1",(project_id,)),"No trained model found")
    try: artifact=joblib.load(row["path"])
    except Exception: raise ModelGuardError("Saved model artifact is unavailable or invalid","artifact_invalid",500) from None
    row["metrics"]=load(row["metrics_json"]); row["details"]=load(row["details_json"])
    return row,artifact


def model_results(project_id: str) -> dict[str, Any]:
    rows=all_rows("SELECT * FROM models WHERE project_id=? ORDER BY version DESC",(project_id,))
    return {"models":[{"id":r["id"],"version":r["version"],"task":r["task"],"selected_model":r["selected_name"],"metrics":load(r["metrics_json"]),"details":load(r["details_json"]),"created_at":r["created_at"]} for r in rows]}
