"""Inference-only classes. No training rows or evaluation partitions are stored."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class NativeFrame(BaseEstimator,TransformerMixin):
    def __init__(self,categorical=()):self.categorical=categorical
    def fit(self,x,y=None):return self
    def transform(self,x):
        out=x.copy()
        for c in out:
            if c in self.categorical:out[c]=out[c].fillna('__MISSING__').astype(str)
            else:out[c]=pd.to_numeric(out[c],errors='coerce').astype(float)
        return out

class EncodedPredictor(BaseEstimator):
    def __init__(self,estimator,classes):self.estimator=estimator;self.classes=classes;self.classes_=np.asarray(classes)
    def __sklearn_is_fitted__(self):return True
    def predict_proba(self,x):
        raw=self.estimator.predict_proba(x);order=np.asarray(self.estimator.classes_,dtype=int);p=np.zeros((len(x),len(self.classes_)));p[:,order]=raw;return p
    def predict(self,x):return self.classes_[self.predict_proba(x).argmax(axis=1)]

class WeightedEnsemble(BaseEstimator):
    def __init__(self,components,weights,task='classification',classes=None,threshold=None,positive_class=None):
        self.components=components;self.weights=weights;self.task=task;self.classes=classes;self.threshold=threshold;self.positive_class=positive_class
        if task=='classification':self.classes_=np.asarray(classes)
        w=np.asarray(weights,dtype=float)
        if len(w)!=len(components) or len(w)==0 or np.any(w<0) or not np.isfinite(w).all() or not np.isclose(w.sum(),1):raise ValueError('Nonnegative normalized weights required')
    def __sklearn_is_fitted__(self):return True
    def component_predictions(self,x):
        if self.task=='regression':return np.asarray([p.predict(x) for _,p in self.components])
        result=[]
        for _,p in self.components:
            raw=p.predict_proba(x);aligned=np.zeros((len(x),len(self.classes_)))
            for j,c in enumerate(p.classes_):
                loc=np.flatnonzero(self.classes_==c)
                if len(loc)!=1:raise ValueError('Class-order mismatch')
                aligned[:,loc[0]]=raw[:,j]
            result.append(aligned)
        return np.asarray(result)
    def predict_proba(self,x):
        if self.task!='classification':raise AttributeError('Regression has no probabilities')
        return np.tensordot(self.weights,self.component_predictions(x),axes=(0,0))
    def predict(self,x):
        if self.task=='regression':return np.tensordot(self.weights,self.component_predictions(x),axes=(0,0))
        p=self.predict_proba(x)
        if len(self.classes_)==2 and self.threshold is not None:
            pos=int(np.flatnonzero(self.classes_==self.positive_class)[0]);return np.where(p[:,pos]>=self.threshold,self.classes_[pos],self.classes_[1-pos])
        return self.classes_[p.argmax(axis=1)]
