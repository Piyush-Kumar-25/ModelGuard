import numpy as np
from sklearn import metrics as m

def unavailable(reason):return {'value':None,'reason':reason}
def value(v):return float(v) if isinstance(v,(int,float,np.number)) and np.isfinite(v) else None
def measured(fn):
    try:
        v=float(fn());return v if np.isfinite(v) else unavailable('Non-finite metric for this sample')
    except (ValueError,ZeroDivisionError):return unavailable('Metric undefined for the available labels or predictions')

def classification_metrics(y,pred,proba=None,classes=None,positive_class=None):
    y=np.asarray(y);pred=np.asarray(pred);classes=np.asarray(classes if classes is not None else sorted(set(y)|set(pred)));binary=len(classes)==2
    positive_class=classes[-1] if binary and positive_class is None else positive_class
    r={'accuracy':float(m.accuracy_score(y,pred)),'error_rate':float(np.mean(y!=pred)),'confusion_matrix':m.confusion_matrix(y,pred,labels=classes).tolist(),'class_order':classes.tolist(),'positive_class':positive_class if binary else None}
    for avg in ['macro','weighted']:
        for name,fn in [('f1',m.f1_score),('precision',m.precision_score),('recall',m.recall_score)]:
            r[f'{avg}_{name}' if name=='f1' else f'{name}_{avg}']=measured(lambda fn=fn,avg=avg:fn(y,pred,labels=classes,average=avg,zero_division=0))
    r['f1']=measured(lambda:m.f1_score(y,pred,pos_label=positive_class,average='binary',zero_division=0)) if binary else r['macro_f1']
    if proba is None:
        r.update({k:unavailable('Probabilities unavailable') for k in ['roc_auc','pr_auc','pr_auc_macro','pr_auc_weighted','log_loss','brier_score','ece']});return r
    p=np.clip(np.asarray(proba),1e-12,1);p=p/p.sum(axis=1,keepdims=True);onehot=(y[:,None]==classes[None,:]).astype(int)
    if np.any(onehot.sum(axis=1)!=1):
        r.update({k:unavailable('Unknown ground-truth class') for k in ['roc_auc','pr_auc','pr_auc_macro','pr_auc_weighted','log_loss','brier_score','ece']});return r
    complete=len(np.unique(y))==len(classes)
    order=np.argsort(classes)
    r['log_loss']=measured(lambda:m.log_loss(y,p[:,order],labels=classes[order]))
    if binary:
        pos=int(np.flatnonzero(classes==positive_class)[0]);truth=onehot[:,pos]
        r['roc_auc']=measured(lambda:m.roc_auc_score(truth,p[:,pos])) if complete else unavailable('Both classes required')
        r['pr_auc']=measured(lambda:m.average_precision_score(truth,p[:,pos])) if complete else unavailable('Both classes required')
        r['brier_score']=float(np.mean((p[:,pos]-truth)**2))
    else:
        r['roc_auc']=measured(lambda:m.roc_auc_score(y,p[:,order],labels=classes[order],multi_class='ovr',average='macro')) if complete else unavailable('All classes required')
        r['brier_score']=float(np.mean(np.sum((p-onehot)**2,axis=1)))
    for avg in ['macro','weighted']:
        r['pr_auc_'+avg]=measured(lambda avg=avg:m.average_precision_score(onehot,p,average=avg)) if complete else unavailable('All classes required')
    if not binary:r['pr_auc']=r['pr_auc_macro']
    confidence=p.max(axis=1);correct=(classes[p.argmax(axis=1)]==y).astype(float);ece=0
    for a,b in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        mask=(confidence>a)&(confidence<=b)
        if mask.any():ece+=mask.mean()*abs(correct[mask].mean()-confidence[mask].mean())
    r['ece']=float(ece);return r

def regression_metrics(y,pred):
    y=np.asarray(y,dtype=float);pred=np.asarray(pred,dtype=float)
    return {'mae':measured(lambda:m.mean_absolute_error(y,pred)),'rmse':measured(lambda:np.sqrt(m.mean_squared_error(y,pred))),'median_absolute_error':measured(lambda:m.median_absolute_error(y,pred)),'r2':measured(lambda:m.r2_score(y,pred,force_finite=False)) if len(y)>1 else unavailable('At least two labels required'),'mape':measured(lambda:np.mean(np.abs((y-pred)/y))) if np.all(np.abs(y)>1e-12) else unavailable('Zero or near-zero targets')}

def evaluate(pipe,x,y,task,positive_class=None):
    pred=pipe.predict(x);p=pipe.predict_proba(x) if task=='classification' else None
    return (classification_metrics(y,pred,p,pipe.classes_,positive_class) if task=='classification' else regression_metrics(y,pred)),pred,p

def utility(metrics,key):
    v=value(metrics.get(key));return None if v is None else -v if key in {'mae','rmse','median_absolute_error','mape','log_loss','brier_score','ece','error_rate'} else v

def relative_drop(before,after,key):
    a=utility(before,key);b=utility(after,key)
    return None if a is None or b is None else (a-b)/max(abs(a),.01)
