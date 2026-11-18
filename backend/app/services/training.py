import copy, inspect, multiprocessing as mp, time
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from threadpoolctl import threadpool_limits
from .registry import preprocessor
from .inference import EncodedPredictor
from .splitting import training_holdout
from ..errors import ModelGuardError


def _fit_base(name,est,x,y,cfg,weights,context):
    prep=preprocessor(x,name);early={'enabled':False,'reason':'Estimator has a fixed bounded fit'}
    if name in {'xgboost','lightgbm','catboost','hist_gradient_boosting'}:
        a,b=training_holdout(y,cfg,context);probe=copy.deepcopy(est);pp=preprocessor(x,name);xa=pp.fit_transform(x.iloc[a]);xb=pp.transform(x.iloc[b]);kw={}
        if weights is not None:kw['sample_weight']=weights[a]
        if name=='xgboost':probe.set_params(early_stopping_rounds=10);kw['eval_set']=[(xb,y[b])];kw['verbose']=False
        elif name=='lightgbm':
            import lightgbm as lgb
            kw['callbacks']=[lgb.early_stopping(10,verbose=False)]
            if 'eval_X' in inspect.signature(probe.fit).parameters:kw.update(eval_X=xb,eval_y=y[b])
            else:kw['eval_set']=[(xb,y[b])]
        elif name=='catboost':kw.update(eval_set=(xb,y[b]),early_stopping_rounds=10,verbose=False)
        else:probe.set_params(early_stopping=True,n_iter_no_change=10);kw.update(X_val=xb,y_val=y[b])
        probe.fit(xa,y[a],**kw)
        if name=='xgboost':steps=probe.best_iteration+1;est.set_params(n_estimators=max(1,steps))
        elif name=='lightgbm':steps=probe.best_iteration_ or 100;est.set_params(n_estimators=steps)
        elif name=='catboost':steps=max(1,probe.get_best_iteration()+1);est.set_params(iterations=steps)
        else:steps=probe.n_iter_;est.set_params(max_iter=max(1,steps),early_stopping=False)
        early={'enabled':True,'selected_iterations':int(steps),'partition':'training only'}
    pipe=Pipeline([('preprocessor',prep),('model',est)])
    pipe.fit(x,y,**({'model__sample_weight':weights} if weights is not None else {}));return pipe,early

def fit_candidate(name,est,x,y,cfg,weights=None,context=None,calibration_method='sigmoid'):
    with threadpool_limits(limits=1):
        weights=np.asarray(weights,dtype=float) if weights is not None else None
        if cfg['task']=='classification':
            encoder=LabelEncoder().fit(y);encoded=encoder.transform(y);a,b=training_holdout(encoded,cfg,context)
            pipe,early=_fit_base(name,est,x.iloc[a],encoded[a],cfg,weights[a] if weights is not None else None,context.iloc[a].reset_index(drop=True) if context is not None else None)
            calibrated=CalibratedClassifierCV(FrozenEstimator(pipe),method=calibration_method)
            calibrated.fit(x.iloc[b],encoded[b],sample_weight=weights[b] if weights is not None else None)
            return EncodedPredictor(calibrated,encoder.classes_.tolist()),{'calibration':{'method':calibration_method,'partition':'training-only independent holdout','fit_rows':len(a),'calibration_rows':len(b),'disjoint':not bool(set(a)&set(b))},'early_stopping':early}
        pipe,early=_fit_base(name,est,x,np.asarray(y),cfg,weights,context);return pipe,{'calibration':{'method':'not applicable'},'early_stopping':early}

def _child(conn,args):
    try:conn.send(('ok',fit_candidate(*args)))
    except Exception as exc:conn.send(('error',{'type':type(exc).__name__,'reason':exc.message if isinstance(exc,ModelGuardError) else 'Candidate fit failed; check sample support and estimator compatibility'}))
    finally:conn.close()

def fit_with_deadline(args,seconds):
    ctx=mp.get_context('spawn');parent,child=ctx.Pipe(False);p=ctx.Process(target=_child,args=(child,args));p.start();child.close()
    try:
        if not parent.poll(seconds):raise ModelGuardError('Candidate exceeded fit time budget','fit_timeout')
        try:status,result=parent.recv()
        except EOFError:raise ModelGuardError('Candidate worker exited without a result','fit_failed') from None
        if status!='ok':raise ModelGuardError(result['reason'],'fit_failed',details={'exception_type':result['type']})
        return result
    finally:
        parent.close();p.join(.2)
        if p.is_alive():p.terminate();p.join(1)
        if p.is_alive():p.kill();p.join()
