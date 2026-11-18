import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from ..errors import ModelGuardError


def split_indices(df,cfg):
    labelled=df.index[df[cfg['target_column']].notna()].to_numpy()
    duplicate_columns=cfg.get('feature_columns') or [c for c in df if c!=cfg.get('record_id_column')]
    duplicate_columns=list(dict.fromkeys(duplicate_columns+[cfg['target_column']]))
    duplicate_mask=df.loc[labelled,duplicate_columns].duplicated();duplicates=df.loc[labelled].index[duplicate_mask].tolist();labelled=labelled[~duplicate_mask.to_numpy()];y=df.loc[labelled,cfg['target_column']]
    if len(labelled)<40:raise ModelGuardError('At least 40 labelled rows required','insufficient_data')
    seed=cfg['random_seed'];v=cfg['validation_size'];t=cfg['test_size'];excluded=[];notes=[]
    if cfg.get('time_column'):
        times=pd.to_datetime(df.loc[labelled,cfg['time_column']],errors='coerce',utc=True)
        if times.isna().any():raise ModelGuardError('Invalid timestamps')
        order=times.sort_values(kind='stable').index.to_numpy();b1=times.loc[order[int(len(order)*(1-v-t))]];b2=times.loc[order[int(len(order)*(1-t))]]
        a=times.index[times<b1].to_numpy();b=times.index[(times>=b1)&(times<b2)].to_numpy();c=times.index[times>=b2].to_numpy();strategy='chronological'
        if cfg.get('group_column'):
            groups=df[cfg['group_column']];sets=[set(groups.loc[z]) for z in [a,b,c]];cross=(sets[0]&sets[1])|(sets[0]&sets[2])|(sets[1]&sets[2]);excluded=df.loc[labelled].index[groups.loc[labelled].isin(cross)].tolist()
            a,b,c=[z[~groups.loc[z].isin(cross).to_numpy()] for z in [a,b,c]];strategy='group_chronological';notes.append('Boundary-spanning groups excluded to preserve chronology and entity separation.')
    elif cfg.get('group_column'):
        g=df.loc[labelled,cfg['group_column']];aidx,hold=next(GroupShuffleSplit(n_splits=1,test_size=v+t,random_state=seed).split(labelled,groups=g));h=labelled[hold];bidx,cidx=next(GroupShuffleSplit(n_splits=1,test_size=t/(t+v),random_state=seed+1).split(h,groups=df.loc[h,cfg['group_column']]))
        a,b,c=labelled[aidx],h[bidx],h[cidx];strategy='group'
    else:
        try:
            strat=y if cfg['task']=='classification' else None
            a,h=train_test_split(labelled,test_size=v+t,random_state=seed,stratify=strat)
            b,c=train_test_split(h,test_size=t/(t+v),random_state=seed+1,stratify=df.loc[h,cfg['target_column']] if strat is not None else None)
            strategy='stratified_random' if strat is not None else 'random'
        except ValueError:raise ModelGuardError('Insufficient support for stratified three-way split','insufficient_data') from None
    if min(map(len,[a,b,c]))<8:raise ModelGuardError('Split has insufficient support','insufficient_data')
    if cfg['task']=='classification' and set(df.loc[a,cfg['target_column']])!=set(y):raise ModelGuardError('Training partition must contain every class','insufficient_data')
    return {'strategy':strategy,'train':a.tolist(),'validation':b.tolist(),'test':c.tolist(),'excluded_indices':sorted(set(excluded+duplicates)),'excluded_duplicate_indices':duplicates,'seed':seed,'notes':notes,'test_locked':True}


def training_holdout(y,cfg,context=None,fraction=.25):
    n=len(y);idx=np.arange(n);context=context if context is not None else pd.DataFrame({'source_row':idx})
    if n<16:raise ModelGuardError('Too few training rows for independent calibration')
    if 'time' in context:
        ts=pd.to_datetime(context['time'],utc=True);boundary=ts.sort_values().iloc[int(n*(1-fraction))];a=idx[(ts<boundary).to_numpy()];b=idx[(ts>=boundary).to_numpy()]
        if 'group' in context:
            cross=set(context.iloc[a]['group'])&set(context.iloc[b]['group']);a=a[~context.iloc[a]['group'].isin(cross).to_numpy()];b=b[~context.iloc[b]['group'].isin(cross).to_numpy()]
    else:
        g=context['group'] if 'group' in context else context['source_row'] if 'source_row' in context and context['source_row'].duplicated().any() else None
        if g is not None:a,b=next(GroupShuffleSplit(n_splits=1,test_size=fraction,random_state=cfg['random_seed']).split(idx,groups=g))
        else:a,b=train_test_split(idx,test_size=fraction,random_state=cfg['random_seed'],stratify=y if cfg['task']=='classification' else None)
    if min(len(a),len(b))<4:raise ModelGuardError('Independent training holdout has insufficient support')
    if cfg['task']=='classification' and (set(np.asarray(y)[a])!=set(y) or set(np.asarray(y)[b])!=set(y)):raise ModelGuardError('Calibration partitions must contain all classes')
    return a,b
