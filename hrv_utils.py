import numpy as np
import torch
from data_utils import get_episode_rri

def extract_hrv(rr_ms,ctm_radius=0.1):
    rr=torch.as_tensor(rr_ms,dtype=torch.float32).flatten()
    d=torch.diff(rr)
    rmssd=torch.sqrt(torch.mean(d**2))
    sdnn=torch.std(rr,unbiased=False)
    sd_diff=torch.std(d,unbiased=False)
    sd1=sd_diff/np.sqrt(2)
    sd2=torch.sqrt(torch.clamp(2*sdnn**2-0.5*sd_diff**2,min=1e-8))
    sd1sd2=sd1/sd2
    x,y=d[:-1],d[1:]
    q1=((x>0)&(y>0)).float().mean()
    ctm=((x**2+y**2)<=ctm_radius**2).float().mean()
    s=torch.sign(d)
    total=0
    i=0
    while i<len(s):
        if s[i]==0:
            i+=1
            continue
        j=i
        while j+1<len(s) and s[j+1]!=0 and s[j+1]==-s[j]:
            j+=1
        n_diff=j-i+1
        if n_diff>=3:
            total+=n_diff+1
        i=j+1
    pas=total/max(len(rr),1)
    return np.array([rmssd.item(),sd1sd2.item(),q1.item(),ctm.item(),float(pas)])


def build_hrv_set(data,subject_ids,input_segment_size,prediction_horizon):
    features,labels,sids=[],[],[]
    for sid in subject_ids:
        label=int(data[sid]["label"])
        for eid in sorted(data[sid]["episodes"]):
            rri=get_episode_rri(data,sid,eid)
            if label==1:
                required=input_segment_size+prediction_horizon
                if len(rri)<required: continue
                seg=rri[-required:-prediction_horizon]
            else:
                if len(rri)<input_segment_size: continue
                seg=rri[:input_segment_size]
            if len(seg)!=input_segment_size: continue
            features.append(extract_hrv(seg))
            labels.append(label)
            sids.append(sid)
    return np.asarray(features),np.asarray(labels),np.asarray(sids)

def build_hrv_groups_for_subject(data,sid,input_segment_size=14400,prediction_horizon=14400,group_len=12):
    label=int(data[sid]["label"])
    groups,labels=[],[]
    for eid in sorted(data[sid]["episodes"]):
        rri=get_episode_rri(data,sid,eid)
        if label==1:
            required=input_segment_size+prediction_horizon
            if len(rri)<required: continue
            seg=rri[-required:-prediction_horizon]
        else:
            if len(rri)<input_segment_size: continue
            seg=rri[:input_segment_size]
        win_size=input_segment_size//group_len
        group=[]
        for start in range(0,input_segment_size,win_size):
            w=seg[start:start+win_size]
            if len(w)==win_size:
                group.append(extract_hrv(w))
        if len(group)==group_len:
            groups.append(group)
            labels.append(label)
    return groups,labels

def get_fold_ids(af_sets,nsr_sets,fold_idx,n_folds=5):
    test_i=fold_idx
    val_i=(fold_idx+1)%n_folds
    train_i=[i for i in range(n_folds) if i not in [test_i,val_i]]
    train_ids=[sid for i in train_i for sid in af_sets[i]+nsr_sets[i]]
    val_ids=af_sets[val_i]+nsr_sets[val_i]
    test_ids=af_sets[test_i]+nsr_sets[test_i]
    return train_ids,val_ids,test_ids