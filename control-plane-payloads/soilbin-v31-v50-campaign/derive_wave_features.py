from __future__ import annotations
import base64,gzip,hashlib,io,json,lzma,math,struct
import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.signal import correlate,correlation_lags

EXPECTED_XZ='6ccca93e5e146e67908e37cf07575bdf1ed44b0f3b84fcf0ee3198cd2ffa02c9'
EXPECTED_BLOB='786cd72be8a6e37aa0e34cee044f70b047c9a84f5a9230abc9028117db765677'
EXPECTED_RAW='ea227f412827a962a1876ceb9aeb9946034b435052e9e74115e32bd7dca73cb0'

def decode_compact(text):
    xz=base64.b64decode(text.strip())
    if hashlib.sha256(xz).hexdigest()!=EXPECTED_XZ: raise RuntimeError('V4_WAVE_XZ_SHA_MISMATCH')
    blob=lzma.decompress(xz)
    if hashlib.sha256(blob).hexdigest()!=EXPECTED_BLOB: raise RuntimeError('V4_WAVE_BLOB_SHA_MISMATCH')
    hlen=struct.unpack('<I',blob[:4])[0]; meta=json.loads(blob[4:4+hlen].decode())
    n=int(meta['n']);off=4+hlen
    run=np.frombuffer(blob,dtype=np.uint8,count=n,offset=off).copy();off+=n
    tim=np.frombuffer(blob,dtype='<u4',count=n,offset=off).copy();off+=4*n
    f1=np.frombuffer(blob,dtype='<f8',count=n,offset=off).copy();off+=8*n
    f5=np.frombuffer(blob,dtype='<f8',count=n,offset=off).copy();off+=8*n
    if off!=len(blob):raise RuntimeError('V4_WAVE_TRAILING_BYTES')
    runs=meta['runs']
    z=pd.DataFrame({'Run_ID':[runs[int(i)] for i in run],'time_ms':tim.astype(float),
                    'LC1_force_delta_N':f1,'LC5_force_delta_N':f5})
    if len(z)!=80958 or z.Run_ID.nunique()!=51:raise RuntimeError('V4_WAVE_SHAPE_MISMATCH')
    return z

def _resample_event(g,n=256):
    g=g.sort_values('time_ms');t=g.time_ms.to_numpy(float)
    x=g[['LC1_force_delta_N','LC5_force_delta_N']].to_numpy(float)
    mag=np.sqrt(np.sum(x*x,axis=1));n0=max(10,min(len(mag)//8,200))
    base=np.r_[mag[:n0],mag[-n0:]] if len(mag)>=2*n0 else mag
    med=float(np.median(base));mad=float(np.median(np.abs(base-med)));sig=max(1e-9,1.4826*mad)
    peak=float(np.max(mag));thr=max(med+5*sig,.10*peak);idx=np.flatnonzero(mag>=thr)
    if len(idx)<3:
        c=int(np.argmax(mag));half=min(max(20,len(mag)//10),len(mag)//2);a=max(0,c-half);b=min(len(mag)-1,c+half)
    else:
        a=int(idx[0]);b=int(idx[-1]);pad=max(5,int(round(.10*(b-a+1))));a=max(0,a-pad);b=min(len(mag)-1,b+pad)
    tt=t[a:b+1];xx=x[a:b+1];u=np.linspace(tt[0],tt[-1],n)
    y=np.column_stack([np.interp(u,tt,xx[:,0]),np.interp(u,tt,xx[:,1])])
    return u,y,float(tt[-1]-tt[0])

def _norm(v):
    v=np.asarray(v,float);v=v-np.mean(v);s=np.linalg.norm(v)
    return v/(s if s>1e-12 else 1.)

def _angle_stats(points):
    d=np.diff(points,axis=0);norm=np.linalg.norm(d,axis=1)
    good=(norm[:-1]>1e-12)&(norm[1:]>1e-12)
    if not np.any(good):return 0.,0.
    a=d[:-1][good];b=d[1:][good]
    cos=np.sum(a*b,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))
    ang=np.arccos(np.clip(cos,-1,1))
    return float(np.mean(ang)),float(np.max(ang))

def _phase_area(x,y):
    return float(.5*abs(np.dot(x,np.roll(y,-1))-np.dot(y,np.roll(x,-1))))

def run_features(g):
    u,w,dur=_resample_event(g);x=w[:,0];y=w[:,1];dt=float(np.median(np.diff(u)))
    xn=_norm(x);yn=_norm(y)
    cc=correlate(yn,xn,mode='full',method='fft');lags=correlation_lags(len(yn),len(xn),mode='full')
    lim=np.abs(lags*dt)<=4000
    kk=np.flatnonzero(lim)[int(np.argmax(cc[lim]))];lag=int(lags[kk]);r=float(cc[kk])
    if lag>=0:
        xa=x[:len(x)-lag] if lag else x.copy();ya=y[lag:]
    else:
        k=-lag;xa=x[k:];ya=y[:len(y)-k]
    if len(xa)<8:xa=x;ya=y;lag=0
    den=float(np.dot(xa,xa));scale=float(np.dot(xa,ya)/den) if den>1e-12 else 0.
    pred=scale*xa;sst=float(np.sum((ya-np.mean(ya))**2))
    r2=float(1-np.sum((ya-pred)**2)/sst) if sst>1e-12 else 0.
    xan=_norm(xa);yan=_norm(ya)
    corr=float(np.corrcoef(xan,yan)[0,1]) if len(xan)>2 else 0.
    mae=float(np.mean(np.abs(xan-yan)))
    tnorm=np.linspace(0,1,len(xa));pts=np.column_stack([tnorm,xan,yan])
    path=float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum());mean_turn,max_turn=_angle_stats(pts)
    if len(pts)>=4:
        v1=np.diff(pts,axis=0)
        cr=np.cross(v1[:-1],v1[1:])
        tors=float(np.mean(np.linalg.norm(np.diff(cr,axis=0),axis=1))) if len(cr)>=2 else 0.
    else:tors=0.
    out={'active_duration_ms':dur,'xcorr_lag_ms':float(lag*dt),'xcorr_r':r,'aligned_scale':scale,'aligned_r2':r2,
         'traj3d_path_length':path,'traj3d_mean_turn':mean_turn,'traj3d_max_turn':max_turn,'traj3d_torsion_proxy':tors,
         'lc1_lc5_phase_area':_phase_area(xn,yn),'aligned_norm_corr':corr,'aligned_norm_mae':mae}
    for name,v in [('lc1',x),('lc5',y)]:
        nv=_norm(v);out[f'shape_norm_area_{name}']=float(np.trapz(np.abs(nv),dx=1/(len(nv)-1)))
        out[f'shape_norm_rms_{name}']=float(np.sqrt(np.mean(nv*nv)))
        out[f'shape_arc_length_{name}']=float(np.sum(np.sqrt((1/(len(nv)-1))**2+np.diff(nv)**2)))
        p=np.column_stack([np.linspace(0,1,len(nv)),nv]);mt,xt=_angle_stats(p)
        out[f'shape_turn_mean_{name}']=mt;out[f'shape_turn_max_{name}']=xt
        co=dct(nv,type=2,norm='ortho')[:10]
        for i,q in enumerate(co):out[f'shape_dct_{name}_{i}']=float(q)
    return out

def build_feature_csv(model_csv_gz_b64,wave_xz_b64):
    raw=gzip.decompress(base64.b64decode(model_csv_gz_b64.strip()))
    d=pd.read_csv(io.BytesIO(raw))
    ts=decode_compact(wave_xz_b64)
    rows=[]
    for rid,g in ts.groupby('Run_ID',sort=False):
        m=d[d.Run_ID==rid]
        if len(m)!=1:raise RuntimeError('RUN_ID_JOIN_MISMATCH:'+rid)
        rr=m.iloc[0]
        z={'Run_ID':rid,'V_code':rr.V_code,'Speed_level':int(rr.Speed_level),'W_code':rr.W_code,
           'Weight_level':int(rr.Weight_level),'Pass_T':int(rr.Pass_T)}
        z.update(run_features(g));rows.append(z)
    out=pd.DataFrame(rows).sort_values(['Speed_level','Weight_level','Pass_T']).reset_index(drop=True)
    if len(out)!=51:raise RuntimeError('EXPECTED_51_WAVE_FEATURE_ROWS')
    b=out.to_csv(index=False).encode()
    return base64.b64encode(gzip.compress(b,mtime=0)).decode(),{'rows':51,'source_xz_sha256':EXPECTED_XZ,
      'source_blob_sha256':EXPECTED_BLOB,'feature_csv_sha256':hashlib.sha256(b).hexdigest(),'columns':list(out.columns)}
