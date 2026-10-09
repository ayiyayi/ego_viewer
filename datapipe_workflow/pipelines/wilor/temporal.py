"""Production MANO v3 temporal processing, promoted from the validated experiment.

Preserve wrist rejection thresholds. Fill <=0.3s with LERP/SLERP, regenerate
MANO meshes and optimize wrist translation subject to 1.5cm/5px bounds.
"""
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation, Slerp
from .postprocess import CAMERA_SPEED_M_S, WRIST_SPEED_M_S, END_DIST_M, POSE_M, _clean

MAX_GAP_S = 0.3

def camera_breaks(camera, fps):
    speed = np.r_[0., np.linalg.norm(np.diff(camera, axis=0), axis=1)*fps]
    return speed, speed > CAMERA_SPEED_M_S


def fill_short_gaps(joints, valid, vertices, breaks, fps, max_gap_s=MAX_GAP_S, interpolation_blocked=None):
    j = joints.copy(); v = vertices.copy(); keep = valid.copy()
    filled = np.zeros_like(valid, dtype=bool)
    counts = []
    for h in range(2):
        reasons = dict(tracking_failure=0, long_gap=0, camera_break=0, endpoint_motion=0, endpoint_pose=0, filled_gaps=0)
        anchors = np.flatnonzero(valid[h])
        for a,b in zip(anchors[:-1], anchors[1:]):
            gap = b-a-1
            if gap == 0: continue
            if interpolation_blocked is not None and interpolation_blocked[h,a+1:b].any():
                reasons['tracking_failure'] += 1; continue
            if gap/fps > max_gap_s + 1e-9:
                reasons['long_gap'] += 1; continue
            if breaks[a+1:b+1].any():
                reasons['camera_break'] += 1; continue
            dist = np.linalg.norm(j[h,b,0]-j[h,a,0])
            if dist > END_DIST_M or dist*fps/(b-a) > WRIST_SPEED_M_S:
                reasons['endpoint_motion'] += 1; continue
            pose = np.linalg.norm((j[h,b]-j[h,b,0])-(j[h,a]-j[h,a,0]), axis=1).mean()
            if pose > POSE_M:
                reasons['endpoint_pose'] += 1; continue
            for t in range(a+1,b):
                alpha = (t-a)/(b-a)
                j[h,t] = (1-alpha)*j[h,a]+alpha*j[h,b]
                v[h,t] = (1-alpha)*v[h,a]+alpha*v[h,b]
                keep[h,t] = filled[h,t] = True
            reasons['filled_gaps'] += 1
        counts.append(reasons)
    return j,keep,v,filled,counts


def stats(joints,valid,breaks,fps):
    out=[]
    for h in range(2):
        visible=valid[h]; edge=np.diff(np.r_[False,~visible,False].astype(int))
        gaps=[(a,b) for a,b in zip(np.flatnonzero(edge==1),np.flatnonzero(edge==-1)) if a>0 and b<len(visible)]
        ok=visible[1:]&visible[:-1]&~breaks[1:]
        step=np.linalg.norm(np.diff(joints[h,:,0],axis=0),axis=1)[ok]*fps
        triple=visible[:-2]&visible[1:-1]&visible[2:]&~breaks[1:-1]&~breaks[2:]
        acc=np.linalg.norm(np.diff(joints[h,:,0],n=2,axis=0),axis=1)[triple]*fps**2
        out.append(dict(valid_frames=int(visible.sum()), visibility_transitions=int(np.count_nonzero(np.diff(visible.astype(int)))),
            short_internal_gaps=int(sum((b-a)/fps<=MAX_GAP_S for a,b in gaps)),
            wrist_speed_p95=float(np.quantile(step,.95)) if len(step) else None,
            wrist_acceleration_p95=float(np.quantile(acc,.95)) if len(acc) else None))
    return out


def optimize_translation(x, weights, strength=2.0, robust_scale=.01):
    """IRLS for sum w|x-y|² + lambda * robust squared second differences.

    Units are meters and frame indices; lambda is an experimental 30fps setting.
    Constant-velocity trajectories have zero regularization cost.
    """
    n = len(x)
    if n < 3:
        return x.copy()
    d = diags([np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)], [0,1,2], shape=(n-2,n)).tocsr()
    w = diags(weights)
    z = x.astype(np.float64).copy()
    for _ in range(5):
        accel = np.linalg.norm(d @ z, axis=1)
        robust = np.minimum(1., robust_scale / np.maximum(accel, 1e-12))
        z = spsolve((w + strength*d.T @ diags(robust) @ d).tocsc(), weights[:,None]*x)
    return z.astype(np.float32)


def segments(valid, breaks):
    start = None
    for t in range(len(valid)+1):
        if start is not None and (t == len(valid) or not valid[t] or breaks[t]):
            yield start,t
            start = None
        if t < len(valid) and valid[t] and start is None:
            start = t


def interpolate_params(params, valid, keep):
    out = {k:v.copy() for k,v in params.items()}
    for h in range(2):
        anchors = np.flatnonzero(valid[h])
        for a,b in zip(anchors[:-1],anchors[1:]):
            ids = np.flatnonzero(keep[h,a+1:b])+a+1
            if not len(ids): continue
            alpha = (ids-a)/(b-a)
            for key in ('wrist','betas'):
                out[key][h,ids] = (1-alpha[:,None])*params[key][h,a]+alpha[:,None]*params[key][h,b]
            for key in ('world_rot','pose'):
                matrices = params[key][h]
                if key == 'world_rot':
                    out[key][h,ids] = Slerp([a,b], Rotation.from_matrix(matrices[[a,b]]))(ids).as_matrix()
                else:
                    for joint in range(15):
                        out[key][h,ids,joint] = Slerp([a,b],Rotation.from_matrix(matrices[[a,b],joint]))(ids).as_matrix()
    return out


def reconstruct(mano, params, valid, camera):
    import torch
    n=valid.shape[1]
    joints=np.zeros((2,n,21,3),np.float32)
    verts=np.zeros((2,n,778,3),np.float32)
    device=next(mano.buffers()).device
    for h in range(2):
        mirror=np.diag([-1.,1.,1.]) if h==0 else np.eye(3)
        ids=np.flatnonzero(valid[h])
        for offset in range(0,len(ids),128):
            ix=ids[offset:offset+128]
            # world_rot = R_camera * mirror * R_MANO * mirror (proper rotation).
            orient=mirror @ np.swapaxes(camera[ix],1,2) @ params['world_rot'][h,ix] @ mirror
            tensor=lambda a:torch.as_tensor(a,dtype=torch.float32,device=device)
            with torch.inference_mode():
                out=mano(global_orient=tensor(orient[:,None]), hand_pose=tensor(params['pose'][h,ix]),betas=tensor(params['betas'][h,ix]),pose2rot=False)
            j=out.joints[:,:21].cpu().numpy();v=out.vertices.cpu().numpy()
            v=(v-j[:,0:1])@mirror;j=(j-j[:,0:1])@mirror
            joints[h,ix]=np.einsum('tij,tkj->tki',camera[ix],j)+params['wrist'][h,ix,None]
            verts[h,ix]=np.einsum('tij,tkj->tki',camera[ix],v)+params['wrist'][h,ix,None]
    return joints,verts


def projected(joints, data):
    cam=np.einsum('tij,htkj->htki',data['R_w2c'],joints)+data['t_w2c'][None,:,None,:]
    uv=cam[...,:2]/np.maximum(cam[...,2:3],1e-6)*float(data['focal'])
    return uv,cam[...,2]


def smooth_interpolated(interp, valid, keep, br, fps, ji, vi, data, filled):
    smooth={k:v.copy() for k,v in interp.items()}
    for h in range(2):
        for a,b in segments(keep[h],br):
            weights=np.where(valid[h,a:b],1.,.25)
            # Keep segment endpoints anchored to prevent boundary drift.
            weights[[0,-1]]=1e6
            smooth['wrist'][h,a:b]=optimize_translation(interp['wrist'][h,a:b],weights,strength=2.*(fps/30.)**4)
    delta=smooth['wrist']-interp['wrist']
    # Bound 3D movement and per-joint image movement relative to the unfiltered MANO result.
    scale=np.minimum(1.,.015/np.maximum(np.linalg.norm(delta,axis=-1),1e-9))
    uv0,z0=projected(ji,data)
    for _ in range(14):
        candidate=ji+delta[:,:,None]*scale[:,:,None,None]
        uv,z=projected(candidate,data)
        bad=keep&((np.linalg.norm(uv-uv0,axis=-1).max(axis=-1)>5.)|(z.min(axis=-1)<=1e-4))
        if not bad.any(): break
        scale[bad]*=.5
    scale[bad]=0.
    shift=delta*scale[:,:,None];smooth['wrist']=interp['wrist']+shift
    js=ji+shift[:,:,None];vs=vi+shift[:,:,None];js[~keep]=0;vs[~keep]=0
    uv1,z1=projected(js,data)
    pixels=np.linalg.norm(uv1-uv0,axis=-1).max(axis=-1)[keep]
    assert pixels.size == 0 or pixels.max()<=5.001
    assert np.all(keep[valid]) and np.array_equal(keep,valid|filled)
    assert np.isfinite(js).all() and np.isfinite(vs).all()
    for h in range(2):
        for a,b in segments(keep[h],br):
            if (filled[h,a:b]).any(): assert valid[h,a] and valid[h,b-1]
    return js,vs,smooth,shift,pixels


def process_v3(data, params, mano):
    original=data['pred_valid']>.5
    joints=data['joints_world'];fps=float(data['fps']);n=original.shape[1]
    if fps<=0:raise ValueError('Invalid fps')
    _,br=camera_breaks(data['cam_pos'],fps)
    valid=np.zeros_like(original)
    starts=[0]+np.flatnonzero(br).tolist()
    for h in range(2):
        for a,b in zip(starts,starts[1:]+[n]):
            valid[h] |= _clean(joints[h,:,0],original[h],a,b,fps)
    _,keep,_,filled,reasons=fill_short_gaps(joints,valid,np.zeros((2,n,1,3),np.float32),br,fps,.3,data.get('interpolation_blocked'))
    interp=interpolate_params(params,valid,keep)
    ji,vi=reconstruct(mano,interp,keep,data['cam_R'])
    js,vs,smooth,shift,pixels=smooth_interpolated(interp,valid,keep,br,fps,ji,vi,data,filled)
    report=dict(method='mano_temporal_v3',gap_seconds=.3,original_left_right=original.sum(1).tolist(),
        accepted_left_right=valid.sum(1).tolist(),masked_left_right=(original&~valid).sum(1).tolist(),
        final_left_right=keep.sum(1).tolist(),camera_breaks=int(br.sum()),
        translation_optimizer=dict(lambda_at_30fps=2.,huber_scale_m=.01,
            observed_weight=1.,interpolated_weight=.25,max_shift_m=.015,max_projection_shift_px=5.),
        filled_left_right=filled.sum(1).tolist(),gap_reasons=reasons,
        max_smoothing_projection_shift_px=float(pixels.max()) if pixels.size else 0.,
        before=stats(joints,valid,br,fps),after_interpolation=stats(ji,keep,br,fps),
        after_smoothing=stats(js,keep,br,fps),camera_changed=False)
    archive=dict(**smooth,valid=keep,accepted=valid,interpolated=filled,camera_breaks=br,translation_shift=shift)
    if 'interpolation_blocked' in data:
        archive['interpolation_blocked'] = data['interpolation_blocked']
    return js,vs,keep,archive,report
