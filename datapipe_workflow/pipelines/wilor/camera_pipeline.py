"""SAM-mask camera-only path. No HaWoR hand network, mesh renderer or infiller.

All SAM jobs share one dedicated process/model. Release it before SLAM. Keep
existing 100-second SLAM chunking/merge semantics; this does not fix chunk seams.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HAWOR = ROOT / 'pipelines/hawor'
sys.path.insert(0, str(ROOT))


def union_mask(row, height, width, target_hw):
    import cv2
    mask = np.zeros((height, width), dtype=np.uint8)
    for side in ('left', 'right'):
        path = row.get(f'{side}_mask_path')
        if not path:
            raise ValueError(f'Missing SAM mask path: {side}, frame {row["frame_idx"]}')
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (height, width):
            raise ValueError(f'Invalid SAM mask dimensions: {path}')
        mask |= (image > 0).astype(np.uint8)
    h, w = target_hw
    # Any covered pixel excludes this region from SLAM; do not invert foreground.
    resized = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    return resized[:h-h%8, :w-w%8] > 0


def setup_hawor():
    os.chdir(HAWOR)
    sys.path.insert(0, str(HAWOR))


def prepare(video):
    setup_hawor()
    from scripts.scripts_test_video.detect_track_video import extract_frames
    from lib.pipeline.tools import detect_track
    seq = video.with_suffix(''); frames = seq / 'extracted_images'
    extract_frames(str(video), str(frames))
    paths = sorted(frames.glob('*.jpg'), key=lambda p:int(p.stem))
    _, tracks = detect_track([str(p) for p in paths], thresh=.2)
    rows = [{} for _ in paths]
    for track in tracks.item().values():
        for r in track:
            h = int(r['det_handedness'][0]); b = r['det_box'][0]
            rows[r['frame']][h] = {'box': b[:4].tolist(), 'score': float(b[4])}
    (seq/'detector_rows.json').write_text(json.dumps(rows))


def slam(video):
    setup_hawor()
    from scripts.scripts_test_video.hawor_slam import hawor_slam
    n = len(list((video.with_suffix('')/'extracted_images').glob('*.jpg')))
    hawor_slam(argparse.Namespace(video_path=str(video), img_focal=None), 0, n)


def run(video, work):
    import cv2
    from pipelines.sam3_hand_tracking.refine import environment_config, save_box_archive
    setup_hawor()
    from scripts.segmented_demo_pipeline import split_video, build_chunk_meta, merge_slam
    from pipelines.wilor.export_viewer import probe_video
    cfg = environment_config()
    if cfg is None: raise ValueError('SAM3 environment is required')
    work.mkdir(parents=True, exist_ok=True)
    from pipelines.wilor.cache import key, stamp, valid, complete, save_json
    run_key = key(dict(video=stamp(video),cfg={k:str(v) for k,v in cfg.items()},weights=stamp(cfg['checkpoint'])),
                  [Path(__file__), *[p for p in (ROOT/'pipelines/sam3_hand_tracking').glob('*.py') if not p.name.startswith('test_')]])
    if valid(work/'complete_receipt.json',run_key):
        print('CACHE HIT: complete camera pipeline',flush=True)
        return
    timings = []
    def command(stage, argv):
        receipt=work/'receipts'/f'{stage}.json'
        dependencies=[]
        if stage.startswith('slam_metric3d'):
            dependencies=[stamp(p) for p in sorted(Path(argv[-1]).with_suffix('').glob('sam3/*/output/slam_masks.npz'))]
        stage_key=key(dict(run=run_key,argv=argv,dependencies=dependencies))
        if stage != 'sam3_resident' and valid(receipt,stage_key):
            print(f'CACHE HIT: {stage}',flush=True)
            timings.append(dict(stage=stage,seconds=0.,cache_hit=True))
            return
        if stage.startswith('slam_metric3d'):
            seq=Path(argv[-1]).with_suffix('')
            for stale in (seq/'SLAM').glob('hawor_slam_w_scale_*.npz'): stale.unlink()
        start = time.perf_counter()
        print(f'STAGE START {stage}', flush=True)
        try:
            subprocess.run(argv, check=True)
        finally:
            timings.append({'stage':stage,'seconds':time.perf_counter()-start})
            (work/'stage_timings.json').write_text(json.dumps(timings, indent=2))
        if stage.startswith('detect_'):
            seq=Path(argv[-1]).with_suffix('')
            complete(receipt,stage_key,[seq/'detector_rows.json', *sorted((seq/'extracted_images').glob('*.jpg'))])
        elif stage.startswith('slam_metric3d'):
            complete(receipt,stage_key,list((seq/'SLAM').glob('hawor_slam_w_scale_*.npz')))
        print(f'STAGE END {stage}: {timings[-1]["seconds"]:.2f}s', flush=True)
    width,height,fps,nvideo = probe_video(video)
    start = time.perf_counter()
    split_key=key(dict(run=run_key,stage='split'))
    if valid(work/'split_receipt.json',split_key):
        chunks=[Path(p) for p in json.loads((work/'chunks.json').read_text())]
    else:
        chunks = [Path(p) for p in split_video(str(video),str(work/'chunks'),100,60,True,False)]
        save_json(work/'chunks.json',[str(p) for p in chunks])
        complete(work/'split_receipt.json',split_key,[work/'chunks.json',*chunks])
    timings.append({'stage':'split','seconds':time.perf_counter()-start})
    jobs=[]; segments=[]
    for ci,chunk in enumerate(chunks):
        command(f'detect_{ci}',[sys.executable,__file__,'--stage','prepare','--video',str(chunk)])
        seq=chunk.with_suffix(''); paths=sorted((seq/'extracted_images').glob('*.jpg'),key=lambda p:int(p.stem))
        raw=json.loads((seq/'detector_rows.json').read_text())
        for start in range(0,len(paths),600):
            segment=seq/'sam3'/f'{start:06d}';(segment/'frames').mkdir(parents=True,exist_ok=True);(segment/'detection').mkdir(exist_ok=True)
            subset=raw[start:start+600]
            for t,path in enumerate(paths[start:start+600]):
                link=segment/'frames'/f'{t:06d}.jpg'
                if not link.exists(): link.symlink_to(path)
            records=[(t,int(h),r) for t,row in enumerate(subset) for h,r in row.items()]
            detection_key=key(dict(run=run_key,source=stamp(seq/'detector_rows.json'),start=start))
            if not valid(segment/'detection/receipt.json',detection_key): np.savez(segment/'detection/bboxes_2d.npz',frame_idx=np.array([r[0] for r in records],dtype=np.int64),track_id=np.array([r[1] for r in records],dtype=np.int64),boxes_xyxy=np.array([r[2]['box'] for r in records],dtype=np.float32).reshape(-1,4),scores=np.array([r[2]['score'] for r in records],dtype=np.float32),handedness=np.array([r[1] for r in records],dtype=np.int64))
            complete(segment/'detection/receipt.json',detection_key,[segment/'detection/bboxes_2d.npz'])
            out=segment/'output'
            jobs.append(['--segment-dir',str(segment),'--output-dir',str(out),'--checkpoint-path',str(cfg['checkpoint']),'--fps',str(fps),'--prompt-geometry','box_below','--propagation-scheme','single_anchor','--tracking-mode','separate','--init-mode','point','--validate-tracks','--compact-masks'])
            segments.append((ci,start,len(subset),out))
    manifest=work/'sam3_jobs.json';manifest.write_text(json.dumps(jobs,indent=2))
    os.environ['SAM3_SRC']=str(cfg['sam3_src'])
    command('sam3_resident',[str(cfg['python_bin']),str(ROOT/'pipelines/sam3_hand_tracking/resident_worker.py'),'--jobs',str(manifest)])
    # DROID's image dimensions, without importing its GPU modules in this process.
    factor=(384*512/(height*width))**.5
    resize_h,resize_w=int(height*factor),int(width*factor)
    h,w=resize_h-resize_h%8,resize_w-resize_w%8
    all_rows=[]; all_blocked=[]; validation_reports=[]; global_frame=0; merged=work/'extracted_images';merged.mkdir(exist_ok=True)
    for ci,chunk in enumerate(chunks):
        seq=chunk.with_suffix('');paths=sorted((seq/'extracted_images').glob('*.jpg'),key=lambda p:int(p.stem));n=len(paths)
        track=seq/f'tracks_0_{n}';track.mkdir(exist_ok=True)
        masks=np.lib.format.open_memmap(track/'model_masks.npy',mode='w+',dtype=bool,shape=(n,h,w))
        rows=[];chunk_blocked=[];start_time=time.perf_counter()
        for c,start,length,out in segments:
            if c!=ci:continue
            exported=[json.loads(line) for line in (out/'sam_tight_bboxes_2d.jsonl').read_text().splitlines()]
            if [r['frame_idx'] for r in exported]!=list(range(length)):raise ValueError('SAM timeline mismatch')
            with np.load(out/'tracking_validation.npz') as validation:
                blocked = validation['interpolation_blocked'].copy()
            if blocked.shape != (length, 2): raise ValueError('Validation timeline mismatch')
            chunk_blocked.extend(blocked.tolist())
            validation_reports.append(dict(chunk=ci, start=start,
                report=json.loads((out/'tracking_validation.json').read_text())))
            compact_path = out/'slam_masks.npz'
            if compact_path.exists():
                with np.load(compact_path) as archive:
                    if tuple(archive['shape']) != (h,w): raise ValueError('Compact mask dimensions mismatch')
                    low_masks = np.unpackbits(archive['packed'],axis=1,count=h*w).reshape(length,h,w).astype(bool)
            else:
                low_masks = None
            for t,row in enumerate(exported):
                masks[start+t]=low_masks[t] if low_masks is not None else union_mask(row,height,width,(resize_h,resize_w))
                rows.append({0:row['left_sam_tight_box_xyxy'],1:row['right_sam_tight_box_xyxy']})
        assert len(rows)==n
        masks.flush();del masks
        save_box_archive(seq/'sam_boxes.npz',rows,width,height,chunk_blocked);all_rows.extend(rows);all_blocked.extend(chunk_blocked)
        for path in paths:
            link=merged/f'{global_frame:06d}.jpg'
            if not link.exists():link.symlink_to(path)
            global_frame+=1
        timings.append({'stage':f'assemble_masks_{ci}','seconds':time.perf_counter()-start_time})
        command(f'slam_metric3d_{ci}',[sys.executable,__file__,'--stage','slam','--video',str(chunk)])
    if global_frame != nvideo: raise ValueError(f'Decoded timeline {global_frame} != input {nvideo}')
    save_box_archive(work/'sam_boxes.npz',all_rows,width,height,all_blocked)
    (work/'tracking_validation.json').write_text(json.dumps(validation_reports,indent=2))
    metas=build_chunk_meta([str(p) for p in chunks],[str(p.with_suffix('')) for p in chunks])
    path=merge_slam(metas,str(work/'SLAM'),'keep_last')
    if path is None: raise ValueError('No SLAM output')
    result={'slam':path,'frames':global_frame,'mask_source':'sam3_left_right_union','mask_hw':[h,w],'sam_model_loads':1,'chunk_frames':[m.frame_count for m in metas],'camera_chunk_merge':'existing keep_last, no new cross-chunk alignment'}
    save_json(work/'camera_pipeline_complete.json',result)
    complete(work/'complete_receipt.json',run_key,[Path(path),work/'camera_pipeline_complete.json',work/'sam_boxes.npz',work/'tracking_validation.json',work/'stage_timings.json',work/'sam3_jobs.timings.json'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--work',type=Path)
    parser.add_argument('--stage',choices=['run','prepare','slam'],default='run')
    args=parser.parse_args();video=args.video.resolve()
    if args.stage=='prepare':prepare(video)
    elif args.stage=='slam':slam(video)
    else:
        from pipelines.wilor.cache import lock
        with lock(args.work.resolve()/'.camera.lock'):run(video,args.work.resolve())
