"""Re-run MANO temporal processing using saved parameters, on CPU by default."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from pipelines.wilor.export_viewer import WILOR_ROOT, write_viewer
from pipelines.wilor.temporal import process_v3


def reprocess(output, raw=None):
    import torch
    torch.set_num_threads(2)
    raw=raw or output/'raw_wilor'
    with np.load(raw/'keypoints.npz') as f:data={k:f[k] for k in f.files}
    with np.load(raw/'mano.npz') as f:params={k:f[k] for k in ('pose','world_rot','wrist','betas')}
    # Import only the MANO wrapper, not the WiLoR network/checkpoint.
    spec=importlib.util.spec_from_file_location('wilor_mano',WILOR_ROOT/'wilor/models/mano_wrapper.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    mano=module.MANO(model_path=str(WILOR_ROOT/'mano_data'),gender='neutral',num_hand_joints=15,create_body_pose=False).eval()
    start=time.perf_counter()
    joints,verts,valid,archive,report=process_v3(data,params,mano)
    output.mkdir(parents=True,exist_ok=True)
    write_viewer(output,joints,verts,valid,mano.faces,float(data['focal']),int(data['width']),int(data['height']),float(data['fps']),data['R_w2c'],data['t_w2c'],data['cam_R'],data['cam_pos'])
    np.savez_compressed(output/'processed_mano.npz',**archive)
    report['postprocess_seconds']=time.perf_counter()-start
    report['source']='cached_mano'
    (output/'temporal_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--raw-dir',type=Path)
    args=parser.parse_args();reprocess(args.output_dir.resolve(),args.raw_dir.resolve() if args.raw_dir else None)
