#!/usr/bin/env python3
"""Run with the dedicated SAM3 interpreter; no weight download or model inference."""
import argparse
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-checkpoint', action='store_true')
    args = parser.parse_args()
    import cv2
    import numpy as np
    import sam3
    import torch
    import torchvision
    from sam3.model_builder import build_sam3_video_predictor
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor

    source = Path(sam3.__file__).resolve().parents[1]
    checkpoint = Path(os.environ.get('EGO_VIEWER_SAM3_CHECKPOINT',
        '/data/heyuping/ego_viewer/assets/weights/sam3/sam3.pt'))
    assert sys.version_info >= (3, 12)
    assert int(torch.__version__.split('.')[0]) >= 2
    assert torch.cuda.is_available(), 'CUDA is unavailable'
    assert callable(build_sam3_video_predictor)
    for name in ('handle_request', 'handle_stream_request'):
        assert callable(getattr(Sam3VideoPredictor, name))
    assert 'checkpoint_path' in inspect.signature(Sam3VideoPredictor).parameters
    with torch.inference_mode():
        x = torch.ones((32, 32), device='cuda', dtype=torch.bfloat16)
        assert torch.all(x @ x == 32).item()
        selected = torchvision.ops.nms(torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.]],device='cuda'),
                                       torch.tensor([.9,.8],device='cuda'), .5)
        assert selected.tolist() == [0]
    torch.cuda.synchronize()
    report = dict(python=sys.version.split()[0], executable=sys.executable, torch=torch.__version__,
                  torchvision=torchvision.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                  numpy=np.__version__, opencv=cv2.__version__, source=str(source),
                  source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
                  checkpoint=str(checkpoint), checkpoint_exists=checkpoint.is_file(),
                  cuda_matmul='passed', torchvision_cuda_nms='passed', predictor_import='passed',
                  model_inference='not_run')
    print(json.dumps(report, indent=2))
    if args.require_checkpoint and not checkpoint.is_file():
        raise SystemExit('SAM3 environment is ready, but the checkpoint is missing')


if __name__ == '__main__':
    main()
