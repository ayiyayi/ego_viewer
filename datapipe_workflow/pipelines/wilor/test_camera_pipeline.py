import json
from pathlib import Path
import sys
import tempfile
import unittest
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from pipelines.wilor.camera_pipeline import union_mask
from pipelines.sam3_hand_tracking.resident_worker import run_jobs


class CameraPipelineTests(unittest.TestCase):
    def test_foreground_union_and_empty_frame(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);l=np.zeros((18,26),np.uint8);r=l.copy();l[4:8,4:8]=255;r[8:12,12:16]=255
            cv2.imwrite(str(p/'l.png'),l);cv2.imwrite(str(p/'r.png'),r)
            row={'frame_idx':0,'left_mask_path':str(p/'l.png'),'right_mask_path':str(p/'r.png')}
            mask=union_mask(row,18,26,(18,26))
            self.assertEqual(mask.shape,(16,24));self.assertEqual(mask.dtype,np.bool_)
            self.assertTrue(mask[5,5] and mask[9,13]);self.assertFalse(mask[0,0])
            cv2.imwrite(str(p/'l.png'),l*0);cv2.imwrite(str(p/'r.png'),r*0)
            self.assertFalse(union_mask(row,18,26,(18,26)).any())
            row['left_mask_path']=str(p/'missing.png')
            with self.assertRaises(ValueError):union_mask(row,18,26,(18,26))

    def test_model_reused_across_jobs(self):
        loads=[];seen=[];model=object()
        def build(*args):loads.append(args);return model
        def execute(args,predictor):seen.append(predictor)
        jobs=[['--segment-dir',str(i),'--output-dir',str(i),'--checkpoint-path','weight'] for i in range(3)]
        self.assertEqual(len(run_jobs(jobs,build,execute)),3)
        self.assertEqual(len(loads),1);self.assertTrue(all(x is model for x in seen))
        jobs[-1][-1]='different-weight'
        with self.assertRaises(ValueError):run_jobs(jobs,build,execute)

if __name__=='__main__':unittest.main()
