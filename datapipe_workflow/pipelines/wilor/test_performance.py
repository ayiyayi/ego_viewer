import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .cache import key, stamp, complete, valid, lock
from .batching import crop_batches, prefetch

class CacheBatchTests(unittest.TestCase):
    def test_receipt_invalidation(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);f=p/'out';f.write_text('one');receipt=p/'receipt.json'
            k=key({'a':1});complete(receipt,k,[f]);self.assertTrue(valid(receipt,k))
            self.assertFalse(valid(receipt,key({'a':2})))
            f.write_text('two');self.assertFalse(valid(receipt,k))
            receipt.write_text('{');self.assertFalse(valid(receipt,k))

    def test_lock(self):
        with tempfile.TemporaryDirectory() as d:
            with lock(Path(d)/'lock'):
                with self.assertRaises(RuntimeError):
                    with lock(Path(d)/'lock'):pass

    def test_batch_mapping_empty_frames_and_tail(self):
        rows=[(t,dict(right=np.float32(side),value=np.array([t,side]))) for t in [0,2,5] for side in [0,1]]
        batches=list(prefetch(crop_batches(iter(rows),4)))
        self.assertEqual([x[0] for x in batches],[[0,0,2,2],[5,5]])
        self.assertEqual(batches[-1][1]['value'].tolist(),[[5,0],[5,1]])
        self.assertEqual(list(prefetch(crop_batches([],4))),[])

    def test_prefetch_exception(self):
        def rows():
            yield 1
            raise ValueError('bad frame')
        with self.assertRaisesRegex(ValueError,'bad frame'):list(prefetch(rows()))

    def test_packed_mask_roundtrip_and_union(self):
        from pipelines.sam3_hand_tracking.mask_io import write_mask,read_mask,reduce_union
        from pipelines.wilor.camera_pipeline import union_mask
        import cv2
        with tempfile.TemporaryDirectory() as d:
            h,w=479,641;a=np.zeros((h,w),bool);a[50:79,8:33]=True
            b=np.zeros_like(a);b[60:180,500:640]=True
            np.testing.assert_array_equal(read_mask(write_mask(Path(d)/'test.png',a)),a)
            row={'frame_idx':0}
            for name,m in [('left',a),('right',b)]:
                path=Path(d)/(name+'.png');cv2.imwrite(str(path),m.astype('uint8')*255);row[name+'_mask_path']=str(path)
            f=(384*512/(h*w))**.5
            np.testing.assert_array_equal(reduce_union([a,b],h,w),union_mask(row,h,w,(int(h*f),int(w*f))))

    def test_compact_export_has_no_pngs_and_preserves_empty_frames(self):
        from pipelines.sam3_hand_tracking.mask_io import write_mask, reduce_union
        from pipelines.sam3_hand_tracking.sam3_point import export_framewise
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);mask=np.zeros((48,64),bool);mask[10:15,20:30]=True
            path=write_mask(out/'hand',mask)
            rows=export_framewise([0,1],{(0,0):dict(mask_path=path,mask_area=50,box_xyxy=[20,10,30,15])},out,64,48,compact=True)
            self.assertIsNone(rows[1]['left_sam_tight_box_xyxy'])
            self.assertFalse(list(out.rglob('*.png')))
            with np.load(out/'slam_masks.npz') as a:
                h,w=a['shape']; masks=np.unpackbits(a['packed'],axis=1,count=int(h*w)).reshape(2,h,w)
            np.testing.assert_array_equal(masks[0],reduce_union([mask],48,64))
            self.assertFalse(masks[1].any())

    def test_shared_loader_restores_after_failure(self):
        import sys,types
        from pipelines.sam3_hand_tracking.frame_cache import shared_frame_cache
        package=types.ModuleType('sam3');models=types.ModuleType('sam3.model');backend=types.ModuleType('sam3.model.sam3_video_inference')
        package.model=models;models.sam3_video_inference=backend;calls=[]
        def load(resource_path,image_size=8):
            calls.append(resource_path);return object()
        backend.load_resource_as_video_frames=load
        with patch.dict(sys.modules,{'sam3':package,'sam3.model':models,'sam3.model.sam3_video_inference':backend}):
            with self.assertRaises(ValueError):
                with shared_frame_cache('/tmp/main_frames'):
                    a=backend.load_resource_as_video_frames('/tmp/main_frames')
                    self.assertIs(a,backend.load_resource_as_video_frames(resource_path='/tmp/main_frames'))
                    backend.load_resource_as_video_frames('/tmp/recovery')
                    raise ValueError()
            self.assertIs(backend.load_resource_as_video_frames,load)
        self.assertEqual(calls,['/tmp/main_frames','/tmp/recovery'])

if __name__=='__main__':unittest.main()
