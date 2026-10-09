import unittest
import numpy as np
from .validation import compatible, assess, validate_and_recover, trusted_detections

class ValidationTests(unittest.TestCase):
    def test_head_and_hand(self):
        hand=[100,200,140,250]
        self.assertTrue(compatible(hand,hand))
        self.assertFalse(compatible([0,0,500,400],hand))
        self.assertFalse(compatible([100,0,140,50],hand))

    def test_medium_confidence_head_is_not_confirmation(self):
        det={0:[dict(side=1,score=.63,box_xyxy=[0,0,1000,500])]}
        self.assertEqual(trusted_detections(det,1,1),[None])

    def test_bounded_confirmation(self):
        box=[10,10,30,40];det=dict(box_xyxy=box,score=.9,side=0)
        ds=[None]*20;ds[3]=ds[7]=det
        keep,_=assess([box]*20,ds,30)
        self.assertEqual(np.flatnonzero(keep).tolist(),list(range(3,8)))
        boxes=[box]*20;boxes[5]=[0,0,400,400]
        keep,_=assess(boxes,ds,30)
        self.assertEqual(np.flatnonzero(keep).tolist(),[3,7])

    def test_recovery_and_failed_recovery(self):
        box=[10,10,30,40];bad=[0,0,500,500]
        det={t:[dict(side=0,score=.9,box_xyxy=box)] for t in range(30)}
        selected={(t,0):dict(box_xyxy=bad) for t in range(30)}
        calls=[]
        def recover(side,start,stop,anchor,detection):
            calls.append((start,stop))
            return {t:dict(box_xyxy=box if start==0 else bad) for t in range(start,stop)}
        result,blocked,_=validate_and_recover(selected,det,30,30,recover)
        self.assertEqual(calls,[(0,15),(15,30)])
        self.assertEqual(len(result),15)
        self.assertFalse(blocked[:15,0].any())
        self.assertTrue(blocked[15:,0].all())
        self.assertTrue(blocked[:,1].all())

    def test_no_retry_without_new_anchor(self):
        box=[10,10,30,40]
        det={t:[dict(side=0,score=.9,box_xyxy=box)] for t in [3,7]}
        selected={(t,0):dict(box_xyxy=box) for t in range(15)}
        def recover(*args):self.fail('Already confirmed anchors must not trigger redundant recovery')
        _,blocked,_=validate_and_recover(selected,det,15,30,recover)
        self.assertEqual(int((~blocked[:,0]).sum()),5)

    def test_archive_retains_blocked_mask(self):
        import tempfile
        from pathlib import Path
        from .refine import save_box_archive, load_box_archive
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'boxes.npz'
            blocked=np.array([[False,True],[True,True]])
            save_box_archive(path,[{0:[1,1,5,5]},{}],10,10,blocked)
            _,valid=load_box_archive(path,2,10,10)
            with np.load(path) as a:np.testing.assert_array_equal(a['interpolation_blocked'],blocked)
            self.assertFalse((valid & blocked).any())
            with self.assertRaises(ValueError):save_box_archive(path,[{0:[1,1,5,5]},{}],10,10,np.ones((2,2),bool))

    def test_forbid_interpolation(self):
        from pipelines.wilor.temporal import fill_short_gaps
        valid=np.array([[1,0,1],[1,0,1]],bool)
        blocked=~valid
        _,keep,_,_,reasons=fill_short_gaps(np.zeros((2,3,21,3)),valid,np.zeros((2,3,1,3)),np.zeros(3,bool),30,interpolation_blocked=blocked)
        np.testing.assert_array_equal(keep,valid)
        self.assertEqual(reasons[0]['tracking_failure'],1)

if __name__=='__main__':unittest.main()
