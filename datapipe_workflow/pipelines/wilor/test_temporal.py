"""Run directly with the existing hawor interpreter; no GPU required."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from pipelines.wilor.temporal import interpolate_params, optimize_translation, segments, fill_short_gaps, smooth_interpolated


class TemporalTests(unittest.TestCase):
    def test_constant_velocity_and_jitter(self):
        x=np.zeros((15,3));x[:,0]=np.arange(15)*.01
        np.testing.assert_allclose(optimize_translation(x,np.ones(15)),x,atol=1e-7)
        noisy=x.copy();noisy[7,1]=.02
        clean=optimize_translation(noisy,np.ones(15))
        self.assertLess(np.linalg.norm(np.diff(clean,n=2,axis=0)),np.linalg.norm(np.diff(noisy,n=2,axis=0)))
        self.assertLess(np.linalg.norm(clean-x),np.linalg.norm(noisy-x))

    def test_no_crossing_camera_or_absence(self):
        valid=np.array([1,1,1,1,0,1,1],bool)
        breaks=np.array([0,0,1,0,0,0,0],bool)
        self.assertEqual(list(segments(valid,breaks)),[(0,2),(2,4),(5,7)])

    def test_slerp_wrap_and_anchors(self):
        p=dict(wrist=np.zeros((2,3,3)),betas=np.zeros((2,3,10)),world_rot=np.tile(np.eye(3),(2,3,1,1)),pose=np.tile(np.eye(3),(2,3,15,1,1)))
        p['world_rot'][:,0]=Rotation.from_euler('z',170,degrees=True).as_matrix()
        p['world_rot'][:,2]=Rotation.from_euler('z',-170,degrees=True).as_matrix()
        p['pose'][:,0]=p['world_rot'][:,0,None];p['pose'][:,2]=p['world_rot'][:,2,None]
        p['wrist'][:,2,0]=.06
        valid=np.array([[1,0,1],[1,0,1]],bool);keep=np.ones_like(valid)
        out=interpolate_params(p,valid,keep)
        np.testing.assert_allclose(out['wrist'][:,1,0],.03)
        for key in ('world_rot','pose'):
            np.testing.assert_array_equal(out[key][valid],p[key][valid])
            mat=out[key][:,1].reshape(-1,3,3)
            np.testing.assert_allclose(np.linalg.det(mat),1.,atol=1e-7)
            np.testing.assert_allclose(Rotation.from_matrix(mat).magnitude(),np.pi,atol=1e-7)

    def test_left_right_world_rotation_roundtrip(self):
        camera=Rotation.from_euler('xyz',[15,30,50],degrees=True).as_matrix()
        mano=Rotation.from_euler('xyz',[-20,10,5],degrees=True).as_matrix()
        for mirror in (np.eye(3),np.diag([-1.,1.,1.])):
            world=camera@mirror@mano@mirror
            self.assertAlmostEqual(np.linalg.det(world),1.)
            np.testing.assert_allclose(mirror@camera.T@world@mirror,mano,atol=1e-7)

    def test_gap_limit_and_camera_break(self):
        n=13
        joints=np.zeros((2,n,21,3),np.float32)
        valid=np.zeros((2,n),bool)
        valid[0,[0,10]]=True  # nine missing frames = 0.3 seconds
        valid[1,[0,11]]=True  # ten missing frames: too long
        vertices=np.zeros((2,n,1,3),np.float32)
        breaks=np.zeros(n,bool)
        _,keep,_,filled,_=fill_short_gaps(joints,valid,vertices,breaks,30)
        self.assertEqual(filled.sum(axis=1).tolist(),[9,0])
        self.assertFalse(keep[:,12].any())
        breaks[5]=True
        _,_,_,filled,_=fill_short_gaps(joints,valid,vertices,breaks,30)
        self.assertFalse(filled.any())

    def test_endpoint_gates_reject_other_hand(self):
        j=np.zeros((2,5,21,3),np.float32)
        valid=np.zeros((2,5),bool);valid[:,[0,4]]=True
        j[0,4,:,0]=.11  # wrist endpoint distance is too large
        j[1,4,1:,1]=.04  # a different finger pose, even at the same wrist
        _,_,_,filled,reasons=fill_short_gaps(j,valid,np.zeros((2,5,1,3)),np.zeros(5,bool),30)
        self.assertFalse(filled.any())
        self.assertEqual(reasons[0]['endpoint_motion'],1)
        self.assertEqual(reasons[1]['endpoint_pose'],1)

    def test_translation_bounds_and_finger_pose(self):
        n=15
        j=np.zeros((2,n,21,3),np.float32);j[:,:,:,2]=.3
        j[:,7,:,0]=.08
        keep=np.ones((2,n),bool)
        params=dict(wrist=j[:,:,0].copy())
        data=dict(R_w2c=np.tile(np.eye(3),(n,1,1)),t_w2c=np.zeros((n,3)),focal=1000)
        js,_,_,shift,pixels=smooth_interpolated(params,keep,keep,np.zeros(n,bool),30,j,j.copy(),data,np.zeros_like(keep))
        self.assertGreater(np.linalg.norm(shift),0)
        self.assertLessEqual(np.linalg.norm(shift,axis=-1).max(),.015001)
        self.assertLessEqual(pixels.max(),5.001)
        np.testing.assert_allclose(js-js[:,:,0:1],j-j[:,:,0:1],atol=1e-7)

    def test_empty_hands(self):
        n=5
        keep=np.zeros((2,n),bool)
        params=dict(wrist=np.zeros((2,n,3),np.float32))
        joints=np.zeros((2,n,21,3),np.float32)
        verts=np.zeros((2,n,778,3),np.float32)
        data=dict(R_w2c=np.tile(np.eye(3),(n,1,1)),t_w2c=np.zeros((n,3)),focal=600)
        js,vs,_,_,pixels=smooth_interpolated(params,keep,keep,np.zeros(n,bool),30,joints,verts,data,keep)
        self.assertEqual(pixels.size,0)
        self.assertFalse(js.any())
        self.assertFalse(vs.any())


if __name__=='__main__':unittest.main()
