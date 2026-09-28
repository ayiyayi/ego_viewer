# P03_05_02_wilor_sam_finebio_cam_v2_stable

Filtered copy of P03_05_02_wilor_sam_finebio_cam_v2. The original export is unchanged.

Same rules as P28_01_01_wilor_sam_finebio_cam_v2_stable.

Abnormal frames are masked and are not replaced:
- A camera step above 8 cm breaks the track. A wrist is not judged against the other side of that break, and nothing is interpolated across it.
- Inside one continuous camera segment, a wrist moving faster than 1.5 m/s is abnormal. If the previous hand comes back within 90 frames, the frames in between stay hidden.

Interpolation is only for a hole of at most 5 frames that originally had no hand. The wrists on both sides must be within 10 cm, each filled step must stay at or below 1.5 m/s, and the finger pose must agree within 3 cm. If those checks fail, the hole stays empty.

Left: kept 7142, masked 4, interpolated 21.
Right: kept 3772, masked 212, interpolated 50.
Max remaining adjacent speed: left 1.37 m/s, right 1.50 m/s.
