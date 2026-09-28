# file-000

Masked copy of file-000. The original export and file-000_stable are unchanged.

Thresholds are speed and time, using the 30 fps meaning of the old rules. No interpolation.

- Camera speed above 2.4 m/s breaks the track. Wrists are not judged across that break.
- Inside one continuous camera segment, a wrist faster than 1.5 m/s is abnormal. If the previous hand returns within 3.0 s at no more than 1.5 m/s, the frames in between stay hidden.
- Empty frames stay empty.

Left: kept 1490, masked 1112.
Right: kept 2206, masked 1330.
Max remaining adjacent speed: left 1.50 m/s, right 1.50 m/s.
Camera breaks: 146.
