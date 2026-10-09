"""Lossless bit-packed masks: cheap writes, no full-resolution PNG codec."""
import struct
import zlib
from pathlib import Path
import numpy as np


def write_mask(path, mask):
    path = Path(path).with_suffix('.mask')
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.asarray(mask, bool)
    path.write_bytes(struct.pack('<II', *mask.shape) + zlib.compress(np.packbits(mask).tobytes(), level=1))
    return str(path)


def read_mask(path):
    path = Path(path)
    if path.suffix != '.mask':
        import cv2
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None: raise ValueError(f'Cannot read mask: {path}')
        return mask > 0
    data = path.read_bytes(); h, w = struct.unpack('<II', data[:8])
    packed = np.frombuffer(zlib.decompress(data[8:]), np.uint8)
    return np.unpackbits(packed, count=h*w).reshape(h,w).astype(bool)


def reduce_union(masks, height, width):
    import cv2
    mask = np.zeros((height, width), bool)
    for value in masks: mask |= value
    scale = (384*512/(height*width))**.5
    h,w = int(height*scale),int(width*scale)
    return cv2.resize(mask.astype(np.float32),(w,h),interpolation=cv2.INTER_AREA)[:h-h%8,:w-w%8] > 0
