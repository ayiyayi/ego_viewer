"""Reuse exactly the backend's preprocessed tensor across left/right sessions.

Scoped to one job and one folder. Recovery folders bypass the cache. No backend
files are changed and the original loader is always restored.
"""
from contextlib import contextmanager
from pathlib import Path
import inspect


@contextmanager
def shared_frame_cache(folder):
    import sam3.model.sam3_video_inference as backend
    original = backend.load_resource_as_video_frames
    signature = inspect.signature(original)
    cached = {}
    root = str(Path(folder).resolve())
    def load(*args, **kwargs):
        bound = signature.bind(*args, **kwargs); bound.apply_defaults()
        values = dict(bound.arguments)
        resource = values.pop('resource_path')
        if not isinstance(resource, str) or str(Path(resource).resolve()) != root:
            return original(*args, **kwargs)
        key = repr(sorted(values.items()))
        if key not in cached:
            cached[key] = original(*args, **kwargs)
        else:
            print('SAM3 reused preprocessed frames for second hand', flush=True)
        return cached[key]
    backend.load_resource_as_video_frames = load
    try:
        yield
    finally:
        backend.load_resource_as_video_frames = original
        cached.clear()
