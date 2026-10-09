"""Stage receipts: atomic publication, source/config keys and output integrity."""
import hashlib
import json
from pathlib import Path
from contextlib import contextmanager
import fcntl


def stamp(path):
    p=Path(path).resolve();s=p.stat()
    return dict(path=str(p),size=s.st_size,mtime_ns=s.st_mtime_ns)


def key(data, sources=()):
    files={str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    return hashlib.sha256(json.dumps(dict(data=data,sources=files),sort_keys=True).encode()).hexdigest()


def save_json(path, value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)


def valid(path, expected):
    try:
        d=json.loads(Path(path).read_text())
        return d['key']==expected and bool(d['outputs']) and all(stamp(x['path'])==x for x in d['outputs'])
    except (OSError,ValueError,KeyError,TypeError):
        return False


def complete(path, expected, outputs):
    save_json(path,dict(key=expected,outputs=[stamp(p) for p in outputs]))


@contextmanager
def lock(path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError(f'Another process owns cache lock: {p}')
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)
