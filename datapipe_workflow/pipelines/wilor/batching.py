"""Bounded CPU crop prefetch; GPU inference remains on the main thread."""
from concurrent.futures import ThreadPoolExecutor


def prefetch(iterator):
    iterator = iter(iterator)
    sentinel = object()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(next, iterator, sentinel)
        while True:
            item = pending.result()
            if item is sentinel:
                return
            pending = pool.submit(next, iterator, sentinel)
            yield item


def crop_batches(rows, batch_size):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    from torch.utils.data._utils.collate import default_collate
    pending, frame_ids = [], []
    for frame, crop in rows:
        pending.append(crop); frame_ids.append(frame)
        if len(pending) == batch_size:
            yield frame_ids, default_collate(pending)
            pending, frame_ids = [], []
    if pending:
        yield frame_ids, default_collate(pending)
