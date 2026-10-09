"""One SAM3 model for all jobs in a video, isolated in the SAM3 environment."""
import argparse
import json
import sys
from pathlib import Path
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sam3_hand_tracking import sam3_point


def run_jobs(jobs, build=None, run=None):
    cache_enabled = run is None and build is None
    from pipelines.wilor.cache import key as cache_key, stamp, valid, complete
    build = build or sam3_point.simple.build_predictor
    run = run or sam3_point.run_point_prompt
    predictor = None
    identity = None
    reports = []
    for argv in jobs:
        args = sam3_point.parse_args(argv)
        key = (args.checkpoint_path, args.default_output_prob_thresh, args.sam_version)
        if identity is not None and key != identity:
            raise ValueError('Resident SAM3 jobs must use the same model configuration')
        out=Path(args.output_dir)
        if cache_enabled:
            job_key=cache_key(dict(args=vars(args),detection=stamp(Path(args.segment_dir)/'detection/bboxes_2d.npz'),
                weights=stamp(args.checkpoint_path),frames=[stamp(p) for p in sorted((Path(args.segment_dir)/'frames').glob('*.jpg'))]),[p for p in Path(__file__).parent.glob('*.py') if not p.name.startswith('test_')])
            if valid(out/'job_receipt.json',job_key):
                reports.append(dict(output=str(out),seconds=0.,cache_hit=True))
                print(f'CACHE HIT: SAM {out}',flush=True)
                continue
        if predictor is None:
            t = time.perf_counter()
            predictor = build(*key)
            identity = key
            print(f'SAM3 resident model loaded in {time.perf_counter()-t:.2f}s', flush=True)
        t = time.perf_counter()
        if args.compact_masks and args.sam_version == 'sam3':
            from sam3_hand_tracking.frame_cache import shared_frame_cache
            with shared_frame_cache(Path(args.segment_dir)/'frames'):
                run(args, predictor=predictor)
        else:
            run(args, predictor=predictor)
        if cache_enabled and args.compact_masks:
            outputs=[out/'slam_masks.npz',out/'sam_tight_bboxes_2d.jsonl',out/'point_prompt_summary.json']
            if args.validate_tracks:outputs += [out/'tracking_validation.npz',out/'tracking_validation.json']
            complete(out/'job_receipt.json',job_key,outputs)
        reports.append({'output': args.output_dir, 'seconds': time.perf_counter()-t})
        print(f'SAM3 resident job {len(reports)}/{len(jobs)} done', flush=True)
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=Path, required=True)
    args = parser.parse_args()
    reports = run_jobs(json.loads(args.jobs.read_text()))
    args.jobs.with_suffix('.timings.json').write_text(json.dumps(reports, indent=2))
