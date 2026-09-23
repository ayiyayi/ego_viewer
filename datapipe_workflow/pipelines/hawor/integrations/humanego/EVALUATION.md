# Evaluating the HaWoR → HumanEgo integration

"Is the integration any good?" splits into two independent questions, evaluated differently:

- **Signal fidelity** — are HaWoR's camera / hand / object geometry *correct*?
- **Task utility** — can a HumanEgo policy actually *learn* from HaWoR-fed data?

This doc gives four evals across both, with the scripts in this folder. All run in the
`humanego` env unless noted.

| # | Eval | Question | GT needed | Status |
|---|------|----------|-----------|--------|
| **B1** | self-consistency (`eval_self_consistency.py`) | fidelity | none | ✅ runs now |
| **A** | training convergence (`FlowMatchingTrainer`) | utility | none (held-out clip) | ✅ runs now |
| **B2** | HaWoR vs Aria-MPS (`compare_hawor_vs_aria.py`) | fidelity (absolute) | Aria MPS | ⚙ needs HaWoR env |
| **Ablation** | Aria- vs HaWoR-annotated training | utility (absolute) | Aria MPS | ⚙ needs HaWoR env |

---

## B1 — Geometric self-consistency (no GT)

Projects each 3D entity into every frame via that frame's HaWoR camera pose and checks it
against an **independently-produced** SAM2 mask; plus metric trajectory sanity. Low
reprojection error + plausible metric motion ⇒ camera/hand/object geometry is mutually
consistent. (Necessary, not sufficient: cannot catch a globally-wrong-but-consistent frame
— that's B2.)

```bash
python eval_self_consistency.py \
    --session $HE/data/scilab/aria/mps_scilab_000_vrs --out b1.json
```

**Result on `mps_scilab_000_vrs` (150 frames):**

| metric | value | reading |
|--------|-------|---------|
| hand reproj vs arm mask | **39.6 px median = 1.28 % of diag** | camera × hand poses consistent ✅ |
| obj2 reproj vs mask | **1.8 px median** | excellent |
| obj1 reproj vs mask | 107 px median | moderate — see parallax below |
| hand speed | median 0.1–0.2, max ~1.7 m/s | human-plausible, metric ✅ |
| camera path length | **0.28 m** over the window | **low parallax** ⇒ weak triangulation for far/large obj1 |
| obj extents | obj1 0.37 m, obj2 0.32 m diag | obj2 large for a tube ⇒ static-object assumption / track spread |

Takeaways: the **camera↔hand** geometry is solid; **object** triangulation quality is
gated by camera parallax (egocentric heads move little during fine manipulation) and by the
static-object assumption. Levers: widen the object-centric window, raise `CoTracker`
frames, or use frames with more head motion.

---

## A — Training convergence (task utility, held-out clip)

`FlowMatchingTrainer` trains on all `scilab` recordings except `000` and evaluates on `000`
every epoch: hand-trajectory **position error (mm)**, **orientation error (deg)**, **grasp
F1**, plus a GT-vs-pred render. A downward eval curve ⇒ the HaWoR-fed data is learnable.

### Run it yourself (step by step)

```bash
# 0. shell vars + env (same as TUTORIAL.md step 0a)
conda activate humanego
export HE=/mnt/data/liuyu/project/HumanEgo
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# 1. make sure the 6 scilab recordings are preprocessed (TUTORIAL.md step 2)
for r in 000 001 002 003 004 005; do
  echo -n "rec $r: "; ls $HE/data/scilab/aria/mps_scilab_${r}_vrs/preprocess/all_data/*/training_data.json | wc -l
done

# 2. (optional) clear the partial run left from the demo so you start fresh
rm -rf $HE/runs/scilab/HumanEgo

# 3. train the full 300 epochs (~1 h on one GPU; ~4 epochs/min)
cd $HE
python -m training.FlowMatchingTrainer --task scilab --use_cfg --job HumanEgo --epochs 300

# 4. watch convergence live in another shell (latest eval snapshot, refreshes when you re-run)
watch -n 30 'ls $HE/runs/scilab/HumanEgo/eval_snapshots/eval_ep_*.json | sort | tail -1 | xargs python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(\"ep%d pos=%.3fm rot=%.1f F1=%.2f\"%(d[\"epoch\"],d[\"pos_err_w_m\"],d[\"rot_err_w_deg\"],d[\"grasp_f1_w\"]))"'
```

### Read the results

- `runs/scilab/HumanEgo/eval_curve.png` — metric curves over epochs (the headline plot).
- `runs/scilab/HumanEgo/eval_snapshots/eval_ep_XXXX.json` — `pos_err_w_m`, `rot_err_w_deg`,
  `grasp_f1_w`, `done_acc` per epoch.
- `runs/scilab/HumanEgo/eval_render/epoch_XXXX/<sess>/teacher_forced_vis/evaluation_vis.mp4`
  — top = GT hand trajectory, bottom = predicted (rendered every 100 epochs).
- `latest.pt` — checkpoint.

### Partial convergence observed (6 clips × 150 frames, 5 train / 1 eval; stopped at ep 61)

| epoch | pos_err_w | rot_err_w | grasp_F1 | done_acc |
|------:|----------:|----------:|---------:|---------:|
| 1  | 0.354 m | 130.4° | 0.55 | 0.33 |
| 13 | 0.322 m | 118.5° | 0.56 | — |
| 30 | 0.288 m | 76.3°  | 0.61 | — |
| 47 | 0.275 m | 43.5°  | 0.66 | — |
| 61 | **0.258 m** | **37.3°** | **0.68** | 0.67 |

All three metrics fall monotonically and grasp/done climb → **the HaWoR-fed data is
learnable**. (This run was stopped at epoch 61 as a convergence check; run the full 300
yourself with the steps above.)

> Caveat: 6 short (~5 s) lab clips of *different* sub-actions is a tiny, heterogeneous set,
> so the absolute numbers (≈0.26 m position) are **not** a performance figure — read A as a
> plumbing/convergence check (loss falls, eval improves, render tracks GT). Real performance
> needs many same-task clips and full-length recordings (drop `--max_frames`).

---

## B2 — HaWoR vs Aria-MPS ground truth (absolute fidelity)

The rigorous fidelity test: run HaWoR on a clip that also has Aria MPS GT (`serve_bread`),
align the two world frames with one Sim(3) (from camera centers, Umeyama), then report
**camera ATE (m)** and **hand position (cm) / orientation (deg)** error vs MPS.

```bash
# 1) (humanego env) Aria .vrs -> mp4 + per-frame timestamps
python extract_vrs_rgb.py \
    --vrs $HE/data/serve_bread/aria/mps_serve_bread_000_vrs/sample.vrs \
    --out /tmp/sb_rgb --fps 30 --rotate cw

# 2) (HAWOR env) annotate that video
conda activate hawor
python /mnt/data/liuyu/project/DataPipeline/pipelines/hawor/scripts/hawor_video_processor.py \
    --video_path /tmp/sb_rgb/video.mp4 --output_dir /tmp/sb_hawor/sb --gpu_ids 0 --vis_mode off

# 3) (humanego env) compare to MPS GT
conda activate humanego
python compare_hawor_vs_aria.py \
    --aria_session $HE/data/serve_bread/aria/mps_serve_bread_000_vrs \
    --hawor_dir /tmp/sb_hawor/sb --frames_ts /tmp/sb_rgb/frames_ts.json --out b2.json
```

Harness is verified correct via `--selftest` (feeds GT as the "HaWoR" side):
camera ATE ≈ 1e-16 m, hand 0.0 cm / 0.0003°. The only blocker to real numbers is the
HaWoR env (py3.10 / torch1.13 + DROID-SLAM + weights); everything else is ready.

What "good" looks like: camera ATE of a few cm and Sim(3) scale ≈ 1 (HaWoR's Metric3D
scale matches MPS metric), hand position error within a few cm.

---

## Ablation — Aria- vs HaWoR-annotated training (absolute utility)

Train the **same** policy on the **same** `serve_bread` clips, annotated two ways, and
compare eval on the same held-out clip. If HaWoR-annotated training reaches comparable
`pos_err`/`grasp_f1`, the integration is a viable Aria-free substitute.

```bash
# Aria-annotated (released): already preprocessed under data/serve_bread/...
python -m training.FlowMatchingTrainer --task serve_bread --use_cfg --job HumanEgo

# HaWoR-annotated: ingest serve_bread HaWoR outputs (B2 step 2) into a parallel task
#   data/serve_bread_hawor/aria/mps_*_vrs  + cfg/preprocess/tasks/serve_bread_hawor.yaml
#   + cfg/training/serve_bread_hawor/HumanEgo.yaml, then:
python -m training.FlowMatchingTrainer --task serve_bread_hawor --use_cfg --job HumanEgo

# compare the two eval curves
python - <<'PY'
import json,glob
for t in ["serve_bread","serve_bread_hawor"]:
    f=sorted(glob.glob(f"runs/{t}/HumanEgo/eval_snapshots/eval_ep_*.json"))[-1]
    d=json.load(open(f)); print(t, "pos_err_w_m=%.3f"%d["pos_err_w_m"], "grasp_f1_w=%.2f"%d["grasp_f1_w"])
PY
```

Depends on B2's HaWoR run; otherwise pure procedure (no new code).

---

## Recommended order

1. **B1** now — cheap fidelity sanity (done; camera/hand consistent, object parallax-limited).
2. **A** — convergence/plumbing on held-out clip (verified to ep 61; run full 300 yourself).
3. **B2** once the `hawor` env exists — the one absolute fidelity number.
4. **Ablation** — the definitive "is HaWoR good enough to replace Aria" answer.
