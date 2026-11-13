# FreeTimeGS++ N3DV benchmark (no-UFM preset)

Upstream: zhixiongzuo/FreeTimeGSPlusPlus `661be49`; fork asd56585452, branch `benchmark-n3dv`.
Run: `/media/tsaichenghan/2TBSSD/FreeTimeGSPlusPlus/benchmark_20261113` (frozen, hashed code).
Launcher: `bash scripts/benchmark/run_n3dv.sh` (resumable; Ctrl-C / `docker stop ftgspp-n3dv` safe).

## Protocol / deviations
- Config: paper final preset (gated marginalization + color correction, max_duration 1) **without UFM**
  init (`temporal_motion_adapted=false`); paper ablation: UFM +0.11 dB on DyNeRF.
- Cameras: poses_bounds.npy (same as other methods), no COLMAP/MVS. Train = all non-cam00 cameras
  (upstream cook_spinach config uses cams 1-19; we use 1-20). cam00 eval only.
- `FTGS_BENCH=1` patches: cam00 excluded from RoMa point init; in-loop cam00 eval (every 500 it) off.
- Images: upstream resize (torchvision Resize, half res = 1352x1014); metrics use the shared
  unified_eval GT (PIL bicubic) on saved uint8 cam00 renders, LPIPS v0.1 on [-1,1].
- Engineering only (no effect on results): 4 decode workers + 22 GB container RAM cap (20 workers
  exhausted host RAM), writable tiny-cuda-nn RTC cache mount, eval run in asdggg/3dgstream (lpips).
- Init_Time = extract + sfm + prep + RoMa points + init. Train_Recon = optimisation + final save
  (dataset load excluded). VRAM = torch max_memory_allocated during training. FPS = rasterization only,
  20 warmup renders. Size = gaussians.pt.

## Results (in progress)
| Scene | PSNR | SSIM | LPIPS-A | LPIPS-V | Init s | Train s | VRAM GiB | FPS | MiB |
|---|---|---|---|---|---|---|---|---|---|
| coffee_martini | 29.81 | 0.923 | 0.100 | 0.196 | 1093 | 1098 | 1.24 | 292.5 | 124.0 |

Stage times coffee_martini: extract 298, sfm 3, prep 118, points 660, init 14, train 1105 (wall), render 69.
Remaining 5 scenes started 2026-11-13 12:47, ~37 min/scene -> expected ~16:00.
