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

## Results (complete, 2026-11-13; cam00, 1352x1014, 300 frames, unified_eval)
| Scene | PSNR | SSIM | LPIPS-A | LPIPS-V | Init s | Train s | VRAM GiB | FPS | MiB |
|---|---|---|---|---|---|---|---|---|---|
| sear_steak | 34.77 | 0.961 | 0.065 | 0.182 | 1049 | 1094 | 1.24 | 336.3 | 124.0 |
| coffee_martini | 29.81 | 0.923 | 0.100 | 0.196 | 1093 | 1098 | 1.24 | 292.5 | 124.0 |
| cook_spinach | 32.66 | 0.954 | 0.074 | 0.192 | 1247 | 1085 | 1.23 | 335.7 | 124.0 |
| cut_roasted_beef | 33.01 | 0.955 | 0.073 | 0.190 | 943 | 1079 | 1.23 | 334.4 | 124.0 |
| flame_salmon_1 | 30.06 | 0.925 | 0.096 | 0.189 | 1130 | 1103 | 1.24 | 298.2 | 124.0 |
| flame_steak | 34.58 | 0.960 | 0.065 | 0.182 | 1038 | 1084 | 1.23 | 341.6 | 124.0 |
| **Mean** | **32.48** | 0.946 | 0.079 | 0.189 | 3.61 s/frame | 3.63 s/frame | 1.23 (max 1.24) | 323.1 | 0.413 MiB/frame |

Total time per frame 7.25 s (Init + Train; Load excluded). Single runs, no tuning on cam00.
- sear_steak alignment check flagged frame 269 (own GT 34.38 vs GT k-5 34.44 dB). GT is near-static
  there (GT k vs k+5 = 39-41 dB); sampled frames 232/274 peak at their own GT. Accepted as false positive.
- Treated as one 300-frame GoF in the Sheet (03_GoF_Details GoF_ID 0).
- Uploaded to Google Sheet 2026-11-13 (01 row 16 + footnote row 17, 02 rows 50-55, 03 rows 284-289,
  04 appended 1800 rows); backup before write in /media/tsaichenghan/2TBSSD/FreeTimeGSPlusPlus/sheet_20261113.
