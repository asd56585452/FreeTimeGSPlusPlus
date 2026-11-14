"""Build Google Sheet rows for FreeTimeGS++ (no UFM) from benchmark_20261113 (read-only on run outputs).

Init = sfm + prep + RoMa points + init (video decode 'extract' excluded) (stage_times.json). Train = optimisation + final save
(benchmark_train.json). VRAM = training peak. FPS = rasterization only, 20 warmup. Size = gaussians.pt.
FreeTimeGS++ trains all 300 frames jointly: treated as one 300-frame GoF (GoF_ID 0, frames 0-299).
"""
import json, sys
from pathlib import Path
import numpy as np

RUN = Path('/media/tsaichenghan/2TBSSD/FreeTimeGSPlusPlus/benchmark_20261113/output')
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
SCENES = ['sear_steak', 'coffee_martini', 'cook_spinach', 'cut_roasted_beef', 'flame_salmon_1', 'flame_steak']
NAME = 'FreeTimeGS++ (no UFM)'
DATE = '2026-11-13'
# 2026-11-14: video decode (extract) excluded, as for all other methods (they read pre-extracted frames).
INIT = ['sfm', 'prep', 'points', 'init']
r = lambda x, n: round(float(x), n)
r2, r3, r4, agg = [], [], [], []
for s in SCENES:
    o = RUN / s
    st = json.load(open(o / 'stage_times.json')); bt = json.load(open(o / 'benchmark_train.json'))
    fp = json.load(open(o / 'render_fps.json')); um = json.load(open(o / 'unified_metrics.json'))
    # sear_steak frame 269: GT is near-static there (GT k vs k+5 = 39-41 dB), own 34.38 vs k-5 34.44 (0.06 dB);
    # frames 232/274 peak clearly at their own GT. Inspected 2026-11-13 -> false positive, accepted.
    ALLOW = {('sear_steak', 269)}
    assert all((s, b[0]) in ALLOW for b in um['mean']['alignment_fail']), (s, um['mean']['alignment_fail'])
    fr = um['frames']; assert len(fr) == 300, s
    I = sum(st[k] for k in INIT); T = bt['train_recon_time']; MB = fp['size_mib']; V = bt['peak_vram_gib']; F = fp['render_fps']
    m = um['mean']
    r2.append([NAME, 'N3DV', s, 300, r(I, 2), r(T, 2), r(I + T, 2), r((I + T) / 300, 3), r(m['psnr'], 3), r(m['ssim'], 4),
               r(m['lpips_alex'], 4), r(MB, 2), r(MB / 300, 4), r(F, 1), r(V, 2), DATE, r(m['lpips_vgg'], 4)])
    # One GoF = the whole 300-frame sequence (trained jointly).
    r3.append([NAME, 'N3DV', s, 0, 0, 299, 300, r(I, 2), r(T, 2), r(I + T, 2), r(I / 300, 3), r(T / 300, 3), r((I + T) / 300, 3),
               r(V, 2), r(m['psnr'], 3), r(m['ssim'], 4), r(m['lpips_alex'], 4), r(MB, 4), r(MB / 300, 6), r(F, 1), r(m['lpips_vgg'], 4)])
    for y in fr:
        r4.append([NAME, 'N3DV', s, 0, y['frame'], y['frame'], f"{y['frame']:05d}.png", r(y['psnr'], 3),
                   r(y['ssim'], 4), r(y['lpips_alex'], 4), r(y['lpips_vgg'], 4)])
    agg.append(((I + T) / 300, MB / 300, V, F, m['psnr'], m['ssim'], m['lpips_alex'], m['lpips_vgg'], I / 300, T / 300))
a = np.array(agg)
lb = [NAME, r(a[:, 0].mean(), 2), r(a[:, 4].mean(), 2), r(a[:, 5].mean(), 4), r(a[:, 6].mean(), 4), r(a[:, 1].mean(), 4),
      r(a[:, 3].mean(), 1), r(a[:, 2].mean(), 2), r(a[:, 2].max(), 2), r(a[:, 7].mean(), 4)]
for n, v in (('02', r2), ('03', r3), ('04', r4), ('01', lb)):
    json.dump(v, open(OUT / f'{n}.json', 'w'))
print('LB', lb)
print(f'time/frame init {a[:, 8].mean():.2f} + train {a[:, 9].mean():.2f}; rows', len(r2), len(r3), len(r4))
for row in r2:
    print(row[2], row[8:11], row[16], 'init', row[4], 'train', row[5], 'vram', row[14], 'fps', row[13], 'MiB', row[11])
