"""Save cam00 renders (uint8 PNG, 1352x1014) for every frame + warmed-up render FPS.

Renders exactly as ftgspp.eval.eval_rgb (no color correction). Camera/time tensors are moved to
the GPU before timing; FPS = rasterization of the 4D Gaussians only.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from ftgspp.data.utils.multiview import MultiViewData
from ftgspp.models.gaussians import Gaussians
from ftgspp.utils import Config
from ftgspp.utils.config import camera_indexer

p = argparse.ArgumentParser()
p.add_argument('config', type=Path)
p.add_argument('run_path', type=Path)
p.add_argument('out', type=Path)
p.add_argument('--warmup', type=int, default=20)
p.add_argument('--repeats', type=int, default=3)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
cfg = Config.load(a.config)
ds = MultiViewData.load(cfg.data.memmap_path)
ev = ds.at(cfg.data.frames.into_slice(), camera_indexer(cfg.data.eval_cameras)).flatten()
gs = Gaussians.load(a.run_path / 'gaussians.pt').cuda().eval()
cams = [(b.time.cuda(), b.w2c.unsqueeze(0).cuda(), b.intrinsic.unsqueeze(0).cuda(), (int(b.height), int(b.width)), int(b.frame))
        for b in ev]
H, W = cams[0][3]
psnrs = []
with torch.inference_mode():
    for (t, w2c, K, shape, frame), b in zip(cams, ev):
        pred, _, _ = gs(t=t, w2c=w2c, intrinsic=K, shape=shape)
        img = (pred[0].clamp(0, 1) * 255 + 0.5).floor().byte().cpu().numpy()
        Image.fromarray(img).save(a.out / f'{frame:05d}.png')
        gt = b.rgb.float() / 255
        psnrs.append(float(-10 * np.log10(((img.astype(np.float32) / 255 - gt.numpy()) ** 2).mean())))
    for i in range(a.warmup):
        t, w2c, K, shape, _ = cams[i % len(cams)]
        gs(t=t, w2c=w2c, intrinsic=K, shape=shape)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(a.repeats):
        for t, w2c, K, shape, _ in cams:
            gs(t=t, w2c=w2c, intrinsic=K, shape=shape)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
rec = dict(width=W, height=H, num_frames=len(cams), render_fps=a.repeats * len(cams) / dt,
           warmup=a.warmup, repeats=a.repeats, psnr_vs_method_gt=float(np.mean(psnrs)),
           size_mib=(a.run_path / 'gaussians.pt').stat().st_size / 1024**2,
           definition='4D Gaussian rasterization only, inputs on GPU, synchronized; size = gaussians.pt')
(a.out.parent / 'render_fps.json').write_text(json.dumps(rec, indent=2))
print('RENDER ' + json.dumps(rec), flush=True)
