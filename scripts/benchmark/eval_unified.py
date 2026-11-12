"""unified_eval_20261108 metrics for FreeTimeGS++ cam00 renders (identical definitions).

GT: raw <raw>/cam00/<i>.png, PIL BICUBIC to 1352x1014, uint8. Renders: <run>/cam00/<i:05d>.png.
PSNR per frame (mean), SSIM 3DGS Gaussian 11x11 sigma 1.5, LPIPS v0.1 Alex/VGG on [-1,1].
Alignment check: frame k must match own GT better than GT k+-5 on sampled frames.
"""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, lpips
from PIL import Image

run, raw = Path(sys.argv[1]), Path(sys.argv[2])
N = 300
g = torch.exp(-(torch.arange(11) - 5.0) ** 2 / (2 * 1.5 ** 2)); g = g / g.sum()
W = (g[:, None] @ g[None]).cuda().expand(3, 1, 11, 11).contiguous()
def ssim(a, b):
    m1, m2 = F.conv2d(a, W, padding=5, groups=3), F.conv2d(b, W, padding=5, groups=3)
    s1 = F.conv2d(a * a, W, padding=5, groups=3) - m1 ** 2; s2 = F.conv2d(b * b, W, padding=5, groups=3) - m2 ** 2
    s12 = F.conv2d(a * b, W, padding=5, groups=3) - m1 * m2; C1, C2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * m1 * m2 + C1) * (2 * s12 + C2)) / ((m1 ** 2 + m2 ** 2 + C1) * (s1 + s2 + C2))).mean().item()
def load(p): return torch.from_numpy(np.asarray(Image.open(p).convert('RGB'))).permute(2, 0, 1)
def gt(i):
    with Image.open(raw / 'cam00' / f'{i}.png') as im:
        return torch.from_numpy(np.asarray(im.convert('RGB').resize((1352, 1014), Image.Resampling.BICUBIC))).permute(2, 0, 1)
def psnr(a, b): return (-10 * torch.log10(((a - b) ** 2).mean())).item()
alex = lpips.LPIPS(net='alex', verbose=False).cuda().eval(); vgg = lpips.LPIPS(net='vgg', verbose=False).cuda().eval()
G = [gt(i) for i in range(N)]
recs = []
with torch.no_grad():
    for i in range(N):
        r = load(run / 'cam00' / f'{i:05d}.png')
        if r.shape != G[i].shape: raise SystemExit(f'frame {i}: {tuple(r.shape)} vs GT {tuple(G[i].shape)}')
        a = (r.float() / 255)[None].cuda(); b = (G[i].float() / 255)[None].cuda()
        recs.append(dict(frame=i, psnr=psnr(a, b), ssim=ssim(a, b),
                         lpips_alex=alex(a * 2 - 1, b * 2 - 1).item(), lpips_vgg=vgg(a * 2 - 1, b * 2 - 1).item()))
    bad = []
    for i in range(10, 290, 37):
        a = load(run / 'cam00' / f'{i:05d}.png').float().cuda() / 255
        own = psnr(a, G[i].float().cuda() / 255); oth = max(psnr(a, G[i + d].float().cuda() / 255) for d in (-5, 5))
        if own <= oth: bad.append((i, round(own, 2), round(oth, 2)))
mean = {k: float(np.mean([x[k] for x in recs])) for k in ('psnr', 'ssim', 'lpips_alex', 'lpips_vgg')}
mean['alignment_fail'] = bad
(run / 'unified_metrics.json').write_text(json.dumps(dict(mean=mean, frames=recs, definition=__doc__), indent=2))
print('UNIFIED ' + json.dumps(mean), flush=True)
if bad: raise SystemExit('alignment check failed')
