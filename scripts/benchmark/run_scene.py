"""Run one N3DV scene through the FreeTimeGS++ pipeline with per-stage wall-clock timing.

Resumable: a stage with a recorded timing is skipped. Each stage is a separate process, as in
upstream ./run. Writes <run>/stage_times.json. cwd must be the method work dir on the SSD
(generated _extracted/_memmap/_points paths are relative to it).
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

STAGES = {
    'extract': ['-m', 'ftgspp.data.extract', '{cfg}'],
    'sfm': ['-m', 'ftgspp.data.sfm', '{cfg}'],
    'prep': ['-m', 'ftgspp.data.prep', '{cfg}'],
    'points': ['-m', 'ftgspp.init.points', '{cfg}'],
    'init': ['-m', 'ftgspp.init', '{cfg}', '{run}'],
    'train': ['-m', 'ftgspp.train', '{cfg}', '{run}'],
    'render': ['/opt/ftgspp/scripts/benchmark/render_cam00.py', '{cfg}', '{run}', '{run}/cam00'],
}

p = argparse.ArgumentParser()
p.add_argument('config', type=Path)
p.add_argument('run', type=Path)
p.add_argument('--to', default='render', choices=list(STAGES))
a = p.parse_args()
a.run.mkdir(parents=True, exist_ok=True)
tf = a.run / 'stage_times.json'
times = json.loads(tf.read_text()) if tf.exists() else {}
env = dict(os.environ, FTGS_BENCH='1')
for name, argv in STAGES.items():
    if name in times:
        print(f'SKIP {name} ({times[name]:.1f}s)', flush=True)
    else:
        # A stage without a recorded time never finished: discard its partial output (timer restarts).
        import shutil, tomllib
        c = tomllib.loads(a.config.read_text())
        partial = {'extract': c['data']['extracted_path'], 'sfm': c['data']['colmap_path'],
                   'prep': c['data']['memmap_path'], 'points': c['init']['points_path']}.get(name)
        if partial and Path(partial).exists():
            shutil.rmtree(partial)
        if name == 'render' and (a.run / 'cam00').exists():
            shutil.rmtree(a.run / 'cam00')
        cmd = [sys.executable] + [x.format(cfg=a.config, run=a.run) for x in argv]
        print('RUN ' + ' '.join(cmd), flush=True)
        t0 = time.perf_counter()
        with (a.run / f'{name}.log').open('a') as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        times[name] = time.perf_counter() - t0
        tf.write_text(json.dumps(times, indent=2))
        print(f'DONE {name} {times[name]:.1f}s', flush=True)
    if name == a.to:
        break
