# Run every check script in one process (one uv env, one sympy import) and print per-script timing.
# Run: uv run --with sympy --with numpy --with scipy python run_all.py
import contextlib, glob, io, os, runpy, sys, time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
scripts = sorted(glob.glob('ga_*_check.py') + glob.glob('inv_*_check.py') + glob.glob('pub_*_check.py') + ['paper2_cs_check.py'])
tot_p = tot_f = 0
for s in scripts:
    out, t0 = io.StringIO(), time.time()
    err = ''
    with contextlib.redirect_stdout(out):
        try:
            runpy.run_path(s, run_name='__main__')
        except BaseException as e:            # ok() asserts; keep going with the next script
            err = f'{type(e).__name__}: {e}'
    txt = out.getvalue()
    p, f = txt.count('PASS '), txt.count('FAIL ') + bool(err)
    tot_p, tot_f = tot_p + p, tot_f + f
    print(f'{s:32s} {p:4d} pass {f:2d} fail {time.time() - t0:6.1f}s {err[:80]}', flush=True)
print(f'TOTAL {len(scripts)} scripts, PASS {tot_p}, FAIL {tot_f}')
