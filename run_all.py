# Run every check script, in parallel, with a content-keyed cache (checkcache.py): a script reruns only when it or what it reads changed.
# Run: ~/.venvs/jev/bin/python run_all.py          (cached)
#      ~/.venvs/jev/bin/python run_all.py --full   (recompute everything, e.g. before publishing)
import glob, os, sys, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
import checkcache

scripts = sorted(glob.glob('ga_*_check.py') + glob.glob('inv_*_check.py') + glob.glob('pub_*_check.py') + ['paper2_cs_check.py'])
t0 = time.time()
res = checkcache.results(scripts, full='--full' in sys.argv)
tot_p = tot_f = n_run = 0
for s in scripts:
    r = res[s]
    p = r['stdout'].count('PASS ')
    f = r['stdout'].count('FAIL ') + (r['rc'] != 0)
    tot_p, tot_f, n_run = tot_p + p, tot_f + f, n_run + (not r['cached'])
    note = 'cached' if r['cached'] else 'ran'
    err = (r['err'].strip().splitlines() or [''])[-1][:80] if r['rc'] else ''
    print(f'{s:36s} {p:4d} pass {f:2d} fail  {note:6s} {err}')
print(f'TOTAL {len(scripts)} scripts ({n_run} ran, {len(scripts) - n_run} cached), PASS {tot_p}, FAIL {tot_f}, {time.time() - t0:.0f}s')
