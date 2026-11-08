# Cache of check results keyed by content: a script is rerun only when its source, the local modules it uses, the files it reads or
# the Python and library versions change. Shared by run_all.py and theorem_base_check.py.
# A script's key covers: its source; every local *.py named in it (import, from ... import, or a quoted 'name.py'); every quoted relative
# path that exists here or in the repository root (a directory counts with all its files); all papers/current/*/main.tex if the source
# mentions papers or .tex; the data/ directory if the source mentions data; the versions of Python, numpy, scipy and sympy.
# Force a full recomputation with: python run_all.py --full
import hashlib, json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
CACHE = os.path.join(HERE, '.check_cache.json')

def _versions():
    v = [sys.version]
    for m in ('numpy', 'scipy', 'sympy'):
        try:
            v.append(m + __import__(m).__version__)
        except Exception:
            v.append(m + '-')
    return '|'.join(v)
VERS = _versions()

def _files_of(path):
    if os.path.isdir(path):
        out = []
        for d, _, fs in os.walk(path):
            out += [os.path.join(d, f) for f in sorted(fs) if not f.startswith('.')]
        return sorted(out)
    return [path] if os.path.isfile(path) else []

def deps(script, _seen=None):
    seen = _seen if _seen is not None else set()
    p = os.path.join(HERE, script)
    if p in seen:
        return set()
    seen.add(p)
    src = open(p, encoding='utf-8', errors='replace').read()
    out = {p}
    names = set(re.findall(r'^\s*(?:from|import)\s+([A-Za-z_]\w*)', src, re.M)) | set(re.findall(r"['\"]([\w\-]+\.py)['\"]", src))
    for n in names:
        f = n if n.endswith('.py') else n + '.py'
        if os.path.isfile(os.path.join(HERE, f)) and f != script:
            out |= deps(f, seen)
    for q in re.findall(r"['\"]([\w\-./]+\.\w{1,5}|[\w\-]+/)['\"]", src):
        for base in (HERE, ROOT):
            out |= set(_files_of(os.path.join(base, q)))
    pc = os.path.join(ROOT, 'papers', 'current')
    if re.search(r"papers|\.tex", src) and os.path.isdir(pc):
        for d in sorted(os.listdir(pc)):
            out |= set(_files_of(os.path.join(ROOT, 'papers', 'current', d, 'main.tex')))
    if re.search(r"\bdata\b", src):
        out |= set(_files_of(os.path.join(HERE, 'data')))
    return out

def key(script):
    h = hashlib.sha256(VERS.encode())
    for f in sorted(deps(script)):
        h.update(f.encode())
        with open(f, 'rb') as fh:
            h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest()

def _load():
    try:
        return json.load(open(CACHE))
    except Exception:
        return {}

def _run(script):
    out = subprocess.run([sys.executable, os.path.join(HERE, script)], cwd=HERE, capture_output=True, text=True, timeout=7200)
    return {'stdout': out.stdout, 'rc': out.returncode, 'err': out.stderr[-400:]}

def results(scripts, full=False, workers=None, progress=None):
    """Return {script: {'stdout', 'rc', 'err', 'cached'}}; run missing or stale scripts in parallel."""
    cache = _load()
    keys = {s: key(s) for s in scripts}
    todo = [s for s in scripts if full or cache.get(s, {}).get('key') != keys[s]]
    res = {s: dict(cache[s], cached=True) for s in scripts if s not in todo}
    with ThreadPoolExecutor(max_workers=workers or os.cpu_count()) as ex:
        for s, r in zip(todo, ex.map(_run, todo)):
            r.update(key=keys[s]); cache[s] = r
            res[s] = dict(r, cached=False)
            if progress:
                progress(s, res[s])
    json.dump(cache, open(CACHE, 'w'))
    return res
