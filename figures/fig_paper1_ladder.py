# Figure for paper1: the ladder of label media. Equation of state against the number k of labels for the additive action
# (w = k/3 - 1 on average over orientations), with radiation (dilation-invariant, w = 1/3) and the four-label vacuum (F'' != 0, w = -1).
# Run: ~/.venvs/jev/bin/python fig_paper1_ladder.py OUT.pdf
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'font.family': 'serif'})
fig, ax = plt.subplots(figsize=(4.4, 3.6))
add = [(0, -1, 'vacuum term\n(value a coupling)'), (1, -2 / 3, 'walls'), (2, -1 / 3, 'strings'), (3, 0, 'dust')]
ax.plot([k for k, _, _ in add], [w for _, w, _ in add], 'o-', color='C0', label=r'additive action, $w=k/3-1$')
for k, w, t in add: ax.annotate(t, (k, w), textcoords='offset points', xytext=(6, -12) if k else (6, -24), fontsize=7)
ax.plot([3], [1 / 3], 's', color='C3', label=r'invariance under the dilation: radiation, $F\propto n^{4/3}$')
ax.annotate('radiation, $w=1/3$', (3, 1 / 3), textcoords='offset points', xytext=(8, -3), fontsize=7)
ax.plot([4], [-1], 'D', color='C2', label=r"four labels, $F''\neq0$: vacuum")
ax.annotate('vacuum energy,\nconstant of integration', (4, -1), textcoords='offset points', xytext=(-95, 6), fontsize=7)
ax.axhline(-1, color='k', lw=0.6, ls=':'); ax.axhline(0, color='k', lw=0.4, ls=':')
ax.set_xlabel('number of labels $k$'); ax.set_ylabel('equation of state $w$'); ax.set_xticks(range(5)); ax.set_ylim(-1.35, 0.55)
ax.set_xlim(-0.3, 4.6); ax.legend(fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False); fig.tight_layout(); fig.savefig(sys.argv[1], metadata={'CreationDate': None, 'ModDate': None} if sys.argv[1].endswith('.pdf') else None); print('saved', sys.argv[1])
