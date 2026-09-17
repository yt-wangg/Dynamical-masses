from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'results/raw_metallicity_delta_vs_delta_mg_20260917'
OUT = RAW
SOURCE = ROOT / 'data/jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits'
SUMMARY = ROOT / 'results/xp_simple_student_t_feh_jcaps_20260910/summary.json'

raw = json.loads((RAW/'bin_stats.json').read_text())
summary = json.loads(SUMMARY.read_text())
rows = np.load(RAW/'source_row_indices.npy')
if len(rows) != 13721: raise RuntimeError(f'Expected 13721 rows, got {len(rows)}')
t = Table.read(SOURCE)
z = np.column_stack([np.asarray(t['feh_jcaps_1'], float), np.asarray(t['feh_jcaps_2'], float)])
err = np.column_stack([np.asarray(t['jc_sigma_m_h_1'], float), np.asarray(t['jc_sigma_m_h_2'], float)])
mg = np.column_stack([np.asarray(t['absg1'], float), np.asarray(t['absg2'], float)])
finite = np.all(np.isfinite(np.column_stack([z, err, mg])), axis=1)
keep = finite & np.all((mg >= 3.5) & (mg <= 13.5), axis=1) & np.all(err > 0, axis=1)
keep &= (np.asarray(t['jc_at_bound_bits_1']) == 0) & (np.asarray(t['jc_at_bound_bits_2']) == 0)
selected = np.flatnonzero(keep)
if not np.array_equal(selected, rows): raise RuntimeError('Saved source rows do not match raw selection')
x = mg[rows, 1] - mg[rows, 0]
raw_y = z[rows, 1] - z[rows, 0]
knots = np.asarray(summary['bias_knots'], float)
values = np.asarray(summary['bias_values'], float)
y = raw_y - (np.interp(mg[rows, 1], knots, values) - np.interp(mg[rows, 0], knots, values))

order = np.argsort(x, kind='mergesort')
groups = [g for g in np.array_split(order, raw['n_bins']) if len(g)]
edges = np.empty(len(groups)+1, float); edges[0] = x.min(); edges[-1] = x.max()
for i, g in enumerate(groups[:-1], 1): edges[i] = 0.5*(x[g[-1]] + x[groups[i]][0])
stats=[]; violin=[]; centers=[]
for i,g in enumerate(groups):
    q16, med, q84 = np.quantile(y[g], [.16,.5,.84])
    stats.append({'bin':i+1,'x_low':float(edges[i]),'x_high':float(edges[i+1]),'x_median':float(np.median(x[g])),'n':int(len(g)),'y_median':float(med),'y_q16':float(q16),'y_q84':float(q84)})
    violin.append(y[g]); centers.append(float(np.median(x[g])))
xlim = [float(raw['x_min']), float(raw['x_max'])]
pad = .04*(float(raw['y_max']) - float(raw['y_min']))
ylim = [float(raw['y_min'])-pad, float(raw['y_max'])+pad]
fig, ax = plt.subplots(figsize=(11.5,7.6), constrained_layout=True)
hb=ax.hexbin(x,y,gridsize=(90,40),mincnt=1,bins='log',cmap='viridis',linewidths=.15,alpha=.88,zorder=1)
cb=fig.colorbar(hb,ax=ax,pad=.012); cb.set_label('Pairs per hexagon (log scale)')
c=np.asarray(centers); widths=1.6*np.minimum(c-edges[:-1], edges[1:]-c)
vp=ax.violinplot(violin,positions=centers,widths=widths,showmeans=False,showmedians=False,showextrema=False,points=120)
for body in vp['bodies']:
    body.set_facecolor('#f28e2b'); body.set_edgecolor('#8c4c13'); body.set_linewidth(.7); body.set_alpha(.28)
med=np.array([s['y_median'] for s in stats]); q16=np.array([s['y_q16'] for s in stats]); q84=np.array([s['y_q84'] for s in stats])
ax.vlines(c,q16,q84,color='#d62728',lw=2,zorder=5,label='16th–84th percentile'); ax.plot(c,med,'o-',color='#d62728',ms=4.5,lw=1.6,zorder=6,label='Median')
ax.axhline(0,color='0.25',lw=.9,ls='--',alpha=.75,zorder=3); ax.set_xlim(*xlim); ax.set_ylim(*ylim)
ax.set_xlabel(r'$\Delta M_G = M_{G,2} - M_{G,1}$ (mag)'); ax.set_ylabel(r'$(z_2-z_1)-[b_G(M_{G,2})-b_G(M_{G,1})]$ (dex)')
ax.set_title('Bias-corrected XP metallicity difference versus absolute-G difference'); ax.legend(loc='upper right',frameon=True,framealpha=.9); ax.grid(color='0.85',lw=.6,alpha=.6)
png=OUT/'bias_corrected_metallicity_delta_vs_delta_mg.png'; fig.savefig(png,dpi=220); plt.close(fig)
result={'source':str(SOURCE),'bias_summary':str(SUMMARY),'y_definition':'(feh_jcaps_2 - feh_jcaps_1) - (b_G(absg2) - b_G(absg1))','bias_knots':knots.tolist(),'bias_values':values.tolist(),'n_pairs':int(len(x)),'n_bins':int(len(groups)),'bin_count_total':int(sum(s['n'] for s in stats)),'x_min':float(x.min()),'x_max':float(x.max()),'y_min':float(y.min()),'y_max':float(y.max()),'fixed_axis_limits':{'x':xlim,'y':ylim},'corrected_points_clipped':int(np.count_nonzero((y<ylim[0])|(y>ylim[1]))),'bins':stats}
(OUT/'bias_corrected_bin_stats.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'png':str(png),'json':str(OUT/'bias_corrected_bin_stats.json'),'n_pairs':len(x),'bin_total':sum(s['n'] for s in stats),'xlim':xlim,'ylim':ylim,'corrected_min':float(y.min()),'corrected_max':float(y.max()),'clipped':result['corrected_points_clipped']},indent=2))
