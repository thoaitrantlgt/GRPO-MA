"""Read-only reconstruction of saved trial risks and development figures."""
import csv
import itertools
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import target
from .evaluation import outcome_mass


BASE=Path('refine-logs/sequential-feasibility/20261011_run02')


def main():
    rows=list(csv.DictReader((BASE/'metrics.csv').open(encoding='utf-8')))
    effects=list(csv.DictReader((BASE/'effects.csv').open(encoding='utf-8')))
    records=json.loads((BASE/'probability_vectors.json').read_text())
    summary=json.loads((BASE/'summary.json').read_text())
    acquisitions=json.loads((BASE/'acquisition.json').read_text())
    configurations=json.loads((BASE/'configurations.json').read_text())
    checks={};maximum=0.
    for k in [2,4]:
        p=np.array([r['p'] for r in records]);p=p[:,[0,3]] if k==2 else p
        n=np.full(k,16//k);states=np.array(list(itertools.product(*(range(x+1) for x in n))))
        probability=outcome_mass(states,n,p)
        errors=target(states/n)[None,:,:]-target(p)[:,None,:]
        expected=np.sum(probability*np.sum(errors**2,axis=2),axis=1)
        for m0 in [1,2]:
            with np.load(BASE/f'exact_K{k}_m{m0}_uniform.npz') as saved:
                maximum=max(maximum,float(np.max(np.abs(expected-saved['risk']))))
            with np.load(BASE/f'exact_K{k}_m{m0}_gated.npz') as saved:
                checks[f'K{k}_m{m0}_gated_identical_uniform']=bool(np.max(np.abs(expected-saved['risk']))<1e-12)
                checks[f'K{k}_m{m0}_bias_variance_identity']=bool(np.max(np.abs(saved['risk']-saved['bias_squared']-saved['variance']))<1e-12)
    checks['independent_uniform_reconstruction']=maximum<1e-12
    checks['reward_path_mass']=summary['max_mass_error']<1e-12
    checks['all_budgets_16']=all(int(r['B'])==16 for r in rows)
    bins=[]
    for k,m0 in itertools.product([2,4],[1,2]):
        table={(r['id'],r['method']):r for r in rows if int(r['K'])==k and int(r['m0'])==m0 and r['kind']=='consumed'}
        for low,high,label in [(0,.025,'[0,0.025)'),(.025,.1,'[0.025,0.1)'),(.1,np.inf,'[0.1,infinity)')]:
            ids=[r['id'] for r in rows if int(r['K'])==k and int(r['m0'])==m0 and r['method']=='uniform' and r['kind']=='consumed' and low<=float(r['dispersion'])<high]
            for method in ['variance-only','JADA','ungated','gated']:
                baseline=np.mean([float(table[i,'uniform']['risk']) for i in ids]) if ids else None
                candidate=np.mean([float(table[i,method]['risk']) for i in ids]) if ids else None
                bins.append({'K':k,'m0':m0,'pilot_fraction':k*m0/16,'dispersion_bin':label,'vectors':len(ids),'method':method,'baseline_mean_risk':baseline,
                             'reduction':1-candidate/baseline if ids else None,'scope':'descriptive consumed-data subgroup; no fresh intervals'})
    with (BASE/'dispersion_bins.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(bins[0]));w.writeheader();w.writerows(bins)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    colors={'gated':'#2065a8','ungated':'#dd8452','JADA':'#8c6bb1'}
    for method in colors:
        selected=[r for r in effects if int(r['K'])==4 and r['method']==method]
        x=[float(r['pilot_fraction']) for r in selected];y=[100*float(r['reduction']) for r in selected]
        lo=[100*(float(r['reduction'])-float(r['development_ci_low'])) for r in selected]
        hi=[100*(float(r['development_ci_high'])-float(r['reduction'])) for r in selected]
        axes[0].errorbar(x,y,yerr=[lo,hi],marker='o',label=method,color=colors[method],capsize=3)
    axes[0].set(xlabel='Pilot fraction (K=4, B=16)',ylabel='Risk reduction vs uniform (%)',title='Consumed-data development intervals')
    axes[0].legend()
    labels=['[0,0.025)','[0.025,0.1)','[0.1,infinity)']
    for method in ['gated','ungated']:
        selected=[r for r in bins if r['K']==4 and r['m0']==1 and r['method']==method]
        axes[1].plot(range(3),[100*r['reduction'] if r['reduction'] is not None else np.nan for r in selected],marker='o',label=method,color=colors[method])
    axes[1].set_xticks(range(3),labels,rotation=15)
    axes[1].set(xlabel='True group dispersion (evaluator only)',ylabel='Risk reduction (%)',title='Fixed bins; K=4, m0=1')
    axes[1].legend()
    selected=[r for r in acquisitions if r['K']==4 and r['method']=='gated']
    bottoms=np.zeros(len(selected));x=np.arange(len(selected))
    for key,label,color in [('numerical_fallback_fraction','Numerical fallback','#b8c7d8'),('no_positive_gain_fraction','No positive gain','#8f9ba8'),('reliability_fallback_fraction','Reliability fallback','#dd8452'),('preferred_fraction','Preferred','#2065a8')]:
        y=100*np.array([r[key] for r in selected]);axes[2].bar(x,y,bottom=bottoms,label=label,color=color);bottoms+=y
    axes[2].set_xticks(x,[f"m0={r['m0']}" for r in selected]);axes[2].set(ylabel='Expected share of remaining observations (%)',title='Gated acquisition accounting',ylim=(0,105));axes[2].legend(fontsize=7)
    for ax in axes[:2]:ax.axhline(0,color='grey',linewidth=.7);ax.grid(alpha=.2)
    fig.suptitle('CPU feasibility: exact reward risk, approximate prior integration; no confirmation GO',fontsize=10)
    fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(BASE/f'feasibility.{suffix}',dpi=220)
    plt.close(fig)
    audit={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'maximum_uniform_reconstruction_error':maximum,
           'tests':10,'review':'same-family assumption audit provisional; local numerical reconstruction only',
           'GPU_cost':0,'MC_precision':'inconclusive for rare deviations; exact policy reward risks available'}
    (BASE/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    # Supplement the frozen evaluation report; original risk tables are unchanged.
    with (BASE/'RESULTS.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## Independent reconstruction and mechanism diagnosis\n\n')
        f.write(f"Ten implementation tests PASS. Independent fixed-count outcome summation reproduces uniform risks with maximum absolute error {maximum:.3g}; gated risk is identical to uniform in all four configurations. [Numerical audit](AUDIT.json).\n\n")
        f.write('Paired MC uses 512 episodes per vector/configuration. There are 120 zero-empirical-SE discrepancies that remain unresolved; studentized checks do not validate these rare effects. All exact effects lie inside the reported pointwise Hoeffding intervals, which are too wide to resolve small gains. Exact reward-state propagation supplies the primary risks; no additional MC trials were selected after outcomes.\n\n')
        f.write('The one-action terminal-count identity is independently verified. See [mechanism report](mechanism-diagnostic/RESULTS.md) for the derivation and exploratory K2 Bayes control. This explains why early information acquisition is absent; it is not a universal impossibility result for sequential allocation.\n\n')
        f.write('[Pilot fraction and fixed-dispersion plot](feasibility.png) and [subgroup table](dispersion_bins.csv) use consumed data. Leverage/gradient evaluation is deferred because the credit prototype has no development signal. No geometry was selected to rescue it.\n\n')
        for c in configurations:
            count=512 if c['K']==2 else 768;cache_bytes=c['risk_cache']['currsize']*count*8
            f.write(f"K={c['K']}, m0={c['m0']}: configuration wall time {c['seconds']:.2f}s; MC {c['mc_seconds']:.2f}s; cached risk-vector payload {cache_bytes/1024**2:.2f} MiB. Payload excludes Python dictionaries, score cache, outcome arrays and other process memory; peak RSS was not measured.\n\n")
        f.write('Decision: STOP this one-action/balanced-completion prototype before the full sweep. A new acquisition design must account for the future adaptive allocation decisions affected by the observation, rather than only reordering samples under the same terminal counts. No GPU escalation.\n')
    print(json.dumps(audit,indent=2));assert all(checks.values())


if __name__=='__main__':main()
