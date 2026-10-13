"""K4 information-control study: resource audit entry point."""
import argparse
import itertools
import json
import csv
import shutil
import hashlib
import time
import gc
from datetime import datetime,timezone
from functools import lru_cache
from math import prod
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .core import target
from .evaluation import outcome_mass, average_permutations
from .symmetry_control import SymmetricControl,SavedControl,ladder_support,build_graph,graph_metrics,canonical,next_key,process_memory


OUT=Path('refine-logs/information-control-k4/20261011_run01')


def complexity(k=4,budget=16,m0=1):
    labelled=[0]*(budget+1)
    for n in itertools.product(range(m0,budget-(k-1)*m0+1),repeat=k):
        if sum(n)<=budget:labelled[sum(n)]+=prod(x+1 for x in n)
    dp=[[0]*(budget+1) for _ in range(k+1)];dp[0][0]=1
    for n in range(m0,budget+1):
        for successes in range(n+1):
            for groups in range(1,k+1):
                for total in range(n,budget+1):dp[groups][total]+=dp[groups-1][total-n]
    return {'K':k,'B':budget,'m0':m0,'labelled_states':sum(labelled),'canonical_states':sum(dp[k]),
            'labelled_terminal_states':labelled[-1],'canonical_terminal_states':dp[k][-1],
            'labelled_by_depth':labelled,'canonical_by_depth':dp[k]}


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n',encoding='utf-8')


def csvsave(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def prior_initial(model,m0):
    states=list(itertools.product(range(m0+1),repeat=model.k))
    probabilities=outcome_mass(states,[m0]*model.k,model.points).mean(axis=0)
    return states,probabilities


def prior_value(model,m0,policy=None):
    states,probabilities=prior_initial(model,m0);n=(m0,)*model.k
    if policy is None:return float(sum(prob*model.value(n,s) for s,prob in zip(states,probabilities)))
    @lru_cache(maxsize=None)
    def fixed(key):
        model.check_deadline()
        if sum(n for n,s in key)==model.budget:return model.values[key]
        j=policy.actions[key];predictive=model.posterior_key(key)@model.points
        return float(predictive[j]*fixed(next_key(key,j,1))+(1-predictive[j])*fixed(next_key(key,j,0)))
    return float(sum(prob*fixed(canonical(n,s)) for s,prob in zip(states,probabilities)))


class UniformControl:
    def __init__(self,model):self.k=model.k;self.budget=model.budget;self.model=model
    def check_deadline(self):self.model.check_deadline()
    def choose(self,n,s,method='uniform'):return int(np.argmin(n)),'uniform'


def residual_audit(model):
    keys=sorted(model.values);rng=np.random.default_rng(3464102027)
    indexes=rng.choice(len(keys),min(256,len(keys)),replace=False)
    selected={keys[int(i)] for i in indexes}
    for m0 in [1,2]:selected.update(canonical((m0,)*4,s) for s in itertools.product(range(m0+1),repeat=4))
    residual=0.;terminal_direct=0.;minimum_margin=float('inf')
    for key in selected:
        weights=model.posterior_key(key)
        if sum(n for n,s in key)==model.budget:
            credit=target(np.array([s/n for n,s in key]));direct=float(weights@np.sum((credit-model.truth)**2,axis=1))
            terminal_direct=max(terminal_direct,abs(direct-model.values[key]))
        else:
            predictive=weights@model.points
            q=np.array([predictive[j]*model.values[next_key(key,j,1)]+(1-predictive[j])*model.values[next_key(key,j,0)] for j in range(4)])
            residual=max(residual,abs(q.min()-model.values[key]),abs(q[model.actions[key]]-model.values[key]))
            minimum_margin=min(minimum_margin,float(np.sort(q)[1]-np.sort(q)[0]))
    return {'sampled_states':len(selected),'maximum_Bellman_residual':residual,'maximum_terminal_direct_error':terminal_direct,'minimum_action_margin':minimum_margin}


def summarize(levels,records,baseline,protocol):
    mask=np.array([r['kind']=='consumed' for r in records]);stress=~mask
    strata=[np.flatnonzero(mask & np.array([r['component']==str(c) for r in records])) for c in range(6)]
    rng=np.random.default_rng(protocol['bootstrap']['seed'])
    draws=np.array([np.concatenate([rng.choice(ix,len(ix),replace=True) for ix in strata]) for _ in range(protocol['bootstrap']['replicates'])])
    effects=[]
    for level in levels:
        for m0 in [1,2]:
            risk=level['risks'][m0];gain=1-risk[mask].mean()/baseline[mask].mean()
            boot=1-risk[draws].mean(axis=1)/baseline[draws].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975])
            effects.append({'base_points_per_scramble':level['base_points'],'support_points':level['support_points'],'m0':m0,
                            'reduction':float(gain),'development_ci_low':float(lo),'development_ci_high':float(hi),
                            'max_original_stress_regression':float(np.max(risk[stress]/baseline[stress]-1))})
    highest=levels[-1];adjacent=levels[-2]
    highrisk=np.stack([highest['risks'][m] for m in [1,2]]);lowrisk=np.stack([adjacent['risks'][m] for m in [1,2]])
    gain=1-highrisk[:,mask].mean()/baseline[mask].mean()
    boot=1-highrisk[:,draws].mean(axis=(0,2))/baseline[draws].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975])
    aggregate_change=float(abs(highrisk[:,mask].mean()-lowrisk[:,mask].mean())/baseline[mask].mean())
    stress_change=float(np.max(np.abs(highrisk[:,stress]-lowrisk[:,stress])/baseline[stress]))
    prior_change=max(abs(highest['prior_values'][m]-adjacent['prior_values'][m])/highest['prior_uniform'] for m in [1,2])
    regrets={f"base{old['base_points']}_m{m}":prior_value(highest['model'],m,old['model'])-highest['prior_values'][m] for old in levels[:-1] for m in [1,2]}
    regret_normalized=max(regrets.values())/highest['prior_uniform']
    stable=aggregate_change<=.01 and stress_change<=.01 and prior_change<=.01 and regret_normalized<=.01
    regression=float(np.max(highrisk[:,stress]/baseline[stress]-1))
    signal=lo>.05 and regression<=.10 and stable
    summary={'verdict':'DEVELOPMENT_HEADROOM_UNDER_CURRENT_PRIOR' if signal else 'NO_ROBUST_STABLE_DEVELOPMENT_SIGNAL',
             'primary_reduction':float(gain),'development_ci':[float(lo),float(hi)],'maximum_original_stress_regression':regression,
             'numerically_stable':stable,'highest_vs_adjacent_aggregate_risk_change_over_uniform':aggregate_change,
             'maximum_stress_risk_change_over_uniform':stress_change,'prior_value_change_over_uniform':float(prior_change),
             'lower_policy_regret_on_highest_support':regrets,'max_lower_policy_regret_over_uniform':float(regret_normalized),
             'confirmation_GO':False,'GPU_cost':0}
    return effects,summary


def write_report(effects,summary,metadata,stressrows):
    lines=['# K4 information-control results','',
           '**Consumed-data development diagnostic. Exact finite-support optimization; continuous-prior quadrature approximate. No confirmation GO or GPU.**','',
           f"Verdict: **{summary['verdict']}**. Highest-level reduction: {100*summary['primary_reduction']:.4f}%, descriptive development interval [{100*summary['development_ci'][0]:.4f}%, {100*summary['development_ci'][1]:.4f}%].",'',
           '| Base points / scramble | Support points | m0 | Reduction vs uniform | Development interval | Maximum original stress regression |',
           '|---:|---:|---:|---:|---|---:|']
    for r in effects:lines.append(f"| {r['base_points_per_scramble']} | {r['support_points']} | {r['m0']} | {100*r['reduction']:.4f}% | [{100*r['development_ci_low']:.4f}%, {100*r['development_ci_high']:.4f}%] | {100*r['max_original_stress_regression']:.4f}% |")
    lines+=['','## Numerical stability','',f"Highest-versus-adjacent aggregate risk change / uniform: {100*summary['highest_vs_adjacent_aggregate_risk_change_over_uniform']:.4f}%. Maximum stress risk change / uniform: {100*summary['maximum_stress_risk_change_over_uniform']:.4f}%. Prior Bayes-value change / prior uniform: {100*summary['prior_value_change_over_uniform']:.4f}%. Maximum lower-policy regret on highest finite support / prior uniform: {100*summary['max_lower_policy_regret_over_uniform']:.4f}%.",'',
            'All four diagnostics require <=1%. These checks do not certify convergence to the continuous Jeffreys prior; no extra resolution levels were selected after winners.','',
            '| Base points | Bellman quotient states | CPU wall seconds | Process peak working set MiB | Prior uniform risk | Prior optimal risk m0=1 | Prior optimal risk m0=2 |',
            '|---:|---:|---:|---:|---:|---:|---:|']
    for r in metadata:lines.append(f"| {r['base_points']} | {r['canonical_states']} | {r['seconds']:.2f} | {r['process_peak_RSS']/1024**2:.2f} | {r['prior_uniform']:.6f} | {r['prior_values'][1]:.6f} | {r['prior_values'][2]:.6f} |")
    lines+=['','Process peak working set is cumulative within this CPU process, including scientific libraries and earlier levels. It is not an isolated policy-table size.','',
            '## Original stress cases at highest resolution','',
            '| Regime | m0 | Uniform risk | Control risk | Regression |',
            '|---|---:|---:|---:|---:|']
    for r in stressrows:lines.append(f"| {r['id']} | {r['m0']} | {r['uniform_risk']:.6g} | {r['control_risk']:.6g} | {100*r['regression']:.4f}% |")
    lines+=['','## Interpretation','',
            'The control optimizes the full adaptive decision tree using only counts and the frozen prior, so it avoids the earlier one-action terminal-count blindness. Its finite-prior optimality does not make it an oracle for the heterogeneous benchmark. Stress performance, integration stability and consumed-data scope must be evaluated separately from average gain.','',
            'No prior, terminal estimator, benchmark vector or threshold was selected after outcomes. The former one-action prototype verdict remains unchanged. Reliability gating and real score geometry are absent from this control; it is not a deployable replacement demonstrated to improve training.','',
            '[Summary](summary.json) · [Effects](effects.csv) · [Stress table](stress.csv) · [Full metrics](metrics.csv) · [Protocol](protocol.json) · [Checks](AUDIT.json).']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run():
    if OUT.exists():raise FileExistsError('Preserve completed or partial control run')
    protocol=json.loads(Path(__file__).with_name('k4_control_protocol.json').read_text())
    OUT.mkdir(parents=True);protocol['frozen_utc']=datetime.now(timezone.utc).isoformat()
    protocol['code_sha256']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in
                             ['k4_control.py','symmetry_control.py','core.py','evaluation.py']}
    dump(OUT/'protocol.json',protocol);(OUT/'code').mkdir()
    for name in protocol['code_sha256']:shutil.copy2(Path(__file__).with_name(name),OUT/'code'/name)
    shutil.copy2(Path(__file__).with_name('k4_control_protocol.json'),OUT/'code/k4_control_protocol.json')
    records=json.loads(Path(protocol['source']).read_text());dump(OUT/'probability_vectors.json',records)
    p=np.array([r['p'] for r in records]);permutations=np.array(list(itertools.permutations(range(4))))
    expanded=p[:,permutations].reshape(-1,4)
    with np.load(protocol['baseline']) as saved:baseline=saved['risk']
    np.save(OUT/'uniform_reference.npy',baseline)
    levels=[];metadata=[];rows=[];checks=[];started=time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            for count in protocol['quadrature']['nested_base_points_per_scramble']:
                begin=time.perf_counter();points=ladder_support(count,protocol['quadrature']['seed']);np.save(OUT/f'prior_support_n{count}.npy',points)
                print(f'K4/B16 Bellman level: base {count}, support {len(points)}',flush=True)
                model=SymmetricControl(points,16,protocol['resource_ceiling']['CPU_wall_seconds_per_level'],protocol['resource_ceiling']['process_RSS_bytes'])
                for s in itertools.product([0,1],repeat=4):model.value((1,)*4,s)
                expected=complexity()['canonical_states']
                if len(model.values)!=expected:raise ArithmeticError('Quotient state coverage differs from independent count')
                model.save(OUT/f'policy_n{count}.npz');check=residual_audit(model);checks.append({'base_points':count,**check})
                if max(check['maximum_Bellman_residual'],check['maximum_terminal_direct_error'])>1e-10:raise ArithmeticError('Bellman audit failed')
                uniform=UniformControl(model);ugraph=build_graph(uniform,1)
                prior_uniform=float(graph_metrics(uniform,ugraph,points)['risk'].mean())
                previous=levels[-1]['model'] if levels else None
                risks={};prior_values={};graphs={};level_diagnostics=[]
                for m0 in [1,2]:
                    graph=build_graph(model,m0,previous);graphs[m0]=graph
                    print(f"  evaluate m0={m0}: {len(graph['terminal'])} terminal labelled states",flush=True)
                    result=average_permutations(graph_metrics(model,graph,expanded),permutations)
                    if result['mass_error']>1e-10:raise ArithmeticError('Outcome probabilities not normalized')
                    risks[m0]=result['risk'];prior_values[m0]=prior_value(model,m0)
                    if prior_values[m0]>prior_uniform+1e-10:raise ArithmeticError('Finite-prior control worse than admissible uniform')
                    np.savez_compressed(OUT/f'exact_n{count}_m{m0}.npz',**{key:value for key,value in result.items() if isinstance(value,np.ndarray)})
                    mask=np.array([r['kind']=='consumed' for r in records])
                    level_diagnostics.append({'m0':m0,'terminal_labelled_states':len(graph['terminal']),'mass_error':result['mass_error'],
                                              'adjacent_policy_action_disagreement_on_current_trajectory':float(result['action_disagreement'][mask].mean()) if previous else None})
                    for ix,record in enumerate(records):rows.append({'base_points':count,'support_points':len(points),'m0':m0,'id':record['id'],'kind':record['kind'],'component':record['component'],
                                                                     'dispersion':float(np.std(p[ix],ddof=1)),'uniform_risk':float(baseline[ix]),'control_risk':float(result['risk'][ix]),
                                                                     'bias_squared':float(result['bias_squared'][ix]),'variance':float(result['variance'][ix])})
                memory=process_memory();peak=memory['peak_rss'] if memory else model.peak_rss
                meta={'base_points':count,'support_points':len(points),'canonical_states':len(model.values),'seconds':time.perf_counter()-begin,
                      'process_peak_RSS':peak,'prior_uniform':prior_uniform,'prior_values':prior_values,'evaluation':level_diagnostics}
                metadata.append(meta);levels.append({'base_points':count,'support_points':len(points),'risks':risks,'prior_values':prior_values,'prior_uniform':prior_uniform,'model':model})
                dump(OUT/'levels.json',metadata);csvsave(OUT/'metrics.csv',rows);dump(OUT/'Bellman_checks.json',checks)
                print(json.dumps(meta,default=lambda x:x.item(),indent=2),flush=True);del graphs;gc.collect()
            effects,summary=summarize(levels,records,baseline,protocol)
    except Exception as exc:
        dump(OUT/'INCOMPLETE.json',{'error':repr(exc),'completed_levels':[x['base_points'] for x in levels],'seconds':time.perf_counter()-started})
        raise
    summary.update({'seconds':time.perf_counter()-started,'vectors':len(records),'consumed_vectors':96,'stress_vectors':6,'levels':len(levels),
                    'evaluations':len(rows),'maximum_Bellman_residual':max(c['maximum_Bellman_residual'] for c in checks)})
    high=levels[-1];stressrows=[]
    for m0 in [1,2]:
        for ix,r in enumerate(records):
            if r['kind']=='stress':stressrows.append({'id':r['id'],'m0':m0,'uniform_risk':float(baseline[ix]),'control_risk':float(high['risks'][m0][ix]),
                                                     'regression':float(high['risks'][m0][ix]/baseline[ix]-1)})
    csvsave(OUT/'effects.csv',effects);csvsave(OUT/'stress.csv',stressrows);dump(OUT/'summary.json',summary)
    dump(OUT/'AUDIT.json',{'status':'PASS','scope':'local finite-support and outcome-propagation checks; scientific interpretation not certified','checks':checks,'GPU_cost':0})
    write_report(effects,summary,metadata,stressrows);print(json.dumps(summary,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit-only',action='store_true');args=parser.parse_args()
    if args.audit_only:
        out=OUT.parent/'COMPLEXITY_AUDIT.json';out.parent.mkdir(parents=True,exist_ok=True)
        result=[complexity(m0=m) for m in [1,2]]
        out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return
    run()


if __name__=='__main__':main()
