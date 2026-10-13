"""Run a frozen CPU headroom screen; no adaptive or learned allocator."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.stats import binom
from threadpoolctl import threadpool_limits
from experiments.sequential_sampling.core import target
from .core import allocations,geometries,moments,update_risks

HERE=Path(__file__).parent


def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def csvsave(path,rows):
    with path.open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def direct_risk(n,p,gram):
    total=0.
    for s in itertools.product(*(range(int(v)+1) for v in n)):
        s=np.array(s);error=target(s/n)-target(p)
        total+=float(binom.pmf(s,n,p).prod())*float(error@gram@error)
    return total


def summary(out,records,risks,selected,counts,names,grams,protocol,audit,seconds):
    # risks axes: m0, method, vector, geometry; method uniform, credit oracle, update oracle.
    mask=np.array([r['kind']=='consumed' for r in records]);component=np.array([str(r['component']) for r in records])
    informative=[i for i,name in enumerate(names) if name!='aligned-null']
    strata=[np.flatnonzero(mask & (component==str(c))) for c in range(6)]
    rng=np.random.default_rng(protocol['bootstrap']['seed'])
    bootstrap=np.concatenate([rng.choice(ix,(protocol['bootstrap']['replicates'],len(ix)),replace=True) for ix in strata],axis=1)
    def effect(indices,gs,ms=(0,1)):
        part=risks[np.array(ms)][:,:,:,np.array(gs)]
        # Pool risks first; never average individual relative reductions.
        means=part[:,:,indices,:].mean(axis=(0,2,3))
        gain=1-means[2]/means[0];advantage=1-means[2]/means[1]
        pool=part.mean(axis=(0,3)) # method, vector
        samples=pool[:,bootstrap].mean(axis=2)
        low_high=np.quantile(1-samples[2]/samples[0],[.025,.975])
        adv_interval=np.quantile(1-samples[2]/samples[1],[.025,.975])
        return {'headroom_vs_uniform':float(gain),'advantage_vs_credit_oracle':float(advantage),
                'development_interval_headroom':low_high.tolist(),'development_interval_advantage':adv_interval.tolist(),
                'mean_uniform_update_risk':float(means[0]),'mean_credit_oracle_update_risk':float(means[1]),'mean_update_oracle_risk':float(means[2])}
    pooled=effect(np.flatnonzero(mask),informative)
    by_geometry={name:effect(np.flatnonzero(mask),[i]) for i,name in enumerate(names) if i in informative}
    by_m0={str(m):effect(np.flatnonzero(mask),informative,[i]) for i,m in enumerate(protocol['m0'])}
    screen=protocol['screen']
    passing=sum(e['headroom_vs_uniform']>.05 and e['advantage_vs_credit_oracle']>.05 for e in by_geometry.values())
    passed=pooled['development_interval_headroom'][0]>screen['pooled_lower_interval_headroom_vs_uniform'] and pooled['development_interval_advantage'][0]>screen['pooled_lower_interval_advantage_vs_credit_oracle'] and passing>=screen['informative_geometries_with_point_headroom_and_advantage_above5percent'] and audit['status']=='PASS'
    report={'verdict':'HEADROOM_FOR_METHOD_DEVELOPMENT' if passed else 'NO_PREDECLARED_GRADIENT_HEADROOM_SIGNAL','primary':pooled,
            'by_geometry':by_geometry,'by_minimum_coverage':by_m0,'geometries_passing_point_screen':passing,
            'confirmation_GO':False,'training_GO':False,'GPU_cost':0,'seconds':seconds,
            'scope':'synthetic fixed-score exact fixed-allocation true-p oracles; consumed-data development only'}
    dump(out/'summary.json',report)
    rows=[];stressrows=[]
    for mi,m0 in enumerate(protocol['m0']):
        for vi,record in enumerate(records):
            for gi,name in enumerate(names):
                u,c,g=risks[mi,:,vi,gi]
                row={'id':record['id'],'kind':record['kind'],'component':record['component'],'m0':m0,'minimum_coverage_fraction':4*m0/16,
                     'sample_std_dispersion':float(np.std(record['p'],ddof=1)),'geometry':name,'uniform_update_risk':u,'credit_oracle_update_risk':c,'update_oracle_risk':g,
                     'headroom_vs_uniform':(1-g/u) if name!='aligned-null' and u>1e-14 else 'NA',
                     'advantage_vs_credit_oracle':(1-g/c) if name!='aligned-null' and c>1e-14 else 'NA'}
                rows.append(row)
                if record['kind']=='stress':stressrows.append(row)
    csvsave(out/'metrics.csv',rows);csvsave(out/'stress.csv',stressrows)
    dispersion=np.array([np.std(r['p'],ddof=1) for r in records]);bins=[0,.1,.25,float('inf')];dispersionrows=[]
    for low,high in zip(bins[:-1],bins[1:]):
        ix=np.flatnonzero(mask & (dispersion>=low) & (dispersion<high))
        if not len(ix):continue
        for mi,m0 in enumerate(protocol['m0']):
            part=risks[mi][:,ix][:,:,informative].mean(axis=(1,2))
            dispersionrows.append({'dispersion_low':low,'dispersion_high':high,'vectors':len(ix),'m0':m0,'minimum_coverage_fraction':m0/4,
                                   'headroom_vs_uniform':1-part[2]/part[0],'advantage_vs_credit_oracle':1-part[2]/part[1]})
    csvsave(out/'dispersion.csv',dispersionrows)
    projector=np.eye(4)-np.ones((4,4))/4;leverage=np.diagonal(projector@grams@projector,axis1=1,axis2=2);leveragerows=[]
    for mi,m0 in enumerate(protocol['m0']):
        for gi,name in enumerate(names):
            average=selected[mi,2,mask,:,gi,:].mean(axis=(0,1))
            for j in range(4):leveragerows.append({'m0':m0,'geometry':name,'thought':j,'contrast_leverage':leverage[gi,j],'average_update_oracle_count':average[j] if name!='aligned-null' else 'NA'})
    csvsave(out/'leverage.csv',leveragerows)
    md=['# Gradient-aware CPU headroom results','','**Consumed-data diagnostic; synthetic scores; oracle knows true p. No confirmation or training GO.**','',
        f"Verdict: **{report['verdict']}**.",'',
        '| Geometry | Headroom vs uniform | Advantage vs credit oracle | Development interval, advantage |',
        '|---|---:|---:|---|']
    for name,e in by_geometry.items():
        interval=e['development_interval_advantage']
        md.append(f"| {name} | {100*e['headroom_vs_uniform']:.3f}% | {100*e['advantage_vs_credit_oracle']:.3f}% | [{100*interval[0]:.3f}%, {100*interval[1]:.3f}%] |")
    md+=['',f"Pooled headroom: {100*pooled['headroom_vs_uniform']:.3f}%, development interval {np.array(pooled['development_interval_headroom'])*100}. Pooled advantage over credit oracle: {100*pooled['advantage_vs_credit_oracle']:.3f}%, interval {np.array(pooled['development_interval_advantage'])*100}.",'',
         'Aligned-null has exactly zero update risk by centering; relative endpoints are NA and excluded by protocol. Orthogonal geometry has identical update and credit objectives. m0 constrains precommitted allocations and is not an adaptive pilot experiment.',
         '',f"Exact lattice: {len(counts)} allocations at m0=1; {sum(np.min(counts,axis=1)>=2)} at m0=2. All102 vectors ×24 orders ×4 geometries ×2 coverage conditions. CPU wall seconds: {seconds:.2f}. Audit: {audit['status']}.",
         '', '[Protocol](protocol.json) · [Raw risk lattice](risk_lattice.npz) · [Selected moments and counts](selected_results.npz) · [Metrics](metrics.csv) · [Stress](stress.csv) · [Dispersion](dispersion.csv) · [Leverage](leverage.csv) · [Audit](AUDIT.json).']
    (out/'RESULTS.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


def run(out):
    if out.exists():raise FileExistsError('Preserve completed runs; choose a fresh --output directory')
    out.mkdir(parents=True);protocol=json.loads((HERE/'protocol.json').read_text())
    source=Path(protocol['source']);records=json.loads(source.read_text());assert len(records)==102
    names,grams=geometries();perms=np.array(list(itertools.permutations(range(4))))
    p=np.array([r['p'] for r in records])[:,perms].reshape(-1,4)
    counts=allocations(4,16,1);uniform_ix=int(np.flatnonzero(np.all(counts==4,axis=1))[0])
    protocol['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    protocol['code_sha256']={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['core.py','run.py']}
    protocol['target_code_sha256']=hashlib.sha256(Path('experiments/sequential_sampling/core.py').read_bytes()).hexdigest()
    dump(out/'protocol.json',protocol);shutil.copyfile(source,out/'probability_vectors.json')
    (out/'code').mkdir()
    for name in ['core.py','run.py']:shutil.copyfile(HERE/name,out/'code'/name)
    shutil.copyfile('experiments/sequential_sampling/core.py',out/'code'/'target_core.py')
    shutil.copyfile('refine-logs/gradient-headroom/DERIVATION_PACKAGE.md',out/'DERIVATION_PACKAGE.md')
    shutil.copyfile('refine-logs/gradient-headroom/EXPERIMENT_PLAN.md',out/'EXPERIMENT_PLAN.md')
    started=time.perf_counter();allrisk=np.empty((len(counts),len(p),len(names)));best_credit=np.full((2,len(p)),np.inf);best_update=np.full((2,len(p),4),np.inf)
    selected_indices=np.zeros((2,3,len(p),4),dtype=int)
    selected_moments=np.zeros((2,3,len(p),4,4,4));uniform_matrix=None;null_max=0.;min_eigenvalue=0.;mean_max=0.
    with threadpool_limits(limits=1):
        for ix,n in enumerate(counts):
            if time.perf_counter()-started>protocol['resource_ceiling']['CPU_wall_seconds']:
                dump(out/'INCOMPLETE.json',{'completed_allocations':ix,'reason':'CPU ceiling','GPU_cost':0});raise TimeoutError('Frozen CPU ceiling')
            matrix,mean=moments(n,p);risk=update_risks(matrix,grams)
            null_max=max(null_max,float(np.max(np.abs(risk[:,1]))));mean_max=max(mean_max,float(np.max(np.abs(matrix.sum(axis=1)))))
            min_eigenvalue=min(min_eigenvalue,float(np.linalg.eigvalsh(matrix).min()))
            if risk.min() < -protocol['numerical_tolerance']:raise ArithmeticError('Substantial negative update risk')
            risk=np.maximum(risk,0);risk[:,1]=0 # Structural identity, not a relative gain.
            allrisk[ix]=risk
            if ix==uniform_ix:uniform_matrix=matrix
            for mi,m0 in enumerate(protocol['m0']):
                if min(n)<m0:continue
                better=risk[:,0]<best_credit[mi];best_credit[mi,better]=risk[better,0]
                selected_indices[mi,1,better,:]=ix
                selected_moments[mi,1,better,:,:,:]=matrix[better,None,:,:]
                for gi in range(4):
                    update=risk[:,gi]<best_update[mi,:,gi];best_update[mi,update,gi]=risk[update,gi]
                    selected_indices[mi,2,update,gi]=ix;selected_moments[mi,2,update,gi]=matrix[update]
            if (ix+1)%50==0:print(f'Exact allocation lattice {ix+1}/{len(counts)}, {time.perf_counter()-started:.1f}s',flush=True)
    selected_indices[:,0,:,:]=uniform_ix
    selected_moments[:,0,:,:,:,:]=uniform_matrix[None,:,None,:,:]
    labelled=np.empty((2,3,len(p),4))
    for mi in range(2):
        for method in range(3):
            for gi in range(4):labelled[mi,method,:,gi]=allrisk[selected_indices[mi,method,:,gi],np.arange(len(p)),gi]
    risks=labelled.reshape(2,3,len(records),24,4).mean(axis=3)
    selected=counts[selected_indices].reshape(2,3,len(records),24,4,4)
    # Deterministic independent direct-error audit across all comparators, geometry and coverage.
    direct_error=0.
    for mi in range(2):
        for method in range(3):
            for gi in [0,2,3]:
                vi=(mi*997+method*191+gi*37)%len(p);n=counts[selected_indices[mi,method,vi,gi]]
                direct_error=max(direct_error,abs(direct_risk(n,p[vi],grams[gi])-labelled[mi,method,vi,gi]))
    with np.load('refine-logs/sequential-feasibility/20261011_run02/exact_K4_m1_uniform.npz') as old:
        uniform_difference=float(np.max(np.abs(risks[0,0,:,0]-old['risk'])))
    exact_search=0.
    for mi,m0 in enumerate(protocol['m0']):
        feasible=np.flatnonzero(np.min(counts,axis=1)>=m0)
        exact_search=max(exact_search,float(np.max(np.abs(allrisk[feasible].min(axis=0)-labelled[mi,2]))))
    moment_difference=float(np.max(np.abs(np.einsum('abvgij,gij->abvg',selected_moments,grams)-labelled)))
    checks={'direct_squared_update_error':direct_error<1e-10,'saved_selected_moment_contraction':moment_difference<1e-10,'unchanged_uniform_credit_risk':uniform_difference<1e-10,
            'exact_global_oracle_minimum':exact_search<1e-10,'oracle_bounds':bool(np.all(labelled[:,2]<=labelled[:,0]+1e-10) and np.all(labelled[:,2]<=labelled[:,1]+1e-10)),
            'orthogonal_credit_update_equivalence':bool(np.allclose(labelled[:,1,:,0],labelled[:,2,:,0],atol=1e-10)),
            'null_geometry':null_max<1e-10,'PSD_error_moments':min_eigenvalue>-1e-10,'centered_error_moments':mean_max<1e-10,
            'budget_and_coverage':bool(np.all(selected.sum(axis=-1)==16) and all(np.min(selected[mi])>=m0 for mi,m0 in enumerate(protocol['m0'])))}
    audit={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'maximum_direct_error_discrepancy':direct_error,'uniform_risk_discrepancy':uniform_difference,
           'global_search_discrepancy':exact_search,'selected_moment_contraction_discrepancy':moment_difference,'maximum_raw_null_risk':null_max,'minimum_second_moment_eigenvalue':min_eigenvalue,'maximum_centering_residual':mean_max,
           'evaluated_allocations':len(counts),'total_labelled_probability_vectors':len(p),'scope':'deterministic numerical checks; no external semantic certification'}
    dump(out/'AUDIT.json',audit)
    np.savez_compressed(out/'risk_lattice.npz',risks=allrisk,counts=counts,probabilities=p,gram=grams,coordinate_orders=perms)
    np.savez_compressed(out/'selected_results.npz',risks=risks,labelled_risks=labelled,selected_indices=selected_indices,selected_counts=selected,
                        selected_moments=selected_moments,gram=grams,counts=counts)
    assert all(checks.values()),audit
    summary(out,records,risks,selected,counts,names,grams,protocol,audit,time.perf_counter()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('refine-logs/gradient-headroom/20261011_run01'))
    run(parser.parse_args().output)
