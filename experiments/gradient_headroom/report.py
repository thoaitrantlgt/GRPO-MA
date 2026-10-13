"""Read-only figures and independent frozen-table checks for the CPU diagnostic."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(out):
    summary=json.loads((out/'summary.json').read_text());protocol=json.loads((out/'protocol.json').read_text())
    with np.load(out/'risk_lattice.npz') as file:
        lattice=file['risks'];counts=file['counts'];probabilities=file['probabilities'];grams=file['gram'];orders=file['coordinate_orders']
    with np.load(out/'selected_results.npz') as file:
        saved=file['risks'];indices=file['selected_indices'];labelled=file['labelled_risks'];selected=file['selected_counts'];matrices=file['selected_moments']
    records=json.loads((out/'probability_vectors.json').read_text());names=list(protocol['geometries'])
    maximum=0.
    for mi,m0 in enumerate(protocol['m0']):
        feasible=np.flatnonzero(np.min(counts,axis=1)>=m0)
        oracle=lattice[feasible].min(axis=0)
        maximum=max(maximum,float(np.max(np.abs(oracle-labelled[mi,2]))))
        expected_credit_indices=feasible[np.argmin(lattice[feasible,:,0],axis=0)]
        assert np.array_equal(indices[mi,1,:,0],expected_credit_indices)
        for method in range(3):
            for gi in range(4):
                reconstructed=lattice[indices[mi,method,:,gi],np.arange(len(probabilities)),gi]
                maximum=max(maximum,float(np.max(np.abs(reconstructed-labelled[mi,method,:,gi]))))
    informative=[0,2,3];consumed=np.array([r['kind']=='consumed' for r in records])
    primary=saved[:,:,consumed,:][:,:,:,informative].mean(axis=(0,2,3))
    effects=[1-primary[2]/primary[0],1-primary[2]/primary[1]]
    reconstruction=max(maximum,float(np.max(np.abs(labelled.reshape(2,3,102,24,4).mean(axis=3)-saved))))
    moment_check=float(np.max(np.abs(np.einsum('abvgij,gij->abvg',matrices,grams)-labelled)))
    file_hashes=[]
    for name,digest in protocol['code_sha256'].items():
        for path in [out/'code'/name,Path(__file__).with_name(name)]:
            file_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest()==digest)
    file_hashes.append(hashlib.sha256((out/'code'/'target_core.py').read_bytes()).hexdigest()==protocol['target_code_sha256'])
    file_hashes.append(hashlib.sha256(Path('experiments/sequential_sampling/core.py').read_bytes()).hexdigest()==protocol['target_code_sha256'])
    file_hashes.append(hashlib.sha256((out/'probability_vectors.json').read_bytes()).hexdigest()==protocol['source_sha256'])
    checks={'all_saved_oracle_minima':reconstruction<1e-10,'selected_moment_contraction':moment_check<1e-10,
            'primary_ratios_match':bool(np.allclose(effects,[summary['primary']['headroom_vs_uniform'],summary['primary']['advantage_vs_credit_oracle']],atol=1e-12)),
            'frozen_source_and_code_hashes':all(file_hashes),'selected_budget_and_coverage':bool(np.all(selected.sum(axis=-1)==16) and np.min(selected[0])>=1 and np.min(selected[1])>=2)}
    # Independently recompute vector-stratified paired bootstrap from saved risks.
    rng=np.random.default_rng(protocol['bootstrap']['seed']);component=np.array([str(r['component']) for r in records])
    draws=np.concatenate([rng.choice(np.flatnonzero(consumed & (component==str(c))),(2000,16),replace=True) for c in range(6)],axis=1)
    pooled=saved[:,:,:,informative].mean(axis=(0,3));samples=pooled[:,draws].mean(axis=-1)
    intervals=[np.quantile(1-samples[2]/samples[0],[.025,.975]),np.quantile(1-samples[2]/samples[1],[.025,.975])]
    checks['primary_intervals_match']=bool(np.allclose(intervals,[summary['primary']['development_interval_headroom'],summary['primary']['development_interval_advantage']],atol=1e-12))
    audit={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'maximum_lattice_reconstruction_error':reconstruction,'maximum_moment_contraction_error':moment_check,
           'scope':'separate saved-artifact and bootstrap reconstruction, no independent-family semantic review'}
    (out/'SAVED_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8');assert all(checks.values()),audit
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    by_geometry=summary['by_geometry'];x=np.arange(len(by_geometry))
    for offset,key,interval,label in [(-.16,'headroom_vs_uniform','development_interval_headroom','vs uniform'),(.16,'advantage_vs_credit_oracle','development_interval_advantage','vs credit oracle')]:
        values=np.array([e[key] for e in by_geometry.values()])*100
        bounds=np.array([e[interval] for e in by_geometry.values()])*100
        axes[0,0].bar(x+offset,values,width=.3,label=label)
        axes[0,0].errorbar(x+offset,values,yerr=[values-bounds[:,0],bounds[:,1]-values],fmt='none',ecolor='black',capsize=3)
    axes[0,0].set(xticks=x,xticklabels=list(by_geometry),ylabel='Update risk reduction (%)',title='Fixed geometries; development intervals')
    axes[0,0].axhline(5,color='gray',linestyle='--');axes[0,0].legend(fontsize=8)
    fractions=[.25,.5]
    for key,interval,label in [('headroom_vs_uniform','development_interval_headroom','vs uniform'),('advantage_vs_credit_oracle','development_interval_advantage','vs credit oracle')]:
        values=np.array([e[key] for e in summary['by_minimum_coverage'].values()])*100
        bounds=np.array([e[interval] for e in summary['by_minimum_coverage'].values()])*100
        axes[0,1].errorbar(fractions,values,yerr=[values-bounds[:,0],bounds[:,1]-values],marker='o',capsize=3,label=label)
    axes[0,1].set(xticks=fractions,xticklabels=['25%','50%'],xlabel='Minimum coverage / total budget',ylabel='Update risk reduction (%)',title='Fixed count constraints; no adaptive pilot')
    axes[0,1].axhline(5,color='gray',linestyle='--');axes[0,1].legend(fontsize=8)
    dispersion=list(csv.DictReader((out/'dispersion.csv').open()))
    for m0 in [1,2]:
        rows=[r for r in dispersion if int(r['m0'])==m0]
        axes[1,0].plot(range(len(rows)),[float(r['advantage_vs_credit_oracle'])*100 for r in rows],marker='o',label=f'm0={m0}')
    labels=[f"[{r['dispersion_low']},{r['dispersion_high']})\nn={r['vectors']}" for r in dispersion if int(r['m0'])==1]
    axes[1,0].set(xticks=range(len(labels)),xticklabels=labels,xlabel='Probability sample standard deviation',ylabel='Advantage over credit oracle (%)',title='Predeclared dispersion bins; point diagnostics')
    axes[1,0].axhline(5,color='gray',linestyle='--');axes[1,0].legend(fontsize=8)
    leverage=list(csv.DictReader((out/'leverage.csv').open()))
    for m0,marker in [(1,'o'),(2,'s')]:
        rows=[r for r in leverage if int(r['m0'])==m0 and r['geometry']=='unequal-norm']
        axes[1,1].plot([float(r['contrast_leverage']) for r in rows],[float(r['average_update_oracle_count']) for r in rows],marker=marker,label=f'm0={m0}')
    axes[1,1].set(xlabel='Contrast leverage: diag(P Q P)',ylabel='Average update-oracle count',title='Unequal-norm geometry; all vectors/orders')
    axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'gradient_headroom.png',dpi=200);fig.savefig(out/'gradient_headroom.pdf');plt.close(fig)
    print(json.dumps(audit,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('refine-logs/gradient-headroom/20261011_run02'))
    main(parser.parse_args().output)
