"""Freeze score-independent prompt selection from an operator-supplied holdout."""
import argparse
import hashlib
import json
from pathlib import Path

SALT='score-geometry-v1'


def select(records,count=128):
    if count<1:raise ValueError('Positive prompt count required')
    ids=[str(row['id']) for row in records]
    if len(set(ids))!=len(ids):raise ValueError('Manifest must have unique prompt IDs')
    ordered=sorted(records,key=lambda row:(hashlib.sha256((SALT+':'+str(row['id'])).encode()).hexdigest(),str(row['id'])))
    if len(ordered)<count:raise ValueError(f'Need {count} independent prompts, found {len(ordered)}; do not silently shrink protocol')
    return ordered[:count]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--images',type=Path,required=True)
    parser.add_argument('--split-description',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Selection is frozen; use a fresh output directory')
    records=json.loads(args.manifest.read_text(encoding='utf-8'))
    if not isinstance(records,list):raise ValueError('Expected ShareRobot JSON list')
    selected=select(records)
    for row in selected:
        if not isinstance(row.get('image'),str) or not row.get('question'):
            raise ValueError('Selected rows need one image path and a question; schema change requires a new preparation contract')
    missing=[row['id'] for row in selected if not (args.images/row['image']).is_file()]
    if missing:raise FileNotFoundError(f'Missing images for preselected IDs: {missing}; repair paths, do not replace prompts')
    args.output.mkdir(parents=True)
    (args.output/'selected_manifest.json').write_text(json.dumps(selected,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    declaration={'source_path':str(args.manifest),'source_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                 'selection_salt':SALT,'prompts':128,'selected_ids':[row['id'] for row in selected],
                 'smoke_ids':[row['id'] for row in selected[:4]],'independence_operator_declaration':args.split_description,
                 'independence_verified':False,'selection_uses_scores_or_rewards':False}
    (args.output/'selection.json').write_text(json.dumps(declaration,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('Frozen128 prompts; first4 smoke IDs declared. Independence is operator-attested, not verified.')

if __name__=='__main__':main()
