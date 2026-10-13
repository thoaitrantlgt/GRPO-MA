"""Read-only server capability check; never starts training or loads weights."""
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--adapter',type=Path)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--images',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a fresh output directory')
    args.output.mkdir(parents=True)
    modules=['torch','transformers','peft','trl','accelerate','numpy','scipy','PIL','qwen_vl_utils']
    distributions={'PIL':'pillow','qwen_vl_utils':'qwen-vl-utils'}
    versions={};missing=[]
    for name in modules:
        if not importlib.util.find_spec(name):missing.append(name);continue
        try:versions[name]=importlib.metadata.version(distributions.get(name,name))
        except importlib.metadata.PackageNotFoundError:versions[name]='unknown'
    checks={'base_config':(args.checkpoint/'config.json').is_file(),
            'base_weights':args.checkpoint.is_dir() and bool(list(args.checkpoint.glob('*.safetensors'))+list(args.checkpoint.glob('pytorch_model*.bin'))),
            'tokenizer':(args.checkpoint/'tokenizer_config.json').is_file(),
            'multimodal_processor':(args.checkpoint/'preprocessor_config.json').is_file(),
            'manifest':args.manifest.is_file(),'image_root':args.images.is_dir()}
    if args.adapter:
        checks['adapter_config']=(args.adapter/'adapter_config.json').is_file()
        checks['adapter_weights']=bool(list(args.adapter.glob('adapter_model.safetensors'))+list(args.adapter.glob('adapter_model.bin')))
    devices=[];torch_error=None
    if 'torch' not in missing:
        try:
            import torch
            devices=[{'index':i,'name':torch.cuda.get_device_name(i),'memory_bytes':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]
        except Exception as error:torch_error=str(error)
    config=None
    if checks['base_config']:
        config=json.loads((args.checkpoint/'config.json').read_text())
        checks['target_model_type']=config.get('model_type')=='qwen2_5_vl'
    result={'status':'PREFLIGHT_PASS' if all(checks.values()) and not missing and not torch_error else 'MISSING_CAPABILITIES',
            'checks':checks,'package_versions':versions,'missing_packages':missing,'torch_error':torch_error,
            'devices':devices,'platform':platform.platform(),'paths':{key:str(getattr(args,key)) if getattr(args,key) else None for key in ['checkpoint','adapter','manifest','images']},
            'scope':'file/package existence and CUDA inventory only; does not certify complete shards, adapter scope, data independence or executable score exporter',
            'real_score_measurement_started':False,'GPU_job_started':False}
    (args.output/'availability.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['status']=='PREFLIGHT_PASS' else 2)

if __name__=='__main__':main()
