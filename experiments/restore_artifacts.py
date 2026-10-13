"""Reassemble byte-identical raw artifacts split to fit GitHub blob limits."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    sha=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):sha.update(block)
    return sha.hexdigest()


def restore(root,manifest):
    root=root.resolve()
    def within(name):
        path=(root/name).resolve()
        if not path.is_relative_to(root):raise ValueError('Artifact path must stay inside the checkout')
        return path
    for record in json.loads(manifest.read_text())['artifacts']:
        target=within(record['path'])
        if target.exists():
            if target.stat().st_size!=record['bytes'] or digest(target)!=record['sha256']:
                raise FileExistsError('Existing artifact differs; refusing to overwrite '+str(target))
            print('Already restored:',record['path']);continue
        for part in record['parts']:
            path=within(part['path'])
            if path.stat().st_size!=part['bytes'] or digest(path)!=part['sha256']:
                raise ValueError('Invalid artifact part: '+part['path'])
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name(target.name+'.restoring')
        with temporary.open('xb') as output:
            for part in record['parts']:
                with within(part['path']).open('rb') as handle:
                    for block in iter(lambda:handle.read(1024*1024),b''):output.write(block)
        if temporary.stat().st_size!=record['bytes'] or digest(temporary)!=record['sha256']:
            raise ValueError('Reassembled artifact checksum mismatch; partial file preserved')
        temporary.rename(target)
        print('Restored:',record['path'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('.'))
    args=parser.parse_args();restore(args.root,args.root/'archive/RAW_ARTIFACT_PARTS.json')
