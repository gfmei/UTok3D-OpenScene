"""Extract only RGB, depth and camera JSON; optionally verify the full-tar MD5."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time

EXPECTED = {'area_5a_no_xyz.tar':'961a490072093fca726a483b74169b48',
            'area_5b_no_xyz.tar':'85fd6de1a7c7247d8563ed5976dbd06d'}


class HashReader:
    def __init__(self, f):
        self.f = f
        self.h = hashlib.md5()

    def read(self, size=-1):
        data = self.f.read(size)
        self.h.update(data)
        return data


def main():
    p=argparse.ArgumentParser()
    p.add_argument('archive',type=Path)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent.parent/'data/stanford2d3d_no_xyz')
    p.add_argument('--verify-md5',action='store_true')
    args=p.parse_args()
    progress=args.archive.with_suffix('.tar.progress.json')
    if progress.exists():
        state=json.loads(progress.read_text())
        total=(state['bytes']+state['chunk']-1)//state['chunk']
        if len(state['completed'])!=total:
            raise ValueError('Download has not finished')
    selected=[]
    start=time.monotonic()
    # Seek over unused meshes, panoramas, normals and annotations. This saves
    # reading tens of GB before reaching the perspective RGB-D files.
    with args.archive.open('rb',buffering=64*1024) as f:
        with tarfile.open(fileobj=f,mode='r:') as tar:
            for m in tar:
                parts=Path(m.name).parts
                if not m.isfile() or len(parts)!=4 or parts[0] not in ('area_5a','area_5b') \
                    or parts[1]!='data' or parts[2] not in ('rgb','depth','pose'):
                    continue
                dest=args.output.joinpath(*parts)
                if '..' in parts or not dest.resolve().is_relative_to(args.output.resolve()):
                    raise ValueError('Invalid archive path')
                dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.is_file() or dest.stat().st_size!=m.size:
                    tmp=dest.with_suffix(dest.suffix+'.part')
                    with tar.extractfile(m) as source,tmp.open('wb') as target:
                        shutil.copyfileobj(source,target,1024**2)
                    if tmp.stat().st_size!=m.size:
                        raise IOError('Incomplete extraction')
                    tmp.replace(dest)
                selected.append(m.name)
                if len(selected)%1000==0:
                    print(args.archive.name,'extracted',len(selected),
                          'files; elapsed',round(time.monotonic()-start),'s',flush=True)
    actual=None
    if args.verify_md5:
        h=hashlib.md5()
        with args.archive.open('rb') as f:
            for data in iter(lambda:f.read(16*1024**2),b''):
                h.update(data)
        actual=h.hexdigest()
        if actual!=EXPECTED[args.archive.name]:
            raise ValueError(f'MD5 mismatch: {actual}')
    report=dict(archive=str(args.archive),md5=actual,verified=actual is not None,files=len(selected),
                output=str(args.output),seconds=time.monotonic()-start)
    (args.output/f'{args.archive.stem}_extraction.json').write_text(json.dumps(report,indent=2))
    print('EXTRACTION COMPLETE',report,flush=True)


if __name__=='__main__':
    main()
