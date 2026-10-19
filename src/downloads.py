"""Resumable, checksum-verified downloads for the exact assignment inputs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.request

CIFAR_URL=('https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python/'
           'resolve/a48007227f9e2cd0af96175f4afb5ac4965e261b/cifar-10-python.tar.gz')

def file_hash(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def fetch_ranges(url,target,size,expected,algorithm='sha256',workers=6):
    target=Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.stat().st_size==size and file_hash(target,algorithm).startswith(expected):
        print('Verified cached',target.name,flush=True)
        return target
    partial=target.with_name(target.name+'.part')
    progress=target.with_name(target.name+'.progress.json')
    step=1024*1024
    done=set()
    if partial.exists() and partial.stat().st_size==size and progress.exists():
        state=json.loads(progress.read_text())
        if state['url']==url and state['size']==size:done=set(state['done'])
    else:
        with partial.open('wb') as f:f.truncate(size)
    lock=threading.Lock()
    def get(start):
        end=min(size-1,start+step-1)
        for attempt in range(8):
            try:
                req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=25) as r:
                    if r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                        raise IOError('Unexpected byte-range response')
                    data=r.read()
                if len(data)!=end-start+1:raise IOError('Incomplete range')
                with partial.open('r+b') as f:f.seek(start);f.write(data)
                with lock:
                    done.add(start)
                    progress.write_text(json.dumps(dict(url=url,size=size,done=sorted(done))))
                return len(data)
            except Exception:
                if attempt==7:raise
                time.sleep(min(5,attempt+1))
    remaining=[i for i in range(0,size,step) if i not in done]
    print(f'Downloading {target.name}: {len(remaining)} ranges remaining',flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(get,i) for i in remaining]
        for i,future in enumerate(as_completed(futures)):
            future.result()
            if i%25==0:print(target.name, f'{len(done)*step/1e6:.0f}/{size/1e6:.0f} MB',flush=True)
    actual=file_hash(partial,algorithm)
    if not actual.startswith(expected):raise IOError(f'Checksum mismatch: {actual}')
    partial.replace(target)
    progress.unlink(missing_ok=True)
    print('Checksum verified:',target.name,actual,flush=True)
    return target

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    fetch_ranges(CIFAR_URL,root/'data'/'cifar-10-python.tar.gz',170498071,
                 'c58f30108f718f92721af3b95e74349a','md5',workers=8)
    fetch_ranges('https://download.pytorch.org/models/resnet18-f37072fd.pth',
                 root/'data'/'torch'/'hub'/'checkpoints'/'resnet18-f37072fd.pth',46830571,'f37072fd',workers=6)
    for name,size,md5 in [
        ('train-images-idx3-ubyte.gz',9912422,'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
        ('train-labels-idx1-ubyte.gz',28881,'d53e105ee54ea40749a09fcbcd1e9432'),
        ('t10k-images-idx3-ubyte.gz',1648877,'9fb629c4189551a2d022fa330f9573f3'),
        ('t10k-labels-idx1-ubyte.gz',4542,'ec29112dd5afa0611ce80d1b7f02629c')]:
        fetch_ranges('https://ossci-datasets.s3.amazonaws.com/mnist/'+name,
                     root/'data'/'MNIST'/'raw'/name,size,md5,'md5')
