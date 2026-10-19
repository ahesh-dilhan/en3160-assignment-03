"""Validate the supplied task's downloaded image data before training."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from torchvision import datasets
from src.experiments import hymenoptera,save_json

def fingerprints(array):
    return {hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in array}

def main():
    results={}
    for label,cls,shape in [('CIFAR10',datasets.CIFAR10,(32,32,3)),('MNIST',datasets.MNIST,(28,28))]:
        splits={}
        hashes={}
        for split in ['train','test']:
            ds=cls(str(ROOT/'data'),train=split=='train',download=True)
            data=np.asarray(ds.data)
            y=np.asarray(ds.targets)
            assert tuple(data.shape[1:])==shape
            assert data.dtype==np.uint8 and data.min()>=0 and data.max()<=255
            assert set(y.tolist())==set(range(10))
            splits[split]=dict(count=len(ds),shape=list(data.shape),dtype=str(data.dtype),
                               counts_per_class=np.bincount(y,minlength=10).tolist())
            hashes[split]=fingerprints(data)
        results[label]=dict(splits=splits,exact_pixel_hashes_shared_between_splits=len(hashes['train']&hashes['test']),
                            policy='Keep the official splits unchanged; report exact pixel matches transparently.')
    target=hymenoptera()
    splits={};hashes={}
    for split in ['train','val']:
        ds=datasets.ImageFolder(target/split)
        hs=set()
        for path,y in ds.samples:
            with Image.open(path) as im:
                im.load()
                assert min(im.size)>0
                hs.add(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
        hashes[split]=hs
        splits[split]=dict(count=len(ds),classes=ds.classes,
                           counts_per_class=np.bincount(ds.targets,minlength=2).tolist())
    results['Hymenoptera']=dict(splits=splits,
        exact_pixel_hashes_shared_between_splits=len(hashes['train']&hashes['val']))
    save_json(ROOT/'results'/'data_audit.json',results)
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
