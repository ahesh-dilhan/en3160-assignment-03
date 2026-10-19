"""Cross-check independently stored artifacts rather than trusting report text."""
import csv
import json
from pathlib import Path
import hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    expected={'cifar_manual':(10,10000,10),'lenet5':(10,10000,10),
              'modern_cnn':(10,10000,10),'resnet18_finetune':(25,153,2),
              'resnet18_features':(25,153,2)}
    for name,(epochs,count,classes) in expected.items():
        folder=ROOT/'results'/name
        s=json.loads((folder/'summary.json').read_text())
        h=list(csv.DictReader((folder/'history.csv').open()))
        assert len(h)==epochs and [int(r['epoch']) for r in h]==list(range(1,epochs+1))
        assert s['final_epoch']==epochs
        key='validation' if classes==2 else 'test'
        p=np.load(folder/'predictions.npz')
        cm=np.loadtxt(folder/'confusion_matrix.csv',delimiter=',',dtype=int)
        direct=np.bincount(p['truth']*classes+p['prediction'],minlength=classes**2).reshape(classes,classes)
        np.testing.assert_array_equal(cm,direct)
        assert cm.sum()==count==s[key]['count']
        assert np.trace(cm)==s[key]['correct']
        assert abs(np.trace(cm)/count-s[key]['accuracy'])<1e-12
        assert abs(float(h[-1][key+'_accuracy'])-s[key]['accuracy'])<1e-12
        assert abs(float(h[-1]['train_accuracy'])-s['train']['accuracy'])<1e-12
        assert digest(ROOT/'checkpoints'/f'{name}.pt')==s['checkpoint_sha256']
        print(name,': metrics, predictions, confusion counts, epochs, checkpoint hash verified')
    print('All experiment artifact checks passed.')

if __name__=='__main__':main()
