"""Generate figures only from recorded predictions, histories and real inputs."""
import csv
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp'/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image
from torchvision import datasets

OUT=ROOT/'results'/'figures'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':'black',
                     'axes.labelcolor':'black','axes.edgecolor':'black',
                     'xtick.color':'black','ytick.color':'black','savefig.facecolor':'white',
                     'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})

def history(name):
    with (ROOT/'results'/name/'history.csv').open() as f:
        return {k:np.array([float(r[k]) for r in rows]) for rows in [list(csv.DictReader(f))] for k in rows[0]}

def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=190,bbox_inches='tight')
    plt.close(fig)

def curves(names,labels,split,out):
    fig,axes=plt.subplots(1,2,figsize=(10,3.3),layout='constrained')
    for name,label,color in zip(names,labels,['#205b8f','#ad4c19']):
        h=history(name)
        for key,ls in [('train','-'),(split,'--')]:
            axes[0].plot(h['epoch'],h[f'{key}_loss'],ls,color=color,label=f'{label}: {key}')
            axes[1].plot(h['epoch'],100*h[f'{key}_accuracy'],ls,color=color,label=f'{label}: {key}')
    for ax,title,ylabel in zip(axes,['Cross-entropy loss','Accuracy'],['Mean loss','Accuracy (%)']):
        ax.set(title=title,xlabel='Epoch',ylabel=ylabel)
        ax.grid(alpha=.18,color='black');ax.legend(fontsize=8,frameon=False)
    save(fig,out)

def matrices(names,labels,classes,out):
    cmap=LinearSegmentedColormap.from_list('lightblue',['#ffffff','#d1e3ed','#91b6d0'])
    fig,axes=plt.subplots(1,len(names),figsize=(10,4.7 if len(classes)==10 else 3.8),layout='constrained')
    for ax,name,title in zip(np.atleast_1d(axes),names,labels):
        cm=np.loadtxt(ROOT/'results'/name/'confusion_matrix.csv',delimiter=',',dtype=int)
        ax.imshow(np.log1p(cm),cmap=cmap,vmin=0,vmax=np.log1p(cm.max()))
        for (i,j),v in np.ndenumerate(cm):
            ax.text(j,i,str(v),ha='center',va='center',fontsize=7 if len(classes)==10 else 13,color='black')
        ax.set(xticks=range(len(classes)),yticks=range(len(classes)),xticklabels=classes,
               yticklabels=classes,xlabel='Predicted label',ylabel='True label',title=title)
        ax.spines[['top','right']].set_visible(True)
    save(fig,out)

def input_examples():
    cf=datasets.CIFAR10(str(ROOT/'data'),train=True,download=False)
    mn=datasets.MNIST(str(ROOT/'data'),train=True,download=False)
    bugs=datasets.ImageFolder(ROOT/'data'/'hymenoptera_data'/'train')
    fig,axes=plt.subplots(3,5,figsize=(10,5.6),layout='constrained')
    for j,i in enumerate([0,1,2,3,4]):
        im,lab=cf[i];axes[0,j].imshow(im);axes[0,j].set_title('CIFAR: '+cf.classes[lab],fontsize=9)
        im,lab=mn[i];axes[1,j].imshow(im,cmap='gray');axes[1,j].set_title('MNIST: '+str(lab),fontsize=9)
        idx=[0,20,125,150,200][j]
        im,lab=bugs[idx];axes[2,j].imshow(im);axes[2,j].set_title('Hymenoptera: '+bugs.classes[lab],fontsize=9)
    for ax in axes.flat:ax.axis('off')
    save(fig,'dataset_examples')

def mnist_errors():
    ds=datasets.MNIST(str(ROOT/'data'),train=False)
    fig,axes=plt.subplots(2,6,figsize=(10,3.8),layout='constrained')
    for row,name in enumerate(['lenet5','modern_cnn']):
        data=np.load(ROOT/'results'/name/'predictions.npz')
        errors=np.flatnonzero(data['truth']!=data['prediction'])[:6]
        for ax,idx in zip(axes[row],errors):
            ax.imshow(ds.data[int(idx)],cmap='gray')
            ax.set_title(f"True {data['truth'][idx]}, pred {data['prediction'][idx]}",fontsize=9)
            ax.axis('off')
        axes[row,0].text(-.15,.5,['LeNet-5','Modern CNN'][row],transform=axes[row,0].transAxes,
                           rotation=90,ha='right',va='center',fontsize=10)
    save(fig,'mnist_errors')

def transfer_examples():
    ds=datasets.ImageFolder(ROOT/'data'/'hymenoptera_data'/'val')
    ft=np.load(ROOT/'results'/'resnet18_finetune'/'predictions.npz')
    fe=np.load(ROOT/'results'/'resnet18_features'/'predictions.npz')
    err=np.flatnonzero((ft['prediction']!=ft['truth']) | (fe['prediction']!=fe['truth']))
    indices=list(err[:3])+[i for i in [0,40,80,120,150] if i not in err][:3]
    fig,axes=plt.subplots(2,3,figsize=(10,5.7),layout='constrained')
    for ax,idx in zip(axes.flat,indices):
        image,label=ds[int(idx)]
        ax.imshow(image)
        ax.set_title(f"True: {ds.classes[label]} | FT: {ds.classes[ft['prediction'][idx]]}\n"
                     f"Fixed features: {ds.classes[fe['prediction'][idx]]}",fontsize=9)
        ax.axis('off')
    save(fig,'transfer_examples')

def main():
    curves(['cifar_manual'],['Manual MLP'],'test','q1_curves')
    curves(['lenet5','modern_cnn'],['LeNet-5','Modern CNN'],'test','q2_curves')
    matrices(['lenet5','modern_cnn'],['LeNet-5: test counts','Modern CNN: test counts'],list(range(10)),'q2_confusion')
    curves(['resnet18_finetune','resnet18_features'],['Fine-tuning','Fixed features'],'validation','q3_curves')
    matrices(['resnet18_finetune','resnet18_features'],['Fine-tuning: validation counts','Fixed features: validation counts'],['ants','bees'],'q3_confusion')
    input_examples();mnist_errors();transfer_examples()
    manifest=[]
    from src.experiments import sha256,save_json
    for path in sorted((ROOT/'data').rglob('*')):
        if path.is_file():manifest.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=sha256(path)))
    save_json(ROOT/'results'/'input_manifest.json',manifest)
    print('Wrote eight figures and input manifest.')

if __name__=='__main__':main()
