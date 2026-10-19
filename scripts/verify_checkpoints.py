"""Reload final model checkpoints and independently repeat held-out inference."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json
import numpy as np
import torch
from torch import nn
from torchvision import datasets,models,transforms
from src.models import ManualMLP,LeNet5,ModernCNN
from src.experiments import evaluate_tensor,evaluate_loader,seed_everything,save_json

def main():
    seed_everything()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results={}
    for name in ['cifar_manual','lenet5','modern_cnn','resnet18_finetune','resnet18_features']:
        state=torch.load(ROOT/'checkpoints'/f'{name}.pt',map_location=device,weights_only=True)
        if name=='cifar_manual':
            ds=datasets.CIFAR10(str(ROOT/'data'),train=False)
            x=torch.as_tensor(ds.data,device=device).permute(0,3,1,2).float().div_(127.5).sub_(1).reshape(len(ds),-1)
            y=torch.tensor(ds.targets,device=device)
            net=ManualMLP(device=device)
            for key,value in state.items():setattr(net,key,value)
            metrics,pred=evaluate_tensor(net,x,y,manual=True)
        elif name in ['lenet5','modern_cnn']:
            ds=datasets.MNIST(str(ROOT/'data'),train=False)
            x=ds.data.to(device).float().unsqueeze(1).div_(255).sub_(.1307).div_(.3081)
            x=nn.functional.pad(x,(2,2,2,2),value=-.1307/.3081)
            y=ds.targets.to(device)
            net=(LeNet5() if name=='lenet5' else ModernCNN()).to(device)
            net.load_state_dict(state)
            metrics,pred=evaluate_tensor(net,x,y,batch_size=256)
        else:
            t=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),
                                   transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
            ds=datasets.ImageFolder(ROOT/'data'/'hymenoptera_data'/'val',t)
            loader=torch.utils.data.DataLoader(ds,batch_size=16,num_workers=0)
            net=models.resnet18(weights=None)
            net.fc=nn.Linear(512,2)
            net.load_state_dict(state)
            net.to(device)
            metrics,y,pred=evaluate_loader(net,loader,device)
        recorded=np.load(ROOT/'results'/name/'predictions.npz')
        np.testing.assert_array_equal(pred.cpu().numpy(),recorded['prediction'])
        np.testing.assert_array_equal(y.cpu().numpy(),recorded['truth'])
        s=json.loads((ROOT/'results'/name/'summary.json').read_text())
        split='validation' if name.startswith('resnet') else 'test'
        assert metrics['correct']==s[split]['correct']
        results[name]=dict(predictions_match=True,correct=metrics['correct'],count=metrics['count'])
        print(name,': independently reloaded predictions match exactly',flush=True)
        del net,state
        if 'x' in locals():del x
        torch.cuda.empty_cache()
    save_json(ROOT/'results'/'checkpoint_verification.json',results)

if __name__=='__main__':main()
