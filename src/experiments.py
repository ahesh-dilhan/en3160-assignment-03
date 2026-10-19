"""Full-data experiments. Run from the repository root with python -m src.experiments."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
import urllib.request
import zipfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from torchvision import datasets, models, transforms
from .models import ManualMLP, LeNet5, ModernCNN, freeze_backbone_training_mode
from .downloads import CIFAR_URL, fetch_ranges

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('TORCH_HOME', str(ROOT/'data'/'torch'))
# Original host currently redirects to an unreachable endpoint. This is the
# byte-identical archive mirror; torchvision still verifies its canonical MD5.
datasets.CIFAR10.url = CIFAR_URL
SEED = 230145


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def save_history(name, history):
    path = ROOT / 'results' / name / 'history.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def environment():
    return dict(torch=torch.__version__, torchvision=__import__('torchvision').__version__,
                numpy=np.__version__, device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                cuda=torch.version.cuda, seed=SEED, threads=torch.get_num_threads())


@torch.no_grad()
def evaluate_tensor(model, x, y, manual=False, batch_size=512):
    if not manual:
        model.eval()
    total_loss, correct = 0., 0
    predictions = []
    for start in range(0, len(y), batch_size):
        xb, yb = x[start:start+batch_size], y[start:start+batch_size]
        logits = model.forward(xb)[0] if manual else model(xb)
        total_loss += nn.functional.cross_entropy(logits, yb, reduction='sum').item()
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        predictions.append(pred.cpu())
    return dict(loss=total_loss/len(y), accuracy=correct/len(y), correct=correct,
                count=len(y)), torch.cat(predictions)


def save_run(name, history, config, train, test, truth, predicted, checkpoint, start):
    save_history(name, history)
    out = ROOT / 'results' / name
    cm = torch.bincount(truth.cpu() * 10 + predicted.cpu(), minlength=100).reshape(10, 10)
    np.savetxt(out / 'confusion_matrix.csv', cm.numpy(), fmt='%d', delimiter=',')
    np.savez_compressed(out / 'predictions.npz', truth=truth.cpu().numpy(), prediction=predicted.numpy())
    checkpoint_path = ROOT / 'checkpoints' / f'{name}.pt'
    checkpoint_path.parent.mkdir(exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    summary = dict(name=name, config=config, environment=environment(), train=train, test=test,
                   duration_seconds=time.perf_counter()-start,
                   checkpoint_sha256=sha256(checkpoint_path), final_epoch=len(history))
    save_json(out / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


def cifar(device):
    seed_everything()
    start = time.perf_counter()
    fetch_ranges(CIFAR_URL, ROOT/'data'/'cifar-10-python.tar.gz',170498071,
                 'c58f30108f718f92721af3b95e74349a','md5')
    tr = datasets.CIFAR10(str(ROOT/'data'), train=True, download=True)
    te = datasets.CIFAR10(str(ROOT/'data'), train=False, download=True)
    def convert(ds):
        x = torch.as_tensor(ds.data, device=device).permute(0, 3, 1, 2).float().div_(127.5).sub_(1)
        return x.reshape(len(ds), -1), torch.tensor(ds.targets, device=device)
    x, y = convert(tr)
    xt, yt = convert(te)
    net = ManualMLP(device=device)
    history = []
    config = dict(epochs=10, batch_size=50, initial_lr=.1, lr_decay=.9, regularization=0,
                  hidden_units=100, activation='sigmoid', initialization='fan-in normal',
                  normalization='RGB channel-wise (x/255 - 0.5)/0.5', optimizer='manual SGD',
                  training_size=len(y), test_size=len(yt))
    for epoch in range(10):
        lr = .1 * .9**epoch
        order = torch.randperm(len(y), device=device)
        for idx in order.split(50):
            _, gradients = net.loss_and_gradients(x[idx], y[idx])
            net.step(gradients, lr)
        train, _ = evaluate_tensor(net, x, y, manual=True)
        test, pred = evaluate_tensor(net, xt, yt, manual=True)
        row = dict(epoch=epoch+1, lr=lr, train_loss=train['loss'], train_accuracy=train['accuracy'],
                   test_loss=test['loss'], test_accuracy=test['accuracy'])
        history.append(row)
        print('cifar_manual', row, flush=True)
        save_history('cifar_manual', history)
    save_run('cifar_manual', history, config, train, test, yt, pred,
             dict(zip(['w1','b1','w2','b2'], [p.cpu() for p in net.parameters()])), start)


def mnist(device):
    seed_everything()
    tr = datasets.MNIST(str(ROOT/'data'), train=True, download=True)
    te = datasets.MNIST(str(ROOT/'data'), train=False, download=True)
    def convert(ds):
        x = ds.data.to(device).float().unsqueeze(1).div_(255)
        x.sub_(.1307).div_(.3081)
        # Raw MNIST black background remains black in intensity before normalization.
        x = nn.functional.pad(x, (2,2,2,2), value=-.1307/.3081)
        return x, ds.targets.to(device)
    x, y = convert(tr)
    xt, yt = convert(te)
    for name, cls in [('lenet5', LeNet5), ('modern_cnn', ModernCNN)]:
        seed_everything()
        start = time.perf_counter()
        net = cls().to(device)
        opt = torch.optim.Adam(net.parameters(), lr=.001)
        # A separate generator gives both models exactly the same minibatch order.
        generator = torch.Generator(device=device).manual_seed(SEED)
        history = []
        for epoch in range(10):
            net.train()
            order = torch.randperm(len(y), generator=generator, device=device)
            for idx in order.split(128):
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(net(x[idx]), y[idx])
                loss.backward()
                opt.step()
            train, _ = evaluate_tensor(net, x, y, batch_size=256)
            test, pred = evaluate_tensor(net, xt, yt, batch_size=256)
            row = dict(epoch=epoch+1, lr=.001, train_loss=train['loss'], train_accuracy=train['accuracy'],
                       test_loss=test['loss'], test_accuracy=test['accuracy'])
            history.append(row)
            print(name, row, flush=True)
            save_history(name, history)
        config = dict(epochs=10, batch_size=128, optimizer='Adam', lr=.001,
                      betas=[.9,.999], weight_decay=0, augmentation='none',
                      normalization='(x/255 - 0.1307)/0.3081, padded from 28 to 32',
                      training_size=len(y), test_size=len(yt),
                      parameters=sum(p.numel() for p in net.parameters()),
                      metric_mode='eval after each epoch on full train and test splits',
                      checkpoint_selection='final epoch 10, no test-based selection')
        save_run(name, history, config, train, test, yt, pred, net.cpu().state_dict(), start)
        del net, opt
        torch.cuda.empty_cache()


def hymenoptera():
    archive = ROOT/'data'/'hymenoptera_data.zip'
    target = ROOT/'data'/'hymenoptera_data'
    if not target.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            urllib.request.urlretrieve('https://download.pytorch.org/tutorial/hymenoptera_data.zip', archive)
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                dest = (archive.parent/item.filename).resolve()
                if not dest.is_relative_to(archive.parent.resolve()):
                    raise ValueError('Unsafe archive member')
            z.extractall(archive.parent)
    return target


@torch.no_grad()
def evaluate_loader(model, loader, device):
    model.eval()
    loss, correct, total = 0., 0, 0
    ys, ps = [], []
    for x, y in loader:
        logits = model(x.to(device))
        yd = y.to(device)
        loss += nn.functional.cross_entropy(logits, yd, reduction='sum').item()
        p = logits.argmax(1).cpu()
        correct += (p == y).sum().item()
        total += len(y)
        ys.append(y); ps.append(p)
    return dict(loss=loss/total, accuracy=correct/total, correct=correct, count=total), torch.cat(ys), torch.cat(ps)


def transfer(device):
    fetch_ranges('https://download.pytorch.org/models/resnet18-f37072fd.pth',
                 ROOT/'data'/'torch'/'hub'/'checkpoints'/'resnet18-f37072fd.pth',
                 46830571,'f37072fd')
    target = hymenoptera()
    normalize = transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    aug = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(), normalize])
    clean = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
                                transforms.ToTensor(), normalize])
    train_ds = datasets.ImageFolder(target/'train', aug)
    train_eval = datasets.ImageFolder(target/'train', clean)
    val_ds = datasets.ImageFolder(target/'val', clean)
    for name in ['resnet18_finetune', 'resnet18_features']:
        seed_everything()
        start = time.perf_counter()
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        if name.endswith('features'):
            for p in model.parameters():
                p.requires_grad_(False)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.to(device)
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.001, momentum=.9)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=7, gamma=.1)
        generator = torch.Generator().manual_seed(SEED)
        train_loader = torch.utils.data.DataLoader(train_ds,batch_size=16,shuffle=True,
                                                  generator=generator,num_workers=0)
        eval_train_loader = torch.utils.data.DataLoader(train_eval,batch_size=16,num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_ds,batch_size=16,num_workers=0)
        history = []
        for epoch in range(25):
            model.train()
            if name.endswith('features'):
                freeze_backbone_training_mode(model)
            lr = opt.param_groups[0]['lr']
            for x, y in train_loader:
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(x.to(device)), y.to(device))
                loss.backward()
                opt.step()
            train, _, _ = evaluate_loader(model, eval_train_loader, device)
            val, truth, pred = evaluate_loader(model, val_loader, device)
            row = dict(epoch=epoch+1,lr=lr,train_loss=train['loss'],train_accuracy=train['accuracy'],
                       validation_loss=val['loss'],validation_accuracy=val['accuracy'])
            history.append(row)
            print(name, row, flush=True)
            save_history(name, history)
            scheduler.step()
        out = ROOT/'results'/name
        cm = torch.bincount(truth*2+pred,minlength=4).reshape(2,2)
        np.savetxt(out/'confusion_matrix.csv',cm.numpy(),fmt='%d',delimiter=',')
        np.savez_compressed(out/'predictions.npz',truth=truth.numpy(),prediction=pred.numpy())
        checkpoint = ROOT/'checkpoints'/f'{name}.pt'
        checkpoint.parent.mkdir(exist_ok=True)
        torch.save(model.cpu().state_dict(),checkpoint)
        summary = dict(name=name,environment=environment(),train=train,validation=val,
            duration_seconds=time.perf_counter()-start,final_epoch=25,
            checkpoint_sha256=sha256(checkpoint),
            config=dict(epochs=25,batch_size=16,optimizer='SGD',lr=.001,momentum=.9,
                scheduler='StepLR(step_size=7,gamma=0.1)',weights='ResNet18_Weights.IMAGENET1K_V1',
                train_size=len(train_ds),validation_size=len(val_ds),classes=train_ds.classes,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                frozen_batchnorm=name.endswith('features'),checkpoint_selection='final epoch 25',
                evaluation_transform='Resize(256), CenterCrop(224), ImageNet normalization',
                training_transform='RandomResizedCrop(224), RandomHorizontalFlip(), ImageNet normalization'))
        save_json(out/'summary.json',summary)
        print(json.dumps(summary),flush=True)
        del model,opt
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['all','q1','q2','q3'],default='all')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything()
    print(environment(),flush=True)
    for task, fun in [('q1',cifar),('q2',mnist),('q3',transfer)]:
        if args.task in ('all',task):
            fun(device)


if __name__ == '__main__':
    main()
