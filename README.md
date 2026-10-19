# EN3160 Assignment 3 — Neural Networks

**Dilhan W.A. · 230145E · University of Moratuwa**

This repository contains the full-data experiments, notebook, figures, recorded metrics,
and LaTeX-generated notebook PDF for the supplied October 9, 2026 assignment.

## Completed deliverables

- [Assignment 3 report, 8 pages](output/230145E_a03.pdf)
- [Submission filename specified literally in the brief](output/230145E_a02.pdf) — identical PDF, Assignment 3 content
- [Executed Jupyter notebook](notebooks/230145E_a03.ipynb)
- [LaTeX source bundle](output/230145E_a03_latex.zip)
- [Recorded results](results/) and [trained checkpoints](checkpoints/)

| Model | Epochs | Training accuracy | Held-out accuracy |
|---|---:|---:|---:|
| Manual CIFAR-10 MLP | 10 | 53.13% | 47.77% test |
| LeNet-5 on MNIST | 10 | 99.60% | 98.78% test |
| Modern CNN on MNIST | 10 | 99.61% | 99.22% test |
| ResNet18 fine-tuning | 25 | 99.59% | 94.12% validation |
| ResNet18 fixed features | 25 | 97.54% | 96.08% validation |

All five saved checkpoints were independently reloaded; their held-out predictions
exactly match the archived arrays. Four model/gradient checks pass. The PDF was
rendered and visually reviewed, including all eight figures and black document colours.

## Experiments

1. CIFAR-10: 3072 → 100 sigmoid → 10 dense network, stable cross-entropy,
   explicitly derived gradients, manual SGD, 10 epochs on all 50,000 training images.
2. MNIST: LeNet-5 style network and a three-convolution CNN with batch normalization
   and dropout, both trained for 10 epochs under matched conditions.
3. Hymenoptera: ImageNet1K-pretrained ResNet18, comparing full fine-tuning with a
   frozen backbone and a trained two-class head, 25 epochs each.

## Reproduce

Use Python 3.12. Install a compatible PyTorch build first (the recorded experiment
uses PyTorch 2.6.0 + CUDA 12.4 and torchvision 0.21.0), then:

```powershell
python -m pip install -r requirements.txt
python -m pytest tests -q
python -m src.downloads
python scripts/audit_data.py
python -m src.experiments --task all
python scripts/verify_results.py
python scripts/verify_checkpoints.py
python scripts/make_figures.py
python scripts/build_notebook.py
python scripts/export_pdf.py
python scripts/verify_pdf.py
```

For the recorded CUDA build, install `torch==2.6.0+cu124` and
`torchvision==0.21.0+cu124` using the additional package index
`https://download.pytorch.org/whl/cu124`. CPU execution is also supported; it is slower
and can give slightly different floating-point results. PDF export needs
[Tectonic](https://tectonic-typesetting.github.io/) on PATH, or pass its executable
with `python scripts/export_pdf.py --tectonic C:/path/to/tectonic.exe`.
Pandoc is provided by `pypandoc_binary`. Training is not repeated by the notebook
build unless `RUN_TRAINING=True` is selected explicitly.

Training automatically downloads the original datasets. `data/` and upstream cached
ImageNet weights are excluded from Git; sources and a generated file-hash manifest
make the exact inputs identifiable. The final trained checkpoints are retained.

All reported metrics use the final scheduled epoch, with full-split evaluation in
inference mode. The Hymenoptera validation split is not described as an independent
test set. The fixed MNIST training order is identical for both architectures.

The assignment's submission paragraph says `your_index_a02.pdf`, although its title
is Assignment 3. Both an Assignment 3 filename and a byte-identical `a02` alias are
provided inside this repository's output directory; neither replaces Assignment 2.

## Sources

- [Supplied assignment](input/en3160_assignment_03.pdf)
- [PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Byte-identical CIFAR archive mirror](https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python/tree/a48007227f9e2cd0af96175f4afb5ac4965e261b) (canonical MD5 verified)
- [MNIST through torchvision](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)
- [Hymenoptera download](https://download.pytorch.org/tutorial/hymenoptera_data.zip)
- [ResNet18 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)

The code is original assignment work. The transfer-learning preprocessing, optimizer,
and schedule follow the official tutorial, with batch size 16 and explicitly frozen
batch-normalization buffers in the feature-extractor experiment.
