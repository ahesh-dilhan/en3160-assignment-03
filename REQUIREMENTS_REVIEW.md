# Assignment review and decisions

Read all three pages, including the provided code and final submission instructions.
No separate data archive was supplied with Assignment 3.

| Requirement | Deliverable |
|---|---|
| Q1: add 100 sigmoid hidden units | `ManualMLP` in `src/models.py` |
| Q1: cross-entropy and manual backpropagation | Stable log-sum-exp loss; all four analytic gradients; autograd oracle tests |
| Q1: ten epochs, train/test accuracy | Full CIFAR-10 run; epoch CSV and final summary |
| Q2a: LeNet-5, MNIST, ten epochs | Padded 32×32 input; tanh and average pooling; modern linear CE output |
| Q2b: three conv layers, BN and dropout | `ModernCNN`; exactly three conv/BN blocks |
| Q2b: same conditions | Same data, split, normalization, optimizer, batch size, epochs, seed, shuffle order |
| Q2c: per-epoch train/test loss and accuracy | Full-split inference metrics and four learning curves |
| Q2c: test confusion matrix for each | Count matrices with true labels as rows and predictions as columns |
| Q3: pretrained ImageNet1K ResNet18 | Explicit `ResNet18_Weights.IMAGENET1K_V1` |
| Q3: fine-tune versus feature extractor | Same schedule and augmentation; all parameters versus final head only |
| PDF of Jupyter notebook with code/results/discussion | Executed notebook, black-text LaTeX export, original-colour images |
| Name/index and GitHub link | Compact first-page header and direct repository link |

## Interpretation choices

- Q1 preserves the original dataset, normalization, batch size 50, zero regularization,
  and 0.9 epoch decay. The example's `std=1e-5` and `lr=2e-6` were intended for its
  different MSE model; this sigmoid/CE network uses fan-in initialization and SGD 0.1.
  These changes are disclosed rather than silently attributed to the listing.
- LeNet uses dense equivalents of its final 5×5 convolution, with a 10-logit
  cross-entropy head rather than the historical Euclidean RBF classifier.
- Q3 has no specified epoch count; the tutorial's 25 epochs are used for both methods.
  Evaluation uses deterministic center crops. The frozen model's BN buffers remain
  fixed, so it is a genuine fixed feature extractor.
- No hyperparameters or checkpoints are selected using the test/validation curves.
  All final results are from the last scheduled epoch. Only one fixed-seed run is
  claimed, so small differences do not establish statistical superiority.
- Submission naming appears copied from Assignment 2; provide both names in the
  isolated Assignment 3 output directory, preserving the old assignment.
