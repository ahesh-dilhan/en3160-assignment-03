import torch
from src.models import ManualMLP, LeNet5, ModernCNN, freeze_backbone_training_mode
from torchvision.models import resnet18


def test_manual_gradients_match_autograd():
    torch.manual_seed(42)
    net = ManualMLP(dtype=torch.float64)
    x = torch.randn(7,3072,dtype=torch.float64)
    y = torch.tensor([0,9,4,2,2,7,1])
    loss, gradients = net.loss_and_gradients(x,y)
    for p in net.parameters():
        p.requires_grad_(True)
    reference = torch.nn.functional.cross_entropy(net.forward(x)[0],y)
    reference.backward()
    torch.testing.assert_close(loss,reference)
    for p,g in zip(net.parameters(),gradients):
        torch.testing.assert_close(g,p.grad,rtol=1e-10,atol=1e-12)


def test_manual_large_logits_stable_and_step_decreases_loss():
    torch.manual_seed(1)
    net = ManualMLP(dtype=torch.float64)
    net.b2[0]=1000
    x=torch.randn(4,3072,dtype=torch.float64)
    y=torch.tensor([1,2,3,4])
    loss,grads=net.loss_and_gradients(x,y)
    assert torch.isfinite(loss)
    net.step(grads,.01)
    updated,_=net.loss_and_gradients(x,y)
    assert updated<loss


def test_network_shapes_and_required_layers():
    x=torch.randn(2,1,32,32)
    for model in [LeNet5(),ModernCNN()]:
        assert model(x).shape==(2,10)
    modern=ModernCNN()
    assert sum(isinstance(m,torch.nn.Conv2d) for m in modern.modules())==3
    assert sum(isinstance(m,torch.nn.BatchNorm2d) for m in modern.modules())==3
    assert sum(isinstance(m,torch.nn.Dropout2d) for m in modern.modules())==3


def test_feature_extraction_freezes_batchnorm_statistics():
    model=resnet18(weights=None)
    for p in model.parameters(): p.requires_grad_(False)
    model.fc=torch.nn.Linear(512,2)
    before=model.bn1.running_mean.clone()
    freeze_backbone_training_mode(model)
    model(torch.randn(2,3,64,64)).sum().backward()
    torch.testing.assert_close(before,model.bn1.running_mean)
    assert model.fc.weight.grad is not None
    assert model.conv1.weight.grad is None
