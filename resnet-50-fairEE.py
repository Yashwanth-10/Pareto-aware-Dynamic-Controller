import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from models.early_exit import (
    ActivationSummaryModule,
    EarlyExitController,
    EarlyExitWrapper,
)

# =====================================================
# Device
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# =====================================================
# CIFAR-10
# =====================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])

trainset = torchvision.datasets.CIFAR100(
    root="./data", train=True, download=True, transform=transform
)
testset = torchvision.datasets.CIFAR100(
    root="./data", train=False, download=True, transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False
)

# =====================================================
# 1️⃣ VANILLA RESNET-50
# =====================================================
vanilla_model = torchvision.models.resnet50(num_classes=10).to(device)

def train_vanilla(model, epochs=20, lr=0.1):
    model.train()
    opt = torch.optim.SGD(
        model.parameters(), lr=lr,
        momentum=0.9, weight_decay=5e-4
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ce = nn.CrossEntropyLoss()

    for e in range(epochs):
        correct, total = 0, 0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        sched.step()
        print(f"[Vanilla] Epoch {e+1}/{epochs} | Acc: {correct/total:.4f}")

print("\nTraining Vanilla ResNet-50")
train_vanilla(vanilla_model)
vanilla_model.eval()

# =====================================================
# 2️⃣ INBUILT EARLY-EXIT RESNET-50 (TRAINED)
# =====================================================
class ResNet50_InbuiltEE(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = torchvision.models.resnet50(num_classes=num_classes)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = base.fc

        self.exit2 = nn.Linear(512, num_classes)
        self.exit3 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)

        x = self.layer2(x)
        p2 = x.mean(dim=[2, 3])
        out2 = self.exit2(p2)

        x = self.layer3(x)
        p3 = x.mean(dim=[2, 3])
        out3 = self.exit3(p3)

        x = self.layer4(x)
        p4 = torch.flatten(self.avgpool(x), 1)
        out4 = self.fc(p4)

        return out2, out3, out4

inbuilt_model = ResNet50_InbuiltEE().to(device)

# Copy trained backbone weights
inbuilt_model.load_state_dict(vanilla_model.state_dict(), strict=False)

def train_inbuilt(model, epochs=10):
    model.train()
    opt = torch.optim.SGD(
        model.parameters(), lr=0.01,
        momentum=0.9, weight_decay=5e-4
    )
    ce = nn.CrossEntropyLoss()

    for e in range(epochs):
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            o2, o3, o4 = model(x)

            loss = (
                0.3 * ce(o2, y) +
                0.3 * ce(o3, y) +
                0.4 * ce(o4, y)
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"[Inbuilt EE] Epoch {e+1}/{epochs}")

print("\nTraining Inbuilt Early-Exit ResNet-50")
train_inbuilt(inbuilt_model)
inbuilt_model.eval()

# =====================================================
# 3️⃣ GENERALIZED EARLY-EXIT (BACKBONE-AGNOSTIC)
# =====================================================
summary_module = ActivationSummaryModule(num_classes=10)
controller = EarlyExitController(
    summary_dim=6, mode="rule", exit_conf_threshold=0.85
)

stages = [
    lambda x: vanilla_model.relu(vanilla_model.bn1(vanilla_model.conv1(x))),
    vanilla_model.layer1,
    vanilla_model.layer2,
    vanilla_model.layer3,
    vanilla_model.layer4,
    lambda x: torch.flatten(vanilla_model.avgpool(x), 1),
]

logits_fns = [
    None, None, None, None, None,
    lambda x: vanilla_model.fc(x)
]

generalized_model = EarlyExitWrapper(
    backbone=vanilla_model,
    stages=stages,
    logits_fns=logits_fns,
    num_layers=len(stages),
    summary_module=summary_module,
    controller=controller,
    exit_on_first=True
).to(device)

generalized_model.eval()

# =====================================================
# EVALUATION
# =====================================================
@torch.no_grad()
def evaluate_vanilla():
    correct, total = 0, 0
    start = time.time()
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        preds = vanilla_model(x).argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct/total, time.time() - start


@torch.no_grad()
def evaluate_inbuilt():
    correct, total = 0, 0
    start = time.time()
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        _, _, out = inbuilt_model(x)
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct/total, time.time() - start


@torch.no_grad()
def evaluate_general():
    correct, total = 0, 0
    start = time.time()
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        logits = generalized_model(x, budget=10.0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct/total, time.time() - start

# =====================================================
# RUN + PLOT
# =====================================================
va, vt = evaluate_vanilla()
ia, it = evaluate_inbuilt()
ga, gt = evaluate_general()

labels = ["Vanilla", "Inbuilt EE", "Generalized EE"]
accs = [va, ia, ga]
times = [vt, it, gt]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(labels, accs)
plt.ylabel("Accuracy")

plt.subplot(1, 2, 2)
plt.bar(labels, times)
plt.ylabel("Inference Time (s)")
plt.tight_layout()
plt.show()

print("\n===== FINAL FAIR COMPARISON =====")
print(f"Vanilla Acc        : {va:.4f} | Time: {vt:.2f}s")
print(f"Inbuilt EE Acc     : {ia:.4f} | Time: {it:.2f}s")
print(f"Generalized EE Acc : {ga:.4f} | Time: {gt:.2f}s")
print("================================")
