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

# =================================================
# Device
# =================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =================================================
# CIFAR-10 Dataset
# =================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False
)

# =================================================
# Vanilla ResNet-50 Backbone
# =================================================
backbone = torchvision.models.resnet50(num_classes=10).to(device)

# =================================================
# Training (Vanilla Backbone)
# =================================================
def train(model, loader, epochs=20, lr=0.1):
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr,
        momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}] Train Acc: {correct/total:.4f}")

print("\nTraining Vanilla ResNet-50...")
train(backbone, trainloader, epochs=20)
backbone.eval()

# =================================================
# Inbuilt Early-Exit ResNet-50
# =================================================
class ResNet50_InbuiltEE(nn.Module):
    def __init__(self, num_classes=10, threshold=0.85):
        super().__init__()
        self.threshold = threshold

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

        # Correct dimensions for ResNet-50
        self.exit2 = nn.Linear(512, num_classes)
        self.exit3 = nn.Linear(1024, num_classes)

    def forward(self, x, early_exit=True):
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        p = x.mean(dim=[2, 3])
        logits2 = self.exit2(p)

        if early_exit and torch.max(F.softmax(logits2, 1)) > self.threshold:
            return logits2, 2

        x = self.layer3(x)
        p = x.mean(dim=[2, 3])
        logits3 = self.exit3(p)

        if early_exit and torch.max(F.softmax(logits3, 1)) > self.threshold:
            return logits3, 3

        x = self.layer4(x)
        p = torch.flatten(self.avgpool(x), 1)
        logits4 = self.fc(p)

        return logits4, 4

inbuilt_model = ResNet50_InbuiltEE().to(device)

# Copy trained backbone weights
inbuilt_model.load_state_dict(backbone.state_dict(), strict=False)
inbuilt_model.eval()

# =================================================
# Generalized Early-Exit Wrapper (External)
# =================================================
stages = [
    lambda x: backbone.relu(backbone.bn1(backbone.conv1(x))),
    backbone.layer1,
    backbone.layer2,
    backbone.layer3,
    backbone.layer4,
    lambda x: torch.flatten(backbone.avgpool(x), 1),
]

logits_fns = [
    None,
    None,
    None,
    None,
    None,
    lambda x: backbone.fc(x),
]

summary_module = ActivationSummaryModule(
    num_classes=10,
    sparsity_threshold=0.05,
)

controller = EarlyExitController(
    summary_dim=6,
    mode="rule",
    exit_conf_threshold=0.85,
)

wrapper = EarlyExitWrapper(
    backbone=backbone,
    stages=stages,
    logits_fns=logits_fns,
    num_layers=len(stages),
    summary_module=summary_module,
    controller=controller,
    per_layer_cost=1.0,
    exit_on_first=True,
).to(device)

wrapper.eval()

# =================================================
# Evaluation Functions
# =================================================
@torch.no_grad()
def evaluate_vanilla(model):
    correct, total = 0, 0
    start = time.time()

    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / total, time.time() - start


@torch.no_grad()
def evaluate_inbuilt(model):
    correct, total = 0, 0
    start = time.time()

    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x, early_exit=True)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / total, time.time() - start


@torch.no_grad()
def evaluate_general(model):
    correct, total = 0, 0
    start = time.time()

    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        logits = model(x, budget=10.0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / total, time.time() - start

# =================================================
# Run Experiments
# =================================================
vanilla_acc, vanilla_time = evaluate_vanilla(backbone)
inbuilt_acc, inbuilt_time = evaluate_inbuilt(inbuilt_model)
general_acc, general_time = evaluate_general(wrapper)

# =================================================
# Plot Results
# =================================================
labels = ["Vanilla", "Inbuilt EE", "Generalized EE"]
accs = [vanilla_acc, inbuilt_acc, general_acc]
times = [vanilla_time, inbuilt_time, general_time]

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.bar(labels, accs)
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")

plt.subplot(1, 2, 2)
plt.bar(labels, times)
plt.ylabel("Inference Time (s)")
plt.title("Latency Comparison")

plt.tight_layout()
plt.show()

# =================================================
# Print Summary
# =================================================
print("\n===== CIFAR-10 ResNet-50 Early-Exit Comparison =====")
print(f"Vanilla Accuracy        : {vanilla_acc:.4f}")
print(f"Inbuilt EE Accuracy     : {inbuilt_acc:.4f}")
print(f"Generalized EE Accuracy : {general_acc:.4f}")
print(f"Vanilla Time            : {vanilla_time:.2f}s")
print(f"Inbuilt EE Time         : {inbuilt_time:.2f}s")
print(f"Generalized EE Time     : {general_time:.2f}s")
print("===================================================")
