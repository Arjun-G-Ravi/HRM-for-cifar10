import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        shear=5,
        scale=(0.8, 1.2)
    ),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])
bs = 512
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2)

class HighBlock(nn.Module):
    '''
    High level CEO planner; slow
    '''
    def __init__(self):
        super(HighBlock, self).__init__()

class LowBlock(nn.Module):
    '''
    Low level worker; super fast
    '''
    def __init__(self):
        super(LowBlock, self).__init__()

class HRM(nn.Module):
    def __init__(self):
        super(HRM, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool1 = nn.MaxPool2d(2, 2, padding=1)
        self.dropout1 = nn.Dropout(0.5)
        
        self.lstm = nn.LSTM(7200, 1024, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 10)
    
    def forward(self, x):
        bs = x.size(0)
        out = F.elu(self.conv1(x))
        out = F.elu(self.conv2(out))
        out = self.pool1(out)
        out = self.dropout1(out)
        out  = out.view(bs,1, -1)

        out, _ = self.lstm(out)
        out = self.fc1(out[:, -1, :])
        out = self.fc2(out)
        return out

# class RecurrentBlock(nn.Module):
#     def __init__(self):
#         super(RecurrentBlock, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, 3)
#         self.conv2 = nn.Conv2d(16, 32, 3)
#         self.pool1 = nn.MaxPool2d(2, 2, padding=1)
#         self.dropout1 = nn.Dropout(0.5)
#         self.lstm = nn.LSTM(7200, 1024, batch_first=True)
#         self.dropout1 = nn.Dropout(0.3)
#         self.fc1 = nn.Linear(1024, 2048)
#         self.fc2 = nn.Linear(2048, 10)
    
#     def forward(self, x):
#         bs = x.size(0)
#         out = F.elu(self.conv1(x))
#         out = F.elu(self.conv2(out))
#         out = self.pool1(out)
#         out = self.dropout1(out)
#         out  = out.view(bs,1, -1)

#         out, _ = self.lstm(out)
#         out = self.fc1(out[:, -1, :])
#         out = self.fc2(out)
#         return out

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = HRM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Lists to store metrics
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
epochs_recorded = []

# Training
for epoch in range(10):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    avg_train_loss = running_loss / len(trainloader)
    train_acc = 100 * correct_train / total_train
    print(f'Epoch {epoch+1}, Loss: {avg_train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')

    # Evaluation every 5 epochs
    if (epoch+1) % 5 == 0 and epoch > 1:
        model.eval()

        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_test_loss = test_loss / len(testloader)
        test_acc = 100 * correct / total
        print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
        
        # Store metrics
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        epochs_recorded.append(epoch + 1)

# Plot graphs
plt.figure(figsize=(15, 5))

# Loss plot
plt.subplot(1, 3, 1)
plt.plot(epochs_recorded, train_losses, 'b-', marker='o', label='Train')
plt.plot(epochs_recorded, test_losses, 'r-', marker='s', label='Test')
plt.title('Train vs Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Combined accuracy plot
plt.subplot(1, 3, 2)
plt.plot(epochs_recorded, train_accuracies, 'g-', marker='o', label='Train')
plt.plot(epochs_recorded, test_accuracies, 'r-', marker='s', label='Test')
plt.title('Train vs Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()