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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2)

class HighBlock(nn.Module):
    '''
    High level CEO planner; slow - processes global context
    '''
    def __init__(self):
        super(HighBlock, self).__init__()
        self.lstm_input_size = 512
        self.lstm_hidden_size = 512
        # Larger kernel for global context
        self.context_conv = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(self.lstm_input_size, self.lstm_hidden_size, batch_first=True)
        self.layer_norm = nn.LayerNorm(512)

    def forward(self, out, hidden=None):
        # Apply global context processing
        context = self.context_conv(out.transpose(1, 2)).transpose(1, 2)
        out = out + context
        out = self.layer_norm(out)
        out, hidden = self.lstm(out, hidden)
        return out, hidden
    
class LowBlock(nn.Module):
    '''
    Low level worker; super fast - processes local features
    '''
    def __init__(self):
        super(LowBlock, self).__init__()
        self.lstm_input_size = 512
        self.lstm_hidden_size = 512
        # Smaller kernel for local processing
        self.local_conv = nn.Conv1d(512, 512, kernel_size=1)
        self.lstm = nn.LSTM(self.lstm_input_size, self.lstm_hidden_size, batch_first=True, num_layers=1)
        self.layer_norm = nn.LayerNorm(512)

    def forward(self, out, hidden=None):
        # Apply local feature processing
        local = self.local_conv(out.transpose(1, 2)).transpose(1, 2)
        out = out + local
        out = self.layer_norm(out)
        out, hidden = self.lstm(out, hidden)
        return out, hidden
    
class HRM(nn.Module):
    '''
    Hierarchical Reasoning Model with proper sequence processing
    '''
    def __init__(self):
        super(HRM, self).__init__()

        self.num_iterations = 3
        self.num_patches = 16  # 4x4 spatial patches

        # CNN
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.bn6 = nn.BatchNorm2d(128)

        self.pool1 = nn.MaxPool2d(2, 2, padding=1) 
        self.pool2 = nn.MaxPool2d(2, 2, padding=1) 
        self.pool3 = nn.MaxPool2d(2, 2, padding=1) 
        
        self.bn_fc1 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.5)
        
        # Dynamically sized projection layer - will be created on first forward pass
        self.patch_proj = None
        self._patch_proj_input_size = None
        
        # LSTM blocks with differentiation
        self.L = LowBlock()
        self.H = HighBlock()

        # FFNN
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(2048, 10)

    def forward(self, x):
        _bs = x.size(0)
        
        # CNN feature extraction
        out = F.elu(self.bn1(self.conv1(x)))
        out = F.elu(self.bn2(self.conv2(out)))
        out = self.pool1(out)
        out = self.dropout1(out)
        out = F.elu(self.bn3(self.conv3(out)))
        out = F.elu(self.bn4(self.conv4(out)))
        out = self.pool2(out)
        out = self.dropout2(out)
        out = F.elu(self.bn5(self.conv5(out)))
        out = F.elu(self.bn6(self.conv6(out)))
        
        # Reshape into patches for hierarchical processing
        out = out.view(_bs, -1)  # Flatten: (batch, total_features)
        total_features = out.size(1)
        features_per_patch = total_features // self.num_patches
        
        # Create projection layer on first forward pass
        if self.patch_proj is None:
            self._patch_proj_input_size = features_per_patch
            self.patch_proj = nn.Linear(features_per_patch, 512).to(x.device)
        
        # Split into patches (sequence)
        out = out.view(_bs, self.num_patches, features_per_patch)
        out = self.patch_proj(out)  # Project to LSTM input size
        
        h_state = None
        l_state = None
        
        # Hierarchical reasoning iterations
        for i in range(self.num_iterations):
            residual = out
            
            # Low-level processes local patches
            out_l, l_state = self.L(out, l_state)
            
            # High-level processes global context
            out_h, h_state = self.H(out_l, h_state)
            
            # Residual connection
            out = out_h + residual
        
        # Aggregate sequence information (mean pooling across patches)
        out = out.mean(dim=1)  # (batch, 512)
        
        out = self.dropout3(out)
        out = F.elu(self.bn_fc1(self.fc1(out)))
        out = self.fc2(out)
        return out   

model = HRM().to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
model = torch.compile(model)

# Metrics storage
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
epochs_recorded = []

# Training loop
for epoch in range(50):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    scheduler.step()
    
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

# Plot results
plt.figure(figsize=(15, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(epochs_recorded, train_losses, 'b-', marker='o', label='Train')
plt.plot(epochs_recorded, test_losses, 'r-', marker='s', label='Test')
plt.title('Train vs Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(epochs_recorded, train_accuracies, 'g-', marker='o', label='Train')
plt.plot(epochs_recorded, test_accuracies, 'r-', marker='s', label='Test')
plt.title('Train vs Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('hrm_training_results.png')
plt.show()