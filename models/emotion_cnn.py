import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7, in_channels=3):
        super(EmotionCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 224 -> 112
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 112 -> 56
        self.dropout2 = nn.Dropout(0.25)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 56 -> 28
        self.dropout3 = nn.Dropout(0.25)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28 -> 14
        self.dropout4 = nn.Dropout(0.25)
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Conv Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Conv Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Conv Block 3
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Conv Block 4
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool4(x)
        x = self.dropout4(x)
        
        # Fully connected layers
        x = self.flatten(x)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout5(x)
        x = self.fc2(x)  # No softmax; handled by loss function
        return x

# Example usage
if __name__ == "__main__":
    model = EmotionCNN(num_classes=7, in_channels=3)
    x = torch.randn(1, 3, 224, 224)  # Batch, Channels, Height, Width
    output = model(x)
    print(f"Output shape: {output.shape}")  # Expected: [1, 7]