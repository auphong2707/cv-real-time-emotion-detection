import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7, in_channels=3, image_size=224):
        super(EmotionCNN, self).__init__()
        
        # Store image size
        self.image_size = image_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # image_size -> image_size/2
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # image_size/2 -> image_size/4
        self.dropout2 = nn.Dropout(0.25)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # image_size/4 -> image_size/8
        self.dropout3 = nn.Dropout(0.25)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # image_size/8 -> image_size/16
        self.dropout4 = nn.Dropout(0.25)
        
        # Calculate the size of the flattened layer
        self._fc_input_size = self._get_fc_input_size(image_size)
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self._fc_input_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def _get_fc_input_size(self, image_size):
        """Calculate the input size for the fully connected layer based on image_size."""
        # Simulate the spatial dimensions through the conv and pooling layers
        size = image_size
        size = size // 2  # After pool1
        size = size // 2  # After pool2
        size = size // 2  # After pool3
        size = size // 2  # After pool4
        # After conv5: 256 filters, size x size spatial dimensions
        return 256 * size * size
    
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
        x = self.fc2(x)
        return x

# Example usage
if __name__ == "__main__":
    # Test with different image sizes
    for img_size in [224, 112, 64]:
        model = EmotionCNN(num_classes=7, in_channels=3, image_size=img_size)
        x = torch.randn(1, 3, img_size, img_size)
        output = model(x)
        print(f"Image size: {img_size}x{img_size}, Output shape: {output.shape}")