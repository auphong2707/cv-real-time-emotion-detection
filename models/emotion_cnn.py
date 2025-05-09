import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block to enhance channel-wise features."""
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7, in_channels=3, image_size=224):
        super(EmotionCNN, self).__init__()
        
        self.image_size = image_size
        
        # Convolutional layers with SE blocks
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.se1 = SEBlock(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.se2 = SEBlock(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # image_size -> image_size/2
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.se3 = SEBlock(128)
        # Residual connection
        self.conv3_residual = nn.Conv2d(64, 128, kernel_size=1)  # Match channels for residual
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # image_size/2 -> image_size/4
        self.dropout2 = nn.Dropout(0.25)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.se4 = SEBlock(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # image_size/4 -> image_size/8
        self.dropout3 = nn.Dropout(0.25)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.se5 = SEBlock(256)
        # Residual connection
        self.conv5_residual = nn.Conv2d(128, 256, kernel_size=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # image_size/8 -> image_size/16
        self.dropout4 = nn.Dropout(0.25)
        
        # Calculate FC input size
        self._fc_input_size = self._get_fc_input_size(image_size)
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self._fc_input_size, 256)  # Reduced from 512
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
    def _get_fc_input_size(self, image_size):
        size = image_size // 16  # After 4 pooling layers
        return 256 * size * size
    
    def forward(self, x):
        # Conv Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.se1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se2(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Conv Block 2 with residual
        identity = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.se3(x)
        identity = self.conv3_residual(identity)  # Match channels
        x = x + identity  # Residual connection
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Conv Block 3
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.se4(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Conv Block 4 with residual
        identity = x
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.se5(x)
        identity = self.conv5_residual(identity)
        x = x + identity
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