import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3
from cbam import CBAM

class EfficientNetCBAM(nn.Module):
    """
    Proposed Architecture for COVID-19 Detection (Assignment 3).
    Combines EfficientNet-B3 for robust feature extraction with
    a CBAM (Convolutional Block Attention Module) to dynamically focus 
    on critical radiological features (e.g., Ground-Glass Opacities).
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientNetCBAM, self).__init__()
        
        # 1. Load EfficientNet-B3 Backbone
        # Using pretrained=pretrained for better compatibility with older torchvision versions
        self.backbone = efficientnet_b3(pretrained=pretrained)
            
        # Extract the feature extraction layers
        self.features = self.backbone.features
        
        # EfficientNet-B3 outputs 1536 channels from its feature extractor
        in_features = 1536
        
        # 2. Add CBAM Module
        self.cbam = CBAM(in_planes=in_features)
        
        # 3. Global Average Pooling
        # Explicitly implementing this per the architectural diagram requirements
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 4. Dense Layer
        # We add dropout to prevent overfitting before the final linear layer
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
        # Note: Softmax is structurally the final layer for probability prediction,
        # but in PyTorch, nn.CrossEntropyLoss() internally applies LogSoftmax.
        # Therefore, we output raw logits here and apply Softmax explicitly 
        # during the prediction/testing phase for the Sugeno Fuzzy Integral.

    def forward(self, x):
        # Step 1: Feature Extraction (EfficientNet-B3)
        x = self.features(x)
        
        # Step 2: Attention Mechanism (CBAM)
        x = self.cbam(x)
        
        # Step 3: Global Average Pooling
        x = self.global_avg_pool(x)
        
        # Flatten the tensor before the dense layer
        x = torch.flatten(x, 1)
        
        # Step 4: Dense Layer
        logits = self.classifier(x)
        
        return logits

# For testing the architecture
if __name__ == "__main__":
    model = EfficientNetCBAM(num_classes=2, pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224) # Batch size of 2
    output = model(dummy_input)
    print(f"Model instantiated successfully.")
    print(f"Output shape (Batch Size, Num Classes): {output.shape}")
