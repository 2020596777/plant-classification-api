import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEmbedding(nn.Module):
    def __init__(self, input_channels=3, feature_dim=64, color_feat_dim=96, texture_feat_dim=10):
        super().__init__()

        # 3-layer CNN with batch normalization and max pooling
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, feature_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_dim)
        self.pool = nn.MaxPool2d(2, 2)  # reduce spatial size by factor of 2

        # After 3 max pools, input size 224x224 -> output 28x28 feature maps
        self.flattened_size = feature_dim * 28 * 28

        # Fully connected layer to compress CNN features into 256-dim vector
        self.fc_img = nn.Linear(self.flattened_size, 256)

        # Fully connected layers to process handcrafted features separately
        self.fc_color = nn.Linear(color_feat_dim, 64)
        self.fc_texture = nn.Linear(texture_feat_dim, 32)

        # Final fully connected layer to combine all features into 128-dim embedding
        self.fc_final = nn.Linear(256 + 64 + 32, 128)

    def forward(self, x_img, x_color, x_texture):
        # Forward pass for image through CNN layers + ReLU activations + pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x_img))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten CNN feature maps to 1D vector
        x = x.view(x.size(0), -1)

        # Pass image features through fully connected layer with ReLU
        x_img_emb = F.relu(self.fc_img(x))

        # Pass handcrafted color histogram features through FC + ReLU
        x_color_emb = F.relu(self.fc_color(x_color))

        # Pass handcrafted texture features through FC + ReLU
        x_texture_emb = F.relu(self.fc_texture(x_texture))

        # Concatenate all feature vectors into one
        x_all = torch.cat([x_img_emb, x_color_emb, x_texture_emb], dim=1)

        # Final embedding vector of size 128
        x_all = self.fc_final(x_all)

        return x_all