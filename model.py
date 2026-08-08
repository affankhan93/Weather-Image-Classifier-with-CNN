import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Define a CNN model 

class CNN(nn.Module):
    def __init__(self, num_classes=11):
        super(CNN, self).__init__()

        self.conv_block = nn.Sequential(
            # First Conv Layer
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),

            # Second Conv layer
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),

            # Thrid Conv layer
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
        )

        self.fc_block = nn.Sequential(
            # Flatten the output 
            nn.Flatten(),

            # Linear combination (input layer) 
            nn.Linear(in_features = 8 * 8 * 128, out_features = 256), # randomly taking out_feature 256 neurons
            nn.ReLU(),
            nn.Dropout(0.4), # It help us to prevent overfitting [dropout randomly turns off 40% of the neurons during training.] 

            # (Hidden layer)
            nn.Linear(in_features=256, out_features=350), # randomly taking out_feature 350 neurons 
            nn.ReLU(),
            nn.Dropout(0.3), # It help us to prevent overfitting [dropout randomly turns off 30% of the neurons during training.] 

            # (output layer) 
            nn.Linear(in_features = 350, out_features = num_classes)
        )


    def forward(self, x):    
        x = self.conv_block(x) 
        output = self.fc_block(x) 
        return output 