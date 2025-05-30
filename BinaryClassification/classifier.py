import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class Classifier(nn.Module):
    def __init__(self, combined_features = 708*14*14 + 29, hidden_dim=256):
        super(Classifier, self).__init__()
        concat_dim = Feature_dim + AU_dim # 138,768 + AU(29)

        # self.sigmoid = nn.Sigmoid() # BCEWithLogitsLoss에서 내부적으로 sigmoid실행(trainer)
        self.fc = nn.Sequential(
            nn.Linear(concat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.RELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        # for layer in self.fc:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        return self.fc(x).squeeze()