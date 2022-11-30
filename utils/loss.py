import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def WeightedCEloss(outputs, labels):
    count = [9534, 4284, 420, 380, 2103, 1320, 3573, 1195, 139, 48, 304, 193, 1001, 190, 534, 1234, 136, 795, 450, 98, 1866, 520, 66, 82, 418, 1130, 166, 40, 155, 96]
    normedWeights = [1 - (x / sum(count)) for x in count]
    normedWeights = torch.FloatTensor(normedWeights).to(device)
    criterion = torch.nn.CrossEntropyLoss(normedWeights)
    return criterion(outputs, labels)

def CEloss(outputs, labels):
    criterion = torch.nn.CrossEntropyLoss()
    
    return criterion(outputs, labels)

def Focalloss(outputs, labels):
    focal_loss = torch.hub.load(
    'adeelh/pytorch-multi-class-focal-loss',
    model='FocalLoss',
    alpha=torch.tensor([.75, .25]),
    gamma=2,
    reduction='mean',
    force_reload=False
    )
    
    return focal_loss(outputs, labels)