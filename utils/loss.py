import torch
import numpy as np

def CEloss(outputs, labels):
    criterion = torch.nn.CrossEntropyLoss()
    
    return criterion(outputs, labels)