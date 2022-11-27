import torch
import numpy as np

def CEloss(outputs, labels):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    return criterion(outputs, labels)