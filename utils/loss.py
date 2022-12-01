import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

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

# https://github.com/wonjun-dev/AI-Paper-Reproduce/blob/master/simCSE-Pytorch/pretrain.py
class SimCSELoss(nn.Module):
    def __init__(self, batch_size, temperature=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()    # negative pair를 indexing 하는 마스크입니다. (자기 자신(대각 성분)을 제외한 나머지) 

    def calc_sim_batch(self, a, b):
        reprs = torch.cat([a, b], dim=0)
        return F.cosine_similarity(reprs.unsqueeze(1), reprs.unsqueeze(0), dim=2)   # 두 representation의 cosine 유사도를 계산합니다.
    
    def calc_align(self, x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean().detach()  # 두 representation의 alignment를 계산하고 반환합니다.

    def calc_unif(self, x, t=2):
        sp_pdist = torch.pdist(x, p=2).pow(2)
        return sp_pdist.mul(-t).exp().mean().log().detach()  # 미니 배치 내의 represenation의 uniformity를 계산하고 반환합니다.
    
    def forward(self, proj_1, proj_2):
        batch_size = proj_1.shape[0]
        if batch_size != self.batch_size:
            mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float() # 에폭 안에서 마지막 미니 배치를 위해서 마스크를 새롭게 정의 합니다.
        else:
            mask = self.mask
            
        z_i = F.normalize(proj_1, p=2, dim=1)   # 모델의 [CLS] represenation을 l2 nomalize 합니다.
        z_j = F.normalize(proj_2, p=2, dim=1)

        sim_matrix = self.calc_sim_batch(z_i, z_j)  # 배치 단위로 두 representation의 cosine 유사도를 계산합니다.

        sim_ij = torch.diag(sim_matrix, batch_size) # sim_matrix에서 positive pair의 위치를 인덱싱 합니다. (대각 성분에서 배치 사이즈만큼 떨어져 있습니다.)
        sim_ji = torch.diag(sim_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = mask.to(sim_matrix.device) * torch.exp(sim_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))  # constrastive loss
        loss = torch.sum(all_losses) / (2 * batch_size) # 샘플 갯수로 나누어 평균 내줍니다.

        lalign = self.calc_align(z_i, z_j)
        lunif = (self.calc_unif(z_i[:batch_size//2]) + self.calc_unif(z_i[batch_size//2:])) / 2

        return loss, lalign, lunif