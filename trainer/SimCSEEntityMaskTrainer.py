from tqdm import tqdm
import torch
import wandb
import numpy as np
import os
import time
import gc
import einops as ein
import torch.nn.functional as F
from torch import nn

import sys

class SimCSEEntityMaskTrainer():
    """
    Entity Mask에 관련된 Trainer입니다.
    기존 Baseline Trainer와 동일하지만 model에
    entity mask1, entity mask2를 넣어주는 것이 다른 점입니다.
    따라서, 해당 입력값이 들어가는 모델인지 확인해주시기 바랍니다.
    """
    def __init__(self, model, criterion, metric, optimizer, device, save_dir,
                 train_dataloader, valid_dataloader=None, lr_scheduler=None, epochs=1, tokenizer=None):
        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.tokenizer = tokenizer

    def train(self):
        """
        train_epoch를 돌고 valid_epoch로 평가합니다.
        """
        for epoch in range(self.epochs):
            standard_time = time.time()
            self._train_epoch(epoch)
            self._valid_epoch(epoch)
            wandb.log({'epoch' : epoch, 'runtime(Min)' : (time.time() - standard_time) / 60})
        torch.cuda.empty_cache()
        del self.model, self.train_dataloader, self.valid_dataloader
        gc.collect()
    
    def _train_epoch(self, epoch):
        gc.collect()
        self.model.train()
        epoch_loss = 0
        steps = 0
        pbar = tqdm(self.train_dataloader)
        for i, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            steps += 1
            
            # https://github.com/wonjun-dev/AI-Paper-Reproduce/blob/master/simCSE-Pytorch/pretrain.py
            logits1 = self.model(input_ids = batch["input_ids"].to(self.device),
                                attention_mask = batch["attention_mask"].to(self.device),
                                entity_mask1 = batch['entity_mask1'].to(self.device),
                                entity_mask2 = batch['entity_mask2'].to(self.device))
            logits2 = self.model(input_ids = batch["input_ids"].to(self.device),
                                attention_mask = batch["attention_mask"].to(self.device),
                                entity_mask1 = batch['entity_mask1'].to(self.device),
                                entity_mask2 = batch['entity_mask2'].to(self.device))
            loss, lalign, lunif = self.criterion(logits1, logits2)
            
            loss.backward()
            epoch_loss += loss.detach().cpu().numpy().item()
            
            self.optimizer.step()
            
            pbar.set_postfix({
                'simCSE loss(unsup.)' : epoch_loss / steps,
                'measure_align' : lalign,
                'measure_unif' : lunif,
                'lr' : self.optimizer.param_groups[0]['lr'],
            })
            wandb.log({'train_loss':epoch_loss/steps})
        pbar.close()

    def _valid_epoch(self, epoch):
        val_loss = 0
        val_steps = 0
        val_loss_values=[sys.maxsize]
        with torch.no_grad():
            self.model.eval()
            for valid_batch in tqdm(self.valid_dataloader):
                val_steps += 1
                
                # https://github.com/wonjun-dev/AI-Paper-Reproduce/blob/master/simCSE-Pytorch/pretrain.py
                logits1 = self.model(input_ids =valid_batch["input_ids"].to(self.device),
                                    attention_mask =valid_batch["attention_mask"].to(self.device),
                                    entity_mask1 = valid_batch['entity_mask1'].to(self.device),
                                    entity_mask2 = valid_batch['entity_mask2'].to(self.device))
                logits2 = self.model(input_ids =valid_batch["input_ids"].to(self.device),
                                    attention_mask =valid_batch["attention_mask"].to(self.device),
                                    entity_mask1 = valid_batch['entity_mask1'].to(self.device),
                                    entity_mask2 = valid_batch['entity_mask2'].to(self.device))
                
                loss, lalign, lunif = self.criterion(logits1, logits2)
                val_loss += loss.detach().cpu().numpy().item()
                

            val_loss /= val_steps
            wandb.log({'epoch' : epoch,'val_loss':val_loss})
            print(f"Epoch [{epoch+1}/{self.epochs}] Val_loss : {val_loss}")

            if min(val_loss_values) >= val_loss:
                print('save checkpoint!')
                if not os.path.exists(f'save/{self.save_dir}'):
                    os.makedirs(f'save/{self.save_dir}')
                torch.save(self.model.state_dict(), f'save/{self.save_dir}/epoch:{epoch}_model.pt')
                val_loss_values.append(val_loss)