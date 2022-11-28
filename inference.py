import torch
import argparse
from tqdm import tqdm
import einops as ein
import numpy as np
import pandas as pd
import pickle
from omegaconf import OmegaConf
import random

import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import dataloader as DataModule
from transformers import AutoTokenizer
import model as Model
from utils.wandb_setting import wandb_setting
from utils.seed_setting import seed_setting

def main(config):
    seed_setting(config.train.seed)
    # 데이터셋 로드 클래스를 불러옵니다.
    
    assert torch.cuda.is_available(), "GPU를 사용할 수 없습니다."
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

    data = pd.read_csv(config.data.test_path)
    with open('data/dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
        
    test = getattr(DataModule, config.model.data_class)(
        mode = "test",
        data=data,
        tokenizer=tokenizer,
        entity_marker_mode= config.data.get('entity_marker_mode'),
        max_length=config.train.max_length)  
    test_dataloader = DataLoader(test, batch_size=16, pin_memory=True, shuffle=False)
    
    # 모델 아키텍처를 불러옵니다.
    model = getattr(Model, config.model.model_class)(
        model_name = config.model.model_name,
        num_labels=30,
        dropout_rate = config.model.dropout_rate,
        add_token_num = config.data.get('entity_marker_num')
        ).to(device)
    checkpoint = torch.load(f'./save/{config.model.saved_dir}/epoch:3_model.pt')
    model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            output = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            # output = ein.rearrange(output, 'batch 1 class -> batch class')

            probs = F.softmax(output, dim=-1).detach().cpu().numpy()
            preds = np.argmax(probs, axis=-1)
            
            all_preds.append(preds)
            all_probs.append(probs.tolist())
        predictions = np.concatenate(all_preds).tolist()
        probablity = np.concatenate(all_probs, axis=0).tolist()
    ## make csv file with predicted answer
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    output_df = pd.DataFrame(data['id'])
    output_df['pred_label'] = predictions
    output_df['probs'] = probablity
    output_df['pred_label'] = output_df['pred_label'].apply(lambda x: dict_num_to_label[x])

    output_df.to_csv(f'save/{config.model.saved_dir}/submission.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    print('추론 완료')

if __name__=='__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='baseline_typed_entity_marker_cls')
    args, _ = parser.parse_known_args()
    
    config_w = OmegaConf.load(f'./configs/{args.config}.yaml')
    print(f'사용할 수 있는 GPU는 {torch.cuda.device_count()}개 입니다.')
    
    main(config_w)
