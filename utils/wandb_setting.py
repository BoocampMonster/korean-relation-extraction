from typing import Callable
import wandb
from omegaconf import OmegaConf, DictConfig

def wandb_setting(
    entity:str,
    project:str,
    group_name:str,
    experiment_name:str,
    arg_config:str):
    """_summary_
        완디비와 오메가컨프를 깔끔하게 정리한 함수입니다.
        1. 오메가컨프를 통해 configs 디렉토리에 있는
           원하는 config가 담긴 yaml 파일을 불러옵니다(json도 됩니다).
        2. wandb를 초기화시켜줍니다.
        3. wandb의 config를 OmegaConf로 불러온 config로 설정해줍니다.
        4. wandb의 config를 반환합니다.
    Args:
        entity : 팀원들과 함께 있는 그룹 이름
        project : 그룹 내에 존재하는 원하는 프로젝트 이름
        group_name (_type_): 프로젝트 내에 하부 그룹 이름
        experiment_name (_type_): 내 실험 이름
        arg_config (_type_): config 이름
    """
    wandb.login()
    if arg_config:
        config = OmegaConf.load(f'./configs/{arg_config}.yaml')
        print(config)
        assert type(config) == DictConfig
    
    wandb.init(project=project, group=group_name, name=experiment_name, entity=entity)
    wandb.config = config
        
    return wandb.config