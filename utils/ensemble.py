import pandas as pd
import numpy as np
from ast import literal_eval
import os
import sys

label = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']

def ensemble(path_to_csv_folder = "./ensemble", output = "output.csv"):
    new_df = pd.DataFrame()
    csv_names = []
    csv_dfs = []
    probs = []
    preds = []
    csv_probs = []

    for i in os.listdir(f"{path_to_csv_folder}"): # csv_dfs에 dataframe을 load
        if i.endswith(".csv"):
            csv_names.append(i.strip(".csv"))
            csv_dfs.append(pd.read_csv(path_to_csv_folder+"/"+i))
    
    for i in range(len(csv_dfs)):
        assert len(csv_dfs[i].columns) == 3, \
        f"csv 파일 중 submission_output이 아닌 파일이 존재합니다.\n파일명 : {csv_names[i]}, column 개수 : {len(csv_dfs[i].columns)}"
        csv_probs.append(csv_dfs[i]['probs'])
    
    for i in zip(*csv_probs):
        temp_prob = list(map(lambda x:np.array(x, dtype=float), list(map(literal_eval,i))))
        new_prob = np.mean(temp_prob, axis=0)
        probs.append(str(new_prob.tolist()))
        preds.append(np.argmax(new_prob, axis=-1))

    new_df['pred_label'] = preds
    new_df['probs'] = probs
    new_df['pred_label'] = new_df['pred_label'].apply(lambda x: label[x])
    
    new_df.to_csv(output)

    return new_df

if __name__ == "__main__":
    """
    python ensemble.py [앙상블할 csv 파일이 들어있는 폴더 경로] [저장할 output 파일명]
    파라미터 미지정시 기본적으로 ensemble폴더를 참조하며, 저장되는 파일명은 output.csv
    """
    if len(sys.argv) == 3:
        ensemble(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        ensemble(sys.argv[1])
    else:
        ensemble()