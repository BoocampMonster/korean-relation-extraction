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

def ensemble(path_to_csv_folder = "ensemble", output = "output.csv"):
    csv_names = []
    csv_dfs = []
    probs = []
    preds = []
    csv_probs = []

    assert os.path.exists(path_to_csv_folder) != False, \
        f"csv파일이 있는 폴더가 존재하지 않습니다. 참조한 폴더 위치 : {os.getcwd() + '/' + path_to_csv_folder}"

    for i in os.listdir(f"{path_to_csv_folder}"): # csv_dfs에 dataframe을 load
        if i.endswith(".csv"):
            csv_names.append(i.strip(".csv"))
            csv_dfs.append(pd.read_csv(path_to_csv_folder+"/"+i))
    
    assert len(csv_names) != 0, \
        "csv 파일이 존재하지 않습니다."

    for i in range(len(csv_dfs)):
        assert len(csv_dfs[i].columns) == 3, \
        f"csv 파일 중 submission_output이 아닌 파일이 존재합니다.\n파일명 : {csv_names[i]}, column 개수 : {len(csv_dfs[i].columns)}"
        csv_probs.append(csv_dfs[i]['probs'])
    
    new_df = pd.DataFrame(csv_dfs[0]['id'])
    
    for i in zip(*csv_probs):
        temp_prob = list(map(lambda x:np.array(x, dtype=float), list(map(literal_eval,i))))
        new_prob = np.mean(temp_prob, axis=0)
        probs.append(str(new_prob.tolist()))
        preds.append(np.argmax(new_prob, axis=-1))
    
    new_df['pred_label'] = preds
    new_df['probs'] = probs
    new_df['pred_label'] = new_df['pred_label'].apply(lambda x: label[x])
    
    yn = ""
    if os.path.exists(output):
        while yn == "":
            yn = input(f"{output}파일이 존재합니다. 덮어쓰기[Y/N] : ")
        if yn in ["Y", "y"]:
            new_df.to_csv(output)
        elif yn in ["N", "n"]:
            pass
    else:
        new_df.to_csv(output)

    return new_df

def diff(ensembled, best_model):
    """
    현재 폴더의 (ensembled, best_model) 두 파일명을 받아 변경된 라벨이 있는 경우 이를 출력합니다.
    결과값이 존재하는 경우, csv로 저장하고, 존재하지 않는 경우 None을 프린트, False를 반환합니다.
    """
    assert os.path.exists(ensembled) != False, \
        f"앙상블된 csv파일이 존재하지 않습니다. 입력된 파일명 : {ensembled}"
    assert os.path.exists(best_model) != False, \
        f"비교할 csv파일이 존재하지 않습니다. 입력된 파일명 : {best_model}"
    
    if ensembled == best_model:
        print("입력된 두 파일명이 같습니다.")
        return -1
    
    target_A = pd.read_csv(ensembled)['pred_label']
    target_B = pd.read_csv(best_model)['pred_label']
    
    id = []
    ensembled = []
    best_model = []
    for idx, (x, y) in enumerate(zip(target_A, target_B)):
        if x != y:
            id.append(idx)
            ensembled.append(x)
            best_model.append(y)

    difference = pd.DataFrame({"id":id, "ensembled" : ensembled, "best_model" : best_model})
    print(f"difference : {len(difference)}")
    if len(difference) == 0:
        print("차이가 존재하지 않습니다.")
        return -1
    else:
        yn = ""
        if os.path.exists("difference.csv"):
            while yn == "":
                yn = input("difference.csv 파일이 존재합니다. 덮어쓰기[Y/N] : ")
            if yn in ["Y", "y"]:
                difference.to_csv("difference.csv")
            elif yn in ["N", "n"]:
                pass
        else:
            difference.to_csv("difference.csv")

#diff("output.csv", "roberta_submission.csv")    

if __name__ == "__main__":
    """
    <ensemble>
    python ensemble.py [앙상블할 csv 파일이 들어있는 폴더 경로] [저장할 output 파일명]
    파라미터 미지정시 기본적으로 ensemble폴더를 참조하며, 저장되는 파일명은 output.csv
    <diff>
    python ensemble.py diff [앙상블된 csv파일 경로] [베스트 모델 csv파일 경로]
    앙상블된 파일과 한 csv파일을 비교해서 라벨이 다른경우만 출력해줍니다.
    차이가 있는 경우, difference.csv에 저장되고, 없는 경우 None을 출력합니다.
    """
    if len(sys.argv) == 1:
        ensemble()
    elif len(sys.argv) > 1:
        if sys.argv[1] == "diff":
            if len(sys.argv) == 4:
                diff(sys.argv[2], sys.argv[3])
            else:
                print("파라미터 수가 적거나 많습니다.")
        else:
            if len(sys.argv) <= 3:
                ensemble(*sys.argv[1:])
            else:
                print("파라미터 수가 너무 많습니다.")