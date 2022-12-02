import pandas as pd
from ast import literal_eval

def swap_entity(data):
    # 바꿔도 라벨의 변화가 없을 것으로 보이는 것들
    change_entitiy = set(["per:alternate_names", "per:siblings", "per:spouse", "per:colleagues", "per:other_family", "org:alternate_names"])

    df = pd.read_csv(data)
    concat_data = []

    for idx, (_, i) in enumerate(df.iterrows()):
        if i['label'] in change_entitiy:
            sub = literal_eval(i[2])
            obj = literal_eval(i[3])

            if sub['start_idx'] < obj['start_idx']:
                sentence = i[1][:sub['start_idx']] + obj['word'] + i[1][sub['end_idx']+1:obj['start_idx']] + sub['word'] + i[1][obj['end_idx']+1:]

            else:
                sentence = i[1][:obj['start_idx']] + sub['word'] + i[1][obj['end_idx']+1:sub['start_idx']] + obj['word'] + i[1][sub['end_idx']+1:]

            concat_data.append({
                "id" : idx,
                "sentence" : sentence,
                "subject_entity" : i[3],
                "object_entity" : i[2],
                "label" : i[4],
                #   "original" : i[1]
            })

    new_data = pd.concat([df, pd.DataFrame(concat_data)], ignore_index = True)
    new_data = new_data.drop('id', axis=1)
    new_data.index.name = 'id'
    new_data.to_csv("subj-obj_change_concat.csv")
    # pd.DataFrame(concat_data).to_csv("subj-obj_change.csv")