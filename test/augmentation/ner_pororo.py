from pororo import Pororo
import pandas as pd
from ast import literal_eval

def get_ner(data:str):
    """
    csv파일을 받아 ner을 수행합니다.
    문장 번호와 ner 결과를 담은 데이터프레임을 반환합니다.
    """
    data = pd.read_csv(data)
    ner = Pororo(task="ner", lang="ko")
    detected_ner = []
    error_list = []
    for _, i in data.iterrows():
        try:
            # apply_wsd를 사용하면 좀 더 자세한 태깅이 가능하다고 하는데, 적용시 에러가 발생해서 False로 설정함.
            # ignore_labels 또한 ignore_labels = ["O"] 로 O태그를 무시할 수 있으나, 몇개가 걸러지지 않는 문제가 있어 if문으로 필터링함.
            detected_ner.append((i["id"],[i for i in ner(i['sentence'], apply_wsd=False) if i[1] != 'O']))
        except KeyError as e: # 특수기호(EMOJI)로 인해 에러 발생으로 예외처리
            detected_ner.append((i["id"], i['sentence']))
            print(f"Error occured at {i['id']} sentence.")
            print(f"Error : {e}")
            error_list.append(i['id'])
            # 에러가 걸러지지 않는 경우, 일단 문장 그대로 추가하고 에러 리스트에 문장번호를 저장함.
            # 에러 개수가 많아지는 경우, 에러 처리를 위한 코드 작성 필요
    
    str_ner = [str(i) for i in detected_ner]
    ner_df = pd.DataFrame(str_ner)
    
    return ner_df

def add_ner(data:str, ner_csv:pd.DataFrame = None):
    df = pd.read_csv(data)
    if ner_csv: # ner 데이터가 데이터프레임으로 주어지면 그대로 사용하고
        ner_df = ner_csv
    else: # 없는 경우, 원본 csv를 통해 ner을 진행한다.
        ner_df = get_ner(df)

    subject = df['subject_entity']
    object = df['object_entity']
    ner_data = ner_df['ner']

    sub_types = []
    obj_types = []

    for i in range(len(df)):
        sub = literal_eval(subject[i])
        obj = literal_eval(object[i])
        ner = literal_eval(ner_data[i])
        sub_flag = False
        obj_flag = False

        for entity in ner:
            # Pororo의 개체명 인식이 데이터와 완전히 일치하지 않는 경우가 있어 pororo가 인식한 개체명 리스트 안에 subject 단어가 존재하는지 in으로 확인해줌.
            # 문장 내 entity가 두번 등장하는 경우가 있으므로 flag를 통해 한번만 추가해줌.
            if sub['word'] in entity[0] and not sub_flag:
                sub_types.append(entity[1])
                sub_flag = True
            elif obj['word'] in entity[0] and not obj_flag:
                obj_types.append(entity[1])
                obj_flag = True

        if not sub_flag: # 개체를 발견하지 못한 경우, 원본 타입으로 지정해줌.
            sub_types.append(sub['type'])
        if not obj_flag:
            obj_types.append(obj['type'])

    # 새로 태깅한 타입들을 새로운 열에 추가함.
    df['subject_type'] = sub_types
    df['object_type'] = obj_types

    df.to_csv("ner_appended.csv")