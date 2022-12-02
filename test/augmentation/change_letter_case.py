import pandas as pd
import re
from ast import literal_eval

def get_alpha_sentence(df:pd.DataFrame):
    """
    정규표현식으로 알파벳이 있는지 확인하고,
    알파벳이 포함된 문장을 반환합니다.
    """
    alpha = re.compile(r'[a-zA-Z]')
    alpha_sentence = []
    
    for _, i in df.iterrows():
        if alpha.match(i['sentence']):
            alpha_sentence.append(i)
    
    return alpha_sentence

def change_case(data, case):
    changed = []
    for i in data:
        sub_entity = literal_eval(i['subject_entity'])
        obj_entity = literal_eval(i['object_entity'])

        if eval(f"i['sentence'].{case}()") != i['sentence']: # 영어를 대/소문자로 변환했을 때 변화가 있으면
            j = i.copy()
            j['sentence'] = eval(f"i['sentence'].{case}()")
            
            sub_entity['word'] = eval(f"sub_entity['word'].{case}()")
            j['subject_entity'] = str(sub_entity)
            
            obj_entity['word'] = eval(f"obj_entity['word'].{case}()") # 대/소문자로 문장, subject, object를 변경해주고
            j['object_entity'] = str(obj_entity) # 원래 형식처럼 스트링 형식으로 저장해준다.

            changed.append(j) # 처리된 문장을 추가
        
    print(f"changed len : {len(changed)}")
    df = pd.DataFrame(changed)

    return df

def change_letter_case(data:str, case:str, mode:str):
    """
    입력된 csv파일 중, 알파벳이 포함된 문장을 추출하고 대소문자로 변환 후 반환합니다.
    data에는 csv파일 경로가, case는 대/소문자에 따라 upper/lower,
    mode는 변경된 부분만 csv파일로 저장하려면 changed, 기존 데이터에 추가하고 새로운 파일로 생성하려면 append를 입력해줍니다.
    """
    df = pd.read_csv(data, index_col = 0)
    alpha_sentence = get_alpha_sentence(df)

    assert case in ["upper", "lower"], \
        f"대소문자 설정이 잘못되었습니다. 대문자는 upper, 소문자는 lower입니다. 현재 입력 : {case}"
    changed = change_case(alpha_sentence, case)

    assert mode in ["changed", "append"], \
        f"모드 설정이 잘못되었습니다. 변경된 부분만 출력하려면 changed,\n\
        입력된 csv 뒤에 새로운 데이터를 추가한 파일을 새로 생성하려면 append입니다.\n\
        현재 입력 : {mode}"
    
    if mode == "changed":
        changed = changed.reset_index(drop = True)
        changed.index.name = 'id'
        changed.to_csv(f"changed_to_{case}.csv")
    else:
        changed = df.append(changed)
        changed.reset_index(drop = True, inplace = True)
        changed.index.name = 'id'
        changed.to_csv(f"appended_with_{case}.csv")