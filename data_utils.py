import dataclasses
import json
import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


@dataclass
class OntologyDSTFeature:
    guid: str
    input_ids: List[int]
    segment_ids: List[int]
    num_turn: int
    target_ids: Optional[List[int]]


@dataclass
class OpenVocabDSTFeature:
    guid: str
    input_id: List[int]
    segment_id: List[int]
    gating_id: List[int]
    target_ids: Optional[Union[List[int], List[List[int]]]]
    slot_positions: Union[List[int]] = None ## 원래 Union없음
    domain_id: int = None


@dataclass
class DSTInputExample:
    guid: str
    context_turns: List[str]
    current_turn: List[str]
    label: Optional[List[str]] = None
    domains: List[str] = None

    def to_dict(self):
        return dataclasses.asdict(self)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2) + "\n"


class WOSDataset(Dataset):
    def __init__(self, features):
        self.features = features
        self.length = len(self.features)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.features[idx]


class DSTPreprocessor:#preprocessor(전처리기), 다음번에알려줄수 있도록
    def __init__(self, slot_meta, src_tokenizer, trg_tokenizer=None, ontology=None):
        self.slot_meta = slot_meta # slot_meta < data-slot_meta.json > \uad00\uad11-\uacbd\uce58 \uc88b\uc740(유니코드) = 관광-경치 좋은 등...
        self.src_tokenizer = src_tokenizer 
        self.trg_tokenizer = trg_tokenizer if trg_tokenizer else src_tokenizer  
        self.ontology = ontology # ontology < data - ontology.json > {"관광-경치 좋은": ["none","dontcare","yes","no"] 등으로 선언되어있다.

    def pad_ids(self, arrays, pad_idx, max_length=-1):
        if max_length < 0:
            max_length = max(list(map(len, arrays)))

        arrays = [
            array + [pad_idx] * (max_length - min(len(array), 512)) for array in arrays
        ]
        return arrays

    def pad_id_of_matrix(self, arrays, padding, max_length=-1, left=False):
        if max_length < 0:
            max_length = max([array.size(-1) for array in arrays])

        new_arrays = []
        for i, array in enumerate(arrays):
            n, l = array.size()
            pad = torch.zeros(n, (max_length - l))
            pad[
                :,
                :,
            ] = padding
            pad = pad.long()
            m = torch.cat([array, pad], -1)
            new_arrays.append(m.unsqueeze(0))

        return torch.cat(new_arrays, 0)

    def _convert_example_to_feature(self): #일부러 에러 발생시켜 예외처리 시키는 부분
        raise NotImplementedError

    def convert_examples_to_features(self): #일부러 에러 발생시켜 예외처리 시키는 부분
        raise NotImplementedError

    def recover_state(self):                #일부러 에러 발생시켜 예외처리 시키는 부분
        raise NotImplementedError

# 데이터 불러오는 부분
def load_dataset(dataset_path, dev_split=0.1): # 전체 training data 10%를 training에 쓰겠다. 90%,   학습 정확성 계산할떄씀(accuraycy) 10%
    #데이터를 연다.
    data = json.load(open(dataset_path))
    #데이터의 글자수 체킹
    num_data = len(data)
    #num_dev 에다가 data글자수 *0.1 이후 int 저장
    num_dev = int(num_data * dev_split) # 정확성(ac)
    #print("시작num_dev=int(num_data * dev_split)")
    #print(num_dev) #800이 출력됨

    #데이터셋이 없을 때 리턴하는 문구
    if not num_dev:
        return data, []  # no dev dataset
    #name_list = [('kim','sungsu'), ('kang','hodong'), ('park','jisung'), ('kim','yuna'), ('park','chanho')]와 같은 형식의 리스트가 있을때
    #defaultdict(<class 'list'>, {'kim': ['sungsu', 'yuna'], 'kang': ['hodong', 'hodong'], 'park': ['jisung', 'chanho']}) 형식으로 키에따른 vaule 값을 모아주는 것
    dom_mapper = defaultdict(list)
    # d에서 data까지
    for d in data:
        #domain의 길이에다가. guid 추가시켜 준다.
        dom_mapper[len(d["domains"])].append(d["guid"]) #defaultdict(<class 'list'>, {2: ['wos-v1_train_00000'], 1: ['wos-v1_train_00001'], 3: ['wos-v1_train_00002']})
        #print("시작dom_mapper.append")                 #도메인 갯수에 따라 guid의 value값이 들어간다.
        #print(dom_mapper.values())
    #글자 총길이수 num_dev/3 을한다 -- 이유불명
    num_per_domain_trainsition = int(num_dev / 3)
    #print("시작num_per_domain_trainsition = int(num_dev / 3) ")
    #print(num_per_domain_trainsition) #266저장됨
    dev_idx = []
    for v in dom_mapper.values(): #dom_mapper.values()의 값은 dict_values([['wos-v1_train_00000'], ['wos-v1_train_00001'], ['wos-v1_train_00002']]) 등등으로 구성됨
        #for 문안에서 if 문을 돌리는데  num_per_domain_trainsition== 글자총길이수/3 dml 값보다 커질떄까지 게속한다.
        if len(v) < num_per_domain_trainsition:
            continue
        #랜덤으로 나온 v의 값을 계쏙해서 idx 에 저장한다.    
        idx = random.sample(v, num_per_domain_trainsition) #['wos-v1_train_05983', 'wos-v1_train_00984', 'wos-v1_train_00222'] 이런식으로 랜덤으로 값이 뽑힘
        #그 값을 dev_idx 에 저장한다.
        dev_idx.extend(idx)

    train_data, dev_data = [], []

    #dev data와 train data 불리하기 위한 구문
    for d in data:
        if d["guid"] in dev_idx:
            dev_data.append(d)  #10%
        else:
            train_data.append(d)    #90%

    dev_labels = {}
    for dialogue in dev_data:
        d_idx = 0
        guid = dialogue["guid"]
        #enumerate는 반복문 사용 시 몇 번째 반복문인지 확인이 필요할 수 있습니다. 
        #인덱스 번호와 컬렉션의 원소를 tuple형태로 반환합니다.
        #role = user라면 if문 중단. 
        #state =turn.pop()은  리스트의 맨 마지막 요소를 돌려주고 그 요소는 삭제한다.
        #이 구문은 user의 대화가 아니라면 state를 받아들인다.
        #그 이후 라벨링 작업을 한다.
        for idx, turn in enumerate(dialogue["dialogue"]): #idx dialogue의 순서, dailogue 자체
            if turn["role"] != "user":
                continue

            state = turn.pop("state") # dialogue(여기선 turn)의 state를 state에 넣고 state 삭제된다.
            guid_t = f"{guid}-{d_idx}" # wos-v1_train_07748-0 이런식으로 guid에다가 d_idx 순서가 붙어버림
            d_idx += 1

            dev_labels[guid_t] = state # 결과적으로 라벨에 따른 ['관광-종류-랜드마크', '관광-역사적-yes'] 이런 state 값이 저장되있는게 dev_labels

    return train_data, dev_data, dev_labels


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def split_slot(dom_slot_value, get_domain_slot=False):
    #dom_slot_value = 숙소-지역-dontcare 이런식임
    try:
        dom, slot, value = dom_slot_value.split("-") #-를 다짤라서 dom = domain slot = slot value = value  넣는다. 
 
    except ValueError:
        tempo = dom_slot_value.split("-")
        if len(tempo) < 2:
            return dom_slot_value, dom_slot_value, dom_slot_value
        dom, slot = tempo[0], tempo[1]
        value = dom_slot_value.replace(f"{dom}-{slot}-", "").strip()
        #value 값이 없을떄를 체크하기 위한 문  value에다가 domain과 slot이란 f"{dom}-{slot}-"이 들어가고 공백 사라짐
    if get_domain_slot:
        return f"{dom}-{slot}", value
    return dom, slot, value


def build_slot_meta(data):
    slot_meta = []
    for dialog in data:
        for turn in dialog["dialogue"]:
            if not turn.get("state"):
                continue

            for dom_slot_value in turn["state"]:
                domain_slot, _ = split_slot(dom_slot_value, get_domain_slot=True)
                if domain_slot not in slot_meta:
                    slot_meta.append(domain_slot)
    return sorted(slot_meta)


def convert_state_dict(state): # label=state,
    """
    :param state: list
    :return: dict
    dic[s] = v : s = domain-slot(관광-종류) , v = value(박물관)
    """
    dic = {}
    for slot in state:
        s, v = split_slot(slot, get_domain_slot=True)   # s = domain-slot(관광-종류) , v = value(박물관)
        dic[s] = v  # dic[관광-종류] = 박물관
    return dic


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

############################################
def get_examples_from_dialogue(dialogue, user_first=False): 
    guid = dialogue["guid"] # guid 즉 이름 불러옴 "guid": "wos-v1_train_02935" 요런거
    examples = [] 
    history = []
    d_idx = 0
    domains = dialogue["domains"]#"domains": [ "관광","택시"] 요런것들이 저장이 된다.
    #제일 처음 집어내는 것은 guid와 domains를 체킹해서 각각 집어넣는다 - 이는 데이터자체의 순서가 guid>domains>dialogue-role-text-state>role-text>role-text-state>role-text 의 흐름으로 구성되어있음
    for idx, turn in enumerate(dialogue["dialogue"]): #dialogue 부분만 집어내서 for문을 돌리는데 idx - 인덱스번호 이는 하나의 대화셋에서 대화만큼 늘어날것
        if turn["role"] != "user":  #user의 질문이 아니라면 if문 넘김                                
            continue
        
        if idx:     #idx가(컴퓨터의 답)sys_utter에다가 dilaogue안의 idx의 text가 sys_utter에 저장    
            sys_utter = dialogue["dialogue"][idx - 1]["text"] #주소는 서울 용산구 14155입니다.  이런식으로 sys_utter이 저장됨
        else:
            sys_utter = "" #아니면 공란

        user_utter = turn["text"] #유저의 대화가 저장됨
        state = turn.get("state")# state가 저장됨
        context = deepcopy(history) # 추가 예정 - 대화만 저장이됨      >>>  감사합니다 .그리고 '덕수 하우스'에 수요일부터 1일간 다섯명 쓸건데요. 예약해주세요. 스파 있나요? 이런식으로
        if user_first:
            current_turn = [user_utter, sys_utter]
        else:
            current_turn = [sys_utter, user_utter]
            #이는 대화의 순서를 체킹해서 current_turn에 넣는 것
        examples.append(
            DSTInputExample( # 35라인 봐야함 전부 str로 선언되어있는 것
                guid=f"{guid}-{d_idx}",
                context_turns=context,
                current_turn=current_turn,
                label=state,
                domains=domains,
            ) #각각의 guid의 인덱스 번호와 최근기록+ 대화 + state(관광-이름-명동 쇼핑거리) + domain이 추가된다.
        )
        history.append(sys_utter)
        history.append(user_utter)
        d_idx += 1
        #history에다가 유저의대화와 컴퓨터대답을 넣고 d_idx(이는 아마 대화의 인덱스번호및 순서체크,처음 0 이후 1씩증가) 이후  examples를 반환한다.
    return examples
    # 유저 - 시스템 한턴의 대화 > 

def get_examples_from_dialogues(data, user_first=False, dialogue_level=False):
    examples = []
    for d in tqdm(data): #데이터의 양만큼 tqdm 실행 somdst_train.py 를 보면 train_data, dev_data 들어감
        example = get_examples_from_dialogue(d, user_first=user_first)
        if dialogue_level:
            examples.append(example)
        else:
            examples.extend(example)
    return examples


def tokenize_ontology(ontology, tokenizer, max_seq_length=12):
    slot_types = []
    slot_values = []
    for k, v in ontology.items():
        tokens = tokenizer.encode(k)
        if len(tokens) < max_seq_length:
            gap = max_seq_length - len(tokens)
            tokens.extend([tokenizer.pad_token_id] *  gap)
        slot_types.append(tokens)
        slot_value = []
        for vv in v:
            tokens = tokenizer.encode(vv)
            if len(tokens) < max_seq_length:
                gap = max_seq_length - len(tokens)
                tokens.extend([tokenizer.pad_token_id] *  gap)
            slot_value.append(tokens)
        slot_values.append(torch.LongTensor(slot_value))
    return torch.LongTensor(slot_types), slot_values

