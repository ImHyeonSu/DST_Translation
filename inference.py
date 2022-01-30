import argparse
import os
import json

import torch
from tqdm import tqdm
from transformers import BertTokenizer

from data_utils import WOSDataset, get_examples_from_dialogues, convert_state_dict
from model.somdst import SOMDST
from preprocessor import SOMDSTPreprocessor
import glob
from pathlib import Path
import re
from torch.cuda.amp import autocast


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def increment_path(path, exist_ok=False):
    """Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"



def postprocess_state(state):
    for i, s in enumerate(state):
        s = s.replace(" : ", ":")
        state[i] = s.replace(" , ", ", ")
    return state


def somdst_inference(model, eval_examples, processor, device):
    processor.reset_state()
    '''
        def reset_state(self): #초기화
        self.prev_example = None
        self.prev_state = {}
        self.prev_domain_id = 0  >> 이전예시와 state domain_id  관광으로 초기화
    '''
    model.eval() # https://bluehorn07.github.io/2021/02/27/model-eval-and-train.html
    predictions = {}
    last_states = {}
    for example in tqdm(eval_examples): #dev_example 중에서 
        if not example.context_turns:
            last_states = {}
        # example.prev_state = last_states
        features = processor._convert_example_to_feature(example)
        """
            return OpenVocabDSTFeature( 
            example.guid,
            tokenized.input_ids,
            tokenized.token_type_ids,
            op_ids,
            target_ids,
            slot_positions,
            domain_id,
        ) 이 값들이 features에 들어감
        """
        features = processor.collate_fn([features])
        '''
            return (
            input_ids,
            input_masks,
            segment_ids,
            slot_position_ids,
            gating_ids,
            domain_ids,
            target_ids,
            max_update,
            max_value,
            guids,
        )  #이를 반환한다.
        '''
        batch = [
            b.to(device) if not isinstance(b, int) and not isinstance(b, list) else b
            for b in features
        ]
        (
            input_ids,
            input_masks,
            segment_ids,
            slot_position_ids,
            gating_ids,
            domain_ids,
            target_ids,
            max_update,
            max_value,
            guids,
        ) = batch
        domain_scores, state_scores, gen_scores = model(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            slot_positions=slot_position_ids,
            attention_mask=input_masks,
            max_value=9,
            op_ids=None,
        ) #model- somdst.py의 return 파일을 받아들임
        _, op_ids = state_scores.view(-1, 4).max(-1) # op_ids는 state_socres의 tensor 크기를 변경한다. max(-1)는 임의의 값중 최고 값을 선언되게 만든다. https://wikidocs.net/52846
        if gen_scores.size(1) > 0: # size가 0보다 크다면
            generated = gen_scores.squeeze(0).max(-1)[1].tolist() #gen_scores의 사이즈가 0인 차원을 없앤 것의 임의의차원에서 값이 젤 큰것을 [1]에 넣고  tolist() 함수를 사용하여 같은 레벨(위치)에 있는 데이터 끼리 묶어준다
        else:
            generated = [] # 아니면 공란

        pred_ops = [processor.id2op[op] for op in op_ids.tolist()] # id2op를 정리해서 pred_ops에 넣고
        processor.prev_state = last_states # example.prev_state = last_states 
        prediction = processor.recover_state(pred_ops, generated) #recovered - slot과 value 값을 반환하는것.
        prediction = postprocess_state(prediction)
        '''
        def postprocess_state(state):
        for i, s in enumerate(state):
        s = s.replace(" : ", ":")
        state[i] = s.replace(" , ", ", ")
        return state - 데이터 전처리
        ''' 
        last_states = convert_state_dict(prediction) # # dic[관광-종류] = 박물관 형태로 반환시켜주는것 즉 prediction의 domain-slot-value 형태로 반환해서
        predictions[guids[0]] = prediction #predictions[guids[0]] 에 넣고 
    return predictions #predictions[guids[0]] 반환


def sumbt_inference(model, eval_loader, processor, device):
    model.eval()
    predictions = {}
    for batch in tqdm(eval_loader):
        input_ids, segment_ids, input_masks, target_ids, num_turns, guids = [
            b.to(device) if not isinstance(b, list) else b for b in batch
        ]
        with torch.no_grad():
            output, pred_slot = model(input_ids, segment_ids, input_masks, labels=None, n_gpu=1)
        batch_size = input_ids.size(0)
        for i in range(batch_size):
            guid = guids[i]
            states = processor.recover_state(pred_slot.tolist()[i], num_turns[i])
            for tid, state in enumerate(states):
                predictions[f"{guid}-{tid}"] = state
    return predictions


def trade_inference(model, eval_loader, processor, device):
    model.eval()
    predictions = {}
    for batch in tqdm(eval_loader):
        input_ids, segment_ids, input_masks, gating_ids, target_ids, guids = [
            b.to(device) if not isinstance(b, list) else b for b in batch
        ]

        with torch.no_grad():
            point_outputs, gate_outputs = model(input_ids, segment_ids, input_masks, 9)
            _, generated_ids = point_outputs.max(-1)
            _, gated_ids = gate_outputs.max(-1)

        for guid, gate, gen in zip(guids, gated_ids.tolist(), generated_ids.tolist()):
            prediction = processor.recover_state(gate, gen)
            prediction = postprocess_state(prediction)
            predictions[guid] = prediction
    return predictions


def direct_output(model_path=None, model=None, processor=None):
    """
    model,processor 혹은 model_path 둘중 하나는 정확히 넣어주어야 실행됩니다.
    모델과 processor를 제대로 맞추지 않으면 오류가 납니다.
    실행 방법 두가지
    1. 저장된 모델 inference -> model_path 넣어주기 (확장자 제외, bin으로 통일)
    2. 학습중에 바로 inference -> model과 processor 넣어주기
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "/gdrive/MyDrive/p3-dst-teamed-st/data" #"" --원래 /opt/ml/input/DST_data/eval_dataset

    eval_data = json.load(open(f"{data_path}/wos-v1_train.json", "r")) #체크 "{data_path}/wos-v1_dev.json" --원래 {data_path}/eval_dials.json
    eval_examples = get_examples_from_dialogues(
        eval_data, user_first=False, dialogue_level=False
    )
    if not processor:
        model_dir_path = os.path.dirname(model_path) #path의 파일/디렉토리 경로를 반환한다.
        model_name = model_path.split('/')[-1]

        config = json.load(open(f"{model_dir_path}/exp_config.json", "r")) #"/gdrive/MyDrive/p3-dst-teamed-st/output 넘버로 선언되어있다./exp_config.json"
        config = argparse.Namespace(**config)

        slot_meta = json.load(open(f"{model_dir_path}/slot_meta.json", "r")) #"/gdrive/MyDrive/p3-dst-teamed-st/data/output 넘버로 선언되어있다./slot_meta.json"
        tokenizer = BertTokenizer.from_pretrained(config.model_name_or_path)
        added_token_num = tokenizer.add_special_tokens(     # config 파일에 이미 vocab size가 변경되어 있음
            {"additional_special_tokens": ["[SLOT]", "[NULL]", "[EOS]"]}
        )
        # Define Preprocessor
        processor = SOMDSTPreprocessor(slot_meta, tokenizer, max_seq_length=512)
        tokenized_slot_meta = []
        for slot in slot_meta:
            tokenized_slot_meta.append(
                tokenizer.encode(slot.replace("-", " "), add_special_tokens=False)
            )

        model = SOMDST(config, 5, 4, processor.op2id["update"])

        ckpt = torch.load(f"{model_path}.bin", map_location="cpu") #불러오기: torch.load(PATH) -> PATH가 가리키는 directory 및 파일 이름에 해당하는 model을 불어옴
        model.load_state_dict(ckpt)
        model.to(device)
        print("Model is loaded")
    else:
        model_dir_path = model_path
        model_name = "output"
    predictions = somdst_inference(model, eval_examples, processor, device)


    json.dump(
        predictions,
        open(f"{model_dir_path}/{model_name}.csv", "w"),
        indent=2,
        ensure_ascii=False,
    )