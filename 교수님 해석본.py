"""
SOM-DST
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

import numpy as np
import json
from torch.utils.data import Dataset
import torch
import random
import re
from copy import deepcopy
from .fix_label import fix_general_label_error

flatten = lambda x: [i for s in x for i in s]
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
domain2id = {d: i for i, d in enumerate(EXPERIMENT_DOMAINS)}

OP_SET = {
    '2': {'update': 0, 'carryover': 1},
    '3-1': {'update': 0, 'carryover': 1, 'dontcare': 2},
    '3-2': {'update': 0, 'carryover': 1, 'delete': 2},
    '4': {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3},
    '6': {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3, 'yes': 4, 'no': 5}
}

# last_dialog_state -> turn_dialog_state로 slot : value 값이 업데이트되는 입력 결과에 대해
# 
def make_turn_label(slot_meta, last_dialog_state, turn_dialog_state,
                    tokenizer, op_code='4', dynamic=False):
    if dynamic:
        gold_state = turn_dialog_state
        turn_dialog_state = {}
        for x in gold_state:
            s = x.split('-')
            k = '-'.join(s[:2])
            turn_dialog_state[k] = s[2]

    op_labels = ['carryover'] * len(slot_meta)
    generate_y = []
    # keys : slot 값들
    keys = list(turn_dialog_state.keys())
    # print("keys : ",keys)
    
    # 현재(v = turn_dialog_state[k])와 이전(vv = last_dialog_state.get(k)) slot 값에 따라 
    # 상태(vv == v: 'carryover', update, dontcare, yes, no)를 결정
    # op_labels[idx] = 'update'에 대해서는 generate_y.append([tokenizer.tokenize(v) + ['[EOS]'], idx])도 수행
    #  generate_y :  [[['moderate', '[EOS]'], 10]] 에서
    #  v!=vv : hotel   None
    #  op_labels[idx] :  update
    #  generate_y :  [[['moderate', '[EOS]'], 10], [['hotel', '[EOS]'], 12]] 로 update
    #
    for k in keys:
        v = turn_dialog_state[k]  # 현재 slot 값
        if v == 'none':
            turn_dialog_state.pop(k)
            continue
        vv = last_dialog_state.get(k) # 이전 slot 값
        
        try:
            idx = slot_meta.index(k)
            if vv != v:
                # print("v!=vv :",v," ",vv)
                if v == 'dontcare' and OP_SET[op_code].get('dontcare') is not None:
                    op_labels[idx] = 'dontcare'
                elif v == 'yes' and OP_SET[op_code].get('yes') is not None:
                    op_labels[idx] = 'yes'
                elif v == 'no' and OP_SET[op_code].get('no') is not None:
                    op_labels[idx] = 'no'
                else:
                    op_labels[idx] = 'update'
                    generate_y.append([tokenizer.tokenize(v) + ['[EOS]'], idx])
                    # print('update :',"k :",k,"v : ",v,"vv : ",vv,"op_labels[",idx,"]=",op_labels[idx],' ',generate_y)
                    # update : k : restaurant-food v :  turkish vv :  vietnamese op_labels[ 17 ]= update   [[['turkish', '[EOS]'], 17]]
                # print("v==vv :",v)
                op_labels[idx] = 'carryover'
            # print("op_labels[idx] : ",op_labels[idx])
            # print("generate_y : ",generate_y)
        except ValueError:
            continue
    #
    # 이전 (slot k : value v) -> for k, v in last_dialog_state.items():
    #                          k : v =slot : value
    #                          k : train-book people, v :  1
    # 현재 (value : vv) -> vv = turn_dialog_state.get(k)
    #
    # 현재 value 값 vv가 None이면
    #      op_labels[idx] = 'delete'로 설정함
    #
    for k, v in last_dialog_state.items():
        vv = turn_dialog_state.get(k)        
        try:
            idx = slot_meta.index(k)
            if vv is None:
                # OP_SET= {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3}
                # 'delete'가 포함되어 있으므로 
                # op_labels[idx] = 'delete'
                if OP_SET[op_code].get('delete') is not None:
                    op_labels[idx] = 'delete'
                    #    update : k : hotel-type v :  hotel vv :  None op_labels[ 12 ]= update   [[['centre', '[EOS]'], 3], [['2', '[EOS]'], 11], [['hotel', '[EOS]'], 12]]
                    # -> delete : k : hotel-type v :  hotel vv :  None op_labels[ 12 ]= delete   []
                    # print('delete :',"k :",k,"v : ",v,"vv : ",vv,"op_labels[",idx,"]=",op_labels[idx],' ',generate_y)
                else: # 수행안됨
                    op_labels[idx] = 'update'
                    generate_y.append([['[NULL]', '[EOS]'], idx])
                    # 출력 안됨 : OP_SET= {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3}
                    # print("k :",k,"v : ",v,"vv : ",vv,"op_labels[",idx,"]=",op_labels[idx],' ',generate_y)
        except ValueError:
            continue
        
    # 현재 slot-value 값으로 gold_state를 설정함
    # turn_dialog_state :  {'hotel-name': 'express by holiday inn cambridge', 'hotel-area': 'east', 'hotel-internet': 'yes', 'hotel-type': 'hotel'}
    # gold_state= ['hotel-name-express by holiday inn cambridge', 'hotel-area-east', 'hotel-internet-yes', 'hotel-type-hotel']
    gold_state = [str(k) + '-' + str(v) for k, v in turn_dialog_state.items()]
    # print("turn_dialog_state : ",turn_dialog_state)
    # print("gold_state=",gold_state)
    
    if len(generate_y) > 0:
        #    [[['la', 'ra', '##za', '[EOS]'], 22], [['alexander', 'bed', 'and', 'breakfast', '[EOS]'], 21], [['1', '[EOS]'], 5]]
        # -> [['1', '[EOS]'], ['alexander', 'bed', 'and', 'breakfast', '[EOS]'], ['la', 'ra', '##za', '[EOS]']]
        # print("1 generate_y=",generate_y)
        generate_y = sorted(generate_y, key=lambda lst: lst[1])
        generate_y, _ = [list(e) for e in list(zip(*generate_y))]
        # print("2 generate_y=",generate_y)

    if dynamic:
        op2id = OP_SET[op_code]
        generate_y = [tokenizer.convert_tokens_to_ids(y) for y in generate_y]
        op_labels = [op2id[i] for i in op_labels]

    # op_labels  : 이전 slot(k) : value(v)와 현재 value(vv)를 비교하여 
    # OP_SET= {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3}중에서 op_labels[idx]의 값을 설정함
    # (1) vv == v : 'carryover'
    # if vv != v: 
    # (2) v == 'dontcare' 
    # (3) v == 'yes' (사용안됨)
    # (4) v == 'no'  (사용안됨)
    # (5) op_labels[idx] = 'update'
    # (6) if vv is None: op_labels[idx] = 'delete'
    # generate_y : op_labels[idx] = 'update'인 경우 
    #              이전 slot value v를 tokenize한 이후에 sorting하고 리스트 데이터 유형으로 저장
    #            (1) generate_y.append([tokenizer.tokenize(v) + ['[EOS]'], idx])
    #            (2) generate_y = sorted(generate_y, key=lambda lst: lst[1])
    # gold_state : gold_state = [str(k) + '-' + str(v) for k, v in turn_dialog_state.items()]
    return op_labels, generate_y, gold_state


def postprocessing(slot_meta, ops, last_dialog_state,
                   generated, tokenizer, op_code, gold_gen={}):
    gid = 0
    for st, op in zip(slot_meta, ops):
        if op == 'dontcare' and OP_SET[op_code].get('dontcare') is not None:
            last_dialog_state[st] = 'dontcare'
        elif op == 'yes' and OP_SET[op_code].get('yes') is not None:
            last_dialog_state[st] = 'yes'
        elif op == 'no' and OP_SET[op_code].get('no') is not None:
            last_dialog_state[st] = 'no'
        elif op == 'delete' and last_dialog_state.get(st) and OP_SET[op_code].get('delete') is not None:
            last_dialog_state.pop(st)
        elif op == 'update':
            g = tokenizer.convert_ids_to_tokens(generated[gid])
            gen = []
            for gg in g:
                if gg == '[EOS]':
                    break
                gen.append(gg)
            gen = ' '.join(gen).replace(' ##', '')
            gid += 1
            gen = gen.replace(' : ', ':').replace('##', '')
            if gold_gen and gold_gen.get(st) and gold_gen[st] not in ['dontcare']:
                gen = gold_gen[st]

            if gen == '[NULL]' and last_dialog_state.get(st) and not OP_SET[op_code].get('delete') is not None:
                last_dialog_state.pop(st)
            else:
                last_dialog_state[st] = gen
    return generated, last_dialog_state


def make_slot_meta(ontology):
    meta = []
    change = {}
    idx = 0
    max_len = 0
    for i, k in enumerate(ontology.keys()):
        d, s = k.split('-')
        if d not in EXPERIMENT_DOMAINS:
            continue
        if 'price' in s or 'leave' in s or 'arrive' in s:
            s = s.replace(' ', '')
        ss = s.split()
        if len(ss) + 1 > max_len:
            max_len = len(ss) + 1
        meta.append('-'.join([d, s]))
        change[meta[-1]] = ontology[k]
    return sorted(meta), change


def prepare_dataset(data_path, tokenizer, slot_meta,
                    n_history, max_seq_length, diag_level=False, op_code='4'):
    dials = json.load(open(data_path))
    data = []
    domain_counter = {}
    max_resp_len, max_value_len = 0, 0
    max_line = None
    t=0
    # print("dials : ",len(dials))
    for dial_dict in dials:
        for domain in dial_dict["domains"]:          
            # print("t=",t," ","domain : ",domain)
            
            #["hotel", "train", "restaurant", "attraction", "taxi"]
            if domain not in EXPERIMENT_DOMAINS:
                continue
            if domain not in domain_counter.keys():
                # print("if domain not in domain_counter.keys():~~~~~~")
                domain_counter[domain] = 0
            domain_counter[domain] += 1

        dialog_history = []
        last_dialog_state = {}
        last_uttr = ""
        t+=1
#        if t<=10 and domain !="hotel" :
#            print("dial_dict : ",len(dial_dict["dialogue"]))
            
        for ti, turn in enumerate(dial_dict["dialogue"]):
            #domain : train, restaurant, police, hotel, attraction, taxi
            turn_domain = turn["domain"]
            if turn_domain not in EXPERIMENT_DOMAINS:
                continue
            turn_id = turn["turn_idx"]
            # 대화문장 : 챗봇 ; 질의문장
            turn_uttr = (turn["system_transcript"] + ' ; ' + turn["transcript"]).strip()
            dialog_history.append(last_uttr)  # 대화문장 들....         
            # {'taxi-destination': 'cambridge', 'taxi-departure': 'clare college', 'taxi-arriveby': '18:00'}
            # {'restaurant-name': 'curry garden', 'train-leaveat': '15:15', 'train-destination': 'cambridge', 'train-day': 'friday', 'train-departure': 'stevenage'}
            # {'hotel-pricerange': 'moderate'}
#            if t<=10 and domain !="hotel" :
#                print("ti=",ti)
#                print('belief_state : ',turn["belief_state"])

            # 
            # fix_general_label_error() : train 데이터의 turn["belief_state"] 값을 이용하여 slot : value 값으로 변환함
            # turn["belief_state"] : train 데이터에서 slot 값들
            # 예 : [{'slots': [['hotel-parking', 'yes']], 'act': 'inform'}, {'slots': [['hotel-pricerange', 'cheap']], 'act': 'inform'}, {'slots': [['hotel-type', 'hotel']], 'act': 'inform'}]
            #
            # slot_meta : domain에서 slot 정의 값들
            # domain=hotel, slot_meta = ['attraction-area', 'attraction-name', 'attraction-type', 'hotel-area', 'hotel-book day', 'hotel-book people', 'hotel-book stay', 'hotel-internet', 'hotel-name', 'hotel-parking', 'hotel-pricerange', 'hotel-stars', 'hotel-type', 'restaurant-area', 'restaurant-book day', 'restaurant-book people', 'restaurant-book time', 'restaurant-food', 'restaurant-name', 'restaurant-pricerange', 'taxi-arriveby', 'taxi-departure', 'taxi-destination', 'taxi-leaveat', 'train-arriveby', 'train-book people', 'train-day', 'train-departure', 'train-destination', 'train-leaveat']
            # 
            turn_dialog_state = fix_general_label_error(turn["belief_state"], False, slot_meta, domain, t)
            # print(ti,"turn_dialog_state : ",turn_dialog_state)
            #
            # turn_dialog_state :  {'hotel-parking': 'yes', 'hotel-pricerange': 'cheap', 'hotel-type': 'hotel'}
            # slot : value 값으로 변환....
            # 입력(turn["belief_state"]) : [{'slots': [['restaurant-food', 'chinese']], 'act': 'inform'}, {'slots': [['restaurant-pricerange', 'cheap']], 'act': 'inform'}]
            # 출력(turn_dialog_state)    : {'restaurant-food': 'chinese', 'restaurant-pricerange': 'cheap'}

            last_uttr = turn_uttr
            # slot_meta : domain에서 slot 정의 값들
            # last_dialog_state(last_dialog_state = turn_dialog_state : 누적) : {} -> {'hotel-book day': 'monday', 'hotel-book people': '1', 'hotel-book stay': '1', 'hotel-name': 'acorn guest house', 'hotel-area': 'north', 'hotel-parking': 'yes', 'hotel-internet': 'no', 'hotel-type': 'guest house', 'train-book people': '6', 'train-destination': 'leicester', 'train-day': 'wednesday', 'train-arriveby': '21:00', 'train-departure': 'cambridge'}
            # turn_dialog_state : slot : value({'restaurant-food': 'chinese', 'restaurant-pricerange': 'cheap'})
            # last_dialog_state -> turn_dialog_state로 천이
            # op_code : 4
            # print(ti,"last_dialog_state : ",last_dialog_state)
            op_labels, generate_y, gold_state = make_turn_label(slot_meta, last_dialog_state,
                                                                turn_dialog_state,
                                                                tokenizer, op_code)
            # op_labels : carryover, update, dontcare ...
            # generate_y : [['08', ':', '00', '[EOS]'], ['birmingham', 'new', 'street', '[EOS]']]
            # gold_state : ['attraction-name-museum of classical archaeology']
            
            if (ti + 1) == len(dial_dict["dialogue"]):
                is_last_turn = True
            else:
                is_last_turn = False
                
            #
#            if t<=3 :      
#                print("TrainingInstance~~~")

            instance = TrainingInstance(dial_dict["dialogue_idx"], turn_domain,
                                        turn_id, turn_uttr, ' '.join(dialog_history[-n_history:]),
                                        last_dialog_state, op_labels,
                                        generate_y, gold_state, max_seq_length, slot_meta,
                                        is_last_turn, op_code=op_code)
            
            #
#            if t<=3 :
#                print("instance.make_instance~~~")
            instance.make_instance(tokenizer)
            data.append(instance)
            last_dialog_state = turn_dialog_state
    return data


class TrainingInstance:
    def __init__(self, ID,
                 turn_domain,
                 turn_id,
                 turn_utter,
                 dialog_history,
                 last_dialog_state,
                 op_labels,
                 generate_y,
                 gold_state,
                 max_seq_length,
                 slot_meta,
                 is_last_turn,
                 op_code='4'):
        self.id = ID
        self.turn_domain = turn_domain
        self.turn_id = turn_id
        self.turn_utter = turn_utter
        self.dialog_history = dialog_history
        self.last_dialog_state = last_dialog_state
        self.gold_p_state = last_dialog_state
        self.generate_y = generate_y
        self.op_labels = op_labels
        self.gold_state = gold_state
        self.max_seq_length = max_seq_length
        self.slot_meta = slot_meta
        self.is_last_turn = is_last_turn
        self.op2id = OP_SET[op_code]

    def shuffle_state(self, rng, slot_meta=None):
        new_y = []
        gid = 0
        # print("shuffle_state")
        for idx, aa in enumerate(self.op_labels):
           
            # carryover
            if aa == 'update':
                new_y.append(self.generate_y[gid])
                gid += 1
                # print(new_y)
            else:
                # 전부 dummy
                new_y.append(["dummy"])
                # print(new_y)
        if slot_meta is None:
            temp = list(zip(self.op_labels, self.slot_meta, new_y))
            rng.shuffle(temp)
        else:
            indices = list(range(len(slot_meta)))
            for idx, st in enumerate(slot_meta):
                indices[self.slot_meta.index(st)] = idx
            temp = list(zip(self.op_labels, self.slot_meta, new_y, indices))
            temp = sorted(temp, key=lambda x: x[-1])
        temp = list(zip(*temp))
        self.op_labels = list(temp[0])
        self.slot_meta = list(temp[1])
        self.generate_y = [yy for yy in temp[2] if yy != ["dummy"]]

    # 
    def make_instance(self, tokenizer, max_seq_length=None,
                      word_dropout=0., slot_token='[SLOT]'):
        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        state = []
        # slot_meta : domain별로 설정된 slot 값
        for s in self.slot_meta:
            state.append(slot_token)  # [SLOT] 추가
            k = s.split('-')
            # 이전 slot : value에서 slot_meta의 value(v) 값을 읽어서
            # if v is not None:
            #    [SLOT]slot:value를 tokenize하여 state에 추가함
            # None이면
            #    [SLOT]slot을 tokenize한 다음에 -[NULL]을 추가한 다음에 state에 추가함
            v = self.last_dialog_state.get(s)
            if v is not None:
                # s:v= attraction-area : north
                # k= ['attraction', 'area', '-', 'north']
                # value(v)를 tokenize함
                # t= ['attraction', 'area', '-', 'north']
                # state= ['[SLOT]', 'attraction', 'area', '-', 'north']으로 t를 계속하여 state에 추가함....
                # state= ['[SLOT]', 'attraction', 'area', '-', 'north', '[SLOT]', 'attraction', 'name', '-', 'milton', 'country', 'park', '[SLOT]', 'attraction', 'type', '-', '[NULL]']
                # print('s:v=',s,':',v)
                k.extend(['-', v])
                # print('k=',k)
                t = tokenizer.tokenize(' '.join(k))
                # print('t=',t)
            else:
                t = tokenizer.tokenize(' '.join(k))
                t.extend(['-', '[NULL]'])
            state.extend(t)
            # print('state=',state)
            
        avail_length_1 = max_seq_length - len(state) - 3
        diag_1 = tokenizer.tokenize(self.dialog_history)
        diag_2 = tokenizer.tokenize(self.turn_utter)
        print("diag_1=",self.dialog_history)
        print("diag_2=",self.turn_utter)
        avail_length = avail_length_1 - len(diag_2)

        if len(diag_1) > avail_length:  # truncated
            avail_length = len(diag_1) - avail_length
            diag_1 = diag_1[avail_length:]

        if len(diag_1) == 0 and len(diag_2) > avail_length_1:
            avail_length = len(diag_2) - avail_length_1
            diag_2 = diag_2[avail_length:]

        drop_mask = [0] + [1] * len(diag_1) + [0] + [1] * len(diag_2) + [0]
        diag_1 = ["[CLS]"] + diag_1 + ["[SEP]"]
        diag_2 = diag_2 + ["[SEP]"]
        segment = [0] * len(diag_1) + [1] * len(diag_2)

        diag = diag_1 + diag_2
        # word dropout
        if word_dropout > 0.:
            drop_mask = np.array(drop_mask)
            word_drop = np.random.binomial(drop_mask.astype('int64'), word_dropout)
            diag = [w if word_drop[i] == 0 else '[UNK]' for i, w in enumerate(diag)]
        input_ = diag + state
        segment = segment + [1]*len(state)
        self.input_ = input_

        self.segment_id = segment
        slot_position = []
        for i, t in enumerate(self.input_):
            if t == slot_token:
                slot_position.append(i)
        self.slot_position = slot_position

        input_mask = [1] * len(self.input_)
        self.input_id = tokenizer.convert_tokens_to_ids(self.input_)
        if len(input_mask) < max_seq_length:
            self.input_id = self.input_id + [0] * (max_seq_length-len(input_mask))
            self.segment_id = self.segment_id + [0] * (max_seq_length-len(input_mask))
            input_mask = input_mask + [0] * (max_seq_length-len(input_mask))

        self.input_mask = input_mask
        self.domain_id = domain2id[self.turn_domain]
        self.op_ids = [self.op2id[a] for a in self.op_labels]
        self.generate_ids = [tokenizer.convert_tokens_to_ids(y) for y in self.generate_y]


class MultiWozDataset(Dataset):
    def __init__(self, data, tokenizer, slot_meta, max_seq_length, rng,
                 ontology, word_dropout=0.1, shuffle_state=False, shuffle_p=0.5):
        self.data = data
        self.len = len(data)
        self.tokenizer = tokenizer
        self.slot_meta = slot_meta
        self.max_seq_length = max_seq_length
        self.ontology = ontology
        self.word_dropout = word_dropout
        self.shuffle_state = shuffle_state
        self.shuffle_p = shuffle_p
        self.rng = rng

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.shuffle_state and self.shuffle_p > 0.:
            if self.rng.random() < self.shuffle_p:
                self.data[idx].shuffle_state(self.rng, None)
            else:
                self.data[idx].shuffle_state(self.rng, self.slot_meta)
        if self.word_dropout > 0 or self.shuffle_state:
            self.data[idx].make_instance(self.tokenizer,
                                         word_dropout=self.word_dropout)
        return self.data[idx]

    def collate_fn(self, batch):
        input_ids = torch.tensor([f.input_id for f in batch], dtype=torch.long)
        # print(input_ids)
        input_mask = torch.tensor([f.input_mask for f in batch], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_id for f in batch], dtype=torch.long)
        state_position_ids = torch.tensor([f.slot_position for f in batch], dtype=torch.long)
        op_ids = torch.tensor([f.op_ids for f in batch], dtype=torch.long)
        domain_ids = torch.tensor([f.domain_id for f in batch], dtype=torch.long)
        gen_ids = [b.generate_ids for b in batch]
        max_update = max([len(b) for b in gen_ids])
        max_value = max([len(b) for b in flatten(gen_ids)])
        for bid, b in enumerate(gen_ids):
            n_update = len(b)
            for idx, v in enumerate(b):
                b[idx] = v + [0] * (max_value - len(v))
            gen_ids[bid] = b + [[0] * max_value] * (max_update - n_update)
        gen_ids = torch.tensor(gen_ids, dtype=torch.long)

        return input_ids, input_mask, segment_ids, state_position_ids, op_ids, domain_ids, gen_ids, max_value, max_update
