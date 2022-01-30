import torch
import numpy as np
from data_utils import DSTPreprocessor, OpenVocabDSTFeature, convert_state_dict, _truncate_seq_pair, OntologyDSTFeature

class SOMDSTPreprocessor(DSTPreprocessor):
    def __init__(
        self,
        slot_meta, #  slot_meta < data-slot_meta.json > \uad00\uad11-\uacbd\uce58 \uc88b\uc740(유니코드) = 관광-경치 좋은 등...
        src_tokenizer, #  
        trg_tokenizer=None, #  
        ontology=None, # ontology < data - ontology.json > {"관광-경치 좋은": ["none","dontcare","yes","no"] 등으로 선언되어있다.
        max_seq_length=512, # 최장 토큰 길이
    ):
        self.slot_meta = slot_meta
        self.src_tokenizer = src_tokenizer 
        self.trg_tokenizer = trg_tokenizer if trg_tokenizer else src_tokenizer
        self.ontology = ontology
        #SLOT의 상태를 체크하기위한 delete~carryover 까지 선언된 부분 update인 경우에만 다음 slot과 value 값이 바뀜
        self.op2id = {"delete": 0, "update": 1, "dontcare": 2, "carryover": 3}
        self.id2op = {v: k for k, v in self.op2id.items()}  #id2op - 0:delete, 1:update ~ 3:carryover 로 저장되어있음
        #Domain은 총 5개로 선언되어 있는 것을 선언하는 부분
        self.domain2id = {"관광": 0, "숙소": 1, "식당": 2, "지하철": 3, "택시": 4}
        self.id2domain = {v: k for k, v in self.domain2id.items()} #id2domain - 0:관광, 1:숙소 ~ 4:택시 로 저장되어있음 WOS - WIZARD OF SEOUL 
        self.prev_example = None  
        self.prev_state = {} # state 딕셔너리형태인것 기억해두기
        self.prev_domain_id = None # 
        self.slot_id = self.src_tokenizer.convert_tokens_to_ids("[SLOT]") #https://github.com/monologg/KoBERT-Transformers, https://thebook.io/080263/ch10/02/02-06/ 토큰의 아이디를 [slot]으로 반환
        self.max_seq_length = max_seq_length
                   
    def _convert_example_to_feature(self, example):
        if not example.context_turns: #context_turns - user 이전 대화를 context_turns에 쌓아나간다. 구체적으론 (guid='wos-v1_train_00098-0', context_turns=[], current_turn=['', '저 미팅 차 서울 온김에 식당도 예약해서 가고, 관광도 하고 싶은데요. 식당부터 좀 도와주실래요/'], label=[], domains=['식당', '관광', '지하철']), DSTInputExample(guid='wos-v1_train_00098-1', context_turns=['', '저 미팅 차 서울 온김에 식당도 예약해서 가고, 관광도 하고 싶은데요. 식당부터 좀 도와주실래요/'], current_turn=['안녕하세요, 식당 도와드리겠습니다. 원하시는 가격대와 지역, 종류 말씀해 주시면 됩니다.', '네 가격대는 적당한 데로 중앙쪽에 양식당이면 좋겠고 차가 없어서 걸어서 갈 수 있는 곳이어야 하겠네요.'], label=['식당-가격대-적당', '식당-지역-서울 중앙', '식당-종류-양식당', '식당-도보 가능-yes'], domains=['식당', '관광', '지하철']), 
            self.reset_state() #이전 대화가 만약 비어있다면 state를 리셋한다.
        #현재 대화인지 이전대화인지 체크하고 현재대화라면 d_t에 이전대화라면 d_prev에 집어넣음
        if self.prev_example:  # 이전 example 파일이 들어가 있으면, 이전예시의 current_trun =[user_utter, sys_utter] or [sys_utter, user_utter] 가 d_prev에 들어감 예시 전화번호는 983880764입니다. 더 필요하신 게 있으실까요? ; 네 관광지와 같은 지역의 한식당을 가고싶은데요 야외석이 있어야되요
            d_prev = " ; ".join(self.prev_example.current_turn) 
            #print("11111")
            #print(d_prev)
        else:
            d_prev = ""
        d_t = " ; ".join(example.current_turn)  # 현재 대화라면 현재대화를 d_t에 넣음 전화번호는 983880764입니다. 더 필요하신 게 있으실까요? ; 네 관광지와 같은 지역의 한식당을 가고싶은데요 야외석이 있어야되요
        #print("000000")
        #print(d_t)
        #d_t가 먼저실행되고 이전대화로써 prev_example이 실행됨
        if not example.label:
            example.label = []

        state = convert_state_dict(example.label) # data_utils.py의 206라인쯤, label=state  이코드는 data_utils.py 273라인에 선언되있음, slot에 대한 value 딕셔너리 리턴받아서 그걸 state에 넣어준다. 즉 state = dic{s} - v 가 저장이 되어있다.
        #print(state)  state = {'숙소-가격대': '적당', '숙소-종류': 'dontcare', '숙소-지역': 'dontcare', '숙소-주차 가능': 'yes'} 이런식임
        b_prev_state = []                         
        op_ids = []
        target_ids = []
        #slot부터 slot_meta의 수까지 무한반복 - 즉 slot을 하나하나 체크하면서 넘어감
        for slot in self.slot_meta:
            #이전slot에 대응되는 value값을  prev_state[]에 넣거나 혹은 [NULL]을 넣는다.
            prev_value = self.prev_state.get(slot, "[NULL]") #get(a, b) : 첫번째 인자에 해당 찾고 싶은 딕셔너리 key 값 입력하고, 두번째 인자에는 첫번째 인자가 없을 경우 출력하고 싶은 값 입력. key값의 value 값 출력
            #value에다가도 slot에 대응되는 value값을 넣거나 혹은 [NULL]을 넣는다..
            value = state.get(slot, "[NULL]")
            #value와 이전 prev_value의 값이 같다면(즉 변화가 없다면)
            if value == prev_value:
                operation = self.op2id["carryover"]     # 변경되지 않는 value 즉 s(t1)과 v(t1-1)이 저장되어야할때, operation= 3
            elif value == "[NULL]":
                operation = self.op2id["delete"]    # 현재 s(t1) - v(t)가null(value)값이 opreation = 0 이겠지만 기록상에선 안보였음(기록이 5천개정도만저장됨)
            elif value == "doncare":
                operation = self.op2id["dontcare"] # 현재 s(t1) - dontcare이 들어감 
            else:
                operation = self.op2id["update"]    # update 라면 즉 value의 값이 update 라면 update "[EOS]" 와 token stiring을 token id의 리스트로 변환해서 target_id에 집어넣는다. token id의 리스트예시 https://ainote.tistory.com/15
                target_id = self.trg_tokenizer.encode( 
                    value + " [EOS]", add_special_tokens=False
                )
                target_ids.append(target_id) # target_ids에 targer_id를 쌓는다.
                #print(value)       박물관               서울 중앙      
                #print(operation)    1                      1
                #print(target_id) [8732, 35002]          [6265, 6672, 35002]
                #print(target_ids) [[8732, 35002]]       [[8732, 35002], [6265, 6672, 35002]] 이런식으로 target_ids에 쌓여나감
            if prev_value == "dontcare":
                prev_value = "dont care"
            #if 문이 돌아가고나서 list.extend(iterable)는 리스트 끝에 가장 바깥쪽에다가 "[SLOT] - prev.value 값" 이 추가됨   
            b_prev_state.extend(["[SLOT]"])
            b_prev_state.extend(slot.split("-"))
            b_prev_state.extend(["-", prev_value])
            #print(b_prev_state) ['[SLOT]', '관광', '경치 좋은', '-', '[NULL]', '[SLOT]', '관광', '교육적', '-', '[NULL]', '[SLOT]', '관광', '도보 가능', '-', '[NULL]', '[SLOT]', '관광', '문화 예술', '-', '[NULL]', '[SLOT]', '관광', '역사적', '-', '[NULL]', '[SLOT]', '관광', '이름', '-', '[NULL]', '[SLOT]', '관광', '종류', '-', '[NULL]']
            #if 문이 돌아가고나서 op.ids 리스트 끝에 opration - delete~update 중 하나의 값이 들어감
            op_ids.append(operation)
            #print(op_ids)  #[3, 3, 3, 3, 3, 3, 1]
            #이 부분 까지 b_prev_state는 domain-slot 까지만 포함되어있고 value값은 [NULL]으로 선언되어있다. 또한 op_ids는 b_prev_state에 매치되는 숫자 (slot의 값에따른)이 저장됨
        #" " + "[SLOT] - prev.value 값" 값이 추가되서 b_prev_state의 값이 저장됨    
        b_prev_state = " ".join(b_prev_state)
        #print(b_prev_state) [SLOT] 관광 경치 좋은 - [NULL] [SLOT] 관광 교육적 - [NULL] [SLOT] 관광 도보 가능 - [NULL] [SLOT] 관광 문화 예술 - [NULL] [SLOT] 관광 역사적 - [NULL] [SLOT] 관광 이름 - [NULL] [SLOT] 관광 종류 - [NULL] [SLOT] 관광 주차 가능 - [NULL] [SLOT] 관광 지역 - [NULL] [SLOT] 숙소 가격대 - [NULL] [SLOT] 숙소 도보 가능 - [NULL] [SLOT] 숙소 수영장 유무 - [NULL] [SLOT] 숙소 스파 유무 - [NULL] [SLOT] 숙소 예약 기간 - [NULL] [SLOT] 숙소 예약 명수 - [NULL] [SLOT] 숙소 예약 요일 - [NULL] [SLOT] 숙소 이름 - [NULL] [SLOT] 숙소 인터넷 가능 - [NULL] [SLOT] 숙소 조식 가능 - [NULL] [SLOT] 숙소 종류 - [NULL] [SLOT] 숙소 주차 가능 - [NULL] [SLOT] 숙소 지역 - [NULL] [SLOT] 숙소 헬스장 유무 - [NULL] [SLOT] 숙소 흡연 가능 - [NULL] [SLOT] 식당 가격대 - [NULL] [SLOT] 식당 도보 가능 - [NULL] [SLOT] 식당 야외석 유무 - [NULL] [SLOT] 식당 예약 명수 - [NULL] [SLOT] 식당 예약 시간 - [NULL] [SLOT] 식당 예약 요일 - [NULL] [SLOT] 식당 이름 - [NULL] [SLOT] 식당 인터넷 가능 - [NULL] [SLOT] 식당 종류 - [NULL] [SLOT] 식당 주류 판매 - [NULL] [SLOT] 식당 주차 가능 - [NULL] [SLOT] 식당 지역 - [NULL] [SLOT] 식당 흡연 가능 - [NULL] [SLOT] 지하철 도착지 - [NULL] [SLOT] 지하철 출발 시간 - [NULL] [SLOT] 지하철 출발지 - [NULL] [SLOT] 택시 도착 시간 - [NULL] [SLOT] 택시 도착지 - [NULL] [SLOT] 택시 종류 - [NULL] [SLOT] 택시 출발 시간 - [NULL] [SLOT] 택시 출발지 - [NULL]
        #문장 토큰단위로 자르는 부분 
        tokenized = self.src_tokenizer(
            d_prev,
            d_t + " [SEP] " + b_prev_state,
            padding=True,
            max_length=self.max_seq_length,
            truncation=True,
            add_special_tokens=True,
        )
        #padding (False) : True로 지정한 경우 batch 내에서 가장 긴 길이에 맞춰서 padding을 한다. False인 경우 padding을 하지 않고 encoding을 한다. 'max_length'로 지정하면 모델이 입력으로 받을 수 있는 최대의 길이에 맞춰서 padding을 한다. 
        #truncation (False): True 인 경우 모델이 입력으로 받을 수 있는 최대의 길이에 맞춰서 글을 자른다  "only_first"인 경우 sentence pair로 입력이 주어지는 경우, 첫 번째 문장만 Truncate 한다. 반대로, "only_second"인 경우 두 번째 문장만 truncate 한다. False로 지정한 경우 trauncate을 하지 않는다.
        #print(tokenized) #{'input_ids': [2, 3, 31, 6265, 6672, 4073, 3249, 4034, 8732, 4292, 6722,~]}

        #print(tokenized.token_type_ids)
        slot_positions = []
        #input_ids
        for i, input_id in enumerate(tokenized.input_ids):
            if input_id == self.slot_id:
                slot_positions.append(i)
                #print(slot_positions) #[70] > [70, 77] > [70, 77, 83] 이런식으로 쌓여나감, 말그대로 slot의 값에 대한 순서를 따지는 듯함
        if not self.prev_example: 
            domain_slot = list(state.keys()) # key값 즉 domain-value 에 값이 domain-slot에저장됨
            #print(domain_slot[0]) - ['식당-가격대', '식당-지역', '식당-종류'] 이런 도메인-슬롯이 저장됨
            if domain_slot:
                domain_id = self.domain2id[domain_slot[0].split("-")[0]]
                #print(domain_id) - 0 > 0 > 2 > 0 이런식으로 출력됨, 계속해서 domain_2id[0]을 짤라나가면서 값을 id에 저장해나감
            else:
                domain_id = self.prev_domain_id
                #print(domain_id) - 간간히 0 > 0 > 0 이런식으로 출력됨, 이것도 마찬가지
                
        else:
            #print(self.prev_example) DSTInputExample(guid='wos-v1_train_00000-0', context_turns=[], current_turn=['', '서울 중앙에 있는 박물관을 찾아주세요'], label=['관광-종류-박물관', '관광-지역-서울 중앙'], domains=['관광', '식당']) 요렇게 들어가있음
            diff_state = set(example.label) - set(self.prev_example.label)
            #print(example.label) ['관광-종류-박물관', '관광-지역-서울 중앙', '관광-이름-문화역서울 284']
            #print(self.prev_example.label) ['관광-종류-박물관', '관광-지역-서울 중앙']
            #print(diff_state){'관광-이름-문화역서울 284'}
            #이런식으로 기존 example.label 에 추가된 값들이 들어 오면 그것만 반환하는 것이 diff_state
            if not diff_state:
                domain_id = self.prev_domain_id # 값이 비었다면 이전 domain_id가 그대로 들어감
            else:
                domain_id = self.domain2id[list(diff_state)[0].split("-")[0]] # 값이 있다면 id로 변환해서 그대로 들어감 0~4까지 겠지
                #print(domain_id)  0 > 0 > 2 > 0 이런식으로 출력됨, 계속해서 domain_2id[0]을 짤라나가면서 값을 id에 저장해나감
        #결국 이 for 문은 prev_example의 값이 있는지 없는지에 따라 domain_id에 값이 들어가는 부분
                
        self.prev_example = example
        self.prev_state = state
        self.prev_domain_id = domain_id
        return OpenVocabDSTFeature( 
            example.guid,
            tokenized.input_ids,
            tokenized.token_type_ids,
            op_ids,
            target_ids,
            slot_positions,
            domain_id,
        )# 여기에다가 그간 작업된 데이터들이 다들어감 

    def reset_state(self): #초기화
        self.prev_example = None
        self.prev_state = {}
        self.prev_domain_id = 0

    def convert_examples_to_features(self, examples): #convert_examples_to_features(train_examples)
        return list(map(self._convert_example_to_feature, examples))

    def recover_state(self, pred_ops, gen_list):
        recovered = []
        gid = 0
        for slot, op in zip(self.slot_meta, pred_ops):
            if op == "dontcare":
                self.prev_state[slot] = "dontcare"
            elif op == "delete":
                if not slot in self.prev_state:
                    print("delete error")
                    continue
                self.prev_state.pop(slot) # slot 빼내고 삭제
            elif op == "update":
                tokens = self.trg_tokenizer.convert_ids_to_tokens(gen_list[gid]) # 한 개의 token id 또는 token id 의 리스트를 token으로 변환한다. skip_special_tokens=True로 하면 decoding할 때 special token들을 제거한다. 
                gen = []                                                        # gen_list[0]인 값을 token에다가 넣고 gen[]선언    
                for token in tokens:
                    if token == "[EOS]":            #[EOS]라는 글이 들어가 있기 때문에 이게 들어가 있으면 멈춘다.
                        break
                    gen.append(token)               # [EOS]라는 글이 없다면 gen[]배열에 추가 gen은 value값
                gen = " ".join(gen).replace(" ##", "")  # 공백을 넣고  ##를 ""로 대체한다
                gid += 1 #gid +
                gen = gen.replace(" : ", ":").replace("##", "") # : 를 :로 대체하고 ##를 ""로 대체한다. 
                if gen == "[NULL]" and slot in self.prev_state:
                    self.prev_state.pop(slot) # 이전값에서 변함이 없다면 slot의 젤마지막을 반환하고 삭제
                else:
                    self.prev_state[slot] = gen # gen과 slot 형태로 연결 즉 slot-value 형태로 반환됨
                    recovered.append(f"{slot}-{gen}")
            else:
                prev_value = self.prev_state.get(slot)
                if prev_value:
                    recovered.append(f"{slot}-{prev_value}") # carryover의 경우인데 이경우 slot-이전 value 값을 넣고 recovered선언됨
        return recovered 

    def collate_fn(self, batch):  #  64-bit integer : LongTensor
        guids = [b.guid for b in batch]
        #guid들을 받아들인 guids
        input_ids = torch.LongTensor(
            self.pad_ids(
                [b.input_id for b in batch],
                self.src_tokenizer.pad_token_id,
                max_length=self.max_seq_length,
            )
        ) #input_ids - 토큰 즉 말을 분리한것의 고유 아이디, #atteintion_mask - 이것은 pad 토큰은 0으로매핑  pad토큰이 아닌 것은 1로매핑
        # 즉 각 input_id 들과 pad_token_id들 그리고 토큰길이 선언
        segment_ids = torch.LongTensor(
            self.pad_ids(
                [b.segment_id for b in batch],
                self.src_tokenizer.pad_token_id,
                max_length=self.max_seq_length,
            )
        ) # segment_id에 대해서도 이를 실행 그렇지만 segement_ids 확인불가
        '''
        LongTensor는 64bit 정수를 선언하기 위한 것 
        '''
        input_masks = input_ids.ne(self.src_tokenizer.pad_token_id) #ne() =! 다르다 를 의미하는 연산자로 pad tokenid와 inpu_ids의 값이 다르다면 masks로 반환

        gating_ids = torch.LongTensor([b.gating_id for b in batch])
        domain_ids = torch.LongTensor([b.domain_id for b in batch])
        target_ids = [b.target_ids for b in batch]
        slot_position_ids = torch.LongTensor([b.slot_positions for b in batch])
        max_update = max([len(b) for b in target_ids])
        max_value = max([len(t) for b in target_ids for t in b] + [10])
        # 각각 값들을 얻어와서 반환
        for bid, b in enumerate(target_ids): # update 라면 즉 value의 값이 update 라면 update "[EOS]" 와 token stiring을 token id의 리스트로 변환해서 target_id에 집어넣는다. token id의 리스트예시 https://ainote.tistory.com/15
            n_update = len(b) # b의 문자열 길이를 n_update에 넣는다.
            for idx, v in enumerate(b): # target_ids의 인덱스 번호와 컬렉션의 원소를 tuple형태로 반환합니다.
                b[idx] = v + [0] * (max_value - len(v)) # 값 전처리 과정인것 같습니다.
            target_ids[bid] = b + [[0] * max_value] * (max_update - n_update) 
        target_ids = torch.LongTensor(target_ids) # target_ids 를 얻고(즉 업데이트 할것)
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
##여기까지

class TRADEPreprocessor(DSTPreprocessor):
    def __init__(
        self,
        slot_meta,
        src_tokenizer,
        trg_tokenizer=None,
        ontology=None,
        max_seq_length=512,
    ):
        self.slot_meta = slot_meta
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer if trg_tokenizer else src_tokenizer
        self.ontology = ontology
        self.gating2id = {"none": 0, "dontcare": 1, "yes": 2, "no": 3, "ptr": 4}
        self.id2gating = {v: k for k, v in self.gating2id.items()}
        self.max_seq_length = max_seq_length
    def _convert_example_to_feature(self, example):
        dialogue_context = " [SEP] ".join(example.context_turns + example.current_turn)

        input_id = self.src_tokenizer.encode(dialogue_context, add_special_tokens=False)
        max_length = self.max_seq_length - 2
        if len(input_id) > max_length:
            gap = len(input_id) - max_length
            input_id = input_id[gap:]

        input_id = (
            [self.src_tokenizer.cls_token_id]
            + input_id
            + [self.src_tokenizer.sep_token_id]
        )
        segment_id = [0] * len(input_id)

        target_ids = []
        gating_id = []
        if not example.label:
            example.label = []

        state = convert_state_dict(example.label)
        for slot in self.slot_meta:
            value = state.get(slot, "none")
            target_id = self.trg_tokenizer.encode(value, add_special_tokens=False) + [
                self.trg_tokenizer.sep_token_id
            ]
            target_ids.append(target_id)
            gating_id.append(self.gating2id.get(value, self.gating2id["ptr"]))
        target_ids = self.pad_ids(target_ids, self.trg_tokenizer.pad_token_id)
        return OpenVocabDSTFeature(
            example.guid, input_id, segment_id, gating_id, target_ids
        )

    def convert_examples_to_features(self, examples):
        return list(map(self._convert_example_to_feature, examples))

    def recover_state(self, gate_list, gen_list):
        assert len(gate_list) == len(self.slot_meta)
        assert len(gen_list) == len(self.slot_meta)

        recovered = []
        for slot, gate, value in zip(self.slot_meta, gate_list, gen_list):
            if self.id2gating[gate] == "none":
                continue

            if self.id2gating[gate] in ["dontcare", "yes", "no"]:
                recovered.append("%s-%s" % (slot, self.id2gating[gate]))
                continue

            token_id_list = []
            for id_ in value:
                if id_ in self.trg_tokenizer.all_special_ids:
                    break

                token_id_list.append(id_)
            value = self.trg_tokenizer.decode(token_id_list, skip_special_tokens=True)

            if value == "none":
                continue

            recovered.append("%s-%s" % (slot, value))
        return recovered

    def collate_fn(self, batch):
        guids = [b.guid for b in batch]
        input_ids = torch.LongTensor(
            self.pad_ids([b.input_id for b in batch], self.src_tokenizer.pad_token_id)
        )
        segment_ids = torch.LongTensor(
            self.pad_ids([b.segment_id for b in batch], self.src_tokenizer.pad_token_id)
        )
        input_masks = input_ids.ne(self.src_tokenizer.pad_token_id)

        gating_ids = torch.LongTensor([b.gating_id for b in batch])
        target_ids = self.pad_id_of_matrix(
            [torch.LongTensor(b.target_ids) for b in batch],
            self.trg_tokenizer.pad_token_id,
        )
        return input_ids, segment_ids, input_masks, gating_ids, target_ids, guids


class SUMBTPreprocessor(DSTPreprocessor):
    def __init__(
            self,
            slot_meta,
            src_tokenizer,
            trg_tokenizer=None,
            ontology=None,
            max_seq_length=64,
            max_turn_length=14,
    ):
        self.slot_meta = slot_meta
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer if trg_tokenizer else src_tokenizer
        self.ontology = ontology
        self.max_seq_length = max_seq_length  # N
        self.max_turn_length = max_turn_length  # M

    def _convert_example_to_feature(self, example):
        guid = example[0].guid.rsplit("-", 1)[0]  # dialogue_idx
        turns = []
        token_types = []
        labels = []
        num_turn = None
        for turn in example[: self.max_turn_length]:
            assert len(turn.current_turn) == 2
            uttrs = []
            for segment_idx, uttr in enumerate(turn.current_turn):
                token = self.src_tokenizer.encode(uttr, add_special_tokens=False)
                uttrs.append(token)

            _truncate_seq_pair(uttrs[0], uttrs[1], self.max_seq_length - 3)
            tokens = (
                    [self.src_tokenizer.cls_token_id]
                    + uttrs[0]
                    + [self.src_tokenizer.sep_token_id]
                    + uttrs[1]
                    + [self.src_tokenizer.sep_token_id]
            )
            token_type = [0] * (len(uttrs[0]) + 2) + [1] * (len(uttrs[1]) + 1)
            if len(tokens) < self.max_seq_length:
                gap = self.max_seq_length - len(tokens)
                tokens.extend([self.src_tokenizer.pad_token_id] * gap)
                token_type.extend([0] * gap)
            turns.append(tokens)
            token_types.append(token_type)
            label = []
            if turn.label:
                slot_dict = convert_state_dict(turn.label)
            else:
                slot_dict = {}
            for slot_type in self.slot_meta:
                value = slot_dict.get(slot_type, "none")
                # TODO
                # raise Exception('label_idx를 ontology에서 꺼내오는 코드를 작성하세요!')
                if value in self.ontology[slot_type]:
                    label_idx = self.ontology[slot_type].index(value)
                else:
                    label_idx = self.ontology[slot_type].index("none")
                label.append(label_idx)  # 45
            labels.append(label)  # turn length, 45
        num_turn = len(turns)
        if num_turn < self.max_turn_length:
            gap = self.max_turn_length - num_turn
            for _ in range(gap):
                dummy_turn = [self.src_tokenizer.pad_token_id] * self.max_seq_length
                turns.append(dummy_turn)
                token_types.append(dummy_turn)
                dummy_label = [-1] * len(self.slot_meta)
                labels.append(dummy_label)
        return OntologyDSTFeature(
            guid=guid,
            input_ids=turns,
            segment_ids=token_types,
            num_turn=num_turn,
            target_ids=labels,
        )

    def convert_examples_to_features(self, examples):
        return list(map(self._convert_example_to_feature, examples))

    def recover_state(self, pred_slots, num_turn):
        states = []
        for pred_slot in pred_slots[:num_turn]:
            state = []
            for s, p in zip(self.slot_meta, pred_slot):
                v = self.ontology[s][p]
                if v != "none":
                    state.append(f"{s}-{v}")
            states.append(state)
        return states

    def collate_fn(self, batch):
        # list를 batch level로 packing
        guids = [b.guid for b in batch]
        input_ids = torch.LongTensor([b.input_ids for b in batch])
        segment_ids = torch.LongTensor([b.segment_ids for b in batch])
        input_masks = input_ids.ne(self.src_tokenizer.pad_token_id)
        target_ids = torch.LongTensor([b.target_ids for b in batch])
        num_turns = [b.num_turn for b in batch]
        return input_ids, segment_ids, input_masks, target_ids, num_turns, guids
