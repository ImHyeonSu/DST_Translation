import argparse
import json
import random
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup, AutoModel
from inference import direct_output
from data_utils import WOSDataset, get_examples_from_dialogues, load_dataset, set_seed
from evaluation import _evaluation
from inference import somdst_inference, increment_path
from model.somdst import SOMDST
from model.somdst import BertEncoder
from preprocessor import SOMDSTPreprocessor

import torch.cuda.amp as amp
from loss import masked_cross_entropy_for_value

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))
if __name__ == "__main__":
    #data는 train(훈련용), 시험용(dev)으로 항상 나뉘어져있음
    #인스턴스 생성 부분
    parser = argparse.ArgumentParser()
    #인스턴스 생성 후 run_name - som dst 로 선언, data_dir - data, save - dir 은 None으로 선언
    parser.add_argument("--run_name", type=str, default="SOMDST")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default=None)
    #머신러닝을 위한 것, 전처리를 위한 인자값 설정 과정
    #한 번의 epoch는 인공 신경망에서 전체 데이터 셋에 대해 forward pass/backward pass 과정을 거친 것을 말함. 즉, 전체 데이터 셋에 대해 한 번 학습을 완료한 상태
    #epochs = 40이라면 전체 데이터를 40번 사용해서 학습을 거치는 것
    #batch size는 한 번의 batch마다 주는 데이터 샘플의 size. 여기서 batch(보통 mini-batch라고 표현)는 나눠진 데이터 셋을 뜻하며 iteration는 epoch를 나누어서 실행하는 횟수라고 생각하면 됨.    
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5) #https://en.wikipedia.org/wiki/Learning_rate , 학습속도를 말하는 것 최적점(최소점을 찾아야한다.)
    parser.add_argument("--adam_epsilon", type=float, default=1e-4) #딥러닝에서 가장 흔히 사용되는 최적화 알고리즘 #https://junklee.tistory.com/25 값은10^-8,10^-4, 10^-2 등으로 사용 보통 10^-4는 적당간 값이라함
    parser.add_argument("--max_grad_norm", type=float, default=1.0) #https://kh-kim.gitbook.io/natural-language-processing-with-pytorch/00-cover-6/05-gradient-clipping 그래디언트 클리핑은 신경망 파라미터 $\theta$ 의 norm(보통 L2 norm)을 구하고, 이 norm의 크기를 제한하는 방법,  기계번역과 같은 문제를 풀 때 학습률을 1과 같은 큰 값으로 학습에 사용할 수 있습니다.
    parser.add_argument("--epochs", type=int, default=1) #전체 데이터 셋에 대해 한 번 학습을 완료한 상태 epochs=1은 https://m.blog.naver.com/qbxlvnf11/221449297033
    parser.add_argument("--warmup_ratio", type=float, default=0.1) #학습 속도를 2e-5라고 명시하면, 첫 번째 단계인 1만 단계 내에서 학습 속도가 약 0e-5에서 2e-5로 선형적으로 증가한다는 것을 의미한다.
    parser.add_argument("--random_seed", type=int, default=42) #재현성을 위해서 random 값을 고정하여 쓰는 방법 어떤 값을 선언해줘도 상관없다.
    parser.add_argument("--max_seq_length", type=int, default=512) #BERT로 입력이 들어갈 때 입력 sequence의 max size https://www.ohsuz.dev/mrc-cookbook/1st-daily-mission#7b9e19aa-a832-48c5-8879-a316582a4469, https://blog.naver.com/PostView.naver?blogId=sooftware&logNo=221790750668
    #Encoder로는 dsksd/bert-ko-small-minimal을 사용
    parser.add_argument("--model_name_or_path", type=str, default="dsksd/bert-ko-small-minimal",) #따로 찾아봐주시면 감사하겠습니다.
    # Model Specific Argument
    parser.add_argument("--hidden_size", type=int, help="GRU의 hidden size", default=768)  #rnn등 인공신경망에서 hidden state(이전 trun의 출력 값을 입력값으로 받아들일때)size가 768 https://dhpark1212.tistory.com/entry/RNN-LSTM-%EC%84%A4%EB%AA%85-%EB%B0%8F-%EA%B5%AC%ED%98%84pytorch
    #vocab_size는 BPE((Byte-Pair Encoding Tokenization)의 단어수를 얼마로 할 것인가 이다.
    parser.add_argument("--vocab_size", type=int, default=None) #텍스트 데이터의 전체 단어 집합의 크기
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1) # 노드를 쉬게하는 퍼센트
    parser.add_argument("--proj_dim", type=int, default=None,)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5) #  실제 데이터를 입력에 넣어주는 방법을 쓴다. 이런 기법을 teacher forcing이라고 한다. http://doc.mindscale.kr/km/data_mining/08.html
    parser.add_argument("--wandb_name", type=str, default=None)
    args = parser.parse_args()

    save = False
    #def increment_path(path, exist_ok=False):
    #"""Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    #Args:
    #    path (str or pathlib.Path): f"{model_dir}/{args.name}".
    #    exist_ok (bool): whether increment path (increment if False).
    #"""
    #path = Path(path)
    #if (path.exists() and exist_ok) or (not path.exists()):
    #   return str(path)
    #else:
    #    dirs = glob.glob(f"{path}*")
    #    matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
    #    i = [int(m.groups()[0]) for m in matches if m]
    #    n = max(i) + 1 if i else 2
    #    return f"{path}{n}"

    if args.save_dir:
        save = True
        save_dir = increment_path(args.save_dir)
    # 난수 고정하는 부분
    #def set_seed(seed):
        #random.seed(seed)
        #np.random.seed(seed)
        #torch.manual_seed(seed)
        #if torch.cuda.device_count() > 0:
            #torch.cuda.manual_seed_all(seed)
    set_seed(args.random_seed)


    # Data Loading
    #json 파일을 불러온다. json 파일은 unicode로 되어있으며 1 - 관광 - 경치 좋은 부터 ~ 45 - 택시-출발지 까지 선언되어 있다.   
    #https://raisonde.tistory.com/entry/%ED%95%9C%EA%B8%80-%E2%86%94-%EC%9C%A0%EB%8B%88%EC%BD%94%EB%93%9C-%EA%B0%84%ED%8E%B8-%EB%B3%80%ED%99%98%EA%B8%B0
    slot_meta = json.load(open(f"{args.data_dir}/slot_meta.json"))  # 45개의 slot DOMAINE과 SLOT을 의미한다.
    #문장을 token단위로 자르기 위한 tokenizer는 구글에서 만든 것을 그대로 사용
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    #특정토큰단위로 만들고 토큰을 SLOT, NULL, EOS로 나눔 여기서, SLOT은 문장(토큰)의 SLOT, NULL이 SLOT의 이름을 명시하는 것이라 예상  EOS 는 한문장의 끝에 붙어서 문장과 문장을 나누는 토큰으로 사용됨 
    added_token_num = tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[SLOT]", "[NULL]", "[EOS]"]}
    )
    #print(added_token_num) = 3
    # Define Preprocessor
    print(f"preprocessing data !! ")
    processor = SOMDSTPreprocessor(slot_meta, tokenizer, max_seq_length=args.max_seq_length) #처리기
    #args.vocab_size 에다가 tokenizer로 자른 vocabsize와 토큰의 size 를 더한다. 보통 vocabsize = 32000정도이지만 여기선 35000 
    args.vocab_size = tokenizer.vocab_size + added_token_num
    #print(added_token_num) = 3
    #print(tokenizer.vocab_size) = 35000
    #print(args.vocab_size) = 35003
    #vocab_size : 텍스트 데이터의 전체 단어 집합의 크기입니다. 
    # wos 파일 불러옴
    train_data_file = f"{args.data_dir}/wos-v1_train.json"
    # 처리과정과 라벨링작업이 들어간다. data_util에 저장되어 있는 load_dataset에서 train_data, dev_data, dev_labels들이 return된다.
    # label작업이 된 것들을 각각 나눠서 dev_examplese와 train_exapmles에 넣는다.
    train_data, dev_data, dev_labels = load_dataset(train_data_file)
    """
    print(train_data[1]) #{'guid': 'wos-v1_train_00001', 'domains': ['관광'], 'dialogue': [{'role': 'user', 'text': '쇼핑을 하려는데 서울 서쪽에 있을까요?', 'state': ['관광-종류-쇼핑', '관광-지역-서울 서쪽']}, {'role': 'sys', 'text': '서울 서쪽에 쇼핑이 가능한 곳이라면 노량진 수산물 도매시장이 있습니다.'}
    print("start1") 
    print(dev_data[1]) #'guid': 'wos-v1_train_00012', 'domains': ['식당', '숙소', '관광'], 'dialogue': [{'role': 'user', 'text': '친구들이랑 갈 일식당 예약 좀 할게요. 서울 중앙쪽에 가격은 상관없으니까 맛있는 곳으로 부탁드려요.'}, {'role': 'sys', 'text': '안녕하세요? 네, 광화문역의 오동통규동동 식당 예약 가능한데요, 규동이 대표 메뉴로 평가도 좋은 곳입니다.'}, {'role': 'user', 'text': '네 예약할게요. 월요일 12시 34분에 10명이거든요. 가능한가요?'},
    print("start2")
    print(dev_labels) #{'wos-v1_train_00005-0': ['식당-지역-서울 북쪽', '식당-종류-중식당', '식당-주차 가능-yes'], 'wos-v1_train_00005-1': ['식당-가격대-dontcare', '식당-지역-서울 북쪽', '식당-종류-중식당', '식당-주차 가능-yes', '식당-주류 판매-yes'], 
    train_data,dev_data 까지 일반적인 딥러닝, 
    dev_labels 정답자체를 체크하기 위한 단순한 답지 -- joint goal accuracy, 
    """
    train_examples = get_examples_from_dialogues(
        train_data, user_first=False, dialogue_level=False
    )
    #print("starte3")
    #print(train_examples)
    dev_examples = get_examples_from_dialogues(
        dev_data, user_first=False, dialogue_level=False
    )
    #print("starte4")
    #print(dev_examples)
    # Extracting Featrues
    #    preprocessor에 들어가 있음
    #    def convert_examples_to_features(self, examples):
    #    return list(map(self._convert_example_to_feature, examples))
    train_features = processor.convert_examples_to_features(train_examples)
    #print("start5")
    #print(train_features) #이건필요없을듯
    # Model 선언
    # 이부분이 SOMDST의 실제 Encoder,decoder가 들어가져 있는부분 model-somdst.py에 저장되어있는 class 파일 가져다 씀 
    print("somdst시작됨")
    model = SOMDST(args, 5, 4, processor.op2id["update"])
    model.to(device)
    print("Model is initialized")

    train_data = WOSDataset(train_features)
    #print("start7")
    #print(train_data)
    #class WOSDataset(Dataset):
    #def __init__(self, features):
    #    self.features = features
    #    self.length = len(self.features)

    #def __len__(self):
    #    return self.length

    #def __getitem__(self, idx):
    #    return self.features[idx]
    
    #데이터 무작위로 불러오는 부분
    #데이터를 RandomSampler로 무작위로 가져온다.
    #https://hulk89.github.io/pytorch/2019/09/30/pytorch_dataset/
    train_sampler = RandomSampler(train_data)
    #print("start8")
    #print(train_sampler)<torch.utils.data.dataloader.DataLoader object at 0x7fedb410f110>
    train_loader = DataLoader(
        train_data,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        collate_fn=processor.collate_fn, ############ collate_fn batch사이즈로 묶일 경우, indices로 batch를 묶을 때 필요한 함수정의
        num_workers=4,
    )
    #print("start9")
    #print(len(train_loader))#6632, <torch.utils.data.dataloader.DataLoader object at 0x7fedb410f110>
    #각각의 파일의 총 길이 출력
    print("# train:", len(train_data))
    print("# dev:", len(dev_examples))

    # Optimizer 및 Scheduler 선언
    #epochs 선언
    #t_total train_loader의 총길이 *1
    #wramup_steps =  train_loader*0.1
    #머신러닝, 딥러닝의 성능을 좌우하는 녀석 중에 옵티마이저(Optimizer, 최적화)가 있는데  최소의 Cost로 결과를 찾아주는 것
    #그 중 AdamW라는 옵티마이저 사용 https://bbdata.tistory.com/17
    n_epochs = args.epochs
    # print(n_epochs) 1
    t_total = len(train_loader) * n_epochs
    #print(t_total) 6632   
    warmup_steps = int(t_total * args.warmup_ratio)
    #print(warmup_steps) 663
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )
    #print(scheduler) <torch.optim.lr_scheduler.LambdaLR object at 0x7fe4880b1990>
    loss_fnc_1 = masked_cross_entropy_for_value  # generation
    loss_fnc_2 = nn.CrossEntropyLoss()  # gating
    #
    # save 파일 만드는 부분
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        json.dump(
            vars(args),
            open(f"{save_dir}/exp_config.json", "w"),
            indent=2,
            ensure_ascii=False,
        )
            

    idx = 0
    best_score, best_checkpoint = 0, 0
    
    for epoch in range(n_epochs): #epoch
        #모델을 epcoh에 따라 교육시키는 부분
        model.train() 
        #머신러닝 진행도를 위한 tqdm 사용하며 진행률 을 나타내는 부분
        for step, batch in enumerate(tqdm(train_loader)): #(tqdm(train_loader))

            batch = [
                b.to(device)
                if not isinstance(b, int) and not isinstance(b, list)
                else b
                for b in batch
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
            ) = batch #>>> 데이터 batch >> somdst.py으로 들어감 

            # teacher forcing
            if (
                    args.teacher_forcing_ratio > 0.0
                    and random.random() < args.teacher_forcing_ratio
            ):
                tf = target_ids
            else:
                tf = None

            with amp.autocast(): # 연산속도 증가함수라고합니다.

                domain_scores, state_scores, gen_scores = model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    slot_positions=slot_position_ids,
                    attention_mask=input_masks,
                    max_value=max_value,
                    op_ids=gating_ids,
                    max_update=max_update,
                    teacher=tf,
                )

                # generation loss
                loss_1 = loss_fnc_1(
                    gen_scores.contiguous(), 
                    target_ids.contiguous(),
                    tokenizer.pad_token_id,
                )# contiguous()는 tensor의 transpose함수를 사용했을때 비연속적인 tensor가 될경우 학습과정에서 에러가걸릴 수 있어 함수를 연속적으로 만들어주는 역할을한다. 즉 러닝효율성을 위한 함수

                # gating loss
                loss_2 = loss_fnc_2(
                    state_scores.contiguous().view(-1, 4),
                    gating_ids.contiguous().view(-1),
                )# 이것도 마찬가지로 연속적인 tensor로 만들어주고  view를 통해 tensor 재배열 
                loss_3 = loss_fnc_2(domain_scores.view(-1, 5), domain_ids.view(-1))
                loss = loss_1 + loss_2 + loss_3

                loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) #https://kh-kim.gitbook.io/natural-language-processing-with-pytorch/00-cover-6/05-gradient-clipping
            optimizer.step() ## optimizer의 step 함수를 호출하면 매개변수가 갱신됩니다.
            scheduler.step() # scheduler의 step 함수를 호출하면 매개변수가 갱신됩니다.
            optimizer.zero_grad() #optimizer.zero_grad()를 실행하므로서 미분을 통해 얻은 기울기를 0으로 초기화합니다. 기울기를 초기화해야만 새로운 가중치 편향에 대해서 새로운 기울기를 구할 수 있습니다.

            if  step % 100 == 0:#step
                print(
                    f"[{epoch}/{n_epochs}] [{step}/{len(train_loader)}] loss: {loss.item()} gen: {loss_1.item()} gate: {loss_2.item()}, domain: {loss_3.item()}"
                )

        predictions = somdst_inference(model, dev_examples, processor, device) # return predictions #predictions[guids[0]] 반환 이는 domain-slot-value 임
        eval_result = _evaluation(predictions, dev_labels, slot_meta) # predictions - domain-slot-value 문자들, dev_labels는 #{'wos-v1_train_00005-0': ['식당-지역-서울 북쪽', '식당-종류-중식당', '식당-주차 가능-yes'], 'wos-v1_train_00005-1': ['식당-가격대-dontcare', '식당-지역-서울 북쪽', '식당-종류-중식당', '식당-주차 가능-yes', '식당-주류 판매-yes'], 
        for k, v in eval_result.items():                                #slot_meta는 ["\uad00\uad11-\uacbd\uce58 \uc88b\uc740", "등등의 관광-교육적 이런 유니코드
            print(f"{k}: {v}") #키와 밸류 값이 프린트 되고
        
        # 모델은 최대 세개만 저장되도록 설정
        if best_score < eval_result["joint_goal_accuracy"]:
            print("Update Best checkpoint!")
            best_score = eval_result["joint_goal_accuracy"]
            best_checkpoint = epoch
        
            if save:
                idx = (idx + 1) % 3
                torch.save(model.state_dict(), f"{save_dir}/best_model{idx}.bin")
                save_info = {"model_name": f"best_model{idx}.bin", "epoch": epoch, "JGA": best_score}
                json.dump(save_info, open(f"{save_dir}/best_model{idx}.json", "w"), indent=2, ensure_ascii=False)#체크
    if save:
        torch.save(model.state_dict(), f"{save_dir}/last_model.bin")
        print(f"Best checkpoint: {save_dir}/model-{best_checkpoint}.bin")
        direct_output(save_dir, model, processor)
