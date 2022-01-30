
from transformers import AutoModel
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
그리고 __init__()에서 모델의 구조와 동작을 정의하는 생성자를 정의합니다. 
이는 파이썬에서 객체가 갖는 속성값을 초기화하는 역할로, 
객체가 생성될 때 자동으호 호출됩니다. super() 함수를 부르면  클래스의 속성들을 가지고 초기화 됩니다. 
foward() 함수는 모델이 학습데이터를 입력받아서 forward 연산을 진행시키는 함수입니다. 
이 forward() 함수는 model 객체를 데이터와 함께 호출하면 자동으로 실행이됩니다. 
예를 들어 model이란 이름의 객체를 생성 후, model(입력 데이터)와 같은 형식으로 객체를 호출하면 자동으로 forward 연산이 수행됩니다.
'''

class SOMDST(nn.Module):
    """Some Information about SOMDST"""

    def  __init__(self, config, n_domain, n_op, update_id):
        super(SOMDST, self).__init__()
        bert = AutoModel.from_pretrained(config.model_name_or_path) # 사전학습된 모델을 가져다 쓴다. automodel
        bert.resize_token_embeddings(config.vocab_size) 
        self.encoder = BertEncoder(config, bert, 5, 4, update_id) #
        self.decoder = Decoder(
            config, self.encoder.bert.embeddings.word_embeddings.weight
        )
         #resize_token_embeddings
         #new_num_tokens 로 지정한 크기만큼의 임베딩을 생성한다. 기존의 임베딩 크기보다 new_num_tokens가 더 크다면 임베딩 제일 뒤에 새로운 임베딩들을 추가하고, 
         #더 작다면 뒤에서부터 임베딩을 삭제해서 임베딩의 크기를 줄인다. 
         #new_num_tokens를 지정하지 않는 경우는 그냥 임베딩의 포인터를 리턴한다.

    def forward(
        self,
        input_ids,
        token_type_ids,
        slot_positions,
        attention_mask,
        max_value,
        op_ids=None,
        max_update=None,
        teacher=None,
    ):
        enc_outputs = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            state_positions=slot_positions,
            attention_mask=attention_mask,
            op_ids=op_ids,
            max_update=max_update,
        )

        domain_scores, state_scores, decoder_inputs, sequence_output, pooled_output = enc_outputs

        gen_scores = self.decoder(
            input_ids,
            decoder_inputs,
            sequence_output,
            pooled_output,
            max_value,
            teacher,
        )

        return domain_scores, state_scores, gen_scores


class BertEncoder(nn.Module):
    """Some Information about BertEncoder"""

    def __init__(self, config, bert, n_domain, n_op, update_id):
        super(BertEncoder, self).__init__()
        self.hidden_size = config.hidden_size # 현재를 T시점으로 했을때  T-1의 시점의 아웃풋값을 T시점의 인풋으로 사용하기 위한 hidden state의 size, 768
        self.config = config #somdst_train.py의 args들
        self.n_op = n_op # 4  {"delete": 0, "update": 1, "dontcare": 2, "carryover": 3}
        self.bert = bert

        self.domain_classifier = nn.Linear(config.hidden_size, n_domain) #hidden_size가 인풋 사이즈 즉 768, n_domain은 아웃풋 사이즈 즉 5
        self.op_classifier = nn.Linear(config.hidden_size, n_op)# hidden_size가 인풋 사이즈 즉 768, n_domain은 아웃풋 사이즈 즉 4
        config.initializer_range = self.bert.config.initializer_range #가중치 초기화를 위한 범위? 0.02
        self.dropout = nn.Dropout(config.hidden_dropout_prob)# 임베딩, 인코더 및 풀러에서 완전히 연결된 모든 계층에 대한 드롭아웃 확률 0.1, dropout이란 쉽게 말해서, 위 그림에서 왼쪽 그림과 같은 모델에서 몇개의 연결을 끊어서, 즉 몇개의 노드를 죽이고 남은 노드들을 통해서만 훈련을 하는 것입니다. 여기서는 0.1이기 때문에 10%의 노드가 쉰다.
        self.update_id = update_id
        '''
        classifier - https://gaebom.tistory.com/7?category=1112232  nn.Linear - https://wikidocs.net/55409
        nn.Linear(input,output)란 input값에 따른 output값으로 분류하는 것을 말하며 domain도 5개 차원, op는 4개 차원으로해서 들어온 입력 값을 출력 차원에 매핑시키는 것이다. 
         분류란 입력 데이터 값을 정해진 몇 개의 부류(Class)로 대응시키는 문제이다. 
         분류 문제의 학습은 학습 데이터를 잘 분류할 수 있는 함수(수학적 함수, 규칙or패턴)를 찾는 것이다. 
         함수의 형태는 수학적 함수일 수도 있고 규칙일 수도 있다. 분류기(Classifier)란 학습된 함수를 이용하여 데이터를 분류하는 것
        ''' 
    def forward(
        self,
        input_ids,
        token_type_ids,
        state_positions,
        attention_mask,
        op_ids=None,
        max_update=None,
    ):

        outputs = self.bert(
            input_ids=input_ids, #{'input_ids': [2, 3, 31, 6265, 6672, 4073, 3249, 4034, 8732, 4292, 6722,~]}
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask, #  attention_mask 라고하면 attention에 영향을 주지 않도록 masking 처리를 한다.(뒤에 말을 예측하기 위한) masking과정에서 가려주고자 하는 토큰에 -무한대의 값을 더해주는 방식(이렇게 하면 softmax함수 후 값이 0이되므로) 
        )
        #input_ids - 토큰 즉 말을 분리한것의 고유 아이디, token_type_ids  - 입력에 문장이 2개 있다고 가정시, 2문장을 구별하는 데 사용 첫번째 문장은 0으로 매핑 두번째 문장은 1로 매핑, 이런식으로 문장구분
        #atteintion_mask - 이것은 pad 토큰은 0으로매핑  pad토큰이 아닌 것은 1로매핑

        sequence_output, pooled_output = outputs[:2]  #bert의 출력, 
        ''' https://www.tensorflow.org/text/tutorials/classify_text_with_bert, https://stackoverflow.com/questions/61331991/bert-pooled-output-is-different-from-first-vector-of-sequence-output

        여길보면pooled_output - [batch_size, H] - 보통은 전체 대화에 대한 임베딩, sequence_output - [batch_size, seq_length, H] - 토큰에 대한 contextembedding, Encoder는 입력 시퀀스를 읽고 단일 벡터를 출력하고 이 단일벡터는 Context Vector라고 불립니다.
        '''
        domain_scores = self.domain_classifier(self.dropout(pooled_output)) # dropout 시킨 대화임베딩을 classifier에 넣어 각각에 도메인에 대한 분류 점수를 만드렁냄
        #domain 예측
        # state_positions: B x J
        # state_pos: B x J x H
        state_pos = state_positions[:, :, None].expand(-1, -1, sequence_output.size(-1)) #https://runebook.dev/ko/docs/pytorch/tensors   .expand()에서 -1이란 차원의 크기를 확정시키는 것으로 state_postions의 경우 1,2차원은 값을 얕은 복사 해두고 크기고정, 및 contextvector의 크기를 고정해서 넣어준다
        # state_output: B x J x H
        #state_pos > state에 따른 텐서 크기선언
        state_output = torch.gather(sequence_output, 1, state_pos)
        # https://gaussian37.github.io/dl-pytorch-snippets/#gather-%EA%B8%B0%EB%8A%A5-%EC%82%AC%EC%9A%A9-%EB%B0%A9%EB%B2%95-1  함수는 tensor에서 인덱스를 기준으로 특정 값들을 추출하기 위해 사용됨
        #sequence_output의 1차원의 값을 state_pos의 값(위치값)에 있는 값들을 출력한다. 이것이 state_output　   
        # state_scores: B x J x n_ops( 4)                           
        state_scores = self.op_classifier(self.dropout(state_output)) # dropout 시킨 state임베딩을 classifier에 넣어 각각에 도메인에 대한 분류 점수를 만들어냄
        #>state_scores예측치 출력
        batch_size = state_scores.size(0) #state_scores의 크기만큼 batch_size를 선언
        if op_ids is None: #self.op2id = {"delete": 0, "update": 1, "dontcare": 2, "carryover": 3} 0~3이 쌓여나간 것이 op_ids
            op_ids = state_scores.view(-1, self.n_op).max(-1)[-1].view(batch_size, -1)
            #https://seungjuitmemo.tistory.com/102 인풋의 사이즈에 따라 tensor재배열,  input.view(a, b) : input의 shape을 a x b로 변경 -- array(4,4)가되어있으면 총 16항이 생긴다. 여기서 view(-1,??)라고 되어있을때 ??의 값에 따라 -1의 값이 알아서정해진다.
            # 또한 op_ids를 추정해보자면 state_scores의 값들을 재정의 한뒤 그 중 제일 큰 값들과 batch_size및 -1(알아서 알맞은 크기 생성됨)의 tensor가 선언된다.라는 의미
            # op_ids를 크기를 재수정함        
        if max_update is None:
            max_update = op_ids.eq(self.update_id).sum(-1).max().item()
            # https://sylaboratory.tistory.com/20 op.ids= update_id 인 값과 배열의 최후에 있는 값들을 더한 후 그 값을 저장한다.(pytoch-item()의 기능)
        gathered = []
        # Operation 이 Update 일 경우에 Value Generation 을 위한 Decoder Input 생성
        for b, a in zip(state_output, op_ids.eq(self.update_id)): # https://www.daleseo.com/python-zip/ state_output과 op.ids.eq(self.update_id)에 대해 토플형태로 반환하는 것
            if a.sum().item() != 0:  # J개의 Slot 중 Update가 1개 이상 존재한다면
                v = b.masked_select(a.unsqueeze(-1)).view(1, -1, self.hidden_size)  # 1 x n_update x H 
                # https://wikidocs.net/52846 unsqueeze(-1)는 제일마지막에 1차원을 추가시킴 차원을 추가시키고 1 x n_update x H 인 값을 v에 저장 
                n = v.size(1)  # num of Update 
                #이후 n에다가 한차원의 사이즈로 v의 값을 n에 집어넣음 https://stackoverflow.com/questions/52772534/what-does-the-1-mean-in-tensor-size-1/52772768
                # 이는 n_update의 값이다.
                gap = max_update - n #그만큼 max_update에서 뺴주고
                if gap > 0: 
                    # 부족한 개수만큼 패딩
                    zeros = torch.zeros(1, 1 * gap, self.hidden_size, device=input_ids.device) 
                    #https://tutorials.pytorch.kr/beginner/blitz/tensor_tutorial.html , https://codlingual.tistory.com/72   zeros를 이용해 zeros(1,gap size)만큼 tensor생성함, device는 추가예정  
                    v = torch.cat([v, zeros], 1) # [v,zeros의 사이즈만큼] [2,2]+[2,2]일시 dim=0 > [4,2] =1 일시 [2,4] 만들어줌 #https://discuss.pytorch.kr/t/torchcat%EA%B3%BC-torchstack%EC%9D%80-%EC%96%B4%EB%96%BB%EA%B2%8C-%EB%8B%A4%EB%A5%B8%EA%B0%80%EC%9A%94/26, https://sanghyu.tistory.com/85
            else:
                # Update 가 존재하지 않으면 dummy 값
                v = torch.zeros(1, max_update, self.hidden_size, device=input_ids.device) 
                #zeros를 이용해 (1,update의 size)만큼 tensor 생성, https://codlingual.tistory.com/72
            gathered.append(v)
            #무엇을하던 gathered 에 v라는tensor가 gathered에 들어감
        decoder_inputs = torch.cat(gathered)  # B x max_update x H  >>> 결국 이것이 decoder의 input으로 만들어짐
        return (
            domain_scores,
            state_scores,
            decoder_inputs,
            sequence_output,
            pooled_output.unsqueeze(0), # domain classification or generation initial input
        )


class Decoder(nn.Module):
    """Some Information about Decoder"""

    def __init__(self, config, bert_model_embedding_weights):
        super(Decoder, self).__init__()
        self.pad_idx = 0
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0) # nn.Embedding(num_embeddings-임베딩을 할 단어들의 개수. 다시 말해 단어 집합의 크기, embedding_dim - 임베딩 할 벡터의 차원, padding_idx -  지정한 인덱스에 대해서는 임베딩 벡터가 0으로 초기화되며 해당 임베딩 벡터는 훈련되지 않는다 )
        self.embed.weight = bert_model_embedding_weights # bert 가중치
        self.gru = nn.GRU(config.hidden_size, config.hidden_size, 1, batch_first=True) #gru(input.size, output.size, num_layers, inputsize의 첫번째 차원을 batch_size로 변경해주는 옵션)         https://zzsza.github.io/data/2018/03/17/pytorch-rnn/, https://wingnim.tistory.com/37, https://velog.io/@sjinu/Pytorch-Implementation-code
        self.w_gen = nn.Linear(config.hidden_size * 3, 1) #hidden_size가 인풋 사이즈 즉 768*3, 아웃풋 사이즈 즉 1
        '''
        classifier - https://gaebom.tistory.com/7?category=1112232  nn.Linear - https://wikidocs.net/55409
        nn.Linear(input,output)란 input값에 따른 output값으로 분류하는 것을 말하며 domain도 5개 차원, op는 4개 차원으로해서 들어온 입력 값을 출력 차원에 매핑시키는 것이다. 
         분류란 입력 데이터 값을 정해진 몇 개의 부류(Class)로 대응시키는 문제이다. 
         분류 문제의 학습은 학습 데이터를 잘 분류할 수 있는 함수(수학적 함수, 규칙or패턴)를 찾는 것이다. 
         함수의 형태는 수학적 함수일 수도 있고 규칙일 수도 있다. 분류기(Classifier)란 학습된 함수를 이용하여 데이터를 분류하는 것
        ''' 
        self.sigmoid = nn.Sigmoid() # https://m.blog.naver.com/rhrkdfus/221473029143 sigmoid는 출력값이 0,1 의 값을 가지며 0은 실패 1은 성공으로 출력하기 위한 함수
        self.dropout = nn.Dropout(config.hidden_dropout_prob) # dropout실행

        for n, p in self.gru.named_parameters(): 
            if "weight" in n:
                p.data.normal_(mean=0.0, std=config.initializer_range) # 평균과 표준 편차가 주어진 별도의 정규 분포에서 추출한 난수의 텐서를 반환, mean (tensor / float) : 요소 당 평균의 텐서 / 모든 분포에 대한 평균, std (Tensor / float) : 요소 별 표준 편차의 텐서 / 모든 분포에 대한 표준 편차
                #named_parameters(): >name과 parameters를 반환하는데 이중 name은 weight 가중치가 n 까지 파라미터에 대한 데이터 텐서 반환을 한다? 
    def forward(
        self, input_ids, decoder_inputs, encoder_output, hidden, max_value, teacher=None
    ):
        mask = input_ids.eq(self.pad_idx) # pad즉 mask된게 있는지 확인한다.
        batch_size, n_update, _ = decoder_inputs.size() ##hidden_size를 더미값으로 처리해준다. 

        state_in = decoder_inputs  # B x max_update x H

        # n_update x B x max_gen_lenth x vocab_size
        all_point_outputs = torch.zeros(
            n_update, batch_size, max_value, self.vocab_size, device=input_ids.device
        ) #zeros를 이용해 zeros(n_update(업데이트할 수),batch_size,max_value, vocab_size)의 tensor생성함, device는 추가예정  

        for j in range(n_update):
            w = state_in[:, j].unsqueeze(1)  # B x 1 x H >> 2차원 tensor에 1차원씩 추가시켜줌 즉  [v,zeros의 사이즈만큼]-update, (1,update의 size)-update없을경우 에다가 차원하나씩을 추가시켜줌 3차원이 되는 것
            slot_value = []
            for k in range(max_value):
                w = self.dropout(w) # dropout 실행
                _, hidden = self.gru(w, hidden)  # 1 x B x H # underscore가 쓰였음 _, 이는 더미값을 쓰이고 최종적으로 hidden에 모든 값이 들어감, inputsize - w, outputsize - hidden_size
                attn_e = torch.bmm(encoder_output, hidden.permute(1, 2, 0))  #B x max_update x H  >>> 결국 이것이 decoder의 input으로 만들어짐 == encoder_output, permute()함수는 tensor 재배열 함수  >  B x T x 1 이라는 tensor가 만들어짐 
                attn_e = attn_e.squeeze(-1).masked_fill(mask, -1e4) #squeeze에 (dim = ?)숫자를 주게되면 해당 차원에 1이 있을 경우 없애는 역할을 한다. mask인 부분을 -1e4로 바꾼다 mask는 -1000의 값이된다.
                attn_history = nn.functional.softmax(attn_e, -1)  # B x T > attn_e의 제일 끝에 차원을 softmax함수를 적용시킴 - softmax함수를 거친 값들의 총합은 1, 또한 거친 값들은 0~1사이에 값들을 가짐

                attn_v = torch.matmul(
                    hidden.squeeze(0), self.embed.weight.transpose(0, 1)
                )  # B x Vocab Size, matmul - 행렬곱,  #squeeze에 (dim = ?)숫자를 주게되면 해당 차원에 1이 있을 경우 없애는 역할을 한 hidden과, bert 가중치의 1,2차원을 변경한 값을 행렬곱함(transpose함수 기능)
                attn_vocab = nn.functional.softmax(attn_v, -1) #attn_v의 제일 끝 차원을 softmax함수 적용시킴

                context = torch.bmm(
                    attn_history.unsqueeze(1), encoder_output
                )  # B x 1 x H ------------  B x T의 차원을 하나 생성한 뒤 #B x max_update x H  - encoder_output와 행렬곱함

                p_gen = self.sigmoid(
                    self.w_gen(torch.cat([w, hidden.transpose(0, 1), context], -1))
                )  # B x 1  >>3차원의 w, hidden.transpose의 1,2차원을 변경한 값과, B x 1 x H 인 context의 제일 끝 값을 합친결과 값에  nn.liner연산을 함 결과적으로 출력값 사이즈 1
                p_gen = p_gen.squeeze(-1) #여기에다가 1차원 더추가해서 2차원이됨

                p_context_ptr = torch.zeros_like(attn_vocab, device=input_ids.device)#attn_vocab을 전부 0으로 채우기
                p_context_ptr.scatter_add(
                    1, input_ids, attn_history
                )  # Copy: B x T -> B x V - 1차원 기준으로 input_ids의 위치의 값을 attn_history의 값으로 바꾼다.
                p_final = p_gen * attn_vocab + (1 - p_gen) * p_context_ptr  # B, V
                _, w_idx = p_final.max(-1)
                slot_value.append([ww.tolist() for ww in w_idx]) #tolist() 함수는 같은 레벨(위치)에 있는 데이터 끼리 묶어준다.
                if teacher is not None:
                    w = self.embed(teacher[:, j, k]).unsqueeze(1) # 3차원에서 1번째 차원은 전체복사, j=update, k=max_value 값을 넣고 차원추가시킨다.
                else:
                    w = self.embed(w_idx).unsqueeze(1) # 3차원의 값 w_idx 에다가 차원을 1차원 추가시킨다.
                all_point_outputs[j, :, k, :] = p_final # p_final을 2,4차원 값은 복사 및 1,3차원은 j,k를 넣고
        return all_point_outputs.transpose(0, 1) #이를 1차원과 2차원을 변경하고 반환한다.