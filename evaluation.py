import json
import argparse
from eval_utils import DSTEvaluator


SLOT_META_PATH = './data/slot_meta.json'


def _evaluation(preds, labels, slot_meta):
    evaluator = DSTEvaluator(slot_meta)

    evaluator.init()
    assert len(preds) == len(labels) # preds- domain-slot-value의 문자의 길이와 labels의 길이가 같다면

    for k, l in labels.items(): #items 함수는 Key와 Value의 쌍을 튜플로 묶은 값을 dict_items 객체로 돌려준다
        p = preds.get(k) # 키값을 p에 넣고
        if p is None:
            raise Exception(f"{k} is not in the predictions!") # 예외발생시켜 예측안된다는것일듯
        evaluator.update(l, p) # evaluator에다가 발류와 키값 저장

    result = evaluator.compute()
    print(result)
    return result#결과값 반환


def evaluation(gt_path, pred_path):
    slot_meta = build_slot_meta(json.load(open(f"{args.data_dir}/wos-v1_train.json")))  # 45개의 slot
    slot_meta = json.load(open(SLOT_META_PATH))
    gts = json.load(open(gt_path))
    preds = json.load(open(pred_path))
    eval_result = _evaluation(preds, gts, slot_meta)
    return eval_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, required=True)
    parser.add_argument('--pred_path', type=str, required=True)
    args = parser.parse_args()
    eval_result = evaluation(args.gt_path, args.pred_path)
