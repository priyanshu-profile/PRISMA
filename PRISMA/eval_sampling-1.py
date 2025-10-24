import json
import re
import os
from math import comb
from omegaconf import DictConfig
import hydra
from sentence_transformers import SentenceTransformer, util

# Evaluation
# For calculating Evaluation metrics
import numpy as np
# import nltk
# nltk.download('wordnet')
# nltk.download('punkt')
# import evaluate
from evaluate import load
bertscore = load("bertscore")
perplexity = load("perplexity", module_type="metric")
bleu = load("bleu")
meteor = load('meteor')
rouge = load('rouge')

# Constants
ANS_RE = re.compile(r"#+ (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
SIMILARITY_THRESHOLD = 0.5

# Load sentence embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def normalize(text):
    return re.sub(r'\s+', ' ', text.strip().lower())

# def extract_answer(completion):
#     ans_lst = re.findall(ANS_RE, completion)
#     if len(ans_lst) > 0:
#         try:
#             ans = re.sub(',', '', ans_lst[-1])
#         except:
#             ans = INVALID_ANS
#     else:
#         ans = INVALID_ANS
#     return ans
def extract_answer(completion):
    try:
        # Extract raw text
        text = completion[0][1]

        # Extract response after "Final Response Response "
        match = re.search(r"Final Response Response (.+)$", text)
        if match:
            return match.group(1).strip()
        else:
            return INVALID_ANS
    except Exception as e:
        return INVALID_ANS


# def parse(lines):
#     all_ans = []
#     for line in lines:
#         ans = extract_answer(json.loads(line)[0][1])
#         # print(ans)
#         all_ans.append(ans)
#     return all_ans
def parse(lines):
    all_ans = []
    for line in lines:
        example = json.loads(line)  # This is [["prompt", "completion"]]
        ans = extract_answer(example)  # Pass the whole list, not just [0][1]
        all_ans.append(ans)
    return all_ans

def get_gold_qa(split, max_data=0):
    with open(f'gsm8k/{split}.jsonl', 'r') as f:
        lines = f.readlines()

    ans_lst = []
    question_lst = []
    solution_lst = []
    for l in lines:
        data = json.loads(l)
        ans_lst.append(data['ans'])
        question_lst.append(data['question'])
        solution_lst.append(data['answer'])
        if max_data > 0 and len(ans_lst) == max_data:
            break
    return question_lst, solution_lst, ans_lst

def eval_json(json_path, gold_ans):
    with open(json_path, 'r') as f:
        lines = f.readlines()
    pred_ans = parse(lines)

    # BERT score
    BERT_score = bertscore.compute(predictions=pred_ans, references=gold_ans, lang="en")
    # print(f'BERT_Score: ', BERT_score)
    BERT_mean = np.mean(BERT_score['f1'])
    print(f'BERT_Score_Mean: ',BERT_mean)

    # Perplexity PPL
    PPL_results = perplexity.compute(predictions=pred_ans, model_id='gpt2')
    PPL_mean = PPL_results["mean_perplexity"]
    print(f'Perplexity_Mean: ',PPL_mean)

    # BLEU Score
    BLEU_results = bleu.compute(predictions=pred_ans, references=gold_ans)
    BLEU_score=BLEU_results['bleu']
    print(f'BLEU score: ', BLEU_score)

    # Meteor score
    meteor_results = meteor.compute(predictions=pred_ans, references=gold_ans)
    meteor_1=meteor_results['meteor']
    print(f'METEOR score: ', meteor_1)

    # ROUGE-L
    rouge_results = rouge.compute(predictions=pred_ans, references=gold_ans)
    rougeL = rouge_results['rougeL']
    print(f'ROUGE_L: ', rougeL)

    cor = 0
    assert len(pred_ans) >= len(gold_ans)

    for i in range(len(gold_ans)):
        if pred_ans[i] != INVALID_ANS:
            pred_emb = embedding_model.encode(normalize(pred_ans[i]), convert_to_tensor=True)
            gold_emb = embedding_model.encode(normalize(str(gold_ans[i])), convert_to_tensor=True)
            similarity = util.cos_sim(pred_emb, gold_emb).item()
            if similarity >= SIMILARITY_THRESHOLD:
                cor += 1

    return {i: json.loads(lines[i])[0][1] for i in range(len(gold_ans))}, \
           {i: pred_ans[i] for i in range(len(gold_ans))}, \
           cor, \
           len(gold_ans)

@hydra.main(version_base=None, config_path="exp_config/llama")
def eval_diverse(cfg: DictConfig):
    exp_name = cfg.exp_name
    run_name = cfg.trainer.run_name
    split = cfg.data.split
    temperature = cfg.eval.sampling.temperature
    json_path = os.path.join('model_outputs_nego_ex_UET_ME/', exp_name, run_name, split)
    print(json_path)

    max_seed = cfg.eval.sampling.max_seed
    path_list = [os.path.join(json_path, f'seed_{idx}-t_{temperature}.json') for idx in range(max_seed)]
    path_list = [f for f in path_list if os.path.exists(f)]
    print(path_list)
    print(len(path_list))
    assert len(path_list) == max_seed

    questions, solutions, gold_ans = get_gold_qa(split)

    all_q, all_ans, new_path_list = [], [], []
    for file_path in path_list:
        res, pred, _, _ = eval_json(file_path, gold_ans)
        all_q.append(res)
        all_ans.append(pred)
        new_path_list.append(file_path)
    path_list = new_path_list

    output = [{} for _ in range(len(gold_ans))]
    for i in range(len(output)):
        output[i]['question'] = questions[i]
        output[i]['gold_ans'] = solutions[i]
        output[i]['positives'] = []
        output[i]['negatives'] = []

    for file_path_idx in range(len(path_list)):
        for idx in range(len(gold_ans)):
            pred = all_ans[file_path_idx][idx]
            gold = gold_ans[idx]
            is_positive = False
            if pred != INVALID_ANS:
                pred_emb = embedding_model.encode(normalize(pred), convert_to_tensor=True)
                gold_emb = embedding_model.encode(normalize(str(gold)), convert_to_tensor=True)
                similarity = util.cos_sim(pred_emb, gold_emb).item()
                is_positive = similarity >= SIMILARITY_THRESHOLD
            key = 'positives' if is_positive else 'negatives'
            solution = all_q[file_path_idx][idx]
            output[idx][key].append(solution)

    no_positives, no_negatives = 0, 0
    with open(f'{json_path}/{split}_dpo_data.jsonl', 'w') as f:
        for item in output:
            if len(item['positives']) == 0:
                no_positives += 1
            if len(item['negatives']) == 0:
                no_negatives += 1
            f.write(json.dumps(item) + '\n')

    if split == 'train':
        print(f'# w/o positives: {no_positives} ({no_positives / len(gold_ans) * 100:.1f}%)')
        print(f'# w/o negatives: {no_negatives} ({no_negatives / len(gold_ans) * 100:.1f}%)')
    corrects = len(gold_ans) - no_positives
    pass_at_10 = corrects / len(gold_ans)
    print(f'\nPass 1@{max_seed}: {corrects} / {len(gold_ans)} = {pass_at_10 * 100:.1f}')

    res = {}
    model_id = run_name
    if os.path.exists(f'results_ex_UET_ME/{exp_name}.json'):
        with open(f'results_ex_UET_ME/{exp_name}.json', 'r') as f:
            res = json.load(f)

    if model_id not in res:
        res[model_id] = {}
    res[model_id][f'{split}@{max_seed}'] = f'{pass_at_10 * 100:.1f}'


    with open(f'results_ex_UET_ME/{exp_name}.json', 'w') as f:
        json.dump(res, f, indent=4)

    print('-=-=-=-=-=-=-=-=-=')

if __name__ == "__main__":
    eval_diverse()
