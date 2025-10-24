import json
import re
import os
from datasets import load_dataset
import hydra
from omegaconf import DictConfig
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
ANS_RE = re.compile(r"#+ (.+)")  # more general than float-only
INVALID_ANS = "[invalid]"
SIMILARITY_THRESHOLD = 0.5

# Sentence Transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())

# def extract_answer(completion):
#     match = re.findall(ANS_RE, completion)
#     return match[-1].strip() if match else INVALID_ANS
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
#     return [extract_answer(json.loads(line)[0][1]) for line in lines]
def parse(lines):
    all_ans = []
    for line in lines:
        example = json.loads(line)  # This is [["prompt", "completion"]]
        ans = extract_answer(example)  # Pass the whole list, not just [0][1]
        all_ans.append(ans)
    return all_ans

@hydra.main(version_base=None, config_path="exp_config/llama")
def eval_json(cfg: DictConfig):
    exp_name = cfg.exp_name
    run_name = cfg.trainer.run_name
    split = cfg.data.split

    # Path to greedy decoded predictions
    json_path = os.path.join("model_outputs_nego_ex_UET_ME", exp_name, run_name, split, "greedy_decode.json")
    print(json_path)
    with open(json_path, "r") as f:
        lines = f.readlines()
    pred_ans = parse(lines)
    # print(pred_ans)

    # Load gold answers
    gold_path = os.path.join("gsm8k", f"{split}.jsonl")
    gold_data = load_dataset("json", data_files=gold_path)["train"]
    gold_ans = [normalize(str(entry["ans"])) for entry in gold_data]

    # Evaluate
    assert len(pred_ans) >= len(gold_ans)

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

    # Evaluating correct and incorrect predicted responses based on cosine similarity 
    correct = 0
    for i in range(len(gold_ans)):
        pred = pred_ans[i]
        gold = gold_ans[i]
        if pred == INVALID_ANS:
            continue
        pred_emb = embedding_model.encode(normalize(pred), convert_to_tensor=True)
        gold_emb = embedding_model.encode(gold, convert_to_tensor=True)
        similarity = util.cos_sim(pred_emb, gold_emb).item()
        if similarity >= SIMILARITY_THRESHOLD:
            correct += 1

    accuracy = correct / len(gold_ans) * 100
    print(f"### {run_name}/{split} — Accuracy: {correct}/{len(gold_ans)} = {accuracy:.1f}%")

    # Log to results
    results_file = f"results_ex_UET_ME/{exp_name}.json"
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            res = json.load(f)
    else:
        res = {}

    if run_name not in res:
        res[run_name] = {}
    res[run_name][f"{split}@1"] = {
        "accuracy": round(accuracy, 2),
        "BERTScore": round(BERT_mean, 4),
        "Perplexity": round(PPL_mean, 4),
        "BLEU": round(BLEU_score, 4),
        "METEOR": round(meteor_1, 4),
        "ROUGE-L": round(rougeL, 4)}

    os.makedirs("results_ex_UET_ME", exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(res, f, indent=4)

    return pred_ans
    print(f'BERT_Score_Mean: ',BERT_mean)

if __name__ == "__main__":
    eval_json()
