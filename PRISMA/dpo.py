import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, 
    HfArgumentParser, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    set_seed,
    )
import hydra
from omegaconf import DictConfig, OmegaConf
from trl import DPOTrainer, DPOConfig
import transformers
# from ruamel.yaml import YAML
import argparse
from dotenv import load_dotenv
load_dotenv()
#DATA_DIR = os.environ.get("DATA_DIR")
DATA_DIR = '/home/priyanshu/Prajwal/DPO-ST-P/dpo-st-copy'
from dataclasses import dataclass, field
@dataclass
class CustomTrainingArguments:
    max_target_length: int = field(default=384)

def make_dataset(data_dir):
    data_files = {
        'train': os.path.join(data_dir, 'train_dpo_processed.jsonl'),
        'eval': os.path.join(data_dir, 'eval_dpo_processed.jsonl'),
    }
    dataset = load_dataset('json', data_files=data_files)
    return dataset['train'], dataset['eval']


@hydra.main(version_base=None, config_path="exp_config/llama")
def main(cfg : DictConfig):
    parser = transformers.HfArgumentParser(DPOConfig)
    trainer_args_dict = OmegaConf.to_container(cfg.trainer)
    # parser = HfArgumentParser((TrainingArguments, CustomTrainingArguments))
    # training_args, custom_args = parser.parse_dict(trainer_args_dict)
    print(trainer_args_dict)
    training_args = parser.parse_dict(trainer_args_dict)[0]
    training_args.output_dir = os.path.join(DATA_DIR, training_args.output_dir)
    
    set_seed(training_args.seed)
    
    model_path = os.path.join(DATA_DIR, cfg.model.model_path)
    if 'llama' in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model_ref = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model_ref = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    train_dataset, eval_dataset = make_dataset(cfg.data.data_dir)

    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    dpo_trainer.train()
    # dpo_trainer.save_model(training_args.output_dir)
    # return
    if 'llama' in model_path:
        state_dict = dpo_trainer.model.state_dict()
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        dpo_trainer._save(training_args.output_dir, state_dict=cpu_state_dict) 
    else:
        dpo_trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
