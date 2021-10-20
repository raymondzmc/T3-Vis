import torch
import torch

from datasets import load_dataset
from transformers import AutoTokenizer

import pdb


def preprocess_function(tokenizer, example):
    example['tokens'] = list(map(tokenizer.convert_ids_to_tokens, example['input_ids']))
    return example


def sst2_train_set():
    dataset = load_dataset('glue', 'sst2')['train']
    visualize_columns = dataset.column_names
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = dataset.map(lambda x: preprocess_function(tokenizer, x), batched=True)

    setattr(dataset, 'visualize_columns', visualize_columns)
    return dataset