import torch
import copy

from datasets import load_dataset
from transformers import AutoTokenizer

import pdb


def preprocess_function(tokenizer, example, max_length):
    example.update(tokenizer(example['sentence'], padding='max_length', max_length=max_length, truncation=True))
    example['tokens'] = list(map(tokenizer.convert_ids_to_tokens, example['input_ids']))
    example['labels'] = copy.copy(example['label'])
    return example

def get_sst_dataset(dataset_type):
    max_length = 128
    dataset = load_dataset('glue', 'sst2')[dataset_type]
    visualize_columns = dataset.column_names
    visualize_columns = ['idx', 'sentence', 'label']

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = dataset.map(lambda x: preprocess_function(tokenizer, x, max_length), batched=True)

    setattr(dataset, 'visualize_columns', visualize_columns)
    setattr(dataset, 'input_columns', ['input_ids', 'attention_mask', 'token_type_ids'])
    setattr(dataset, 'target_columns', ['labels'])
    setattr(dataset, 'max_length', max_length)
    setattr(dataset, 'tokenizer', tokenizer)
    return dataset

def sst2_train_set():
    return get_sst_dataset('train')

def sst2_validation_set():
    return get_sst_dataset('validation')

def sst2_test_set():
    return get_sst_dataset('test')