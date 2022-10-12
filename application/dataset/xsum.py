import torch
import copy

from datasets import load_dataset
from transformers import AutoTokenizer, PegasusTokenizer

import pdb


def preprocess_function_abs(tokenizer, examples):
    document = examples['document']
    examples.update(tokenizer(document, truncation=True, padding="longest", return_tensors="pt"))
    return examples

def xsum_test_set():
    max_length = 512
    dataset = load_dataset('xsum')['test']
    visualize_columns = dataset.column_names
    tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
    dataset = dataset.map(lambda x: preprocess_function_abs(tokenizer, x), batched=False)

    setattr(dataset, 'visualize_columns', visualize_columns)
    setattr(dataset, 'input_columns', ['input_ids', 'attention_mask'])
    setattr(dataset, 'target_columns', ['highlights'])
    setattr(dataset, 'max_length', max_length)
    setattr(dataset, 'tokenizer', tokenizer)
    return dataset