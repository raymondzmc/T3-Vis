import torch
import copy

from datasets import load_dataset
from transformers import AutoTokenizer

import pdb



class OpusDataset(object):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return self.data

    def __getitem__(self, index):
        return self.data[index]


def preprocess_function_abs(tokenizer, doc):
    return tokenizer(doc, truncation=True, padding="longest", return_tensors="pt")

def opus_test_set():
    max_length = 512
    with open('../resources/opus-2020-07-17.test.txt', 'r') as f:
        lines = f.readlines()

    dataset = []
    for line in lines:
        if line.startswith('>>yue_Hant<<'):
            dataset.append(line)

    # visualize_columns = dataset.column_names
    tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-zh')
    data = [preprocess_function_abs(tokenizer, x) for x in dataset]

    dataset = OpusDataset(data)
    # setattr(dataset, 'visualize_columns', visualize_columns)
    setattr(dataset, 'input_columns', ['input_ids', 'attention_mask'])
    # setattr(dataset, 'target_columns', ['highlights'])
    setattr(dataset, 'max_length', max_length)
    setattr(dataset, 'tokenizer', tokenizer)
    return dataset