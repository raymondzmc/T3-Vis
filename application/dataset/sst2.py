from datasets import load_dataset


def preprocess_function(examples):
    examples['input'] = examples['sentence']
    return examples


def sst2_train_set():
    dataset = load_dataset('glue', 'sst2')['train']
    dataset = dataset.map(preprocess_function, batched=True)
    return dataset