import torch
import copy

from datasets import load_dataset
from transformers import AutoTokenizer, PegasusTokenizer

import pdb

def clean(x):
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)

def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}

def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)

def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)



def preprocess_function(tokenizer, example):
    max_src_nsents = 100
    max_length = 512
   
    example.update(tokenizer(example['article'], padding='max_length', max_length=max_length, truncation=True))

    sent_labels = greedy_selection(example['article'][:args.max_src_nsents], tgt, 3)
    example['tokens'] = list(map(tokenizer.convert_ids_to_tokens, example['input_ids']))
    example['labels'] = copy.copy(example['label'])

    # Use separator tokens to determine the length of each sentence
    _segs = [-1] + [i for (i, t) in enumerate(example['input_ids']) if t == tokenizer.vocab['[SEP]']]
    segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]

    # Create binary masks as segment (sentence) embeddings 
    segments_ids = []
    for i, s in enumerate(segs):
        if (i % 2 == 0):
            segments_ids += s * [0]
        else:
            segments_ids += s * [1]

    cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == tokenizer.vocab['[CLS]']]
    sent_labels = sent_labels[:len(cls_ids)]

    example['cls_idx'] = cls_ids
    example['seg_ids'] = segments_ids
    
    return example

def preprocess_function_abs(tokenizer, examples):
    article = examples['article']
    examples.update(tokenizer(article, truncation=True, padding="longest", return_tensors="pt"))
    return examples



def cnndm_train_set():
    max_length = 512
    dataset = load_dataset('cnn_dailymail', '3.0.0', ignore_verifications=True)['train']
    visualize_columns = dataset.column_names
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = dataset.map(lambda x: preprocess_function(tokenizer, x, max_length), batched=True)

    setattr(dataset, 'visualize_columns', visualize_columns)
    setattr(dataset, 'input_columns', ['input_ids', 'attention_mask', 'token_type_ids'])
    setattr(dataset, 'target_columns', ['labels'])
    setattr(dataset, 'max_length', max_length)
    setattr(dataset, 'tokenizer', tokenizer)
    return dataset

def cnndm_test_set():
    max_length = 1024
    dataset = load_dataset('cnn_dailymail', '3.0.0', ignore_verifications=True)['test']
    visualize_columns = dataset.column_names
    tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-cnn_dailymail')
    dataset = dataset.map(lambda x: preprocess_function_abs(tokenizer, x), batched=False)

    setattr(dataset, 'visualize_columns', visualize_columns)
    setattr(dataset, 'input_columns', ['input_ids', 'attention_mask'])
    setattr(dataset, 'target_columns', ['highlights'])
    setattr(dataset, 'max_length', max_length)
    setattr(dataset, 'tokenizer', tokenizer)
    return dataset




