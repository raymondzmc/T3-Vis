import os
import pdb
import json
import torch
import argparse
import umap

from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np
from os.path import join as pjoin
from rouge_score import rouge_scorer
import spacy

from run_abs_sum import tensor2list, get_entities, approximate_match_number, compute_entity_score
from utils.head_importance import get_head_importance_gptneo


def main(args):
    device = args.device
    model_name = args.model_name
    dataset = load_dataset('xsum')['test']
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)
    # model = GPT2Model.from_pretrained(model_name).to(device)

    max_output_len = 2048
    max_input_len = model.config.max_position_embeddings

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = 'left'

    all_input_len = []
    hiddens = []
    hiddens_len = []

    aggregate_attn = np.zeros((
        model.config.num_layers,
        model.config.num_heads,
        max_output_len,
        max_output_len,
    ), dtype=np.float16)

    input_position_count = torch.zeros(max_input_len, device=device)
    output_position_count = torch.zeros(max_output_len, device=device)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    nlp = spacy.load("en_core_web_sm")

    all_rouge_scores = []
    source_entity_precision = []
    gt_entity_f1 = []

    head_importance = np.zeros((model.config.num_heads, model.config.num_layers))

    model.eval()

    num_steps = 0
    for i, example in enumerate(tqdm(dataset)):
        article = example['document']
        highlights = example['summary']
        _id = example['id']
        batch = tokenizer([article], truncation=True, padding="longest", return_tensors="pt").to(device)
        batch['max_length'] = max_output_len
        batch['return_dict_in_generate'] = True
        batch['output_attentions'] = False
        batch['output_hidden_states'] = True
        batch['output_scores'] = True
        batch['num_beams'] = 2

        # pdb.set_trace()
        with torch.no_grad():
            output = model.generate(**batch)

        prediction = tokenizer.decode(output['sequences'][0])
        avg_score = 0.
        rouge_results = scorer.score(highlights, prediction)
        for metric, score in rouge_results.items():
            avg_score += score.fmeasure
        all_rouge_scores.append(avg_score / 3)

        source_entity_precision.append(compute_entity_score(prediction, article, nlp)[0])
        gt_entity_f1.append(compute_entity_score(prediction, highlights, nlp)[-1])

        input_len = len(batch.input_ids[0])
        all_input_len.append(input_len)
        # pdb.set_trace()
        beam_indices = output['beam_indices'][0, :-1]
        output_len = len(beam_indices)
        hiddens_len.append(output_len)

        # output_len x (1 + n_layer) x beam_size x batch_size x hidden_dim
        hidden_states = output.hidden_states
        # pdb.set_trace()
        hidden_states = torch.stack([hidden[-1][:, 0] for hidden in hidden_states])[:output_len]

        beam_search_hidden_states = []
        for step, hidden in enumerate(hidden_states):
            beam_idx = beam_indices[step].item()
            beam_search_hidden_states.append(hidden[beam_idx])
        # pdb.set_trace()
        hiddens.append(torch.stack(beam_search_hidden_states).half().cpu().numpy())
        head_importance += get_head_importance_gptneo(model)

        if batch['output_attentions']:
            output_len += 1

            # output_len x n_layers x beam_size x n_heads x 1 x output_len
            self_attention = output.attentions
            self_attention = [torch.stack(attn)[:, 0].squeeze(2).to(dtype=torch.float16) for attn in self_attention] # output_len x n_layers x n_heads x n_tokens_prior

            for decoder_step, row in enumerate(self_attention):
                aggregate_attn[:, :, decoder_step, :row.shape[-1]] += row.cpu().numpy()

            input_position_count[:input_len] += 1
            output_position_count[:output_len] += 1

        num_steps += 1

    head_importance /= num_steps
    torch.save(head_importance, pjoin(args.output_dir, 'head_importance.pt'))
    # pdb.set_trace()

    torch.save(hiddens, pjoin(args.output_dir, 'hidden_states.pt'))
    # pdb.set_trace()

    umap_projector = umap.UMAP(
        n_neighbors=5,
        min_dist=0.1,
        n_components=2,
        metric='cosine',
        verbose=True,
        low_memory=True,
        init='random',
    )

    hiddens_len = [len(h) for h in hiddens]
    all_hiddens = np.concatenate(hiddens, axis=0)
    all_projections = umap_projector.fit_transform(all_hiddens)
    accumulated_all_projections = []
    start = 0
    for output_len in hiddens_len:
        accumulated_all_projections.append(all_projections[start: start + output_len].tolist())
        start = start + output_len

    # pdb.set_trace()
    torch.save(accumulated_all_projections, pjoin(args.output_dir, 'projections.pt'))

    # pdb.set_trace()
    input_projection_data = {
        'continuous': {
            'input_len': np.array(all_input_len),
            'avg_rouge':  np.array(all_rouge_scores),
            'source_entity_precision': np.array(source_entity_precision),
            'gt_entity_f1': np.array(gt_entity_f1),
        }
    }
    torch.save(input_projection_data, os.path.join(args.output_dir, 'input_projection_data.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generation')
    parser.add_argument('-dataset', type=str, default='xsum', choices=['cnndm', 'xsum'])
    parser.add_argument('-model_name', type=str, default='EleutherAI/gpt-neo-125M')
    parser.add_argument('-hidden_aggregate_method', type=str, default='mean')
    parser.add_argument('-output_dir', type=str, default='resources/gpt-neo')

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # args.device = torch.device('cuda:0')
    main(args)
