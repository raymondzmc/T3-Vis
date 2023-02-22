import os
import pdb
import json
import torch
import argparse
import umap

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np
from scipy import linalg as la
from utils.head_importance import get_head_importance_pegasus
from utils.helpers import normalize, format_attention, format_attention_image
from os.path import join as pjoin
import chinese_converter



def main(args):

    device = args.device

    gold_dir = 'resources/gpt2-metaicl-commonsenseqa/out/channel-metaicl-gold'
    random_dir = 'resources/gpt2-metaicl-commonsenseqa/out/channel-metaicl-random'
    encoder_hiddens = []
    decoder_hiddens = []
    head_attributions = torch.zeros(36, 20)

    for _dir in [gold_dir, random_dir]:
        tensorized_inputs = torch.load(pjoin(gold_dir, 'tensorized_inputs.pt'))
        input_len = [x.sum().item() for x in tensorized_inputs['attention_mask']]

        hidden_states = torch.load(pjoin(gold_dir, 'hidden_states.pt'))
        input_attributions = torch.load(pjoin(gold_dir, 'input_attributions.pt'))
        head_attributions += torch.load(pjoin(gold_dir, 'head_attributions.pt'))
        _decoder_hiddens = [x[:input_len[i]].numpy() for i, x in enumerate(hidden_states)]
        decoder_hiddens.extend(_decoder_hiddens)
        encoder_hiddens.extend(np.stack([x.mean(0) for x in _decoder_hiddens]).tolist())

    # Projection Data
    fit = umap.UMAP(
        n_neighbors=5,
        min_dist=0.1,
        n_components=2,
        metric='cosine',
        verbose=True,
        low_memory=True,
        init='random',
    )

    encoder_projections = fit.fit_transform(encoder_hiddens)
    torch.save(encoder_projections, pjoin(args.output_dir, 'encoder_mean_projections.pt'))
    decoder_len = [len(h) for h in decoder_hiddens]

    decoder_all_hiddens = np.concatenate(decoder_hiddens, axis=0)
    decoder_all_projections = fit.fit_transform(decoder_all_hiddens)
    _decoder_all_projections = []
    start = 0
    for output_len in decoder_len:
        _decoder_all_projections.append(decoder_all_projections[start: start + output_len].tolist())
        start = start + output_len
    torch.save(_decoder_all_projections, pjoin(args.output_dir, 'decoder_projections.pt'))

    # pdb.set_trace()
    encoder_projection_data = {
        'x': encoder_projections[:, 0].tolist(),
        'y': encoder_projections[:, 1].tolist(),
        'ids': np.arange(len(encoder_projections)),
        'continuous': {
            'input_len': np.array(decoder_len),
            # 'avg_rouge':  np.array(all_rouge_scores),
            # 'source_entity_precision': np.array(source_entity_precision),
            # 'gt_entity_f1': np.array(gt_entity_f1),
        },
        'discrete': {
            'context_type': np.concatenate((np.zeros(500), np.ones(500))),
        }
    }
    torch.save(encoder_projection_data, os.path.join(args.output_dir, 'encoder_projection_data.pt'))
    

    # Place-holders
    head_attributions = torch.rand(36, 20)
    torch.save(head_attributions, os.path.join(args.output_dir, 'decoder_head_importance.pt'))

    with torch.no_grad():
        aggregate_decoder_attn = torch.softmax(torch.rand((36, 20, 1024, 1024), dtype=torch.float32), -1).half().numpy()

    decoder_attn_dir = os.path.join(args.output_dir, 'decoder_attentions')
    os.makedirs(decoder_attn_dir, exist_ok=True)
    decoder_attn_img = format_attention_image(aggregate_decoder_attn, output_dir=decoder_attn_dir)

    torch.save(aggregate_decoder_attn, os.path.join(args.output_dir, 'aggregate_attn.pt'))
    # torch.save(decoder_attn_img, pjoin(args.output_dir, 'aggregate_decoder_attn_img.pt'))
    
    pdb.set_trace()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPT-2 Channel CommonsenseQA')
    parser.add_argument('-hidden_aggregate_method', type=str, default='mean')
    parser.add_argument('-output_dir', type=str, default='resources/gpt2-metaicl-commonsenseqa')

    args = parser.parse_args()
    # args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = torch.device('cuda:0')
    main(args)
