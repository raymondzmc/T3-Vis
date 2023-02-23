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


    # labels = []
    # with open('resources/gpt2-metaicl-commonsenseqa/commonsense_qa_16_100_test.jsonl', 'r') as json_file:
    #     json_list = list(json_file)

    # for json_str in json_list:
    #     result = json.loads(json_str)
    #     options = result['options']
    #     output = result['output']

    #     label = np.zeros((len(options)))
    #     label[options.index(output)] = 1
    #     labels.append(label)
    # labels = np.concatenate(labels)
    

    device = args.device
    n_layers = 36
    n_heads = 20

    gold_dir = 'resources/gpt2-metaicl-commonsenseqa/out/channel-metaicl-gold'
    random_dir = 'resources/gpt2-metaicl-commonsenseqa/out/channel-metaicl-random'
    encoder_hiddens = []
    decoder_hiddens = []
    max_len = 1024
    n_examples = 0

    for _dir in [gold_dir, random_dir]:
        tensorized_inputs = torch.load(pjoin(gold_dir, 'tensorized_inputs.pt'))
        input_len = [x.sum().item() for x in tensorized_inputs['attention_mask']]
        if max(input_len) < max_len:
            max_len = max(input_len)
        n_examples += len(input_len)


    head_attributions = torch.zeros(0, n_layers, n_heads)
    input_attributions = torch.zeros(0, max_len, max_len)
    aggregate_decoder_attn = torch.zeros(0, )
    context_types = []
    all_prediction_type = []
    all_losses = []
    for _dir in [gold_dir, random_dir]:
        metadata = torch.load(pjoin(_dir, 'metadata.pt'))[:100]
        losses = torch.load(pjoin(_dir, 'losses.pt'))[:500]
        predictions = torch.load(pjoin(_dir, 'predictions.pt'))[:100]
        groundtruths = torch.load(pjoin(_dir, 'groundtruths.pt'))[:100]

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tensorized_inputs = torch.load(pjoin(gold_dir, 'tensorized_inputs.pt'))
        tokens = tokenizer.convert_ids_to_tokens(tensorized_inputs['input_ids'][0])

        prediction_types = []

        for idx, example in enumerate(metadata):
            gt, pred = np.zeros(len(example['options'])), np.zeros(len(example['options']))
            gt[example['answer'][0]] = 1
            try:
                pred_idx = example['options'].index(predictions[idx])
            except:
                pdb.set_trace()
            pred[pred_idx] = 1

            pred_types = []
            for t, y in zip(gt, pred):
                if t == 0 and y == 0:
                    pred_types.append('tn')
                elif t == 1 and y == 1:
                    pred_types.append('tp')
                elif t == 1 and y == 0:
                    pred_types.append('fn')
                elif t == 0 and y == 1:
                    pred_types.append('fp')

            assert len(pred_types) == len(gt)
            prediction_types += pred_types

        hidden_states = torch.load(pjoin(_dir, 'hidden_states.pt'))
        _decoder_hiddens = [x[:input_len[i]].numpy() for i, x in enumerate(hidden_states)]
        decoder_hiddens.extend(_decoder_hiddens)
        encoder_hiddens.extend(np.stack([x.mean(0) for x in _decoder_hiddens]).tolist())
        input_attributions = torch.cat((input_attributions, torch.load(pjoin(_dir, 'input_attributions.pt'))[:, :max_len, :max_len]), dim=0)
        head_attributions = torch.cat((head_attributions, torch.load(pjoin(_dir, 'head_attributions.pt'))), dim=0)

        context_types += ['gold' if _dir == gold_dir else 'random' for _ in range(len(hidden_states))] 

        aggregate_decoder_attn = torch.load(pjoin(_dir, 'aggregate_attentions.pt'))[:, :, :max_len, :max_len]

        all_losses += losses
        all_prediction_type += prediction_types

    # Place-holders
    torch.save(head_attributions.mean(0), os.path.join(args.output_dir, 'decoder_head_importance.pt'))
    torch.save(input_attributions, os.path.join(args.output_dir, 'input_attributions.pt'))

    decoder_attn_dir = os.path.join(args.output_dir, 'decoder_attentions')
    os.makedirs(decoder_attn_dir, exist_ok=True)
    decoder_attn_img = format_attention_image(aggregate_decoder_attn, output_dir=decoder_attn_dir)
    torch.save(aggregate_decoder_attn, os.path.join(args.output_dir, 'aggregate_attn.pt'))






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

    # decoder_all_hiddens = np.concatenate(decoder_hiddens, axis=0).astype(np.float16)
    # decoder_all_projections = fit.fit_transform(decoder_all_hiddens)
    # _decoder_all_projections = []
    # start = 0
    # for output_len in decoder_len:
    #     _decoder_all_projections.append(decoder_all_projections[start: start + output_len].tolist())
    #     start = start + output_len
    # torch.save(_decoder_all_projections, pjoin(args.output_dir, 'decoder_projections.pt'))

    encoder_projection_data = {
        'x': encoder_projections[:, 0].tolist(),
        'y': encoder_projections[:, 1].tolist(),
        'ids': np.arange(len(encoder_projections)),
        'continuous': {
            'input_len': np.array(decoder_len),
            'losses': np.array(all_losses),
            # 'avg_rouge':  np.array(all_rouge_scores),
            # 'source_entity_precision': np.array(source_entity_precision),
            # 'gt_entity_f1': np.array(gt_entity_f1),
        },
        'discrete': {
            'context_type': np.array(context_types),
            'pred_types': np.array(all_prediction_type),
        }
    }
    torch.save(encoder_projection_data, os.path.join(args.output_dir, 'encoder_projection_data.pt'))
    

    
    # torch.save(decoder_attn_img, pjoin(args.output_dir, 'aggregate_decoder_attn_img.pt'))
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPT-2 Channel CommonsenseQA')
    parser.add_argument('-hidden_aggregate_method', type=str, default='mean')
    parser.add_argument('-output_dir', type=str, default='resources/gpt2-metaicl-commonsenseqa')

    args = parser.parse_args()
    # args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = torch.device('cuda:0')
    main(args)
