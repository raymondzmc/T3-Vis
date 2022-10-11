import os
import pdb
import json
import torch
import argparse
import umap

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from datasets import load_dataset
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np
from scipy import linalg as la
from utils.helpers import format_attention_image
from os.path import join as pjoin

def tensor2list(tensor, decimals=3):
    return torch.round(tensor, decimals=decimals).cpu().tolist()


def main(args):

    device = args.device
    model_name = args.model_name
    dataset = load_dataset('xsum')['test']
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    hidden_size = model.model.config.hidden_size
    max_output_len = 512
    max_input_len = model.config.max_position_embeddings


    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # id_list = []
    encoder_hiddens = np.zeros((len(dataset), hidden_size), dtype=np.float16)
    decoder_hiddens = []
    decoder_len = []

    aggregate_encoder_attn = np.zeros((
        model.config.encoder_layers,
        model.config.encoder_attention_heads,
        max_input_len,
        max_input_len,
    ), dtype=np.float16)


    aggregate_cross_attn = np.zeros((
        model.config.decoder_layers,
        model.config.decoder_attention_heads,
        max_output_len,
        max_input_len,
    ), dtype=np.float16)

    aggregate_decoder_attn = np.zeros((
        model.config.decoder_layers,
        model.config.decoder_attention_heads,
        max_output_len,
        max_output_len,
    ), dtype=np.float16)

    input_position_count = torch.zeros(max_input_len, device=device)
    output_position_count = torch.zeros(max_output_len, device=device)
    cross_attn_position_count = torch.zeros((max_output_len, max_input_len), device=device)


    # for i, example in enumerate(tqdm(dataset)):
    #     article = example['document']
    #     highlights = example['summary']
    #     _id = example['id']
    #     batch = tokenizer([article], truncation=True, padding="longest", return_tensors="pt").to(device)
    #     batch['max_length'] = max_output_len
    #     batch['return_dict_in_generate'] = True
    #     batch['output_attentions'] = True
    #     batch['output_hidden_states'] = True
    #     # batch['output_scores'] = True

    #     with torch.no_grad():
    #         output = model.generate(**batch)

    #     input_len = len(batch['input_ids'][0])
    #     output_len = len(output.cross_attentions)

    # #   output_ids = output['sequences']

    # #   # # Output vocab probability distribution for each generation step (removed to save memory)
    # #   # # scores = output.scores

    #     if batch['output_attentions']:
    #         # n_layer x batch_size x n_heads x input_len x input_len
    #         encoder_attention = output.encoder_attentions
    #         encoder_attention= torch.stack(encoder_attention).squeeze(1).to(dtype=torch.float16) # n_layer x n_heads x input_len x input_len 

    #         # output_len x n_layers x beam_size x n_heads x 1 x output_len
    #         decoder_attention = output.decoder_attentions
    #         decoder_attention = [torch.stack(attn)[:, 0].squeeze(2).to(dtype=torch.float16) for attn in decoder_attention] # output_len x n_layers x n_heads x n_tokens_prior

    #         # output_len x n_layers x beam_size x n_heads x 1 x input_size
    #         cross_attention = output.cross_attentions
    #         cross_attention = torch.stack([torch.stack(attn)[:, 0].squeeze(2) for attn in cross_attention]).to(dtype=torch.float16)
    #         cross_attention = cross_attention.transpose(0, 1).transpose(1, 2)


    #         aggregate_encoder_attn[:, :, :input_len, :input_len] += encoder_attention.cpu().numpy()
    #         aggregate_cross_attn[:, :, :output_len, :input_len] += cross_attention.cpu().numpy()
    #         for decoder_step, row in enumerate(decoder_attention):
    #             aggregate_decoder_attn[:, :, decoder_step, :row.shape[-1]] += row.cpu().numpy()

    #         input_position_count[:input_len] += 1
    #         output_position_count[:output_len] += 1
    #         cross_attn_position_count[:output_len, :input_len] += 1

    # #   # # (1 + n_layer) x batch_size x input_len x hidden_dim
    #     encoder_hidden_states = output.encoder_hidden_states[-1][0].mean(0)
    #     encoder_hiddens[i] = encoder_hidden_states.half().cpu().numpy()

    # #   # # output_len x (1 + n_layer) x beam_size x batch_size x hidden_dim
    #     decoder_hidden_states = output.decoder_hidden_states
    #     decoder_len.append(len(decoder_hidden_states))
    #     decoder_hiddens.append(torch.stack([hiddens[-1][0][0] for hiddens in decoder_hidden_states]).half().cpu().numpy())

    # torch.save(encoder_hiddens, pjoin(args.output_dir, 'encoder_hidden_states.pt'))
    # torch.save(decoder_hiddens, pjoin(args.output_dir, 'decoder_hidden_states.pt'))


    # max_input_len = len(input_position_count.nonzero(as_tuple=False))
    # max_output_len = len(output_position_count.nonzero(as_tuple=False))
    # aggregate_encoder_attn = aggregate_encoder_attn[:, :, :max_input_len, :max_input_len] / \
    #                          input_position_count.cpu().numpy()[:max_input_len]

    # aggregate_decoder_attn = aggregate_decoder_attn[:, :, :max_output_len, :max_output_len] /\
    #                          output_position_count.cpu().numpy()[:max_output_len]

    # cross_attn_position_count[cross_attn_position_count == 0] = 1
    # aggregate_cross_attn = aggregate_cross_attn[:, :, :max_output_len, :max_input_len] /\
    #                        cross_attn_position_count.cpu().numpy()[:max_output_len, :max_input_len]

    # torch.save(aggregate_encoder_attn, pjoin(args.output_dir, 'aggregate_encoder_attn.pt'))
    # torch.save(aggregate_decoder_attn, pjoin(args.output_dir, 'aggregate_decoder_attn.pt'))
    # torch.save(aggregate_cross_attn, pjoin(args.output_dir, 'aggregate_cross_attn.pt'))

    # encoder_attn_img = format_attention_image(aggregate_encoder_attn)
    # decoder_attn_img = format_attention_image(aggregate_decoder_attn)
    # cross_attn_img = format_attention_image(aggregate_cross_attn)
    # torch.save(encoder_attn_img, pjoin(args.output_dir, 'aggregate_encoder_attn_img.pt'))
    # torch.save(decoder_attn_img, pjoin(args.output_dir, 'aggregate_decoder_attn_img.pt'))
    # torch.save(cross_attn_img, pjoin(args.output_dir, 'aggregate_cross_attn_img.pt'))

    fit = umap.UMAP(
        n_neighbors=5,
        min_dist=0.1,
        n_components=2,
        metric='cosine',
        verbose=True,
        low_memory=True,
        init='random',
    )

    # encoder_mean_projections = fit.fit_transform(encoder_hiddens)
    # torch.save(encoder_mean_projections, pjoin(args.output_dir, 'encoder_mean_projections.pt'))
    encoder_projections = torch.load(pjoin(args.output_dir, 'encoder_hidden_states.pt'))
    decoder_hiddens = torch.load(pjoin(args.output_dir, 'decoder_hidden_states.pt'))
    decoder_len = [len(h) for h in decoder_hiddens]

    decoder_all_hiddens = np.concatenate(decoder_hiddens, axis=0)
    decoder_all_projections = fit.fit_transform(decoder_all_hiddens)
    _decoder_all_projections = []
    start = 0
    for output_len in decoder_len:
        _decoder_all_projections.append(decoder_all_projections[start: start + output_len])
        start = start + output_len
    torch.save(_decoder_all_projections, pjoin(args.output_dir, 'decoder_all_projections.pt'))

    with open(os.path.join(args.output_dir, 'projections.json'), 'w+') as fp:
        projection_data = {
            'encoder': encoder_projections.tolist(),
            'decoder': _decoder_all_projections.tolist(),
            'ids': id_list,
        }
        json.dump(projection_data, fp)

    pdb.set_trace()

    # encoder_mean_projections = fit.fit_transform(torch.stack([h.mean(dim=0) for h in encoder_hiddens]))
    # decoder_mean_projections = fit.fit_transform(torch.stack([h.mean(dim=0) for h in decoder_hiddens]))
    # torch.save(encoder_mean_projections, 'encoder_mean_projections.pt')
    # torch.save(decoder_mean_projections, 'decoder_mean_projections.pt')
    
    # encoder_all_projections = fit.fit_transform(encoder_hiddens)
    # torch.save(encoder_all_projections, 'encoder_all_projections.pt')
    # del encoder_hiddens, encoder_all_projections

    # decoder_hiddens = torch.load('decoder_hidden_states.pt')
    # decoder_hiddens_30D = pca.fit(decoder_hiddens)
    # decoder_all_projections = fit.fit_transform(decoder_all_hiddens)
    # torch.save(decoder_hiddens_30D, 'decoder_hidden_states_30D.pt')


    # encoder_projections = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(encoder_hiddens.numpy())
    # decoder_projections = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(torch.cat(decoder_hiddens).numpy())
    
    # Decoder hidden states grouped by examples
    

    # Save projection data
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarization')
    parser.add_argument('-dataset', type=str, default='xsum', choices=['cnndm', 'xsum'])
    parser.add_argument('-model_name', type=str, default='google/pegasus-xsum')
    parser.add_argument('-hidden_aggregate_method', type=str, default='mean')
    parser.add_argument('-output_dir', type=str, default='resources/pegasus_xsum')
    args = parser.parse_args()
    # args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = torch.device('cuda:0')
    main(args)
