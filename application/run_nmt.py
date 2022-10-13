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
# from rouge_score import rouge_scorer
# import spacy

# def tensor2list(tensor, decimals=3):
#     return torch.round(tensor, decimals=decimals).cpu().tolist()


# def get_entities(nlp,doc):
#     doc=doc.replace('(CNN)','')
#     s = nlp(doc)
#     all_entities = [ent.text for ent in s.ents]
#     return all_entities
# def approximate_match_number(entity_list1,entity_list2):
#     match_num=0
#     for e1 in entity_list1:
#         for e2 in entity_list2:
#             if e1.lower()==e2.lower() or e1.replace('the','').strip()==e2 or e1==e2.replace('the','').strip():
#                 match_num+=1
#                 break
#     return match_num
# def compute_entity_score(text1, text2, nlp):
#     text1_entities = set(get_entities(nlp,text1))
#     # text1_entity_num.append(len(text1_entities))

#     # get the entities in the ground-truth summary
#     text2_entities = set(get_entities(nlp,text2))
#     # text2_entity_num.append(len(text2_entities))

#     num_intersect=approximate_match_number(text1_entities,text2_entities)
#     r= num_intersect/float(len(text2_entities)) if len(text2_entities)!=0 else 0
#     p= num_intersect/float(len(text1_entities)) if len(text1_entities)!=0 else 0
#     f = 2*p*r/(p+r) if r!=0 or p!=0 else 0
#     return p, r, f




def main(args):

    device = args.device
    model_name = args.model_name
    # dataset = load_dataset('xsum')['test']

    model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-en-zh').to(device)
    tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-zh')
    with open('../resources/opus-2020-07-17.test.txt', 'r') as f:
        lines = f.readlines()

    dataset = []
    for line in lines:
        if line.startswith('>>yue_Hant<<'):
            dataset.append(line)

    # pdb.set_trace()
    # model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    hidden_size = model.model.config.hidden_size
    max_output_len = 512
    max_input_len = model.config.max_position_embeddings


    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # id_list = []
    encoder_hiddens = np.zeros((len(dataset), hidden_size), dtype=np.float16)
    decoder_hiddens = []
    decoder_len = []

    # aggregate_encoder_attn = np.zeros((
    #     model.config.encoder_layers,
    #     model.config.encoder_attention_heads,
    #     max_input_len,
    #     max_input_len,
    # ), dtype=np.float16)


    # aggregate_cross_attn = np.zeros((
    #     model.config.decoder_layers,
    #     model.config.decoder_attention_heads,
    #     max_output_len,
    #     max_input_len,
    # ), dtype=np.float16)

    # aggregate_decoder_attn = np.zeros((
    #     model.config.decoder_layers,
    #     model.config.decoder_attention_heads,
    #     max_output_len,
    #     max_output_len,
    # ), dtype=np.float16)

    # input_position_count = torch.zeros(max_input_len, device=device)
    # output_position_count = torch.zeros(max_output_len, device=device)
    # cross_attn_position_count = torch.zeros((max_output_len, max_input_len), device=device)

    # scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    # nlp = spacy.load("en_core_web_sm")



    encoder_len = [] 
    # all_rouge_scores = []
    # source_entity_precision = []
    # gt_entity_f1 = []
    # encoder_head_importance = np.zeros((16, 16))
    # decoder_head_importance = np.zeros((16, 16, 2))

    # Neede for importance
    model.eval()

    num_steps = 0

    for i, example in enumerate(tqdm(dataset)):
        # article = example['document']
        # highlights = example['summary']
        _id = i
        batch = tokenizer([example], truncation=True, padding="longest", return_tensors="pt").to(device)
        batch['max_length'] = max_output_len
        batch['return_dict_in_generate'] = True
        batch['output_attentions'] = False
        batch['output_hidden_states'] = True
        batch['output_scores'] = True

        with torch.no_grad():
            output = model.generate(**batch)

        prediction = tokenizer.decode(output['sequences'][0])
        chinese_converter.to_simplified(prediction)


        # avg_score = 0.
        # rouge_results = scorer.score(highlights, prediction)
        # for metric, score in rouge_results.items():
        #     avg_score += score.fmeasure
        # all_rouge_scores.append(avg_score / 3)

        # source_entity_precision.append(compute_entity_score(prediction, article, nlp)[0])
        # gt_entity_f1.append(compute_entity_score(prediction, highlights, nlp)[-1])

        beam_indices = output['beam_indices'][0, :-1]

        input_len = len(batch.input_ids[0])
        encoder_len.append(input_len)
        output_len = len(beam_indices)
        # decoder_len.append(output_len)

            # n_layer x batch_size x n_heads x input_len x input_len

    #   # # (1 + n_layer) x batch_size x input_len x hidden_dim
        encoder_hidden_states = output.encoder_hidden_states[-1][0].mean(0)
        encoder_hiddens[i] = encoder_hidden_states.half().cpu().numpy()

    #   # # output_len x (1 + n_layer) x beam_size x batch_size x hidden_dim
        decoder_hidden_states = output.decoder_hidden_states
        decoder_hidden_states = torch.stack([hidden[-1][:, 0] for hidden in decoder_hidden_states])[:output_len]
        beam_search_hidden_states = []
        for step, hidden in enumerate(decoder_hidden_states):
            beam_idx = beam_indices[step].item()
            beam_search_hidden_states.append(hidden[beam_idx])

        decoder_hiddens.append(torch.stack(beam_search_hidden_states).half().cpu().numpy())

        # head_importance = get_head_importance_pegasus(model)
        # encoder_head_importance += head_importance['encoder']
        # decoder_head_importance[:, :, 0] += head_importance['decoder']
        # decoder_head_importance[:, :, 1] += head_importance['cross']

    #     if batch['output_attentions']:
    #         output_len += 1
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
        num_steps += 1


    # encoder_head_importance /= num_steps
    # decoder_head_importance /= num_steps
    # torch.save(encoder_head_importance, pjoin(args.output_dir, 'encoder_head_importance.pt'))
    # torch.save(decoder_head_importance, pjoin(args.output_dir, 'decoder_head_importance.pt'))

    # max_input_len = len(input_position_count.nonzero(as_tuple=False))
    # max_output_len = len(output_position_count.nonzero(as_tuple=False))
    # aggregate_encoder_attn = aggregate_encoder_attn[:, :, :max_input_len, :max_input_len] / \
                             # input_position_count.cpu().numpy()[:max_input_len]

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

    torch.save(encoder_hiddens, pjoin(args.output_dir, 'encoder_hidden_states.pt'))
    torch.save(decoder_hiddens, pjoin(args.output_dir, 'decoder_hidden_states.pt'))

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
            'input_len': np.array(encoder_len),
            # 'avg_rouge':  np.array(all_rouge_scores),
            # 'source_entity_precision': np.array(source_entity_precision),
            # 'gt_entity_f1': np.array(gt_entity_f1),
        }
    }
    torch.save(encoder_projection_data, os.path.join(args.output_dir, 'encoder_projection_data.pt'))
    # with open(os.path.join(args.output_dir, 'projections.json'), 'w+') as fp:
    #     json.dump(projection_data, fp)

    # pdb.set_trace()

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
    parser.add_argument('-output_dir', type=str, default='resources/nmt')

    args = parser.parse_args()
    # args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = torch.device('cuda:0')
    main(args)
