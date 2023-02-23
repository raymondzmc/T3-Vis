import os
import json
import torch
import flask
import pickle
import argparse
import collections

import numpy as np
from os.path import join as pjoin

from utils.input_saliency import register_hooks, compute_input_saliency
from utils.head_importance import get_head_importance_pegasus
from utils.helpers import normalize, format_attention, format_attention_image
import chinese_converter

import pdb

################### Global Objects ###################
app = flask.Flask(__name__, template_folder='templates')


class T3_Visualization(object):
    """
    Visualization class for tracking the backend state as well as methods for processing data
    """

    def __init__(self, args):

        self.args = args
        try:
            exec(f"from dataset import {args.dataset}")
        except ImportError:
            print(
                f"\nWarning: Cannot import function \"{args.dataset}\" from directory \"dataset\", please ensure the function is defined in this file!")

        if args.device == None:
            self.device = 'cuda' if torch.cuda.is_available() and args.device else 'cpu'
        else:
            self.device = args.device
        print(self.device)

        self.resource_dir = args.resource_dir
        self.checkpoint_dirs = [subdir for subdir in os.listdir(self.resource_dir) if
                                os.path.isdir(pjoin(self.resource_dir, subdir))]
        self.curr_checkpoint_dir = None

        self.filter_paddings = args.filter_paddings

        self.model_type = args.model_type

        # TO DO: Set the pretrained model as an attribute of the model to get called

        self.init_model(args.model)
        self.dataset = eval(f"{args.dataset}()")
        self.num_hidden_layers = self.model.num_hidden_layers
        self.num_attention_heads = self.model.num_attention_heads
        print(self.num_hidden_layers, self.num_attention_heads)
        self.pruned_heads = collections.defaultdict(list)

        # self.table_headings = tuple(self.dataset.visualize_columns)
        # self.table_content = [{col_name: row[col_name] for col_name in self.table_headings} for i, row in
        #                       enumerate(self.dataset) if (args.n_examples and i < args.n_examples)]

        # Temp
        self.decoder_projections = torch.load(pjoin(self.resource_dir, 'decoder_projections.pt'))
        self.encoder_attentions = None
        self.decoder_attentions = None
        self.cross_attentions = None

        if os.path.exists(pjoin(self.resource_dir, 'input_attributions.pt')):
            self.input_attributions = torch.load(pjoin(self.resource_dir, 'input_attributions.pt')).float()
            self.input_attributions[torch.isnan(self.input_attributions)] = 1
            # self.input_attributions[torch.isneginf(self.input_attributions)] = -65504

    def init_model(self, model_name):
        try:
            exec(f"from models import {args.model}")
        except ImportError:
            print(
                f"\nWarning: Cannot import function \"{args.model}\" from directory \"models\", please ensure the function is defined in this file!")

        self.model = eval(f"{model_name}()")
        self.model.requires_grad_(True)
        self.model.to(self.device)

        if self.curr_checkpoint_dir != None:
            self.model.load_state_dict(torch.load(pjoin(curr_checkpoint_dir, 'model.pt')))

    def prune_heads(self, heads_to_prune):
        if heads_to_prune == {} and self.pruned_heads != {}:
            self.init_model(self.args.model)
        else:
            for layer, heads in heads_to_prune.items():
                pruned_heads = self.pruned_heads[layer]
                heads_to_prune[layer] = list(set(heads) - set(pruned_heads))
                self.pruned_heads[layer] = set(heads + pruned_heads)
            self.model.prune_heads(heads_to_prune)

    def get_attentions(self, attn_type, layer, head):
        results = {}

        if attn_type == 'encoder':
            results['encoder_attentions'] = self.encoder_attentions[layer][head]
        elif attn_type == 'decoder':
            results['cross_attentions'] = self.cross_attentions[layer][head]
            results['decoder_attentions'] = [a[layer][head] for a in self.decoder_attentions]

        return results

    def evaluate_example(self, idx, encoder_head=None, decoder_head=None):
        """
        Perform inference on a single data example,
        return output logits, attention scores, saliency maps along with other attributes 
        """
        results = {}

        # print(idx)
        idx = 530
        # self.model.zero_grad()
        # register_hooks(self.model)
        example = self.dataset[idx]

        # for col in self.dataset.target_columns:
        #     example[col] = torch.tensor(example[col])

        model_input = {k: v for (k, v) in example.items() if k in self.dataset.input_columns}
        input_len = example['attention_mask'].sum().item()

        if self.filter_paddings:
            for input_key in self.dataset.input_columns:
                if len(model_input[input_key].shape) == 1:
                    model_input[input_key] = model_input[input_key][:input_len].to(self.device)
                elif len(model_input[input_key].shape) == 2:
                    model_input[input_key] = model_input[input_key][:, :input_len].to(self.device)
        # print(model_input['input_ids'].shape)
        # results['decoder_projections'] = {}
        # results['decoder_projections']['x'] = np.array(self.decoder_projections[idx])[:, 0].tolist()
        # results['decoder_projections']['y'] = np.array(self.decoder_projections[idx])[:, 1].tolist()
        # batch['output_hidden_states'] = True

        self.model.eval()
        with torch.no_grad():

            if self.model_type == 'encoder-decoder':
                model_input['max_length'] = 512
                model_input['return_dict_in_generate'] = True
                model_input['output_attentions'] = True
                model_input['output_scores'] = True
                output = self.model.generate(**model_input)
            else:
                # model_input = {k: v[:ma]}
                model_input['output_attentions'] = True
                output = self.model(**model_input)

        if hasattr(output, 'loss'):
            results['loss'] = output.loss.item() if type(output.loss) == torch.Tensor else output.loss
        else:
            results['loss'] = 0
        # output_ids = output['sequences']
        # logits = output['logits']

        # output['loss'].backward(retain_graph=True)
        # results['loss'] = output['loss'].item()
        # results['input_saliency'] = input_saliency
        # results['output'] = output['sequences'].squeeze(0).tolist()
        # results['attn'] = format_attention(output['attentions'], self.num_attention_heads, self.pruned_heads)
        # results['attn_pattern'] = format_attention_image(np.array(results['attn']))
        # results['head_importance'] = normalize(get_taylor_importance(self.model)).tolist()
        if self.model_type == 'encoder-decoder':
            self.encoder_attentions = (torch.stack(output['encoder_attentions']).squeeze(1) * 100).byte().cpu().tolist()
            self.cross_attentions = torch.stack(
                [(torch.stack(a)[:, 0].squeeze(2) * 100).byte().cpu() for a in output['cross_attentions']]
            ).transpose(0,1).transpose(1, 2).tolist()
            self.decoder_attentions = [(torch.stack(a)[:, 0].squeeze(2) * 100).byte().cpu().tolist() for a in output['decoder_attentions']]
        else:
            self.decoder_attentions = [(a.squeeze() * 100).byte().cpu().tolist() for a in output['attentions']]


        if self.model_type == 'encoder-decoder':
            if encoder_head:
                layer, head = int(encoder_head[0]) - 1, int(encoder_head[1]) - 1
                results['encoder_attentions'] = self.encoder_attentions[layer - 1][head - 1]

            if decoder_head:

                layer, head = int(decoder_head[0]) - 1, int(decoder_head[1]) - 1
                if self.model_type == 'encoder-decoder':
                    results['cross_attentions'] = self.cross_attentions[layer - 1][head - 1]
                results['decoder_attentions'] = [a[layer - 1][head - 1] for a in self.decoder_attentions]
        else:
            if encoder_head:
                layer, head = int(encoder_head[0]) - 1, int(encoder_head[1]) - 1
                results['encoder_attentions'] = self.decoder_attentions[layer - 1][head - 1]

        # head_importance = get_head_importance_pegasus(self.model)
        # Note: Still work to do in compute_ig_importance_score()
        # head_importance_ig = get_head_importance_pegasus(self.model, 'ig', model_input['input_ids'].squeeze(0))

        results['input_tokens'] = self.dataset.tokenizer.convert_ids_to_tokens(example['input_ids'].squeeze(0))[:input_len]
        results['input_tokens'] = [tok.replace('Ġ', '') for tok in results['input_tokens']]

        if self.model_type == 'encoder-decoder':
            results['output_tokens'] = self.dataset.tokenizer.convert_ids_to_tokens(output['sequences'].squeeze(0))
            results['output_tokens'] = [chinese_converter.to_simplified(x) for x in results['output_tokens']]
            output_len = len(results['output_tokens']) - 1
        else:
            results['output_tokens'] = []
            output_len = input_len

        output_projection = {}
        output_projection['ids'] = np.arange(len(self.decoder_projections[idx])).tolist()[:output_len]
        output_projection['x'] = np.array(self.decoder_projections[idx])[:, 0].tolist()[:output_len]
        output_projection['y'] = np.array(self.decoder_projections[idx])[:, 1].tolist()[:output_len]
        output_projection['domain'] = (min(min(output_projection['x']), min(output_projection['y'])), max(max(output_projection['x']), max(output_projection['y'])))
        output_projection['continuous'] = [{
            'name': 'ids',
            'values': np.arange(len(self.decoder_projections[idx])).tolist()[:output_len],
            'max': output_len - 1,
            'min': 0,
        }]
        output_projection['discrete'] = []
        results['output_projections'] = output_projection
        # Head Importance and attribution
        # results['encoder_head_importance'] = normalize(head_importance['encoder']).tolist()
        # results['decoder_head_importance'] = [normalize(head_importance['decoder']).tolist(),
        #                                       normalize(head_importance['cross']).tolist()]

        # Note: Still work to do in compute_input_saliency()
        # input_saliency = compute_input_saliency(self.model, len(example['input_ids']), example['input_ids'],
        #                                         output['sequences'])


        if self.model_type == 'encoder-decoder':
            model_input['interpretation'] = True
            # model_input['best_indices'] = output['sequences'][0].tolist()
            # model_input['best_beam_indices'] = output['beam_indices'][0, :-1].tolist()
            # model_input['input_indices'] = model_input['input_ids'][0].tolist()
            # Dummy values for attribution
            results['attributions'] = [{
                'input': np.random.rand(len(results['input_tokens'])).tolist() if step > 0 else [],
                'output': np.random.rand(step).tolist(),
            } for step in range(len(results['output_tokens']))]
            
            # self.model.train()
            # output = self.model.generate(**model_input)
            # results['attributions'] = output.saliency
            # results['attributions'].insert(0, {'input': [],'output': [],})
        else:
            results['attributions']  = self.input_attributions[idx][:input_len, :input_len]
            for i in range(len( results['attributions'] )):
                results['attributions'][i][i:] = 0
                if i > 1:
                    if not results['attributions'][i][:i].sum().item() == 0:
                        results['attributions'][i][:i] = normalize(results['attributions'][i][:i]) * 80
                elif i == 1:
                    results['attributions'][i][:i] = 1
            results['attributions'] = results['attributions'].tolist()
            # [x]
            # results['attributions'] = normalize(self.input_attributions[idx][:input_len, :input_len] ** 5).tolist()

        
        return results


# redirect requests from root to index.html
@app.route('/')
def index():
    return flask.render_template(
        'index.html',
        # headings=t3_vis.table_headings,
        # content=t3_vis.table_content,
        checkpoints=t3_vis.checkpoint_dirs,
        num_hidden_layers=range(t3_vis.num_hidden_layers + 1),
    )


def check_resource_dir(resource_dir):
    # Check all subdirectories in the "resource_dir" for data files needed for visualization
    sub_dirs = os.listdir(resource_dir)
    # print(sub_dirs)
    required_files = [
        # (File Name, File Information, Required)
        ('aggregate_attn.pt', 'Aggregated Attention Matrices', True),
        ('head_importance.pt', 'Head Importance Scores', True),
        ('projection_data.pt', 'Dataset Projection Data', True),
        ('model.pt', 'Model Parameters', False)]  # By default use the randomly initialized model parameters

    for subdir in sub_dirs:
        subdir_path = pjoin(resource_dir, subdir)
        if os.path.isdir(subdir_path):
            subdir_files = set(os.listdir(subdir_path))

            for (filename, info, required) in required_files:
                if not (filename in subdir_files):
                    print(f"Cannot find file \"{filename}\" containing \"{info}\" in subdirectory \"{subdir_path}\"")
                    if required:
                        return False

    return True


@app.route('/api/data', methods=['POST'])
def get_data():
    """
    Function for retrieving and transforming data for visualization from "resource_dir"

    TODO: Move this function into a method for "T3-Vis"
    """
    # projection_type = flask.request.json['projectionType']
    # pdb.set_trace()
    # if projection_type == 'cartography' and flask.request.json['checkpointName'] in ['pretrained', 'epoch_1']:
        # flask.request.json['checkpointName'] = 'epoch_2'

    # t3_vis.curr_checkpoint_dir = flask.request.json['checkpointName']

    # if t3_vis.curr_checkpoint_dir != None:
    #     checkpoint_dir = pjoin(t3_vis.resource_dir, t3_vis.curr_checkpoint_dir)
    #     model_weights_file = pjoin(checkpoint_dir, 'model.pt')
    #     if os.path.exists(model_weights_file):
    #         t3_vis.model.load_state_dict(torch.load(model_weights_file))

    # layer = int(flask.request.json['hiddenLayer'])

    # TODO: This should be formatted by the user during preprocessing
    # projection_keys = {
    #     # 'hidden': (f'projection_{layer}_0', f'projection_{layer}_1'),
    #     # 'hidden': (f'projection_{layer}_1', f'projection_{layer}_2'),
    #     # 'cartography': ('avg_variability', 'avg_confidence'),
    #     'discrete': ['labels', 'predictions'],
    #     'continuous': ['gt_confidence', 'loss'],
    # }

    # projection_data = torch.load(pjoin(checkpoint_dir, 'projection_data.pt'))
    projection_path = pjoin(t3_vis.resource_dir, 'encoder_projection_data.pt')
    projection_data = torch.load(projection_path)

    # input_len = np.array([len(x['input_ids'][0]) for x in t3_vis.dataset]).astype(int)
    # projection_data['continuous'] = {'input_len': input_len}

    results = {}
    results['ids'] = projection_data['ids'].tolist()


    if ('x' in projection_data.keys() and 'y' in projection_data.keys()):
        results['x'] = list(projection_data['x'])
        results['y'] = list(projection_data['y'])
        # Avoid scaling t-SNE embeddings
        # TODO: need another option to determine when or when not to scale
        results['domain'] = (min(min(results['x']), min(results['y'])), max(max(results['x']), max(results['y'])))
    else:
        assert Error(f"Missing \"x\" and \"y\" keys in \"{projection_path}\"")

    results['discrete'] = []
    results['continuous'] = []

    # Process discrete data attributes
    non_discrete_types = [np.float32, np.float64, np.float16, np.double]
    if 'discrete' in projection_data.keys():
        for attr_name in projection_data['discrete'].keys():
            attr_val = {}
            attr_val['name'] = attr_name

            if np.array(projection_data['discrete'][attr_name]).dtype in non_discrete_types:
                projection_data['discrete'][attr_name] = np.array(projection_data['discrete'][attr_name]).astype(int)

            attr_val['values'] = np.array(projection_data['discrete'][attr_name]).tolist()
            attr_val['domain'] = np.unique(np.array(projection_data['discrete'][attr_name]).astype(str)).tolist()
            results['discrete'].append(attr_val)

    # Process continuous data attributes
    if 'continuous' in projection_data.keys():
        for attr_name in projection_data['continuous'].keys():
            attr_val = {}
            attr_val['name'] = attr_name
            attr_val['values'] = projection_data['continuous'][attr_name].tolist()
            attr_val['max'] = int(projection_data['continuous'][attr_name].max())
            attr_val['min'] = int(projection_data['continuous'][attr_name].min())
            attr_val['mean'] = int(projection_data['continuous'][attr_name].mean())
            attr_val['median'] = int(np.median(projection_data['continuous'][attr_name]))
            results['continuous'].append(attr_val)

    # Load attention images (stored in backend)
    # print("got here")
    # aggregate_encoder_attn = torch.load(pjoin(t3_vis.resource_dir, 'aggregate_encoder_attn_img.pt'))
    # t3_vis.aggregate_encoder_attn = aggregate_encoder_attn
    # for i in range(len(aggregate_encoder_attn)):
    #     layer = i // 16
    #     head = i % 16
    #     torch.save(np.array(aggregate_encoder_attn[i]['attn'], dtype=np.uint8), pjoin(t3_vis.resource_dir, f'aggregate_encoder_attn_img_{layer}_{head}.pt'))

    # aggregate_decoder_attn = torch.load(pjoin(t3_vis.resource_dir, 'aggregate_decoder_attn_img.pt'))
    # t3_vis.aggregate_decoder_attn = aggregate_decoder_attn
    # pdb.set_trace()
    # for i in range(len(aggregate_decoder_attn)):
    #     layer = i // 16
    #     head = i % 16
    #     torch.save(np.array(aggregate_decoder_attn[i]['attn'], dtype=np.uint8), pjoin(t3_vis.resource_dir, f'aggregate_decoder_attn_img_{layer}_{head}.pt'))
    # attn_img_path = pjoin(t3_vis.resource_dir, 'aggregate_cross_attn_img.pt')
    # if os.path.exists(attn_img_path):
    #     aggregate_cross_attn = torch.load(pjoin(attn_img_path))
    #     t3_vis.aggregate_cross_attn = aggregate_cross_attn
    #     for i in range(len(aggregate_cross_attn)):
    #         layer = i // 16
    #         head = i % 16
    #         torch.save(np.array(aggregate_cross_attn[i]['attn'], dtype=np.uint8), pjoin(t3_vis.resource_dir, f'aggregate_cross_attn_img_{layer}_{head}.pt'))

    # pdb.set_trace()
    # Do this for now, need to send an image file to be more efficient
    # for i in range(len(aggregate_encoder_attn)):
    #     aggregate_encoder_attn[i]['attn'] = json.dumps(aggregate_encoder_attn[i]['attn'])

    # importance = torch.load(pjoin(checkpoint_dir, 'head_importance.pt'))
    # results['decoder_head_importance'] = np.random.rand(t3_vis.model.config.num_hidden_layers, t3_vis.model.config.num_attention_heads, 2).tolist()
    # results['encoder_head_importance'] = np.random.rand(t3_vis.model.config.num_hidden_layers, t3_vis.model.config.num_attention_heads).tolist() # Deco
    decoder_head_importance = torch.load(pjoin(t3_vis.resource_dir, 'decoder_head_importance.pt'))
    # decoder_head_importance = torch.rand(36, 20)

    # Decoder-only
    if len(decoder_head_importance.shape) == 2:
        decoder_head_importance = normalize(decoder_head_importance)
    # Encoder-Decoder
    elif len(decoder_head_importance.shape) == 3:
        decoder_head_importance[:, :, 0] = normalize(decoder_head_importance[:, :, 0])
        decoder_head_importance[:, :, 1] = normalize(decoder_head_importance[:, :, 1])

    encoder_head_importance = torch.zeros(36, 20)
    # encoder_head_importance = normalize(torch.load(pjoin(t3_vis.resource_dir, 'encoder_head_importance.pt')))


    results['decoder_head_importance'] = decoder_head_importance.tolist()
    # results['encoder_head_importance'] = encoder_head_importance.tolist()
    # results['aggregate_attn'] = aggregate_encoder_attn

    return flask.jsonify(results)


@app.route('/api/eval_one', methods=['POST'])
def eval_one():
    """
    Evaluate a single example in the back-end, specified by the example_id
    """
    request = flask.request.json
    # heads_to_prune = {int(k): v for k, v in flask.request.json['pruned_heads'].items()}
    # if heads_to_prune != {}: 
    #     t3_vis.prune_heads(heads_to_prune)
    results = t3_vis.evaluate_example(request['example_id'], request['encoder_head'], request['decoder_head'])
    return flask.jsonify(results)


@app.route('/api/attentions', methods=['POST'])
def get_attentions():
    """
    Evaluate a single example in the back-end, specified by the example_id
    """
    request = flask.request.json

    attention_type = request['attention_type']
    layer = int(request['layer']) - 1
    head = int(request['head']) - 1

    
    if (attention_type == 'encoder' and t3_vis.encoder_attentions != None) or \
       (attention_type == 'decoder' and t3_vis.decoder_attentions != None):
        results = t3_vis.get_attentions(attention_type, layer, head)
    else:
        results = {}

    return flask.jsonify(results)


@app.route('/api/agg_attentions', methods=['POST'])
def get_aggregate_attentions():
    request = flask.request.json

    attention_type = request['attention_type']
    layer = int(request['layer']) - 1
    head = int(request['head']) - 1
    print(attention_type, layer, head)

    results = {}

    if t3_vis.model_type == 'encoder-decoder':

        if attention_type == 'encoder':
            aggregate_encoder_attn = torch.load(pjoin(t3_vis.resource_dir, f'aggregate_encoder_attn_img_{layer}_{head}.pt'))
            results['attn'] = aggregate_encoder_attn.tolist()
        elif attention_type == 'decoder':
            aggregate_decoder_attn = torch.load(pjoin(t3_vis.resource_dir, f'aggregate_cross_attn_img_{layer}_{head}.pt'))
            aggregate_cross_attn = torch.load(pjoin(t3_vis.resource_dir, f'aggregate_cross_attn_img_{layer}_{head}.pt'))
            results['attn'] = [aggregate_decoder_attn.tolist(), aggregate_cross_attn.tolist()]
        else:
            raise NotImplementedError("Attention Type Not Supported!")
    else:
        aggregate_decoder_attn = torch.load(pjoin(t3_vis.resource_dir, 'decoder_attentions', f'attn_img_{layer}_{head}.pt'))
        results['attn'] = aggregate_decoder_attn.tolist()

    results['input_len'] = 512
    results['output_len'] = 511

    return flask.jsonify(results)


if __name__ == '__main__':

    cwd = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=True)
    parser.add_argument("--port", default="8888")
    parser.add_argument("--host", default=None)

    # TODO: Let user select the model/dataset from UI interaction
    parser.add_argument("--model", required=True, help="Method for returning the model")
    parser.add_argument("--dataset", required=True, help="Method for returning the dataset")

    # This should be based on the number of examples saved
    parser.add_argument("--n_examples", default=10, type=int, help="The maximum number of data examples to visualize")
    parser.add_argument("--device", default=None, type=str)

    parser.add_argument("--model_type", default='decoder', type=str, choices=['decoder', 'encoder-decoder'])

    parser.add_argument("--filter_paddings", default=True, type=bool, help="Filter padding tokens for visualization")
    parser.add_argument("--resource_dir", default=pjoin(cwd, 'resources', 'pegasus_xsum'), \
                        help="Directory containing the necessary visualization resources for each model checkpoint")

    args = parser.parse_args()

    # if not check_resource_dir(args.resource_dir):
    #     exit(1)

    global t3_vis

    t3_vis = T3_Visualization(args)

    app.run(host=args.host, port=int(args.port), debug=args.debug)
