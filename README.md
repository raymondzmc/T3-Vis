# T<sup>3</sup>-Vis

T<sup>3</sup>-Vis is a visual analytic framework designed to assist in the training and fine-tuning of Transformer-based models.

![T3-Vis Interface Teaser](documentation/images/teaser.png)


## Install and Getting Started

We suggest installing within a virtual environment such as `virtualenv` or [`conda`](https://docs.conda.io/en/latest/)

```sh
git clone https://github.com/raymondzmc/T3-Vis.git ~/T3-Vis

# Set up Python environment
cd ~/T3-Vis
pip3 install -r requirements.txt
```

## Setting up the Task

### Dataset
To visualize a specific dataset, create a new file in `application/dataset/` containing a function that returns an iterable object without taking any arguments (see [`sst2.py`](application/dataset/sst2.py) for examples), then import the dataset function in `application/models/__init__.py`. 

We suggest following the format for the [`Dataset`](https://huggingface.co/docs/datasets/access.html) object in the [`datasets`](https://huggingface.co/docs/datasets/index.html) library

The `dataset` object needs to have the attribute `visualize_columns` corresponding to a list of columns to be visualized in the Data Table View, while all items in the iterable object needs have the following keys:

* __`id`__: Index of the example
* __`tokens`__: String representation of input tokens for visualizing the saliency maps in the Instance Investigation View

### Model

For a customized transformer-based model, create a new file in `application/models/` containing a function that returns the initialized model and tokenizer without taking any arguments (see [`bert_classification.py`](application/models/bert_classification.py) and [`bert_sum.py`](application/models/bert_sum.py) for examples). Finally, import the model function in `application/models/__init__.py`.

We suggest the model should be implemented in such a way where an item from the `dataset` could be directly used as input (to avoid modifying the functions in the backend), such that:
```python
example = dataset[0]
output = model(example)
```

Additionally, the model needs to have the following attributes/methods for our backend algorithms to work:

* __`hidden_size`__: The hidden state dimensions
* __`num_hidden_layers`__: Number of hidden layers
* __`num_attention_heads`__: Number of attention heads in each layer
* __`prune_heads(heads_to_prune)`__ (optional): Operation for pruning attention heads (see the [implementation](https://huggingface.co/transformers/main_classes/model.html?highlight=prune_heads#transformers.PreTrainedModel.prune_heads) from Hugging Face's `transformers` library)

## Data Preprocessing
Coming soon...

In the meantime, please check out our [paper](https://arxiv.org/abs/2108.13587) to appear in the EMNLP 2021 System Demonstration track.
