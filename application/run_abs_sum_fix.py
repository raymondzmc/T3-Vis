import torch
from utils.helpers import normalize, format_attention, format_attention_image
from os.path import join as pjoin

aggregate_cross_attn = torch.load(pjoin('resources/pegasus_xsum', 'aggregate_cross_attn.pt'))
cross_attn_img = format_attention_image(aggregate_cross_attn, color='blue')
torch.save(cross_attn_img, pjoin('resources/pegasus_xsum', 'aggregate_cross_attn_img.pt'))