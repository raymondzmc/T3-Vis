from transformers import PegasusForConditionalGeneration, PegasusTokenizer


def pegasus():
    model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-cnn_dailymail')

    # Hard-code for now
    setattr(model, 'num_hidden_layers', 12)
    setattr(model, 'num_attention_heads', 12)
    setattr(model, 'hidden_size', 1024)
    return model