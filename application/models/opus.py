from transformers import AutoModelForSeq2SeqLM


def opus():
    # model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-cnn_dailymail')
    model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-en-zh')

    # Hard-code for now
    setattr(model, 'num_hidden_layers', model.config.num_hidden_layers)
    setattr(model, 'num_attention_heads', model.config.num_attention_heads)
    setattr(model, 'hidden_size', model.config.hidden_size)
    return model
