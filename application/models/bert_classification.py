from transformers import AutoModelForSequenceClassification, AutoTokenizer


def bert_classifier():
	model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
	tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
	return model, tokenizer