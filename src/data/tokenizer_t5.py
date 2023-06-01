from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

class T5Tokenizer:
    def __init__(self, model_config, data_config):

        self.model_config = model_config
        self.data_config = data_config
        self.tokenizer = AutoTokenizer.from_pretrained(model_config['token'], use_auth_token=True)

    def __len__(self):
        return len(self.tokenizer)

    def add_tokens(self, tokens_list):
        self.tokenizer.add_tokens(tokens_list)

    def __call__(self, input_sequences):
            
        encoding = self.tokenizer(
            [sequence for sequence in input_sequences],
            padding="longest",
            max_length=self.data_config['max_length'],
            truncation=True,
            return_tensors="pt"
        )

        return encoding

    def decode(self, token_list):
        predicted = self.tokenizer.decode(token_list)
        for item in self.tokenizer.all_special_tokens:
            predicted = predicted.replace(item, '')
        
        return predicted
