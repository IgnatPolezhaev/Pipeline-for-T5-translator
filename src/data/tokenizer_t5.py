from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

class T5Tokenizer:
    def __init__(self, model_config, data_config, language):

        self.model_config = model_config
        self.data_config = data_config
        self.language = language
        self.tokenizer = AutoTokenizer.from_pretrained(model_config['token'], use_auth_token=True)
        
        self.word2index = dict()
        self.index2word = dict()
        self.built_token_mapping()
        
    def built_token_mapping(self):
        self.word2index = self.tokenizer.get_vocab()
        self.index2word = {v: k for k, v in self.word2index.items()}

    def __len__(self):
        return len(self.word2index)

    def __call__(self, input_sequences):

        if self.language == 'source':
            task_prefix = "translate English to Russian: "
        elif self.language == 'target':
            task_prefix = ""
            
        encoding = self.tokenizer(
            [task_prefix + sequence for sequence in input_sequences],
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
        
        if self.language == 'source':
            return predicted[30:]
        elif self.language == 'target':
            return predicted
