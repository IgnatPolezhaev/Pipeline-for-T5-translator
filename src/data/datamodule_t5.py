from torch.utils.data import Dataset, DataLoader
from data.tokenizer_t5 import T5Tokenizer
from data.utils import TextUtils, short_text_filter_function

class DataManager:
    def __init__(self, data_config, model_config):
        super().__init__()
        self.model_config = model_config
        self.data_config = data_config
        self.tokenizer = T5Tokenizer(self.model_config, self.data_config)
        self.tokenizer.add_tokens(list(tuple('абвгдежзийклмнопрстуфхцчшщъыьэюя')))
        
    def collate_wrapper(self, batch):
        source_data = []
        target_data = []
        for i in range(self.data_config["batch_size"]):
            source_data.append(batch[i][0])
            target_data.append(batch[i][1])
        
        tokenized_source = self.tokenizer(source_data)
        tokenized_target = self.tokenizer(target_data)
        tokenized_source["labels"] = tokenized_target["input_ids"]

        return tokenized_source
        
    def prepare_data(self):
        pairs = TextUtils.read_langs_pairs_from_file(filename=self.data_config["filename"])
        prefix_filter = self.data_config['prefix_filter']
        if prefix_filter:
            prefix_filter = tuple(prefix_filter)

        source_sentences, target_sentences = [], []
        unique_sources = set()
        for pair in pairs:
            source, target = pair[0], pair[1]
            if short_text_filter_function(pair, self.data_config['max_length'], prefix_filter) and source not in unique_sources:
                source_sentences.append(source)
                target_sentences.append(target)
                unique_sources.add(source)

        train_size = int(len(source_sentences)*self.data_config["train_size"])
        source_train_sentences, source_val_sentences = source_sentences[:train_size], source_sentences[train_size:]
        target_train_sentences, target_val_sentences = target_sentences[:train_size], target_sentences[train_size:]

        train_dataloader = DataLoader(list(zip(source_train_sentences, target_train_sentences)),
                                      shuffle=True,
                                      batch_size=self.data_config["batch_size"],
                                      collate_fn=self.collate_wrapper,
                                      drop_last = True
                                     )

        val_dataloader = DataLoader(list(zip(source_val_sentences, target_val_sentences)),
                                    shuffle=True,
                                    batch_size=self.data_config["batch_size"],
                                    collate_fn=self.collate_wrapper,
                                    drop_last = True
                                   )
        
        return train_dataloader, val_dataloader
