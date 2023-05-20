# Pipeline-for-T5-translator
This repository implements a pipeline for training the T5 model for translation from English to Russian. The model with huggingface t5-small is used as the T5 model.

- configs
  - data_config.yaml
  - model_config.yaml
- src
  - data
    - datamodule_t5.py
    - tokenizer_t5.py
    - utils.py
  - models
     - seq2seq_t5.py
     - trainer.py
  - training_logs
    - progress_log.txt
  - metrics.py
  - txt_logger.py

data_config.yaml and model_config.yaml contain training parameters

datamodule_t5.py prepares data and creates dataloads

tokenizer_t5.py tokenizers the data

utils.py contains some additionals functions

seq2seq_t5.py contains model T5

trainer.py 

metrics.py counts the BLEU metric

txt_logger.py writes logs to file progress_log.txt
