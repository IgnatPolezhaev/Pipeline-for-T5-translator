# Pipeline-for-T5-translator
This repository implements a pipeline for training the T5 model for translation from English to Russian.

The configs folder contains yaml files with training parameters.

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
