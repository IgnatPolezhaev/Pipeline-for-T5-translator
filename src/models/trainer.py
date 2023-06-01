from tqdm import tqdm

class Trainer:
    def __init__(self, model, model_config, logger):
        self.model = model
        self.epoch_num = model_config['epoch_num']
        self.logger = logger
        self.logger.log(model_config)

    def train(self, train_dataloader, val_dataloader):
        train_loss_list, validation_loss_list = [], []
        try:
            for epoch in tqdm(range(self.epoch_num)):
                train_epoch_loss = 0
                self.model.train()
                for batch in train_dataloader:
                    train_loss = self.model.training_step(batch)
                    train_epoch_loss += train_loss
                train_epoch_loss = train_epoch_loss / len(train_dataloader)

                val_epoch_loss, val_epoch_bleu = 0, 0
                self.model.eval()
                for batch in val_dataloader:
                    result_dict = self.model.validation_step(batch)
                    val_epoch_loss += result_dict['loss']
                    val_epoch_bleu += result_dict['bleu']
                val_epoch_loss = val_epoch_loss / len(val_dataloader)
                val_epoch_bleu = val_epoch_bleu / len(val_dataloader)

                self.logger.log({"val_loss": val_epoch_loss,
                                 "train_loss": train_epoch_loss ,
                                 "bleu_score": val_epoch_bleu})

        except KeyboardInterrupt:
            pass

        print(f"Last {epoch} epoch train loss: ", train_epoch_loss)
        print(f"Last {epoch} epoch val loss: ", val_epoch_loss)
        print(f"Last {epoch} epoch val bleu: ", val_epoch_bleu)
