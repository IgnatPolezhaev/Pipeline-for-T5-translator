import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.optimization import Adafactor
from torch.optim.lr_scheduler import StepLR
import metrics

class Seq2SeqT5(torch.nn.Module):
    def __init__(self, model_config, device, tokenizer):
        super(Seq2SeqT5, self).__init__()
        self.device = device
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.t5_model = T5ForConditionalGeneration.from_pretrained(self.model_config['model'], 
                                                                   use_auth_token=True).to(self.device)
        self.t5_model.resize_token_embeddings(len(tokenizer))
        self.optimizer = Adafactor(self.t5_model.parameters(), 
                                   lr=self.model_config['learning_rate'], 
                                   relative_step=False)
        self.scheduler = StepLR(self.optimizer, step_size=2000, gamma=0.1)

    def training_step(self, batch):
        batch['input_ids'] = batch['input_ids'].to(self.device)
        batch['attention_mask'] = batch['attention_mask'].to(self.device)
        batch['labels'] = batch['labels'].to(self.device)

        self.t5_model.train()
        self.optimizer.zero_grad()
        
        batch['labels'][batch['labels'] == self.tokenizer.tokenizer.pad_token_id] = -100
        loss = self.t5_model(input_ids=batch['input_ids'], 
                             attention_mask=batch['attention_mask'], 
                             labels=batch['labels']
                            ).loss

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def validation_step(self, batch):
        self.t5_model.eval()
        
        batch['input_ids'] = batch['input_ids'].to(self.device)
        batch['attention_mask'] = batch['attention_mask'].to(self.device)
        batch['labels'] = batch['labels'].to(self.device)
        
        result_dict = {}
        with torch.no_grad():
            output = self.t5_model(input_ids=batch['input_ids'], 
                                   attention_mask=batch['attention_mask'], 
                                   labels=batch['labels']
                                  )
        loss = output.loss
        decoder_result_list = torch.argmax(output.logits, dim=-1).cpu().numpy()
        result_dict['loss'] = loss.item()

        decoded_list = []
        for seq in decoder_result_list:
            decoded_seq = self.tokenizer.decode(seq)
            decoded_list.append(decoded_seq)

        result_dict['predicted'] = decoded_list
        
        bleu_score, _, _ = self.eval_bleu(torch.argmax(output.logits, dim=-1), batch['labels'])
        result_dict['bleu'] = bleu_score
        
        return result_dict
    
    def predict(self, list_sentence):
        translate_sentence = []
        inputs = self.tokenizer.tokenizer([sentence for sentence in list_sentence], return_tensors="pt", padding=True).to(self.device)
        output_sequences = self.t5_model.generate(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"],do_sample=False)
        translate_sentence = self.tokenizer.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        return translate_sentence

    def eval_bleu(self, predicted_ids, target_tensor):
        predicted = predicted_ids.squeeze().detach().cpu().numpy()
        actuals = target_tensor.squeeze().detach().cpu().numpy()
        bleu_score, actual_sentences, predicted_sentences = metrics.bleu_scorer(
            predicted=predicted, actual=actuals, tokenizer=self.tokenizer
        )
        return bleu_score, actual_sentences, predicted_sentences
