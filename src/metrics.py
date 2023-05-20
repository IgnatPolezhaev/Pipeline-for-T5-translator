from typing import Tuple, List
import numpy as np
from torchtext.data.metrics import bleu_score

def bleu_scorer(predicted: np.ndarray, actual: np.ndarray, target_tokenizer):
    batch_bleu = []
    predicted_sentences = []
    actual_sentences = []
    for a, b in zip(predicted, actual):
        words_predicted = target_tokenizer.decode(a)
        words_actual = target_tokenizer.decode(b)
        predicted_sentences.append(words_predicted.split())
        actual_sentences.append([words_actual.split()])

    BLEU = bleu_score(predicted_sentences, actual_sentences)
    
    return BLEU, actual_sentences, predicted_sentences