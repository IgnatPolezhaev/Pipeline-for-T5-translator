from typing import Tuple, List
import numpy as np
from torchtext.data.metrics import bleu_score

def bleu_scorer(predicted: np.ndarray, actual: np.ndarray, tokenizer):
    batch_bleu = []
    predicted_sentences = []
    actual_sentences = []
    for a, b in zip(predicted, actual):
        words_predicted = tokenizer.decode(a)
        words_actual = tokenizer.decode(b)
        predicted_sentences.append(words_predicted.split())
        actual_sentences.append([words_actual.split()])

    BLEU = bleu_score(predicted_sentences, actual_sentences)
    
    return BLEU, actual_sentences, predicted_sentences