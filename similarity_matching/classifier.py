from collections import Counter

import numpy as np

from transformers import pipeline
from typing import List
from functools import lru_cache
from math import log2


class TopicClassifier:
    """
    Uses a pretrained Natural Language Inference model to classify sentences
    into topics.

    While this uses the default English Bart Large MNLI, this can be swapped out
    for a model similar to https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
    to scale to multiple languages.

    Source: https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.ZeroShotClassificationPipeline
    """

    def __init__(self, labels: List[str]) -> None:
        self.classifier = pipeline(
            task="zero-shot-classification", model="facebook/bart-large-mnli"
        )
        self.labels = sorted(labels)

    def predict(self, sequence: str) -> List[float]:
        prediction = self.classifier(sequence, self.labels, multi_label=True)
        return [
            prediction["scores"][i] for i in np.argsort(prediction["labels"]).tolist()
        ]

    def top_prediction(self, sequence: str) -> str:
        predictions = self.predict(sequence)
        top_pred = np.argmax(predictions)
        return self.labels[top_pred]


@lru_cache(maxsize=200)
def shannon_entropy(term: str) -> float:
    c = Counter(iter(term.lower()))
    total = c.total()
    return sum((c[i] / total) * log2(total / c[i]) for i in term.lower())
