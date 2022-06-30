from similarity_matching.classifier import TopicClassifier
import pytest


@pytest.fixture
def topic_classifier():
    return TopicClassifier(labels=["purchasing", "stock trading", "excel macros"])
