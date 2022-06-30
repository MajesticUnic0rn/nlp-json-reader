import pytest


@pytest.mark.parametrize(
    "sequence,expected",
    [
        ("purchase #: 123 is rejected", "purchasing"),
        (
            "Dr. F, confirming AMZN current price $3265.08. Please acknowledge.",
            "stock trading",
        ),
    ],
)
def test_topic_classifier(topic_classifier, sequence, expected):
    likely_topic = topic_classifier.top_prediction(sequence)
    assert likely_topic == expected
