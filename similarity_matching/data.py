from functools import partial
import pathlib
import json
from uuid import UUID, uuid4
import pandas as pd  # type: ignore # no stubs
from typing import Dict, Iterable, List, Tuple

from similarity_matching.classifier import TopicClassifier, shannon_entropy


def load_data(
    data_path: pathlib.Path,
) -> Tuple[List[List[Dict[str, str]]], List[List[Dict[str, str]]]]:
    with open(data_path, "r") as f:
        json_data = json.load(f)
        return json_data["contact"], json_data["respond"]


def _record_item_to_record(
    record: List[Dict[str, str]], label: str
) -> Tuple[UUID, str, str]:
    record_id = uuid4()
    # Ideally - would tackle these with their structural data
    total_text = " ".join(
        [
            event.get("event_type", "")
            + " "
            + event.get("element_name", "")
            + " "
            + event.get("text_value", "")
            for event in record
        ]
    )
    return (
        record_id,
        total_text,
        label,
    )


def _data_records(
    contact_data: List[List[Dict[str, str]]], label: str
) -> Iterable[Tuple[UUID, str, str]]:
    for record in contact_data:
        yield _record_item_to_record(record, label)


def _filter_entropy(term: str, entropy_cutoff: float) -> str:
    return " ".join([t for t in term.split() if shannon_entropy(t) > entropy_cutoff])


def _process_data(
    contact_data: List[List[Dict[str, str]]],
    topic_classifier: TopicClassifier,
    label: str,
    entropy_cutoff: float,
) -> pd.DataFrame:
    df = pd.DataFrame.from_records(
        _data_records(contact_data, label),
        columns=["record_id", "total_text", "label"],
    )
    df["topic"] = df.total_text.apply(topic_classifier.top_prediction)
    df["original_text"] = df.total_text
    df["total_text"] = df.total_text.apply(
        partial(_filter_entropy, entropy_cutoff=entropy_cutoff)
    )
    return df


def training_data(file: pathlib.Path, classifier: TopicClassifier, entropy_cutoff=2.5):
    contact, response = load_data(file)
    return _process_data(contact, classifier, "contact", entropy_cutoff), _process_data(
        response, classifier, "response", entropy_cutoff
    )
