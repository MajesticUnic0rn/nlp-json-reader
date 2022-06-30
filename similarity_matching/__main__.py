import pathlib
import argparse
from typing import List
from similarity_matching.classifier import TopicClassifier
from similarity_matching.data import training_data
from similarity_matching.model import SimilarityModel


def main(filepath: pathlib.Path, labels: List[str] = ["purchasing", "stock trading"]):
    classifier = TopicClassifier(labels)
    data, response = training_data(filepath, classifier)
    model = SimilarityModel(labels)
    results = model(data, response)
    # In typical systems would simply store the linked IDs, but for debugging given the datasize,
    #   it seems appropriate to just print the 3 results.
    for source_idx, target_idx in results.items():
        print(
            f"Matching: \n {data.iloc[source_idx].original_text}\nwith:\n{response.iloc[target_idx].original_text}\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Match Similar Events")
    parser.add_argument("--file", dest="file", help="JSON File for Matching")
    parser.add_argument(
        "--labels",
        dest="labels",
        help="Text Labels for Topics",
        default="",
    )
    args = parser.parse_args()
    if len(args.labels) == 0:
        main(pathlib.Path(args.file))
    else:
        main(pathlib.Path(args.file), args.labels.split(","))
