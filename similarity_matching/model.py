import pandas as pd  # type: ignore # no stubs
import numpy as np

from sklearn.metrics.pairwise import linear_kernel  # type: ignore # no stubs
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore # no stubs

from typing import Dict, List


class SimilarityModel:
    def __init__(
        self,
        topics: List[str],
        similarity_cutoff: float = 0.7,
        language: str = "english",
    ) -> None:
        self.topics = topics
        self.tf_idf = TfidfVectorizer(min_df=1, stop_words=language, lowercase=True)
        self.similarity_cutoff = similarity_cutoff

    def __call__(self, df: pd.DataFrame, target_df: pd.DataFrame) -> Dict[int, int]:
        lhs_items: List[int] = []
        rhs_items: List[int] = []
        df.reset_index()
        target_df.reset_index()
        for topic in self.topics:
            # Filter to subset of similar items, with the added benefit
            #   of filtering out already matched items.
            topic_df = df.loc[(df.topic == topic)]
            topic_target_df = target_df.loc[(target_df.topic == topic)]
            if topic_df.shape[0] == 0 or topic_target_df.shape[0] == 0:
                continue

            self.tf_idf.fit(topic_df.total_text)
            lhs_term_matrix = self.tf_idf.transform(topic_df.total_text)
            rhs_term_matrix = self.tf_idf.transform(topic_target_df.total_text)
            term_similarity = linear_kernel(lhs_term_matrix, rhs_term_matrix)

            # Term Similarity is Commutative - finding the max across
            #   columns or rows is sufficient to identifying links
            max_term_similarity_idx = np.argmax(
                term_similarity, axis=0
            )  # (lhs_terms, rhs_terms)
            max_term_similarity = np.max(term_similarity, axis=1)
            if max(max_term_similarity) < self.similarity_cutoff:
                continue

            similar_lhs_terms = np.arange(len(max_term_similarity))[
                max_term_similarity > self.similarity_cutoff
            ]
            similar_rhs_terms = max_term_similarity_idx[
                max_term_similarity > self.similarity_cutoff
            ]

            lhs_items += topic_df.iloc[similar_lhs_terms].index.tolist()
            rhs_items += topic_target_df.iloc[similar_rhs_terms].index.tolist()

        return dict(zip(lhs_items, rhs_items))
