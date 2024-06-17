import math


class Metrics:
    def __init__(self, relevant_docs: list[list[dict]]):
        self.relevant_docs = relevant_docs
        self.num_relevant_docs = len(self.relevant_docs)

    def discounted_cumulutive_gain_at_k(
        self, retrieved_docs_all_queries: list[list[str]], query_index: int, k: int
    ):
        """Calculates DCG@k.

        Args:
            retrieved_docs_all_queries: A list of list of query stings.
            query_index: The index of the query for which DCG@k should be
                         calculated.
            k: Number up to which DCG@k should be calculated.
        Returns:
            DCG@k
        """
        retrieved_docs: list[str] = retrieved_docs_all_queries[query_index]
        relevant_docs: list[dict] = self.relevant_docs[query_index]

        dcg = 0.0

        for i in range(k):
            if i < min(len(retrieved_docs), self.num_relevant_docs):
                relevance_score = relevant_docs[i].get("relevance")
                dcg += relevance_score / (math.log2(i + 2))
        return dcg

    def normalized_discounted_cumulutive_gain_at_k(
        self, retrieved_docs_all_queries: list[list[str]], query_index: int, k: int
    ):
        """Calculated NDCG@k.

        Args:
            retrieved_docs_all_queries: A list of list of query stings.
            query_index: The index of the query for which DCG@k should be
                         calculated.
            k: Number up to which DCG@k should be calculated.

        Returns:
            DCG@k
        """

        retrieved_docs: list[str] = retrieved_docs_all_queries[query_index]
        relevant_docs: list[dict] = self.relevant_docs[query_index]

        # Sort relevance scores
        relevance_scores = []
        for i in range(k):
            if i < min(len(retrieved_docs), self.num_relevant_docs):
                relevance_scores.append(relevant_docs[i].get("relevance"))

        # Calculate IDCG@k
        relevance_scores.sort(reverse=True)
        idcg_at_k = 0.0
        for i in range(k):
            if i < min(len(retrieved_docs), self.num_relevant_docs):
                idcg_at_k += relevance_scores[i] / (math.log2(i + 2))

        # Calculate DCG@k
        dcg_at_k = self.discounted_cumulutive_gain_at_k(
            retrieved_docs_all_queries, query_index, k
        )

        return float(dcg_at_k / idcg_at_k)
