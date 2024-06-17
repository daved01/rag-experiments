class OrderUnawareMetrics:
    def __init__(self, relevant_docs: list[str]):
        self.relevant_docs = relevant_docs

    def precision_at_k(self, retrieved_docs: list[str], k: int) -> float:
        """Calculates Precision@k."""
        retrieved_docs_at_k = retrieved_docs[:k]
        relevant_count = sum(
            [1 for doc in retrieved_docs_at_k if doc in self.relevant_docs]
        )
        return float(relevant_count / k)

    def recall_at_k(self, retrieved_docs: list[str], k: int) -> float:
        """Calculates Recall@k."""
        retrieved_docs_at_k = retrieved_docs[:k]
        relevant_count = sum(
            [1 for doc in retrieved_docs_at_k if doc in self.relevant_docs]
        )
        return float(relevant_count / len(self.relevant_docs))

    def f1_at_k(self, retrieved_docs: list[str], k: int) -> float:
        """Calculates F1@k."""
        precision = self.precision_at_k(retrieved_docs, k)
        recall = self.recall_at_k(retrieved_docs, k)
        if precision + recall == 0:
            return 0.0
        return float(2 * (precision * recall) / (precision + recall))


class OrderAwareMetrics:
    """Calculates order aware metrics.

    Supported metrics:
    - RR, MRR
    - AP, MAP

    Attributes:
        relevant_docs_all_queries: List of list of strings representing the
            ground-truth of relevant documents of evaluation.
            The outer list corresponds to the list of queries, while the innner
            list is the number of relevant documents for that query.
    """

    def __init__(self, relevant_docs_all_queries: list[list[str]]):
        self.relevant_docs_all_queries = relevant_docs_all_queries

    def reciprocal_rank(
        self, retrieved_docs_all_queries: list[list[str]], query_index: int
    ) -> float:
        """Caluclates RR.

        Args:
            retrieved_docs_all_queries: list[list[str]]: A list of list of strings,
                                        where each outer list represents a query, and
                                        the inner list the relevant documents for
                                        that query.
            query_index: The index of the query to process in the list of all queries.
        Returns:
            reciprocal_rank: The RR as a float.
        """
        relevant_docs_lookup = set(self.relevant_docs_all_queries[query_index])
        for ind, retrieved_doc in enumerate(retrieved_docs_all_queries[query_index]):
            if retrieved_doc in relevant_docs_lookup:
                return 1 / (1 + ind)
        return 0

    def mean_reciprocal_rank(
        self, retrieved_docs_all_queries: list[list[str]]
    ) -> float:
        """Calculates MRR."""

        num_queries = len(self.relevant_docs_all_queries)

        mrr = 0
        for ind in range(num_queries):
            mrr += self.reciprocal_rank(retrieved_docs_all_queries, ind)

        return float(1 / num_queries * mrr)

    def average_precision(
        self, retrieved_docs_all_queries: list[list[str]], query_index: int
    ) -> float:
        """Calculates AP."""
        relevant_docs_lookup = set(self.relevant_docs_all_queries[query_index])

        score = 0.0
        num_hits = 0.0
        for ind, retrieved_doc in enumerate(retrieved_docs_all_queries[query_index]):
            if retrieved_doc in relevant_docs_lookup:
                num_hits += 1.0
                score += num_hits / (ind + 1)

        return float(score / len(self.relevant_docs_all_queries[query_index]))

    def mean_average_precision(
        self, retrieved_docs_all_queries: list[list[str]]
    ) -> float:
        """Calculates MAP."""
        num_queries = len(self.relevant_docs_all_queries)

        nominator = 0
        for ind in range(num_queries):
            nominator += self.average_precision(retrieved_docs_all_queries, ind)

        return float(nominator / num_queries)
