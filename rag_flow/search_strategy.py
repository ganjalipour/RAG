from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from weaviate.classes.query import BM25Operator, Filter, HybridFusion, MetadataQuery

from logger import LOG


class SearchMode(Enum):
    """Search mode enumeration"""

    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class SearchStrategy(ABC):
    """Abstract base class for search strategies"""

    @abstractmethod
    def search(
        self,
        collection,
        query: str,
        k: int,
        min_score: Optional[float],
        where_filter,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        pass

    def _process_search_results(
        self, response, min_score: Optional[float], search_type: str, query: str
    ) -> List[Dict[str, Any]]:
        """Process and format search results"""
        results = []
        for obj in response.objects:
            score = obj.metadata.score if obj.metadata else 0.0
            if min_score is not None and score < min_score:
                continue
            result = {
                "uuid": obj.uuid,
                "chunk": obj.properties.get("chunk", ""),
                "source": obj.properties.get("source", ""),
                "active": obj.properties.get("active", False),
                "chunk_num": obj.properties.get("chunk_num", 0),
                "score": score,
                "search_type": search_type,
                "distance": (
                    obj.metadata.distance if hasattr(obj.metadata, "distance") else None
                ),
                "creation_time": obj.metadata.creation_time if obj.metadata else None,
            }
            if search_type == "hybrid" and hasattr(obj.metadata, "explain_score"):
                result["explain_score"] = obj.metadata.explain_score
            results.append(result)
        LOG.info(
            f"Found {len(results)} results using {search_type} search for query: '{query[:50]}...'"
        )
        return results


class VectorSearchStrategy(SearchStrategy):
    """Vector search strategy"""

    def search(
        self,
        collection,
        query: str,
        k: int,
        min_score: Optional[float],
        where_filter,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        search_params = {
            "query": query,
            "limit": k,
            "return_metadata": MetadataQuery(
                score=True, creation_time=True, distance=True
            ),
            "return_properties": ["chunk", "source", "active", "chunk_num"],
        }
        if where_filter:
            search_params["filters"] = where_filter
        if "distance" in kwargs:
            search_params["distance"] = kwargs["distance"]
        response = collection.query.near_text(**search_params)
        return self._process_search_results(response, min_score, "vector", query)


class KeywordSearchStrategy(SearchStrategy):
    """Keyword BM25 search strategy"""

    def search(
        self,
        collection,
        query: str,
        k: int,
        min_score: Optional[float],
        where_filter,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        search_params = {
            "query": query,
            "limit": k,
            "return_metadata": MetadataQuery(score=True, creation_time=True),
            "return_properties": ["chunk", "source", "active", "chunk_num"],
        }
        if where_filter:
            search_params["filters"] = where_filter
        if "bm25_operator" in kwargs:
            search_params["operator"] = kwargs["bm25_operator"]
        if "query_properties" in kwargs:
            search_params["query_properties"] = kwargs["query_properties"]
        response = collection.query.bm25(**search_params)
        return self._process_search_results(response, min_score, "keyword", query)


class HybridSearchStrategy(SearchStrategy):
    """Hybrid search strategy"""

    def search(
        self,
        collection,
        query: str,
        k: int,
        min_score: Optional[float],
        where_filter,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        search_params = {
            "query": query,
            "alpha": kwargs.get("alpha", 0.5),
            "limit": k,
            "return_metadata": MetadataQuery(
                score=True, creation_time=True, distance=True, explain_score=True
            ),
            "return_properties": ["chunk", "source", "active", "chunk_num"],
        }
        if where_filter:
            search_params["filters"] = where_filter
        if "fusion_type" in kwargs:
            search_params["fusion_type"] = kwargs["fusion_type"]
        if "vector" in kwargs:
            search_params["vector"] = kwargs["vector"]
        if "max_vector_distance" in kwargs:
            search_params["max_vector_distance"] = kwargs["max_vector_distance"]
        if "bm25_operator" in kwargs:
            search_params["bm25_operator"] = kwargs["bm25_operator"]
        if "query_properties" in kwargs:
            search_params["query_properties"] = kwargs["query_properties"]
        response = collection.query.hybrid(**search_params)
        return self._process_search_results(response, min_score, "hybrid", query)


class SearchStrategyFactory:
    """Factory for creating search strategies"""

    @staticmethod
    def create_strategy(search_mode: SearchMode) -> SearchStrategy:
        if search_mode == SearchMode.VECTOR:
            return VectorSearchStrategyQdrant()
        elif search_mode == SearchMode.KEYWORD:
            return KeywordSearchStrategy()
        elif search_mode == SearchMode.HYBRID:
            return HybridSearchStrategy()
        else:
            raise ValueError(f"Unsupported search mode: {search_mode}")


class VectorSearchStrategyQdrant(SearchStrategy):
    def search(self, client, query_vector, k, min_score, where_filter, **kwargs):
        response = client.search(
            collection_name="rag_embeddings",
            query_vector=query_vector,
            limit=k
        )
        results = []
        for hit in response:
            if min_score and hit.score < min_score:
                continue
            results.append({
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            })
        return results
