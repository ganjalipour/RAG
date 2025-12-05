import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from langchain_core.documents import Document
from weaviate.classes.query import Filter

from logger import LOG
from rag_flow.config import EmbeddingConfig
from rag_flow.content_enhancer import ContentEnhancer
from rag_flow.document_splitter import DocumentSplitter
from rag_flow.qdrant_embedding_Store import QdrantEmbeddingStore
from rag_flow.search_strategy import SearchMode, SearchStrategyFactory
from openai import OpenAI

if TYPE_CHECKING:
    from rag_flow.answer_generator import AnswerGenerator


class EmbeddingManager:
    """Main orchestrator for RAG flow"""

    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, config: Optional[EmbeddingConfig] = None):
        config = config or EmbeddingConfig.load_from_env()
        collection_name = config.collection_name

        if collection_name not in cls._instances:
            with cls._lock:
                if collection_name not in cls._instances:
                    instance = super(EmbeddingManager, cls).__new__(cls)
                    cls._instances[collection_name] = instance
        return cls._instances[collection_name]

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        if hasattr(self, "_initialized"):
            return
        self.config = config or EmbeddingConfig.load_from_env()
        self.splitter = DocumentSplitter(
            separator=self.config.separator,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        # self.store = EmbeddingStore(self.config)

        self.config.qdrant_url = "https://030204a0-d677-4608-abae-9886fe552ce4.eu-central-1-0.aws.cloud.qdrant.io:6333"
        self.config.collection_name = "rag_embeddings"
        self.config.qdrant_api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.jxJCFk4DogmElkTus0kBi5vGszFRX1Y0APwJoBzKvGo"
        print("qdrant 1111")

        self.embedding_model = self._init_embedding_model()
        self.store = QdrantEmbeddingStore(self.config)

        self.content_enhancer = ContentEnhancer(self.config)
        self._initialized = True
        LOG.info(
            f"Initialized EmbeddingManager with collection: {self.config.collection_name}"
        )
    
    def _init_embedding_model(self):
        """Initialize OpenAI embedding API client"""
        try:
            # مدل و کلید از config یا env
            model_name = getattr(self.config, "embedding_model_name", "text-embedding-3-large")
            api_key = getattr(self.config, "openai_api_key", None)

            if not api_key:
                raise ValueError("❌ OpenAI API key در config تعریف نشده است.")

            print(f"🔹 استفاده از OpenAI embedding model: {model_name}")
            self.openai_client = OpenAI(api_key=api_key)

            # برمی‌گردونیم تابعی که مستقیماً embedding را تولید می‌کند
            def embed_text(text: str) -> list[float]:
                response = self.openai_client.embeddings.create(
                    model=model_name,
                    input=text
                )
                return response.data[0].embedding

            return embed_text

        except Exception as e:
            raise ValueError(f"❌ خطا در بارگذاری مدل embedding: {e}")

    @classmethod
    def get_instance(cls, collection_name: Optional[str] = None) -> "EmbeddingManager":
        config = EmbeddingConfig.load_from_env()
        if collection_name:
            config.collection_name = collection_name
        return cls(config)

    @classmethod
    def clear_instances(cls):
        with cls._lock:
            cls._instances.clear()

    def save_embeddings(self, documents: List[Document]) -> Dict[str, Any]:
        documents = self.splitter.split_documents(documents)

        # Enhance chunks with questions and entities
        enhanced_documents = self.content_enhancer.enhance_documents(documents)

        return self.store.save_embeddings(enhanced_documents)

    def get_objects_by_source(self, source_value: str) -> List[str]:
        return self.store.get_objects_by_source(source_value)

    def update_active_status_by_source(
        self, source_value: str, active_status: bool
    ) -> Dict[str, Any]:
        return self.store.update_active_status_by_source(source_value, active_status)

    def delete_objects_by_source(self, source_value: str) -> Dict[str, Any]:
        return self.store.delete_objects_by_source(source_value)

    def delete_collection(self, collection_name: str = None) -> Dict[str, Any]:
        """Delete the entire Weaviate collection"""
        return self.store.delete_collection(collection_name)

    def _build_where_filter(
        self,
        active_only: bool = True,
        source_filter: Optional[Union[str, List[str]]] = None,
    ):
        filters = []
        if active_only:
            filters.append(Filter.by_property("active").equal(True))
        if source_filter:
            if isinstance(source_filter, str):
                filters.append(Filter.by_property("source").equal(source_filter))
            elif isinstance(source_filter, list) and len(source_filter) > 0:
                source_conditions = [
                    Filter.by_property("source").equal(source)
                    for source in source_filter
                ]
                if len(source_conditions) == 1:
                    filters.append(source_conditions[0])
                else:
                    combined_filter = source_conditions[0]
                    for condition in source_conditions[1:]:
                        combined_filter = combined_filter | condition
                    filters.append(combined_filter)
        if len(filters) == 0:
            return None
        elif len(filters) == 1:
            return filters[0]
        else:
            final_filter = filters[0]
            for filter_condition in filters[1:]:
                final_filter = final_filter & filter_condition
            return final_filter

    def _get_surrounding_chunks(
        self,
        original_results: List[Dict[str, Any]],
        previous_k: int,
        next_k: int,
        active_only: bool = True,
        source_filter: Optional[Union[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Get surrounding chunks (previous and next) for the original results"""
        all_chunks = []
        chunk_keys_added = set()

        # Build filter for active chunks and source filtering
        where_filter = self._build_where_filter(active_only, source_filter)

        for result in original_results:
            source = result["source"]
            chunk_num = result.get("chunk_num", 0)
            # Add original chunk
            chunk_key = (source, chunk_num)

            if chunk_key not in chunk_keys_added:
                LOG.debug(
                    f"[ORIGINAL] UUID: {result.get('uuid', 'N/A')} | Source: {source} | Chunk: {chunk_num}"
                )
                all_chunks.append(result)
                chunk_keys_added.add(chunk_key)

            # Get previous chunks
            if previous_k > 0:
                prev_chunks = self._fetch_chunks_by_range(
                    source, chunk_num - previous_k, chunk_num - 1, where_filter
                )
                for chunk in prev_chunks:
                    chunk_source = chunk.get("source", "")
                    chunk_num_key = chunk.get("chunk_num", 0)
                    chunk_key = (chunk_source, chunk_num_key)
                    if chunk_key not in chunk_keys_added:
                        LOG.debug(
                            f"[PREVIOUS] UUID: {chunk.get('uuid', 'N/A')} | Source: {chunk_source} | Chunk: {chunk_num_key}"
                        )
                        all_chunks.append(chunk)
                        chunk_keys_added.add(chunk_key)

            # Get next chunks
            if next_k > 0:
                next_chunks = self._fetch_chunks_by_range(
                    source, chunk_num + 1, chunk_num + next_k, where_filter
                )
                for chunk in next_chunks:
                    chunk_source = chunk.get("source", "")
                    chunk_num_key = chunk.get("chunk_num", 0)
                    chunk_key = (chunk_source, chunk_num_key)
                    if chunk_key not in chunk_keys_added:
                        LOG.debug(
                            f"[NEXT] UUID: {chunk.get('uuid', 'N/A')} | Source: {chunk_source} | Chunk: {chunk_num_key}"
                        )
                        all_chunks.append(chunk)
                        chunk_keys_added.add(chunk_key)

        # Sort by source and chunk_num to maintain order
        all_chunks.sort(key=lambda x: (x["source"], x.get("chunk_num", 0)))
        return all_chunks

    def _fetch_chunks_by_range(
        self,
        source: str,
        start_chunk_num: int,
        end_chunk_num: int,
        where_filter: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch chunks within a specific chunk_num range for a source"""
        if start_chunk_num > end_chunk_num:
            return []

        try:
            # Build range filter
            range_filter = (
                Filter.by_property("source").equal(source)
                & Filter.by_property("chunk_num").greater_or_equal(start_chunk_num)
                & Filter.by_property("chunk_num").less_or_equal(end_chunk_num)
            )

            # Combine with existing where filter if provided
            if where_filter:
                range_filter = where_filter & range_filter

            # Query chunks in range
            response = self.store.collection.query.fetch_objects(
                filters=range_filter,
                return_properties=["chunk", "source", "chunk_num"],
                limit=end_chunk_num - start_chunk_num + 1,
            )

            # Convert to result format
            chunks = []
            for obj in response.objects:
                chunks.append(
                    {
                        "chunk": obj.properties.get("chunk", ""),
                        "source": obj.properties.get("source", ""),
                        "chunk_num": obj.properties.get("chunk_num", 0),
                    }
                )

            return chunks

        except Exception as e:
            LOG.warning(
                f"Error fetching chunks in range {start_chunk_num}-{end_chunk_num} for source '{source}': {e}"
            )
            return []

    def search_similar_chunks(
        self,
        query: str,
        search_mode: SearchMode = SearchMode.VECTOR,
        k: int = 5,
        min_score: Optional[float] = None,
        active_only: bool = True,
        source_filter: Optional[Union[str, List[str]]] = None,
        **search_kwargs,
    ) -> List[Dict[str, Any]]:
        try:
            strategy = SearchStrategyFactory.create_strategy(search_mode)
            where_filter = self._build_where_filter(active_only, source_filter)
            return strategy.search(
                self.store.collection,
                query,
                k,
                min_score,
                where_filter,
                **search_kwargs,
            )
        except Exception as e:
            LOG.error(f"Error in search ({search_mode.value}): {e}")
            return []

    def get_context_for_query(
        self,
        query: str,
        search_mode: SearchMode = SearchMode.VECTOR,
        k: int = 5,
        max_context_length: int = 8000,
        active_only: bool = True,
        source_filter: Optional[Union[str, List[str]]] = None,
        previous_k: int = 0,
        next_k: int = 0,
        **search_kwargs,
    ) -> Dict[str, Any]:
        results = self.search_similar_chunks(
            query,
            search_mode,
            k,
            active_only=active_only,
            source_filter=source_filter,
            **search_kwargs,
        )
        if not results:
            return {
                "context": "",
                "sources": [],
                "chunk_count": 0,
                "total_length": 0,
                "search_mode": search_mode.value,
            }

        # Get surrounding chunks if requested
        if previous_k > 0 or next_k > 0:
            enhanced_results = self._get_surrounding_chunks(
                results, previous_k, next_k, active_only, source_filter
            )
        else:
            enhanced_results = results

        context_parts = []
        sources = set()
        current_length = 0
        for result in enhanced_results:
            chunk = result["chunk"]
            source = result["source"]
            if current_length + len(chunk) > max_context_length and context_parts:
                break
            context_parts.append(chunk)
            sources.add(source)
            current_length += len(chunk)

        context = "\n\n".join(context_parts)
        return {
            "context": context,
            "sources": list(sources),
            "chunk_count": len(context_parts),
            "total_length": len(context),
            "search_mode": search_mode.value,
        }

    def search_and_generate_answer(
        self,
        query: str,
        answer_generator: "AnswerGenerator",
        search_mode: SearchMode = SearchMode.VECTOR,
        k: int = 5,
        active_only: bool = True,
        source_filter: Optional[Union[str, List[str]]] = None,
        bot_context: Optional[Dict[str, str]] = None,
        max_context_length: int = 5000,
        previous_k: int = 0,
        next_k: int = 0,
        **search_kwargs,
    ) -> Dict[str, Any]:

        try:
            # Use existing get_context_for_query method for search and context building
            context_result = self.get_context_for_query(
                query=query,
                search_mode=search_mode,
                k=k,
                max_context_length=max_context_length,
                active_only=active_only,
                source_filter=source_filter,
                previous_k=previous_k,
                next_k=next_k,
                **search_kwargs,
            )

            generated_answer = answer_generator.generate_answer(
                query=query,
                context=context_result["context"],
                sources=context_result["sources"],
                bot_context=bot_context,
            )

            LOG.info(
                f"Generated answer using {search_mode.value} search with {context_result['chunk_count']} chunks for query: '{query[:50]}...'"
            )
            return {
                "answer": generated_answer,
                "query": query,
                "search_mode": search_mode.value,
                "sources": context_result["sources"],
                "chunk_count": context_result["chunk_count"],
                "context_length": context_result["total_length"],
                "generation_type": "separate_llm",
            }
        except Exception as e:
            LOG.error(f"Error generating answer with {search_mode.value} search: {e}")
            return {
                "answer": None,
                "error": str(e),
                "query": query,
                "search_mode": search_mode.value,
                "sources": [],
                "chunk_count": 0,
                "context_length": 0,
                "generation_type": "no_llm",
            }

    def update_chunk_by_uuid(
        self,
        chunk_uuid: str,
        new_text: str,
        questions: List[str] = None,
        entities: List[str] = None,
    ) -> bool:
        """
        Update a specific chunk in Weaviate by its UUID

        Args:
            chunk_uuid: UUID of the chunk to update
            new_text: New text content for the chunk
            questions: List of related questions (optional)
            entities: List of related entities (optional)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # First check if the object exists
            try:
                existing = self.store.collection.query.fetch_object_by_id(
                    uuid=chunk_uuid, return_properties=["chunk"]
                )
                if not existing:
                    LOG.warning(f"Chunk with UUID {chunk_uuid} not found in Weaviate")
                    return False
            except Exception as e:
                LOG.warning(f"Chunk with UUID {chunk_uuid} not found: {e}")
                return False

            # Prepare update properties
            update_properties = {"chunk": new_text}

            # Add questions and entities if provided
            if questions is not None:
                update_properties["questions"] = questions
            if entities is not None:
                update_properties["entities"] = entities

            # Update the object in Weaviate (Weaviate will auto-generate embeddings)
            self.store.collection.data.update(
                uuid=chunk_uuid, properties=update_properties
            )

            LOG.info(f"Successfully updated chunk {chunk_uuid} in Weaviate")
            return True

        except Exception as e:
            LOG.exception(f"Error updating chunk {chunk_uuid} in Weaviate: {e}")
            return False

    def delete_chunk_by_uuid(self, chunk_uuid: str) -> bool:
        """
        Delete a specific chunk from Weaviate by its UUID

        Args:
            chunk_uuid: UUID of the chunk to delete

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # First check if the object exists
            try:
                existing = self.store.collection.query.fetch_object_by_id(
                    uuid=chunk_uuid, return_properties=["chunk"]
                )
                if not existing:
                    LOG.warning(f"Chunk with UUID {chunk_uuid} not found in Weaviate")
                    return False
            except Exception as e:
                LOG.warning(f"Chunk with UUID {chunk_uuid} not found: {e}")
                return False

            # Delete the object from Weaviate
            self.store.collection.data.delete_by_id(uuid=chunk_uuid)

            LOG.info(f"Successfully deleted chunk {chunk_uuid} from Weaviate")
            return True

        except Exception as e:
            LOG.exception(f"Error deleting chunk {chunk_uuid} from Weaviate: {e}")
            return False
