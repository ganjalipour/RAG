# rag_flow

`rag_flow` is a modular Python package for managing a Retrieval-Augmented Generation (RAG) pipeline using Weaviate for vector storage, document splitting, and search capabilities. It supports vector, keyword (BM25), and hybrid search modes, making it suitable for building question-answering systems and other text-retrieval applications.

## Features
- **Document Splitting**: Splits text into chunks for efficient embedding.
- **Embedding Storage**: Stores and manages embeddings in Weaviate with support for collections.
- **Search Modes**:
  - Vector search (dense, similarity-based).
  - Keyword search (BM25, sparse).
  - Hybrid search (combines vector and keyword).
- **RAG Support**: Generates answers using retrieved context with customizable prompts.
- **Modular Design**: Separates concerns into configuration, splitting, storage, and search strategies.

## Configuration
Set the following environment variables:
```bash
export WEAVIATE_URL="your_weaviate_cluster_url"
export WEAVIATE_API_KEY="your_weaviate_api_key"
export OPENAI_API_KEY="your_openai_api_key"
export OPENAI_MODEL_NAME="gpt-4o-mini"
```

## Directory Structure
- `config.py`: Manages configuration loading.
- `document_splitter.py`: Handles document splitting into chunks.
- `embedding_store.py`: Manages Weaviate storage operations.
- `search_strategy.py`: Implements search strategies (vector, keyword, hybrid).
- `embedding_manager.py`: Orchestrates the RAG pipeline.

## Usage
### Initializing the Embedding Manager
```python
from rag_flow.embedding_manager import EmbeddingManager, SearchMode
from rag_flow.config import EmbeddingConfig
from langchain_core.documents import Document

# Load configuration
config = EmbeddingConfig.load_from_env()
config.collection_name = "my_collection"

# Initialize manager
em = EmbeddingManager(config=config)
```

### Saving Documents
```python
# Example document
docs = [Document(page_content="Sample text about AI", metadata={"source": "ai_doc"})]

# Save embeddings
result = em.save_embeddings(docs)
print(result)  # {'success': True, 'added': 1, 'total': 1}
```

### Deleting Documents
```python
# Delete documents by source
result = em.delete_objects_by_source("ai_doc")
print(result)  # {'deleted': 1, 'failed': 0, 'confirmed': True, 'message': '...'}
```

### Search Examples
#### Vector Search
```python
results = em.search_similar_chunks(
    query="artificial intelligence",
    search_mode=SearchMode.VECTOR,
    k=5,
    min_score=0.7,
    distance=0.3
)
print(results)  # List of matching chunks with metadata
```

#### Keyword Search (BM25)
```python
from weaviate.classes.query import BM25Operator

results = em.search_similar_chunks(
    query="python programming",
    search_mode=SearchMode.KEYWORD,
    k=5,
    bm25_operator=BM25Operator.or_(minimum_match=2),
    query_properties=["chunk"]
)
print(results)  # List of matching chunks with metadata
```

#### Hybrid Search
```python
from weaviate.classes.query import HybridFusion

results = em.search_similar_chunks(
    query="neural networks",
    search_mode=SearchMode.HYBRID,
    k=5,
    alpha=0.7,
    fusion_type=HybridFusion.RELATIVE_SCORE,
    max_vector_distance=0.4
)
print(results)  # List of matching chunks with metadata
```

### Generating Answers (RAG)
```python
from rag_flow.answer_generator import AnswerGenerator
from llm_toolkit import LLMFactory

# Create AnswerGenerator instance
answer_llm = LLMFactory.create_model().get_model(temperature=0.1)
answer_generator = AnswerGenerator(answer_llm)

answer = em.search_and_generate_answer(
    query="What is machine learning?",
    search_mode=SearchMode.HYBRID,
    k=3,
    answer_generator=answer_generator,
    alpha=0.6,
    fusion_type=HybridFusion.RELATIVE_SCORE
)
print(answer["answer"])  # Generated answer
```

### Example in a Chatbot
```python
from rag_flow.embedding_manager import EmbeddingManager, SearchMode
from rag_flow.config import EmbeddingConfig
from rag_flow.answer_generator import AnswerGenerator
from llm_toolkit import LLMFactory
from weaviate.classes.query import BM25Operator

# Configure for a specific bot
config = EmbeddingConfig.load_from_env()
config.collection_name = "bot_123"

# Create AnswerGenerator instance
answer_llm = LLMFactory.create_model().get_model(temperature=0.1)
answer_generator = AnswerGenerator(answer_llm)

# Generate answer
answer = EmbeddingManager(config=config).search_and_generate_answer(
    query="What is AI?",
    search_mode=SearchMode.KEYWORD,
    k=8,
    answer_generator=answer_generator,
    bm25_operator=BM25Operator.or_(minimum_match=2),
    query_properties=["chunk"]
)
print(answer["answer"])
```

## Notes
- Ensure Weaviate is running and accessible at the specified `WEAVIATE_URL`.
- The package uses a singleton pattern for `EmbeddingManager` per collection to optimize resource usage.
- For testing, use `EmbeddingManager.clear_instances()` to reset singleton instances.

## Troubleshooting
- **Missing Environment Variables**: Verify all required environment variables are set.
- **Weaviate Connection Issues**: Check the Weaviate cluster URL and API key.
- **Search Errors**: Ensure `query_properties` and `bm25_operator` are correctly configured for keyword/hybrid searches.