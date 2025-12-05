from typing import Any, Dict, List, Optional
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_core.documents import Document

from logger import LOG
from rag_flow.config import EmbeddingConfig

from qdrant_client import models

import openai

class QdrantEmbeddingStore:
    """Manages Qdrant storage operations (equivalent to Weaviate version)"""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.client = QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key or None,
            check_compatibility=False,
            timeout=60.0,
        )
   
        self.collection_name = config.collection_name
        self._get_or_create_collection(self.collection_name)

    def _get_or_create_collection(self, name: str):
        collections = self.client.get_collections().collections
        if any(c.name == name for c in collections):
            LOG.info(f"✅ Collection '{name}' already exists.")
        else:
            LOG.info(f"📦 Creating Qdrant collection '{name}' ...")
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=3072,  # برای text-embedding-3-large
                    distance=models.Distance.COSINE,
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0
                ),
            )

        # ✅ اطمینان از اینکه فیلدهای مهم ایندکس شده‌اند
        indexed_fields = {
            "source": models.PayloadSchemaType.KEYWORD,
            "active": models.PayloadSchemaType.BOOL,
            "chunk_num": models.PayloadSchemaType.INTEGER,
        }

        for field, schema in indexed_fields.items():
            try:
                self.client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=schema
                )
                LOG.info(f"🟢 Indexed field '{field}' successfully.")
            except Exception as e:
                if "already exists" in str(e):
                    continue
                LOG.warning(f"⚠️ Could not index field '{field}': {e}")

    # def _get_or_create_collection(self, name: str):
    #     """Get or create Qdrant collection"""
    #     collections = self.client.get_collections().collections
    #     if any(c.name == name for c in collections):
    #         LOG.info(f"Collection '{name}' already exists.")
    #         return

    #     LOG.info(f"Creating Qdrant collection '{name}' ...")
    #     self.client.create_collection(
    #         collection_name=name,
    #         vectors_config=models.VectorParams(
    #             size=3072,  # text-embedding-3-large
    #             distance=models.Distance.COSINE
    #         )
    #     )

    # def save_embeddings(self, documents: List[Document], embeddings: List[List[float]]):
    #     """
    #     Save document embeddings to Qdrant (with rollback on failure).
    #     :param documents: list of LangChain Document objects
    #     :param embeddings: list of corresponding embeddings (same length)
    #     """
    #     if len(documents) != len(embeddings):
    #         raise ValueError("documents and embeddings length must match")

    #     added_ids = []
    #     try:
    #         points = []
    #         next_chunk_num = self._get_next_chunk_num(documents)

    #         for idx, (doc, vector) in enumerate(zip(documents, embeddings)):
    #             point_id = str(uuid4())
    #             payload = {
    #                 "chunk": doc.page_content,
    #                 "source": doc.metadata.get("source", "unknown"),
    #                 "active": True,
    #                 "chunk_num": doc.metadata.get("chunk_num", next_chunk_num + idx),
    #                 "questions": doc.metadata.get("questions", []),
    #                 "entities": doc.metadata.get("entities", []),
    #             }

    #             points.append(
    #                 models.PointStruct(
    #                     id=point_id,
    #                     vector=vector,
    #                     payload=payload
    #                 )
    #             )
    #             added_ids.append(point_id)

    #         self.client.upsert(
    #             collection_name=self.collection_name,
    #             points=points
    #         )
    #         LOG.info(f"Successfully added {len(points)} documents")
    #         return {"status": "success", "added": len(points)}

    #     except Exception as e:
    #         LOG.error(f"Error saving embeddings, starting rollback: {e}")
    #         if added_ids:
    #             self._rollback_added_objects(added_ids)
    #         raise

    def save_embeddings(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
    ):
        if embeddings is None:
            from openai import OpenAI
            client = OpenAI(api_key=self.config.openai_api_key)
            texts = [doc.page_content for doc in documents]
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=texts
            )
            embeddings = [d.embedding for d in response.data]

        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings length must match")

        added_ids = []
        try:
            points = []
            for idx, (doc, vector) in enumerate(zip(documents, embeddings)):
                point_id = str(uuid4())
                payload = {
                    "chunk": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "active": True,
                    "chunk_num": doc.metadata.get("chunk_num", idx),
                    "questions": doc.metadata.get("questions", []),
                    "entities": doc.metadata.get("entities", []),
                }
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                )
                added_ids.append(point_id)

            # ✅ ارسال در batch برای جلوگیری از timeout
            batch_size = 50
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                )
                LOG.info(f"✅ Uploaded batch {i // batch_size + 1}")

            LOG.info(f"🎯 Successfully added {len(points)} embeddings to Qdrant")
            return {"status": "success", "added": len(points)}

        except Exception as e:
            LOG.error(f"❌ Error saving embeddings, starting rollback: {e}")
            if added_ids:
                self._rollback_added_objects(added_ids)
            raise

    def _rollback_added_objects(self, added_ids: List[str]):
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=added_ids),
            )
            LOG.info(f"Rollback successful for {len(added_ids)} objects")
        except Exception as e:
            LOG.warning(f"Rollback failed: {e}")

    def _get_next_chunk_num(self, documents: List[Document]) -> int:
        """Get the max chunk_num for sources"""
        sources = set(doc.metadata.get("source", "unknown") for doc in documents)
        max_chunk_num = 0
        for source in sources:
            try:
                hits = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=models.Filter(
                        must=[models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=source)
                        )]
                    ),
                    limit=1000
                )
                points = hits[0]
                if points:
                    max_chunk_num = max(
                        p.payload.get("chunk_num", 0) for p in points
                    )
            except Exception as e:
                LOG.warning(f"Error querying chunk_num for source {source}: {e}")
        return max_chunk_num + 1

    def _rollback_added_objects(self, ids: List[str]):
        """Rollback added points by deleting them"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=ids)
            )
            LOG.info(f"Rollback successful for {len(ids)} objects")
        except Exception as e:
            LOG.error(f"Rollback failed: {e}")

    def get_objects_by_source(self, source_value: str) -> List[str]:
        """Get IDs of objects by source"""
        try:
            hits = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=source_value)
                    )]
                ),
                limit=1000
            )
            ids = [p.id for p in hits[0]]
            LOG.info(f"Found {len(ids)} objects with source='{source_value}'")
            return ids
        except Exception as e:
            LOG.error(f"Error getting objects by source: {e}")
            return []

    def update_active_status_by_source(self, source_value: str, active_status: bool) -> Dict[str, Any]:
        """Update active status of points by source"""
        try:
            ids = self.get_objects_by_source(source_value)
            if not ids:
                return {"success": 0, "failed": 0, "total": 0}

            for pid in ids:
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload={"active": active_status},
                    points=[pid]
                )

            LOG.info(f"Updated active status for {len(ids)} points")
            return {"success": len(ids), "failed": 0, "total": len(ids)}
        except Exception as e:
            LOG.error(f"Error updating active status: {e}")
            return {"success": 0, "failed": 0, "total": 0}

    def delete_objects_by_source(self, source_value: str) -> Dict[str, Any]:
        """Delete all points by source value"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=source_value)
                        )]
                    )
                )
            )
            LOG.info(f"Deleted all objects with source='{source_value}'")
            return {"deleted": True, "message": "Objects deleted"}
        except Exception as e:
            LOG.error(f"Error deleting by source: {e}")
            return {"deleted": False, "error": str(e)}

    def delete_collection(self, collection_name: str = None) -> Dict[str, Any]:
        """Delete entire collection"""
        collection_name = collection_name or self.collection_name
        try:
            self.client.delete_collection(collection_name)
            LOG.info(f"Deleted Qdrant collection: {collection_name}")
            return {"success": True, "message": f"Collection {collection_name} deleted"}
        except Exception as e:
            LOG.error(f"Error deleting collection: {e}")
            return {"success": False, "error": str(e)}

    def search(self, query_vector: List[float], limit: int = 5):
        """Search by vector"""
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            return results
        except Exception as e:
            LOG.error(f"Search error: {e}")
            return []
