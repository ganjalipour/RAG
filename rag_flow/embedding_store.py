from typing import Any, Dict, List

import weaviate
from langchain_core.documents import Document
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter

from logger import LOG
from rag_flow.config import EmbeddingConfig


class EmbeddingStore:
    """Manages Weaviate storage operations"""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=config.weaviate_url,
            auth_credentials=Auth.api_key(config.weaviate_api_key),
            headers={
                "X-OpenAI-Api-Key": config.openai_api_key,
            },
        )
        self.collection = self.get_or_create_collection(config.collection_name)

    def get_or_create_collection(
        self, name: str
    ) -> weaviate.collections.collection.Collection:
        """Get or create a Weaviate collection"""
        if self.client.collections.exists(name):
            collection = self.client.collections.get(name)
            return collection

        self.client.collections.create(
            name=name,
            vectorizer_config=[
                Configure.NamedVectors.text2vec_openai(
                    name="vector",
                    source_properties=["chunk", "source", "questions", "entities"],
                    model="text-embedding-3-large",
                )
            ],
            generative_config=Configure.Generative.openai(
                model=self.config.openai_model_name
            ),
            properties=[
                Property(
                    name="chunk",
                    data_type=DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                Property(
                    name="source",
                    data_type=DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                Property(
                    name="active",
                    data_type=DataType.BOOL,
                    index_filterable=True,
                    index_searchable=False,
                ),
                Property(
                    name="chunk_num",
                    data_type=DataType.INT,
                    index_filterable=True,
                    index_searchable=False,
                ),
                Property(
                    name="questions",
                    data_type=DataType.TEXT_ARRAY,
                    index_filterable=False,
                    index_searchable=True,
                ),
                Property(
                    name="entities",
                    data_type=DataType.TEXT_ARRAY,
                    index_filterable=False,
                    index_searchable=True,
                ),
            ],
        )
        return self.client.collections.get(name)

    def save_embeddings(self, documents: List[Document]):
        """Save document embeddings to Weaviate with transactional rollback on failure"""
        added_object_ids = []  # Track successfully added objects for potential rollback

        try:
            # Get the next chunk number by counting existing chunks for this source
            next_chunk_num = self._get_next_chunk_num(documents)

            # Use larger batch size for better performance (200 is recommended)
            with self.collection.batch.fixed_size(batch_size=200) as batch:
                for idx, doc in enumerate(documents):
                    # Generate deterministic UUID for tracking
                    from weaviate.util import generate_uuid5

                    obj_uuid = generate_uuid5(
                        {
                            "chunk": doc.page_content,
                            "source": doc.metadata.get("source", "unknown"),
                        }
                    )

                    properties = {
                        "chunk": doc.page_content,
                        "source": doc.metadata.get("source", "unknown"),
                        "active": True,
                        "chunk_num": doc.metadata.get(
                            "chunk_num", next_chunk_num + idx
                        ),
                        "questions": doc.metadata.get("questions", []),
                        "entities": doc.metadata.get("entities", []),
                    }

                    batch.add_object(properties=properties, uuid=obj_uuid)

                    # Track the UUID for potential rollback
                    added_object_ids.append(obj_uuid)

                    # Stop if too many errors occur during batching
                    if batch.number_errors > 10:
                        LOG.error(
                            f"Batch import stopped due to excessive errors: {batch.number_errors}"
                        )
                        break

            # Check for failed objects after batch completion
            failed_objects = self.collection.batch.failed_objects
            failed_references = self.collection.batch.failed_references

            # Calculate success metrics
            total_attempted = len(documents)
            failed_count = len(failed_objects) if failed_objects else 0
            successful_count = total_attempted - failed_count

            # Log detailed results
            if failed_objects:
                LOG.warning(f"Number of failed imports: {len(failed_objects)}")
                LOG.warning(f"First failed object: {failed_objects[0]}")

            if failed_references:
                LOG.warning(f"Number of failed references: {len(failed_references)}")
                LOG.warning(f"First failed reference: {failed_references[0]}")

            # ALL-OR-NOTHING: If any failures occurred, rollback and raise exception
            if failed_count > 0:
                LOG.error(
                    f"Rolling back due to {failed_count} failed imports out of {total_attempted} attempted"
                )

                # Perform rollback of successfully added documents
                rollback_result = self._rollback_added_objects(
                    added_object_ids, successful_count
                )

                if rollback_result["rollback_success"]:
                    # Verify rollback was actually complete
                    verification_success = self._verify_rollback_complete(
                        added_object_ids
                    )

                    if verification_success:
                        LOG.info(
                            f"Successfully rolled back and verified deletion of {rollback_result['deleted_count']} documents"
                        )
                        raise Exception(
                            f"Batch import failed: {failed_count} documents failed. All {successful_count} successfully added documents have been rolled back and verified."
                        )
                    else:
                        LOG.error(
                            "CRITICAL: Rollback reported success but verification failed - some objects still exist!"
                        )
                        raise Exception(
                            f"Batch import failed AND rollback verification failed. Some documents may still exist in database. Manual cleanup required."
                        )
                else:
                    LOG.error(
                        f"CRITICAL: Rollback partially failed! {rollback_result['failed_deletions']} documents could not be deleted"
                    )
                    raise Exception(
                        f"Batch import failed AND rollback partially failed. {rollback_result['failed_deletions']} documents remain in database. Manual cleanup required."
                    )

            # Only reach here if no failures occurred
            LOG.info(f"Successfully added all {successful_count} documents")
            return {
                "status": "success",
                "success": True,
                "added": successful_count,
                "failed": failed_count,
                "total_attempted": total_attempted,
                "chunk_uuids": added_object_ids,
            }

        except Exception as e:
            # If an exception occurred during the batch process itself, attempt rollback
            if added_object_ids:
                LOG.error(
                    f"Exception during batch import, attempting rollback of {len(added_object_ids)} tracked objects"
                )
                rollback_result = self._rollback_added_objects(
                    added_object_ids, len(added_object_ids)
                )

                if rollback_result["rollback_success"]:
                    # Verify emergency rollback
                    verification_success = self._verify_rollback_complete(
                        added_object_ids
                    )
                    if verification_success:
                        LOG.info(
                            f"Emergency rollback successful and verified: deleted {rollback_result['deleted_count']} objects"
                        )
                    else:
                        LOG.error(
                            f"CRITICAL: Emergency rollback reported success but verification failed!"
                        )
                else:
                    LOG.error(
                        f"CRITICAL: Emergency rollback failed! {rollback_result['failed_deletions']} objects may remain"
                    )

            LOG.error(f"Error during batch import: {e}")
            raise

    def _get_next_chunk_num(self, documents: List[Document]) -> int:
        """Get the next chunk number by counting existing chunks for the sources in documents"""
        sources = set(doc.metadata.get("source", "unknown") for doc in documents)

        max_chunk_num = 0
        for source in sources:
            try:
                # Query for the highest chunk_num for this source
                response = self.collection.query.fetch_objects(
                    filters=Filter.by_property("source").equal(source),
                    return_properties=["chunk_num"],
                    limit=1000,  # Adjust based on expected chunk count
                )

                if response.objects:
                    # Find the maximum chunk_num for this source
                    source_max = max(
                        obj.properties.get("chunk_num", 0) for obj in response.objects
                    )
                    max_chunk_num = max(max_chunk_num, source_max)

            except Exception as e:
                LOG.warning(f"Error querying chunk numbers for source '{source}': {e}")
                # If we can't query, assume 0 as safe default
                pass

        return max_chunk_num + 1

    def _rollback_added_objects(self, object_ids, expected_count):
        """Rollback successfully added objects by deleting them"""
        if not object_ids:
            return {"rollback_success": True, "deleted_count": 0, "failed_deletions": 0}

        failed_deletions = 0

        try:
            # Use batch delete for efficiency
            with self.collection.batch.fixed_size(batch_size=100) as delete_batch:
                for obj_id in object_ids:
                    try:
                        # Delete by UUID
                        delete_batch.delete_object(uuid=obj_id)
                    except Exception as delete_error:
                        LOG.warning(f"Failed to delete object {obj_id}: {delete_error}")
                        failed_deletions += 1

            # Check batch delete results
            delete_failed_objects = self.collection.batch.failed_objects
            if delete_failed_objects:
                failed_deletions += len(delete_failed_objects)
                LOG.warning(f"Batch delete had {len(delete_failed_objects)} failures")

            deleted_count = expected_count - failed_deletions

            return {
                "rollback_success": failed_deletions == 0,
                "deleted_count": deleted_count,
                "failed_deletions": failed_deletions,
            }

        except Exception as rollback_error:
            LOG.error(f"Critical error during rollback: {rollback_error}")
            return {
                "rollback_success": False,
                "deleted_count": 0,
                "failed_deletions": expected_count,
                "error": str(rollback_error),
            }

    def _verify_rollback_complete(self, original_object_ids):
        """Verify that rollback was complete by checking if objects still exist"""
        remaining_objects = []

        for obj_id in original_object_ids:
            try:
                # Try to fetch the object
                obj = self.collection.query.fetch_object_by_id(obj_id)
                if obj:
                    remaining_objects.append(obj_id)
            except Exception:
                # Object not found (good - it was deleted)
                pass

        if remaining_objects:
            LOG.error(
                f"Rollback verification failed: {len(remaining_objects)} objects still exist"
            )
            return False
        else:
            LOG.info("Rollback verification successful: all objects removed")
            return True

    def get_objects_by_source(self, source_value: str) -> List[str]:
        """Get UUIDs of objects by source"""
        try:
            response = self.collection.query.fetch_objects(
                filters=Filter.by_property("source").equal(source_value)
            )
            uuids = [obj.uuid for obj in response.objects]
            LOG.info(f"Found {len(uuids)} objects with source='{source_value}'")
            return uuids
        except Exception as e:
            LOG.error(f"Error querying objects by source: {e}")
            return []

    def update_active_status_by_source(
        self, source_value: str, active_status: bool
    ) -> Dict[str, Any]:
        """Update active status for objects by source"""
        try:
            uuids = self.get_objects_by_source(source_value)
            if not uuids:
                LOG.info(f"No objects found with source='{source_value}'")
                return {"success": 0, "failed": 0, "total": 0}

            success_count = 0
            failed_count = 0
            with self.collection.batch.dynamic() as batch:
                for uuid in uuids:
                    try:
                        self.collection.data.update(
                            uuid=uuid, properties={"active": active_status}
                        )
                        success_count += 1
                    except Exception as e:
                        failed_count += 1
                        LOG.error(f"Failed to update UUID {uuid}: {e}")

            LOG.info(
                f"Updated {success_count}/{len(uuids)} objects to active={active_status}"
            )
            return {
                "success": success_count,
                "failed": failed_count,
                "total": len(uuids),
            }
        except Exception as e:
            LOG.error(f"Error updating objects by source: {e}")
            return {"success": 0, "failed": 0, "total": 0}

    def delete_objects_by_source(self, source_value: str) -> Dict[str, Any]:
        """Delete objects by source"""
        try:
            result = self.collection.data.delete_many(
                where=Filter.by_property("source").equal(source_value)
            )
            deleted_count = getattr(result, "successful", 0)
            failed_count = getattr(result, "failed", 0)
            LOG.info(f"Deleted {deleted_count} objects with source='{source_value}'")
            return {
                "deleted": deleted_count,
                "failed": failed_count,
                "confirmed": True,
                "message": f"Successfully deleted {deleted_count} objects",
            }
        except Exception as e:
            LOG.error(f"Error deleting objects by source: {e}")
            return {
                "deleted": 0,
                "failed": 0,
                "confirmed": False,
                "error": str(e),
                "message": f"Failed to delete objects: {str(e)}",
            }

    def delete_collection(self, collection_name: str = None) -> Dict[str, Any]:
        """Delete the entire Weaviate collection"""
        collection_name = collection_name or self.config.collection_name

        try:
            if self.client.collections.exists(collection_name):
                self.client.collections.delete(collection_name)
                LOG.info(f"Successfully deleted Weaviate collection: {collection_name}")
                return {
                    "success": True,
                    "collection_name": collection_name,
                    "message": f"Successfully deleted collection {collection_name}",
                }
            else:
                LOG.info(f"Collection {collection_name} does not exist")
                return {
                    "success": True,
                    "collection_name": collection_name,
                    "message": f"Collection {collection_name} does not exist",
                }
        except Exception as e:
            LOG.error(f"Error deleting collection {collection_name}: {e}")
            return {
                "success": False,
                "collection_name": collection_name,
                "error": str(e),
                "message": f"Failed to delete collection {collection_name}: {str(e)}",
            }
