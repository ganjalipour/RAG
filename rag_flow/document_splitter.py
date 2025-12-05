from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

from logger import LOG


class DocumentSplitter:
    """Handles document splitting into chunks"""

    def __init__(self, separator: str, chunk_size: int, chunk_overlap: int):
        self.text_splitter = CharacterTextSplitter(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=True,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        try:
            # Normalize text from image OCR - convert literal \n to actual newlines
            normalized_documents = []
            for doc in documents:
                normalized_content = doc.page_content.replace("\\n", "\n")
                normalized_doc = Document(
                    page_content=normalized_content, metadata=doc.metadata.copy()
                )
                normalized_documents.append(normalized_doc)

            texts = self.text_splitter.split_documents(normalized_documents)
            LOG.info(f"Split {len(documents)} documents into {len(texts)} chunks")

            # Add chunk numbering to initial splits (start from 1 to match Weaviate)
            for i, doc in enumerate(texts):
                doc.metadata["chunk_num"] = i + 1

            # Calculate original content length for validation
            original_content_length = sum(len(doc.page_content) for doc in texts)

            # Ensure no chunk is less than 200 characters
            filtered_texts = self._merge_small_chunks(texts, min_chars=50)
            LOG.info(f"After merging small chunks: {len(filtered_texts)} chunks")

            # Renumber chunks after merging (start from 1 to match Weaviate)
            for i, doc in enumerate(filtered_texts):
                doc.metadata["chunk_num"] = i + 1

            # Validate content preservation
            merged_content_length = sum(len(doc.page_content) for doc in filtered_texts)
            content_diff = abs(original_content_length - merged_content_length)

            # Allow for some difference due to added spaces during merging
            max_allowed_diff = (
                len(texts) * 2
            )  # Up to 2 chars per original chunk for spaces
            if content_diff > max_allowed_diff:
                LOG.warning(
                    f"Content length mismatch: original={original_content_length}, "
                    f"merged={merged_content_length}, diff={content_diff}"
                )
            else:
                LOG.info(
                    f"Content validation passed: diff={content_diff} chars (within {max_allowed_diff} allowed)"
                )

            for i, doc in enumerate(filtered_texts):
                LOG.debug(f"Chunk {i} metadata: {doc.metadata}")
            return filtered_texts
        except Exception as e:
            LOG.error(f"Error splitting documents: {e}")
            raise

    def _merge_small_chunks(
        self, chunks: List[Document], min_chars=200
    ) -> List[Document]:
        """Merge chunks that are less than min_chars characters with adjacent chunks"""
        if not chunks:
            return chunks

        merged_chunks = []
        i = 0

        while i < len(chunks):
            current_chunk = chunks[i]

            # If current chunk is less than min_chars characters
            if len(current_chunk.page_content) < min_chars:
                merged_content = current_chunk.page_content
                merged_metadata = current_chunk.metadata.copy()
                j = i + 1

                # Try to merge with next chunks, but stop if we encounter a large chunk
                while j < len(chunks) and len(merged_content) < min_chars:
                    next_chunk = chunks[j]

                    # If next chunk is large enough on its own, only take part of it
                    if len(next_chunk.page_content) >= min_chars:
                        # Only add enough content to reach min_chars, leave rest as separate chunk
                        chars_needed = min_chars - len(merged_content)
                        if chars_needed > 0:
                            # Find a good break point (space) near the needed length
                            content_to_add = next_chunk.page_content[
                                : chars_needed + 100
                            ]  # Add buffer
                            last_space = content_to_add.rfind(" ")
                            if (
                                last_space > chars_needed // 2
                            ):  # Use space if it's not too early
                                content_to_add = content_to_add[:last_space]
                            else:
                                content_to_add = content_to_add[:chars_needed]

                            merged_content += " " + content_to_add

                            # Create a new chunk with remaining content
                            remaining_content = next_chunk.page_content[
                                len(content_to_add) :
                            ].strip()
                            if remaining_content:
                                remaining_chunk = Document(
                                    page_content=remaining_content,
                                    metadata=next_chunk.metadata.copy(),
                                )
                                # Preserve original chunk_num for the remaining chunk
                                remaining_chunk.metadata["chunk_num"] = (
                                    next_chunk.metadata.get("chunk_num", j + 1)
                                )
                                # Insert the remaining chunk back into the list
                                chunks[j] = remaining_chunk
                            else:
                                # No remaining content, skip this chunk
                                j += 1
                        break
                    else:
                        # Next chunk is also small, merge it completely
                        merged_content += " " + next_chunk.page_content
                        j += 1

                # If still less than min_chars and we have previous chunks, merge with the last one
                if len(merged_content) < min_chars and merged_chunks:
                    # Merge with the previous chunk
                    last_chunk = merged_chunks.pop()
                    merged_content = last_chunk.page_content + " " + merged_content
                    merged_metadata = last_chunk.metadata.copy()

                # Create merged document with original chunk_num from first chunk
                merged_doc = Document(
                    page_content=merged_content, metadata=merged_metadata
                )
                # Keep the chunk_num from the first chunk in the merge
                merged_doc.metadata["chunk_num"] = current_chunk.metadata.get(
                    "chunk_num", i + 1
                )
                merged_chunks.append(merged_doc)
                i = j
            else:
                # Chunk is already >= min_chars characters
                merged_chunks.append(current_chunk)
                i += 1

        return merged_chunks
