import json
import re
from typing import List

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from logger import LOG
from rag_flow.config import EmbeddingConfig


class ContentEnhancer:
    """Generates questions and extracts entities from document chunks"""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.llm = ChatOpenAI(
            model=config.openai_model_name,
            api_key=config.openai_api_key,
            temperature=0.3,
        )

    def enhance_documents(self, documents: List[Document]) -> List[Document]:
        """Enhance documents with questions and entities"""
        enhanced_docs = []

        for doc in documents:
            try:
                # Generate questions about the chunk
                questions = self._generate_questions(
                    doc.page_content, doc.metadata.get("source", "")
                )

                # Extract entities from the chunk
                entities = self._extract_entities(doc.page_content)

                # Add to metadata
                doc.metadata["questions"] = questions
                doc.metadata["entities"] = entities

                enhanced_docs.append(doc)

                LOG.debug(
                    f"Enhanced chunk with {len(questions)} questions and {len(entities)} entities"
                )

            except Exception as e:
                LOG.error(f"Error enhancing document chunk: {e}")
                # Add empty arrays to maintain schema consistency
                doc.metadata["questions"] = []
                doc.metadata["entities"] = []
                enhanced_docs.append(doc)

        return enhanced_docs

    def _generate_questions(self, text: str, source: str = "") -> List[str]:
        """Generate 1-3 questions about the text chunk"""
        source_context = f"\n\nDocument source: {source}" if source else ""

        prompt = f"""
        Based on the following text, generate 1-3 specific questions that this text answers.
        Generate only the number of questions that are meaningful and relevant - don't force 3 questions if the text only supports 1-2 good questions.
        
        The questions should be:
        - In the same language as the original text
        - Specific and directly answerable by the text
        - Diverse (covering different aspects if possible)
        - Natural and realistic (questions a user might actually ask)
        - Consider the document source/filename for context if provided and if it's meaningful (ignore generic filenames like "document.pdf" or "file123.txt")
        
        Text:
        {text}{source_context}
        
        Return only the questions as a JSON array of strings, no other text.
        Example: ["What is the price of X?", "How does X work?"]
        """

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()

            # Try to parse JSON
            try:
                questions = json.loads(content)
                if isinstance(questions, list):
                    return [q.strip() for q in questions if q.strip()]
            except json.JSONDecodeError:
                # Fallback: extract questions from text
                questions = self._extract_questions_from_text(content)
                return questions

        except Exception as e:
            LOG.error(f"Error generating questions: {e}")

        return []

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from the text"""
        prompt = f"""
        Extract the main entities from the following text. Focus on:
        - People names
        - Services, companies, organizations
        - Locations (cities, countries, addresses)
        - Products, technologies, systems
        - Important concepts or terms
        
        Keep entities in the same language as the original text.
        
        Text:
        {text}
        
        Return only the entities (2-6 entities) as a JSON array of strings, no other text.
        Example: ["OpenAI", "ChatGPT", "San Francisco", "AI technology"]
        """

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()

            # Try to parse JSON
            try:
                entities = json.loads(content)
                if isinstance(entities, list):
                    return [e.strip() for e in entities if e.strip()]
            except json.JSONDecodeError:
                # Fallback: extract entities from text
                entities = self._extract_entities_from_text(content)
                return entities

        except Exception as e:
            LOG.error(f"Error extracting entities: {e}")

        return []

    def _extract_questions_from_text(self, text: str) -> List[str]:
        """Fallback method to extract questions from text response"""
        # Look for lines that end with question marks
        lines = text.split("\n")
        questions = []

        for line in lines:
            line = line.strip()
            if line.endswith("?"):
                # Remove any numbering or bullet points
                cleaned = re.sub(r"^\d+\.\s*", "", line)
                cleaned = re.sub(r"^\-\s*", "", cleaned)
                cleaned = re.sub(r"^\*\s*", "", cleaned)
                if cleaned:
                    questions.append(cleaned)

        return questions[:3]  # Max 3 questions

    def _extract_entities_from_text(self, text: str) -> List[str]:
        """Fallback method to extract entities from text response"""
        # Look for lines that might contain entities
        lines = text.split("\n")
        entities = []

        for line in lines:
            line = line.strip()
            if line and not line.endswith("?"):
                # Remove any numbering or bullet points
                cleaned = re.sub(r"^\d+\.\s*", "", line)
                cleaned = re.sub(r"^\-\s*", "", cleaned)
                cleaned = re.sub(r"^\*\s*", "", cleaned)
                if cleaned:
                    entities.append(cleaned)

        return entities[:10]  # Max 10 entities
