from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from logger import LOG
from rag_flow.config import EmbeddingConfig


class ConversationEnhancer:
    """Enhances user queries with conversation context and reformulates questions"""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.llm = ChatOpenAI(
            model=config.openai_model_name,
            api_key=config.openai_api_key,
            temperature=0.1,  # Lower temperature for more consistent reformulation
        )

    def reformulate_question(
        self, current_query: str, conversation_history: List[Dict[str, Any]]
    ) -> str:
        """
        Reformulate the current query to be standalone using conversation history

        Args:
            current_query: The user's current question/message
            conversation_history: List of previous messages in format:
                [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

        Returns:
            Reformulated standalone question
        """
        # If no history, return as-is
        if not conversation_history or len(conversation_history) == 0:
            LOG.info(
                f"No conversation history, returning original query: {current_query}"
            )
            return current_query

        LOG.info(
            f"Reformulating query with {len(conversation_history)} history messages: {current_query}"
        )

        try:
            # Create reformulation prompt with conversation context
            reformulation_prompt = self._create_reformulation_prompt(
                current_query, conversation_history
            )

            # Get reformulated question
            messages = [
                SystemMessage(content=reformulation_prompt),
                HumanMessage(content=current_query),
            ]

            response = self.llm.invoke(messages)
            reformulated_query = response.content.strip()

            # Validate the reformulated query
            if reformulated_query and len(reformulated_query) > 10:
                LOG.info(
                    f"Reformulated query: '{current_query}' -> '{reformulated_query}'"
                )
                return reformulated_query
            else:
                LOG.warning(
                    f"Invalid reformulated query, using original: {current_query}"
                )
                return current_query

        except Exception as e:
            LOG.error(f"Error reformulating question: {e}")
            return current_query

    def _build_context_messages(
        self, conversation_history: List[Dict[str, Any]]
    ) -> List[str]:
        """Build context from conversation history"""
        context_parts = []

        # Take last 6 messages for context (3 exchanges)
        recent_history = (
            conversation_history[-6:]
            if len(conversation_history) > 6
            else conversation_history
        )

        for msg in recent_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                context_parts.append(f"User: {content}")
            elif role == "assistant":
                context_parts.append(f"Assistant: {content}")

        return context_parts

    def _create_reformulation_prompt(
        self, current_query: str, conversation_history: List[Dict[str, Any]]
    ) -> str:
        """Create prompt for question reformulation"""

        # Build conversation context
        context_parts = self._build_context_messages(conversation_history)
        context_text = "\n".join(context_parts)

        prompt = f"""
You are a conversation context analyzer. Your task is to reformulate user questions to be standalone and clear.

Given the conversation history and the current user query, reformulate the current query to be a complete, standalone question that includes all necessary context from the conversation.

Conversation History:
{context_text}

Current User Query: {current_query}

Instructions:
1. If the current query references previous context (like "it", "this", "that", "more about X"), replace those references with specific details from the conversation history
2. If the query is incomplete or unclear, add context to make it complete
3. Keep the same language as the original query
4. Make the reformulated question natural and conversational
5. If the query is already standalone and clear, you may return it unchanged
6. Focus on maintaining the user's intent while making the question self-contained

Return only the reformulated question, no additional text or explanation.
"""

        return prompt

    def prepare_conversation_context(
        self, conversation_history: List[Dict[str, Any]], max_context_length: int = 2000
    ) -> str:
        """
        Prepare conversation context for RAG system

        Args:
            conversation_history: List of conversation messages
            max_context_length: Maximum length of context to include

        Returns:
            Formatted conversation context string
        """
        if not conversation_history:
            return ""

        # Take recent history that fits within max_context_length
        context_parts = []
        current_length = 0

        # Process in reverse order to get most recent messages first
        for msg in reversed(conversation_history):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                formatted_msg = f"User: {content}"
            elif role == "assistant":
                formatted_msg = f"Assistant: {content}"
            else:
                continue

            # Check if adding this message would exceed max length
            if current_length + len(formatted_msg) > max_context_length:
                break

            context_parts.insert(
                0, formatted_msg
            )  # Insert at beginning to maintain order
            current_length += len(formatted_msg)

        return "\n".join(context_parts)
