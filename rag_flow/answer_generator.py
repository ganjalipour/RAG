"""Answer generator module for creating responses based on search results and bot context."""

from typing import Dict, List, Optional

from logger import LOG


class AnswerGenerator:
    """A class to handle answer generation based on search chunks and bot context using a language model.

    Attributes:
        llm: The language model instance.
    """

    def __init__(self, llm):
        """Initializes the AnswerGenerator with a language model.

        Args:
            llm: The language model to use for generation.
        """
        self.llm = llm

    def generate_answer(
        self,
        query: str,
        context: str,
        sources: List[str] = None,
        bot_context: Optional[Dict[str, str]] = None,
    ) -> str:
        """Generate an answer based on query, context, and bot context.

        Args:
            query: The user's question
            context: Pre-built context string from search results
            sources: List of source names (for reference) - currently unused
            bot_context: Dict containing bot's name, description, system_prompt

        Returns:
            Generated answer as a string
        """
        try:
            # Build bot identity
            bot_identity = self._build_bot_identity(bot_context)

            # Create prompt
            prompt = self._create_prompt(bot_identity, query, context)

            # Generate response
            response = self.llm.invoke(prompt)
            generated_answer = (
                response.content.strip()
                if hasattr(response, "content")
                else str(response).strip()
            )

            LOG.debug(f"Generated answer for query: {query[:50]}...")
            return generated_answer

        except Exception as e:
            LOG.error(f"Failed to generate answer: {e}")
            return "I don't have specific information about your question in my knowledge base. Please try rephrasing your question or provide more details."

    def _build_bot_identity(self, bot_context: Optional[Dict[str, str]]) -> str:
        """Build bot identity from context."""
        if not bot_context:
            return "You are a helpful assistant."

        bot_name = bot_context.get("name", "Assistant")
        bot_desc = bot_context.get("description", "")
        bot_company_name = bot_context.get("company_name", "")
        system_prompt = bot_context.get("system_prompt", "")

        # Build comprehensive bot identity
        if system_prompt:
            bot_identity = system_prompt
        elif bot_desc:
            bot_identity = f"You are {bot_name}, a specialized assistant. Your role and expertise: {bot_desc}"
        else:
            bot_identity = f"You are {bot_name}, a helpful assistant."

        # Add name and description as additional context if we have a system prompt
        if system_prompt and bot_name:
            bot_identity += f"\n\nYour name is {bot_name}"
        if bot_company_name:
            bot_identity += (
                f"\n\nYou represent the following company: {bot_company_name}"
            )
        if system_prompt and bot_desc:
            bot_identity += (
                f"\n\nAdditional context about your role and expertise: {bot_desc}"
            )

        return bot_identity

    def _create_prompt(self, bot_identity: str, query: str, context: str) -> str:
        """Create the prompt for answer generation."""
        prompt = f"""{bot_identity}

User's question: "{query}"

"""

        if context:
            prompt += f"""Relevant context from your knowledge base:\n{context}"""

            prompt += """

Please respond according to these guidelines:
1. Use the provided context to give a comprehensive and accurate answer
2. If the context doesn't fully answer the question but is related, use it as much as possible and supplement with your general knowledge within your area of expertise
3. Always maintain your defined personality and role
4. Be helpful and professional

Your response:"""
        else:
            prompt += """No specific context was found in your knowledge base for this question.

Please respond according to these guidelines:
1. If the question is within your area of expertise as defined above, provide a helpful general answer based on your knowledge
2. If the question is outside your specialty, politely explain that you don't have specific information about this topic
3. In both cases, suggest the user try rephrasing their question or provide more specific details
4. If the question is small talk, go along with it and try to shift the conversation to your area of expertise
5. Always maintain your defined personality and role
6. Be helpful and professional

Your response:"""

        return prompt
