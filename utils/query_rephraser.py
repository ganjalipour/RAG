import re
from typing import Dict, List

from langchain_openai import ChatOpenAI


class QueryRephraser:
    def __init__(self, model_name: str, api_key: str):
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.3,
        )

    def rephrase_to_standalone(
        self, chat_history: List[Dict[str, str]], query: str
    ) -> str:
        formatted_history = ""
        for msg in chat_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history += f"{role}: {msg['content'].strip()}\n"

        prompt = f"""You are a helpful assistant. Your task is to rewrite the latest user question so that it is fully self-contained.
            Include any relevant information from earlier in the chat history that is needed to make the question understandable on its own.
            Chat history:
            {formatted_history}
            user:{query}
            Rewrite user question to be standalone:
            Return only the new standalone question as plain text.
        """

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            return query


def extract_last_turns(history: Dict, turns: int = 3) -> List[Dict[str, str]]:
    """
    Extract the last `turns` assistant-user message pairs.
    Returns: a flat list of message dicts (assistant/user) sorted chronologically.
    """
    messages = history["items"]
    pairs = []
    buffer = []

    # Walk backward to collect latest assistant-user pairs
    for item in reversed(messages):
        if item["type"] != "message":
            continue
        role = item["role"]
        content = " ".join(item["content"]).strip()

        if not content:
            continue

        buffer.insert(0, {"role": role, "content": content})
        if role == "user" and len(buffer) >= 2:
            # One pair completed (assistant followed by user)
            pairs.insert(0, buffer.copy())
            buffer.clear()

        if len(pairs) >= turns:
            break

    # Flatten the selected pairs
    flattened = [msg for pair in pairs for msg in pair]
    return flattened
