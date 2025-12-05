import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
from livekit import api
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    get_job_context,
    llm,
)
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.plugins.cartesia.models import TTSDefaultVoiceId

from weaviate.collections.classes.grpc import HybridFusion

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from helpers.decorator import TimeDecorator
from logger import LOG
from rag_flow.config import EmbeddingConfig
from utils.query_rephraser import QueryRephraser, extract_last_turns
from rag_flow.embedding_manager import EmbeddingManager
from rag_flow.search_strategy import SearchMode
from utils import get_conf

LANGUAGE_DICT = {
    "Urdu": "ur",
    "English": "en",
    "Turkish": "tr",
    "Japanese": "ja",
    "Hindi": "hi",
    "Afrikaans": "af",
    "Arabic": "ar",
    "Armenian": "hy",
    "Azerbaijani": "az",
    "Belarusian": "be",
    "Bosnian": "bs",
    "Bulgarian": "bg",
    "Catalan": "ca",
    "Chinese": "zh",
    "Cantonese": "yue",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "Estonian": "et",
    "Filipino": "fil",
    "Finnish": "fi",
    "French": "fr",
    "Galician": "gl",
    "Georgian": "ka",
    "German": "de",
    "Greek": "el",
    "Gujarati": "gu",
    "Hebrew": "iw",
    "Hungarian": "hu",
    "Icelandic": "is",
    "Indonesian": "id",
    "Italian": "it",
    "Kannada": "kn",
    "Kazakh": "kk",
    "Korean": "ko",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Macedonian": "mk",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Spanish": "es",
    "Thai": "th",
    "Ukrainian": "uk",
    "Vietnamese": "vi",
    "Persian": "fa",
    "Serbian": "sr",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Swahili": "sw",
    "Swedish": "sv",
    "Tagalog": "tl",
    "Tamil": "ta",
    "Welsh": "cy",
    "Malay": "ms",
    "Marathi": "mr",
    "Maori": "mi",
    "Nepali": "ne",
}
DEEPGRAM_CARTESIA_LANGUAGES = [
    "English",
    "Dutch",
    "French",
    "German",
    "Hindi",
    "Italian",
    "Japanese",
    "Korean",
    "Polish",
    "Portuguese",
    "Russian",
    "Spanish",
    "Swedish",
    "Turkish",
]

VOICE_WEBHOOK_ENDPOINT = "calls/call-webhook/"
BOT_CONFIG_ENDPOINT = "bots/api/config/"

# Config
CONFIG = {
    "project_env": get_conf("PROJECT_ENV"),
    "livekit_url": get_conf("LIVEKIT_URL"),
    "livekit_api_key": get_conf("LIVEKIT_API_KEY"),
    "livekit_api_secret": get_conf("LIVEKIT_API_SECRET"),
    "bot_manager_url": get_conf("BOT_MANAGER_URL"),
    "openai_model_name": get_conf("OPENAI_MODEL_NAME"),
    "openai_api_key": get_conf("OPENAI_API_KEY"),
    "cartesia_tts": get_conf("CARTESIA_TTS_MODEL"),
    "cartesia_tts_speed": get_conf("CARTESIA_TTS_SPEED"),
    "cartesia_voice_id": get_conf("CARTESIA_TR_VOICE_ID"),
    "cartesia_api_key": get_conf("CARTESIA_API_KEY"),
    "deepgram_stt": get_conf("DEEPGRAM_STT_MODEL"),
    "deepgram_api_key": get_conf("DEEPGRAM_API_KEY"),
    "openai_voice": get_conf("OPENAI_VOICE"),
    "activation_threshold": float(get_conf("ACTIVATION_THRESHOLD", 0.5)),
    "min_silence_duration": float(get_conf("MIN_SILENCE_DURATION", 0.9)),
    "min_speech_duration": float(get_conf("MIN_SPEECH_DURATION", 0.05)),
    "auth_key": get_conf("ADMIN_KEY"),
}


class ConversationRecorder:
    """Handles saving conversation data to the Django webhook."""

    @staticmethod
    @TimeDecorator.exe_time
    async def save_conversation(
        bot_id: str,
        call_id: str,
        conversation: List[Tuple[str, str]]= [],
        call_start_time: float = 0,
        call_status: str = None,
        history: Optional[Dict] = None,
    ) -> None:
        """Save conversation data to Dashboard webhook.

        Args:
            bot_id: Identifier for the bot.
            call_id: Identifier for the call.
            conversation: List of (user_message, bot_response) tuples.
            call_start_time: Timestamp when the call started.
            call_status: Status of the call (e.g., CONNECTED, ANSWERED).
            history: Session history in JSON format.
        """
        try:
            history_items = []

            if history and "items" in history:
                for item in history["items"]:
                    if item.get("type") == "message" and item.get("content"):
                        content = (
                            "\n".join(item["content"])
                            if isinstance(item["content"], list)
                            else item["content"]
                        )
                        history_items.append(
                            {"role": item.get("role"), "content": content}
                        )
            else:
                # Fallback: build from flat conversation tuples (in order)
                for user_msg, bot_resp in conversation:
                    if user_msg:
                        history_items.append({"role": "user", "content": user_msg})
                    if bot_resp:
                        history_items.append({"role": "assistant", "content": bot_resp})

            call_duration = time.time() - call_start_time if call_start_time else 0

            if not history_items and not call_status:
                call_status = "Not ANSWERED"
                call_duration = 0
            payload = {
                "bot_id": bot_id,
                "call_id":call_id,
                "history": history_items,
                "call_duration": call_duration,
                "end_time": datetime.utcnow().isoformat(),
                "status": call_status or "COMPLETED",
            }
            if call_id:
                response = requests.post(
                    f"{CONFIG['bot_manager_url']}{VOICE_WEBHOOK_ENDPOINT}",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=5,
                )
                response.raise_for_status()
                LOG.info("Conversation saved successfully")
        except requests.RequestException as e:
            LOG.error(f"Failed to save conversation: {e}")


def build_bot_identity(bot_context: Optional[Dict[str, str]], language: str) -> str:
    """Build bot identity from context."""
    if not bot_context:
        return "You are a helpful assistant."

    bot_name = bot_context.get("name", "")
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
    if bot_name:
        bot_identity += (
            f"\n\nYour name is {bot_name}. \n Response in {language} language."
        )
    if bot_company_name:
        bot_identity += f"\n\nYou represent the following company: {bot_company_name}"
    if system_prompt and bot_desc:
        bot_identity += (
            f"\n\nAdditional context about your role and expertise: {bot_desc}"
        )
    bot_identity += "Keep your response ideal for a phone call, not too long, but provides full answer"
    bot_identity += "Do not use commas in prices, use points when needed. (7,000 -> 7000, or 7,500 -> 7.500)"
    bot_identity += "Correct the text to make it accurate, natural, and aligned with standard dental terminology"
    bot_identity += "Always use _search_knowledge_base_direct to get info"
    return bot_identity


class ContextAwareAgent(Agent):
    """An agent that can answer questions using RAG (Retrieval Augmented Generation)."""

    def __init__(self, bot_id: str, language: str, bot_config: dict, call_start_time: float = 0) -> None:
        """Initialize the RAG-enabled agent with bot_id."""
        bot_identity = build_bot_identity(bot_config, language)
        super().__init__(instructions=bot_identity)
        self._seen_results = set()
        self.bot_id = bot_id
        self.language = language
        self.greeting_message = bot_config.get(
            "first_message", "Hi, How can i assist you?"
        )
        self.call_id = bot_config.get('call_id')
        self.processed_call_ids = set()
        self.call_start_time = call_start_time
        self.__init_embed()

    @TimeDecorator.exe_time
    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> str | None:
        """Called when the user has finished speaking, before the LLM responds.

        This ensures EVERY user query gets knowledge base context automatically.
        """
        # Extract user message text
        print("userrrrrturn")
        user_text = ""
        if isinstance(new_message.content, str):
            user_text = new_message.content
        elif isinstance(new_message.content, list):
            user_text = " ".join(str(item) for item in new_message.content)

        LOG.info(f"🎤 USER INPUT: {user_text}")

        if self.call_id not in self.processed_call_ids:
            await ConversationRecorder.save_conversation(
                bot_id=self.bot_id, call_id=self.call_id, call_status="ONGOING", call_start_time=self.call_start_time
            )
            self.processed_call_ids.add(self.call_id)

        chat_history = extract_last_turns(self.session.history.to_dict(), turns=3)
        if chat_history:
            rephraser = QueryRephraser(
                model_name="gpt-4.1-nano", api_key=CONFIG["openai_api_key"]
            )
            standalone = rephraser.rephrase_to_standalone(chat_history, user_text)
            LOG.info(f"Rephrased USER INPUT: {standalone=}")
        else:
            standalone = user_text

        if standalone:
            try:
                context = await self._search_knowledge_base_direct(standalone)
                if context:
                    turn_ctx.add_message(
                        role="assistant",
                        content=f"RELEVANT CONTEXT FROM KNOWLEDGE BASE:\n{context}\n\nUse the context to provide accurate and helpful responses. If the context doesn't contain relevant information for the user's question, you can still provide a helpful response based on your general knowledge.",
                    )
                else:
                    LOG.info("📚 RAG CONTEXT: No relevant context found")

            except Exception as e:
                LOG.error(f"Error retrieving knowledge base context: {e}")

    def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool | llm.RawFunctionTool],
        model_settings,
    ):
        """Override LLM node to capture responses for quality monitoring."""
        result = self._llm_node_with_logging(chat_ctx, tools, model_settings)
        return result

    async def _llm_node_with_logging(self, chat_ctx, tools, model_settings):
        """LLM node wrapper that logs responses for quality monitoring."""
        llm_start = time.time()
        response_chunks = []

        async for chunk in Agent.default.llm_node(
            self, chat_ctx, tools, model_settings
        ):
            response_chunks.append(chunk)
            yield chunk

        llm_time = time.time() - llm_start
        LOG.debug(f"LLM answer generation time: {llm_time:.4f} seconds")

    @TimeDecorator.exe_time
    def __init_embed(self):
        """Initialize the embedding manager with configuration."""
        try:
            config = EmbeddingConfig.load_from_env()
            config.collection_name = f"Bot_{self.bot_id}"
            self.embedding_manager = EmbeddingManager(config=config)
        except Exception as e:
            raise RuntimeError(f"EmbeddingProcessor initialization failed: {str(e)}")

    @function_tool()
    @TimeDecorator.exe_time
    async def _search_knowledge_base_direct(self, query: str) -> str:
        """Direct search of knowledge base without RunContext dependency.

        Args:
            query: The user's query to search for in the document collection.

        Returns:
            A string containing the relevant document context.
        """
        if not query or not isinstance(query, str):
            LOG.warning("Invalid or empty query received")
            return ""

        LOG.info(f"Processing document search for query: {query}")

        try:
            results = self.embedding_manager.get_context_for_query(
                query=query,
                search_mode=SearchMode.HYBRID,
                k=7,
                previous_k=0,
                next_k=0,
                alpha=0.7,
                fusion_type=HybridFusion.RELATIVE_SCORE,
                max_vector_distance=0.8,
            )
            results = results.get("context", "")
            if not results:
                LOG.info("No relevant documents found")
                return ""

            # Handle different result types
            if isinstance(results, list):
                # If it's a list of strings, join them
                context_response = " ".join(str(item) for item in results)
            elif isinstance(results, str):
                # If it's already a string, use it directly
                context_response = results
            else:
                # Convert to string if it's something else
                context_response = str(results)

            LOG.debug(f"Raw results type: {type(results)}, Raw results: {results}")
            return context_response
        except Exception as e:
            LOG.error(f"Document search failed: {str(e)}")
            return ""

    async def on_enter(self):
        """Called when the agent enters the session."""
        self.session.generate_reply(
            instructions=f"Always say: {self.greeting_message} in {self.language} language",
            allow_interruptions=True,
        )

    # @function_tool
    # @TimeDecorator.exe_time
    # async def search_knowledge_base(self, query: str) -> str:
    #     """Direct search of knowledge base without RunContext dependency.
    #
    #     Args:
    #         query: The user's query to search for in the document collection.
    #
    #     Returns:
    #         A string containing the relevant document context.
    #     """
    #     if not query or not isinstance(query, str):
    #         LOG.warning("Invalid or empty query received")
    #         return ""
    #
    #     LOG.info(f"Processing document search for query: {query}")
    #
    #     try:
    #         results = self.embedding_manager.get_context_for_query(
    #             query=query,
    #             search_mode=SearchMode.HYBRID,
    #             k=15,
    #             previous_k=0,
    #             next_k=0,
    #             alpha=0.7,
    #             fusion_type=HybridFusion.RELATIVE_SCORE,
    #             max_vector_distance=0.8,
    #         )
    #         results = results.get("context", "")
    #         if not results:
    #             LOG.info("No relevant documents found")
    #             return ""
    #
    #         if isinstance(results, list):
    #             context_response = " ".join(str(item) for item in results)
    #         elif isinstance(results, str):
    #             context_response = results
    #         else:
    #             context_response = str(results)
    #
    #         LOG.debug(f"Raw results type: {type(results)}, Raw results: {results}")
    #
    #         return context_response
    #     except Exception as e:
    #         LOG.error(f"Document search failed: {str(e)}")
    #         return ""

    @function_tool
    @TimeDecorator.exe_time
    async def end_call(self):
        """Permanently end the call session.
        Use this function only after the conversation has been fully completed,
        meaning the user has explicitly indicated they wish to end the interaction
        (e.g., saying goodbye, thanking you, or stating that no further help is needed).

        Do not call this just because the user stops speaking;
        always ensure any final response has been delivered before closing the session.
        """
        try:
            await self.session.generate_reply(
                instructions=f"Good bye message in {self.language} language.",
            )
        except Exception as e:
            LOG.error(f"Error during goodbye message playback: {e}")

        try:
            ctx = get_job_context()
            if ctx is None:
                LOG.warning("Not running in a job context, cannot hang up call")
                return
            try:
                await ctx.api.room.delete_room(
                    api.DeleteRoomRequest(room=ctx.room.name)
                )
                LOG.info(f"Call ended.")
                LOG.info(f"Room `{ctx.room.name}` deleted successfully")
            except Exception as e:
                LOG.error(f"Failed to delete room: {e}")
        except Exception as e:
            LOG.error(f"Error during hangup: {e}")


class VoiceAgent:
    """Manages the voice agent's lifecycle and interactions."""

    def __init__(self):
        self.call_id = None
        self.bot_config = {}
        self.call_start_time = 0
        self.conversation: List[Tuple[str, str]] = []
        self.bot_id: str = ""
        self.latest_user_message: Optional[str] = None
        self.session = None

    @TimeDecorator.exe_time
    def setup_speech(self, language: str, lang_code: str):
        """Configures speech-to-text (STT) and text-to-speech (TTS) services based on the input language.

        Args:
            language: The name of the language (e.g., 'English', 'Spanish').
            lang_code: The language code (e.g., 'en', 'es').

        Returns:
            tuple: A pair of (stt, tts) service objects configured for the specified language.
        """
        # tts = openai.TTS(
        #     voice=CONFIG["openai_voice"],
        #     instructions=f"You are a friendly and knowledgeable assistant with a pleasant tone."
        #                  f" Please respond in the {language} (White dialect).",
        # )
        if language in DEEPGRAM_CARTESIA_LANGUAGES:
            tts = cartesia.TTS(
                api_key=CONFIG["cartesia_api_key"],
                voice=CONFIG["cartesia_voice_id"]
                if lang_code == "tr"
                else TTSDefaultVoiceId,
                speed=CONFIG["cartesia_tts_speed"],
                language=lang_code,
                model=CONFIG["cartesia_tts"],
            )
            stt = deepgram.STT(
                api_key=CONFIG["deepgram_api_key"],
                model=CONFIG["deepgram_stt"],
                language=lang_code,
            )
        else:
            tts = openai.TTS(
                voice=CONFIG["openai_voice"],
                instructions=f"You are a friendly and knowledgeable assistant with a pleasant tone."
                f" Please respond in the {language} (White dialect).",
            )
            stt = openai.STT(model="gpt-4o-transcribe", language=lang_code)
        return stt, tts

    @TimeDecorator.exe_time
    def fetch_bot_config(self, bot_id: str) -> dict:
        """Fetches the bot configuration from the API.

        Args:
            bot_id: The ID of the bot to fetch config for.

        Returns:
            dict: The bot configuration dictionary if successful, else empty dict.
        """

        if not bot_id:
            return {}

        url = f"{CONFIG['bot_manager_url']}{BOT_CONFIG_ENDPOINT}"
        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={"bot_id": int(bot_id), "auth_key": CONFIG["auth_key"]},
                verify=False,
                timeout=5,
            )
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success" and "bot" in data:
                LOG.info(f"Fetched config for bot_id={bot_id}: {data['bot']}")
                return data["bot"]
            LOG.warning(f"Unexpected response for bot_id={bot_id}: {data}")
        except Exception as e:
            LOG.error(f"Failed to fetch bot config for bot_id={bot_id}: {e}")
        return {}

    @TimeDecorator.exe_time
    async def initialize(self, ctx: JobContext) -> None:
        """Initialize the voice agent with room settings and start the session."""
        room_name = ctx.room.name or ""
        call_id, self.bot_id, language = '3', '8', "turkish"

        self.bot_config = self.fetch_bot_config(self.bot_id)
        self.call_id = self.bot_config.get('call_id')
        stt, tts = self.setup_speech(
            language=language, lang_code=LANGUAGE_DICT.get(language, "tr")
        )
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

        if ctx.room.connection_state == 1:
            await ConversationRecorder.save_conversation(
                bot_id=self.bot_id,call_id=self.call_id, call_status="INITIATED"
            )

        self.session = AgentSession(
            llm=openai.realtime.RealtimeModel(model='gpt-4o-realtime-preview-2025-06-03', voice='sage'),
            vad=silero.VAD.load(
                activation_threshold=CONFIG["activation_threshold"],
                min_silence_duration=CONFIG["min_silence_duration"],
                min_speech_duration=CONFIG["min_speech_duration"],
            ),
            turn_detection="vad",
        )
        self.call_start_time = time.time()
        await self.session.start(
            agent=ContextAwareAgent(
                bot_id=self.bot_id, language=language, bot_config=self.bot_config, call_start_time=time.time()
            ),
            room=ctx.room,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )
        async def save_final_conversation():
            if self.session and self.session.history:
                await ConversationRecorder.save_conversation(
                    bot_id=self.bot_id,
                    call_id=self.call_id,
                    conversation=self.conversation,
                    call_start_time=self.call_start_time,
                    history=self.session.history.to_dict(),
                )

        ctx.add_shutdown_callback(save_final_conversation)


async def entrypoint(ctx: JobContext) -> None:
    """Main entrypoint for the agent."""
    voice_agent = VoiceAgent()
    await voice_agent.initialize(ctx)


def run_app() -> None:
    """Run the voice agent application."""
    worker_options = WorkerOptions(
        entrypoint_fnc=entrypoint,
        ws_url=CONFIG["livekit_url"],
        api_key=CONFIG["livekit_api_key"],
        api_secret=CONFIG["livekit_api_secret"],
    )
    LOG.info(f"Starting worker with ws_url: {worker_options.ws_url}")

    if len(sys.argv) > 1 and sys.argv[1] in ["dev", "start"]:
        cli.run_app(worker_options)


if __name__ == "__main__":
    run_app()
