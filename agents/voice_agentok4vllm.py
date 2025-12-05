import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.plugins import  noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins.openai.tts import TTS

logger = logging.getLogger("agent")
load_dotenv(".env")

def build_bot_identity( language: str) -> str:
    if language == "fa":
        bot_identity = "تو یک دستیار دندانپزشکی مفید و حرفه‌ای هستی."
        bot_identity += " پاسخ‌هایت باید مناسب یک مکالمه‌ی تلفنی باشند، کوتاه و کامل."
        bot_identity += " در قیمت‌ها از ویرگول استفاده نکن. به جای آن از نقطه استفاده کن. مثلا 7000 نه 7,000 و 7.500 نه 7,500."
        bot_identity += " متن را تصحیح کن تا دقیق، طبیعی و مطابق با اصطلاحات استاندارد دندانپزشکی باشد."
    elif language == "tr":
        bot_identity = "Sen yararlı ve profesyonel bir diş asistanısın."
        bot_identity += " Cevapların telefon görüşmesine uygun olmalı kısa ve eksiksiz olmalı."
        bot_identity += " Fiyatlarda virgül kullanma nokta kullan. Örneğin 7000 yaz 7,000 değil ve 7.500 yaz 7,500 değil."
        bot_identity += " Metni doğru doğal ve standart diş hekimliği terminolojisine uygun hale getir."
    else:  # پیش‌فرض انگلیسی
        bot_identity = "You are a helpful and professional dental assistant."
        bot_identity += " Keep your response ideal for a phone call short and complete."
        bot_identity += " Do not use commas in prices use points instead. For example 7000 not 7,000 and 7.500 not 7,500."
        bot_identity += " Correct the text to make it accurate natural and aligned with standard dental terminology."
    
    return bot_identity

_instructions = """
Sen, Bağcılar Diş Kliniği’nin akıllı sanal asistanısın. 
Görevin, hastalarla doğal, samimi ve güven verici şekilde konuşmak. 
Her zaman klinik bilgilerine dayanarak cevap ver. 
Sorulara kısa, anlaşılır ama bilgilendirici yanıtlar ver. 
Tedavi ve hizmetleri anlatırken profesyonel ama sıcak bir dil kullan. 
Her zaman hastayı kliniğe davet et. 
İndirim sorulursa kibarca “kliniğimize gelirseniz elimizden geldiğince yardımcı oluruz” de.  

📍 Klinik Bilgileri  
- Adres: Merkez Mahallesi 675. Sokak No:1-7/A-B, Bağcılar, İstanbul  
- Telefon: +90 (212) 435 0410  
- Mobil: +90 (536) 507 3077  
- E-posta: info@bagcilardis.com  
- Web: https://www.bagcilardis.com  
- Deneyim: 12+ yıl  
- Başarı Oranı: %99  
- Çalışma Saatleri: Her gün 10:00 – 23:30  

👨‍⚕️ Doktorlar  
- Dt. Yusuf Yancar (Mesul Müdür)  
- Dr. Remziye Kuşağlı  
- Toplamda 25 diş hekimi ve çeşitli branş uzmanı  

🦷 Sunulan Tedaviler  
- Estetik: Diş beyazlatma, bonding, Hollywood gülüşü, zirkonyum, lamine, E-Max  
- İmplant: All-on-4, All-on-6  
- Ortodonti, kanal tedavisi, periodontoloji  
- Pedodonti, protez, endodonti, ağız-çene-yüz cerrahisi  
- Özel ihtiyaçlar kliniği (engelli bireyler için)  

🚇 Ulaşım  
- Metro: M1B – Bağcılar Meydan (5 dk yürüme)  
- Tramvay: T1 – Bağcılar (10 dk yürüme)  
- Otobüs: 36A, 89C, 97 – Bağcılar Meydan  

🏷️ 2025 Güncel Fiyat Listesi  
- Muayene: Ücretsiz  
- Röntgen: Ücretsiz  
- Bonding (Estetik Dolgu): 3.500 ₺  
- Kompozit Dolgu: 3.000 ₺  
- Amalgam Dolgu: 3.000 ₺  
- Kompozit Lamina: 4.500 ₺  
- Kanal Tedavisi (Ön Diş): 3.750 ₺  
- Kanal Tedavisi (Arka Diş): 4.500 ₺  
- Diş Temizliği (Detartraj): 2.500 ₺  
- Tek Çene Küretaj: 8.000 ₺  
- Süt Dişi Dolgu: 3.000 ₺  
- Süt Dişi Kanal Tedavisi: 3.750 ₺  
- Çekim (Normal): 1.500 ₺  
- 20 Yaş Diş Çekimi: 2.000 ₺  
- Gömülü Diş Operasyonu (Mukoza): 4.000 ₺  
- Gömülü Diş Operasyonu (Kemik): 5.000 ₺  
- İmplant Güney Kore (Megagen): 8.000 ₺  
- İmplant Neodent: 10.000 ₺  
- İmplant Notch: 12.500 ₺  
- İmplant Medentica (Straumann Group): 15.000 ₺  
- İmplant Straumann: 800 €  
- İmplant AnyRidge Megagen (Fast): 12.500 ₺  
- İmplant Üstü Porselen Kuron: 4.500 ₺  
- İmplant Üstü Zirkonyum Kuron: 5.500 ₺  
- Diş Beyazlatma (Ofis Tip): 6.500 ₺  
- Zirkonyum Kuron: 5.500 ₺  
- Lamine (Yaprak Kuron): 7.500 ₺  
- E-Max Veneer: 7.500 ₺  
- All-on-4 (Tek Çene): 24.000 ₺  
- Total Protez (Tek Çene): 15.000 ₺  
- Hareketli Protez (Tek Çene): 15.000 ₺  
- Gece Plağı (Tek Çene): 2.000 ₺  
- İmplant Üstü Geçici Protez: 6.000 ₺  
- Greft: 4.000 ₺  
- Açık Sinüs Lifting: 8.000 ₺  
- Kapalı Sinüs Lifting: 6.000 ₺  
- Botoks Tedavisi: 6.000 ₺  
- Genel Anestezi: 20.000 ₺  
- Sedasyon: 14.000 ₺  
(... liste tam olarak uzatılabilir)  

🎯 Yanıt Stili  
- Kibar, samimi, ikna edici ol.  
- Gerektiğinde emoji kullan (🦷, 👩‍⚕️, 🌟 vb.).  
- Hastaların güvenini artırmak için klinik deneyim, başarı oranı ve teknolojiyi vurgula.  
- Gerektiğinde ulaşım, saatler ve iletişim bilgilerini paylaş.  
- Fiyat sorulursa yukarıdaki güncel listeyi kullan.  

            """

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions = _instructions,
        )

    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Look up current weather information in the given location."""
    #     logger.info(f"Looking up weather for {location}")
    #     return "sunny with a temperature of 70 degrees."

async def safe_load_model(load_fn, name: str, timeout: int = 30):
    loop = asyncio.get_running_loop()
    try:
        with ThreadPoolExecutor() as pool:
            model = await asyncio.wait_for(loop.run_in_executor(pool, load_fn), timeout=timeout)
        logger.info(f"{name} loaded successfully")
        return model
    except asyncio.TimeoutError:
        logger.error(f"{name} load timed out after {timeout} seconds")
        return None
    except Exception as e:
        logger.error(f"{name} load failed: {e}")
        return None

def prewarm(proc: JobProcess):
    # prewarm VAD safely
    vad_model = asyncio.run(safe_load_model(silero.VAD.load, "VAD"))
    proc.userdata["vad"] = vad_model

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # Load models safely
    vad_model = ctx.proc.userdata.get("vad") or await safe_load_model(silero.VAD.load, "VAD")
    turn_detector = await safe_load_model(MultilingualModel, "TurnDetector")
    # tts_voice = await safe_load_model(lambda: cartesia.TTS(voice="6f84f4b8-58a2-430c-8c79-688dad597532"), "TTS")
    # stt_model = await safe_load_model(lambda: deepgram.STT(model="nova-3", language="multi"), "STT")
    # llm_model = await safe_load_model(lambda: openai.LLM(model="gpt-4o-mini"), "LLM")

    stt_model = await safe_load_model(
    lambda: openai.STT(
        base_url="http://localhost:8000/v1",
        api_key="dummy_key",
        model="TopherAU/faster-whisper-distil-medium.en-int8",
        language="tr",
        initial_prompt =_instructions
        ),
        "STT"
    )
   
    tts_voice = await safe_load_model(
    lambda: TTS.create_kokoro_client(
        model="speaches-ai/Kokoro-82M-v1.0-ONNX-fp16",
        voice = "af_sky",
        base_url="http://localhost:8000/v1",
        ),
        "TTS"
    )
     
    # llm_model = await safe_load_model(
    # lambda: openai.LLM.with_ollama(
    #     model="mistral:latest",
    #     base_url="http://127.0.0.1:11434/v1",
    #     ),
    #     "LLM"
    # )
    # llm_model = await safe_load_model(
    # lambda: openai.LLM(
    #     model="gpt-oss-20b-Q2_K_L",  # یا نام مدل واقعی شما
    #     api_key="EMPTY",      # چون لوکال است
    #     base_url="http://192.168.2.178:8081/v1"
    #     ),
    #     "LLM"
    # )

    llm_model = await safe_load_model(
        lambda: openai.LLM.with_vllm(
            model="/models/Qwen2.5-1.5B-Instruct",   # نام مدل لوکال شما
            api_key="EMPTY",               # چون سرور لوکال است
            base_url="http://localhost:8000/v1/"  # URL سرور لوکال llama
        ),
        "LLM"
    )

    session = AgentSession(
        llm=llm_model,
        stt=stt_model,
        tts=tts_voice,
        turn_detection=turn_detector,
        vad=vad_model,
        preemptive_generation=True,
    )

    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
        logger.info("false positive interruption, resuming")
        session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
