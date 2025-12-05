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
    get_job_context,
    StopResponse
)
from livekit.agents.llm import function_tool
from livekit.plugins import  noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins.openai.tts import TTS
from livekit import api

logger = logging.getLogger("agent")
load_dotenv(".env")


_instructions = """
Sen, Bağcılar Diş Kliniği’nin akıllı sanal asistanısın.
cevablar kisa olsun

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

💡 Örnekler (function trigger için):
- Eğer kullanıcı gerçek bir insan asistanla konuşmak isterse `call_real_assistant` kullan.
- Örnek cümleler:
    - "Beni gerçek asistana bağla"
    - "Bir insan asistanla konuşmak istiyorum"
    - "Lütfen gerçek asistana devret"
- Bu durumlarda kullanıcıya şöyle yanıt ver:
    - "Tamam, sizi gerçek asistana bağlıyorum..."  

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
"""

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=_instructions)
        self.real_assistant_called = False  # پرچم برای خاموش کردن ایجنت
            
    async def on_user_message(self, message: str, ctx: RunContext):
        # اگر دستیار واقعی وصل شده باشد، ایجنت پاسخی ندهد
        if self.real_assistant_called:
            return None
        return await super().on_user_message(message, ctx)
    
    async def on_user_turn_completed(self, turn_ctx, new_message):
        if new_message == "#" : 
            self.real_assistant_called = True
            print("################################")
            # return "masalan call kardim agent bayad harf nazaneh"
            raise StopResponse()
        
        print("newwwwwwww1111111")
        print(new_message)
        if self.real_assistant_called:
            print("newwwwwwww22222222222222")
            logger.info("Agent muted, skipping reply in pipeline.")
            # return await super().on_user_turn_completed(turn_ctx, "")
            raise StopResponse()
        return await super().on_user_turn_completed(turn_ctx, new_message)
    
    

    @function_tool
    async def call_real_assistant(self, phone_number: str = "+905325747455", user_message: str = ""):
        """
        Klinik asistanını arar ve konuşmayı devreder.
        Bu fonksiyon yalnızca kullanıcı açıkça 'gerçek asistana bağla' derse çalışır.
        """
        print("11111111111111111111 88888---")
        if user_message == "#" : 
            self.real_assistant_called = True
            print("################################")
            raise StopResponse()
            #return "masalan call kardim agent bayad harf nazaneh"

        if self.real_assistant_called:
            print("baadeh tamaaaaaaaaaas ")
            raise StopResponse()


     
        trigger_words = ["asistan","asistana","asistana bağla", "gerçek asistan", "canlı destek", "operatör"]
        print(user_message)
        if not any(t in user_message.lower() for t in trigger_words):
            print("2222222222222222222222222222222222222 ------------")
            print("Bu işlem yalnızca kullanıcı açıkça gerçek asistana bağlanmak istediğinde yapılır.")
            return "Bu işlem yalnızca kullanıcı açıkça gerçek asistana bağlanmak istediğinde yapılır."


        ctx = get_job_context()
        if ctx is None:
            return "Context bulunamadı, çağrı başlatılamadı."

        try:
            sip_trunk_id = "ST_ofpmSZ8gZzr2"
            logger.info(f"Gerçek asistana bağlanıyor... room={ctx.room.name}")
            print("aistan33333333333333333333")
            await ctx.api.sip.create_sip_participant(api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=sip_trunk_id,
                sip_call_to= "+905325747455",
                participant_identity="1",
                wait_until_answered=True,
            ))
            self.real_assistant_called = True
            raise StopResponse()
            #return f"Asistan {phone_number} numarasından arandı ve odaya katıldı."
        except Exception as e:
            logger.error(f"SIP çağrısı başarısız: {e}")
            return f"Çağrı başlatılamadı: {e}"



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
    vad_model = asyncio.run(safe_load_model(silero.VAD.load, "VAD"))
    proc.userdata["vad"] = vad_model

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    vad_model = ctx.proc.userdata.get("vad") or await safe_load_model(silero.VAD.load, "VAD")
    turn_detector = await safe_load_model(MultilingualModel, "TurnDetector")

    assistant=Assistant()
    
    stt_model = await safe_load_model(
        lambda: openai.STT(
            base_url="http://localhost:8001/v1",
            api_key="dummy_key",
            model="Systran/faster-whisper-large-v3",
            language="tr",
        ),
        "STT"
    )
   
    tts_voice = await safe_load_model(
        lambda: TTS.create_kokoro_client(
            model="speaches-ai/piper-tr_TR-fahrettin-medium",
            voice = "af_sky",
            base_url="http://localhost:8001/v1",
        ),
        "TTS"
    )
     
    llm_model = await safe_load_model(
        lambda: openai.LLM.with_ollama(
            model="gpt-oss:20b",
            base_url="http://34.134.173.200:11434/v1", # 127.0.0.1  http://34.134.173.200:11434/

        ),
        "LLM"
    )


    session = AgentSession(
        llm=llm_model,
        stt=stt_model,
        tts=tts_voice,
        turn_detection=turn_detector,
        vad=vad_model,
        preemptive_generation=False,
    )

    # ✅ Patch generate_reply
    orig_generate_reply = session.generate_reply

    def patched_generate_reply(*args, **kwargs):
        if assistant.real_assistant_called:
            logger.info("🔇 Mute mode: blocked generate_reply")
            return None
        return orig_generate_reply(*args, **kwargs)

    session.generate_reply = patched_generate_reply

    @session.on("reply_generated")
    def _on_reply(ev):
        if assistant.real_assistant_called:
            print("replaaaaaaaaaaaaaaaaaay")
            ev.prevent_send()
            ev.prevent_store()

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
        agent=assistant,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
