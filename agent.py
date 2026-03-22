import asyncio
import logging
import os
import httpx
from datetime import datetime
from dotenv import load_dotenv
import pytz

load_dotenv()

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
    llm,
)
from livekit.plugins import openai, sarvam, silero, deepgram

logger = logging.getLogger("priya-clinic")
logging.basicConfig(level=logging.INFO)

# ENV
LIVEKIT_URL        = os.getenv("LIVEKIT_URL", "")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
SARVAM_API_KEY     = os.getenv("SARVAM_API_KEY", "")
DEEPGRAM_API_KEY   = os.getenv("DEEPGRAM_API_KEY", "")
CAL_API_KEY        = os.getenv("CAL_API_KEY", "")
CAL_EVENT_TYPE_ID  = os.getenv("CAL_EVENT_TYPE_ID", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# CLINIC CONFIG
CLINIC_NAME   = "Priya Clinic"
DOCTOR_NAME   = "Dr. Sharma"
CLINIC_TIMING = "Somvar se Shanivar, subah 9 baje se shaam 7 baje tak"
TIMEZONE      = "Asia/Kolkata"


async def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"})
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")


async def book_cal_appointment(patient_name: str, date: str, time: str, reason: str) -> dict:
    try:
        ist = pytz.timezone(TIMEZONE)
        dt_str = f"{date}T{time}:00"
        dt_naive = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S")
        dt_ist   = ist.localize(dt_naive)
        start_utc = dt_ist.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        payload = {
            "eventTypeId": int(CAL_EVENT_TYPE_ID),
            "start": start_utc,
            "attendee": {
                "name":     patient_name,
                "email":    f"{patient_name.replace(' ','').lower()}@patient.priyaclinic.com",
                "timeZone": TIMEZONE,
                "language": "en"
            },
            "metadata": {"reason": reason}
        }
        headers = {
            "Authorization": f"Bearer {CAL_API_KEY}",
            "Content-Type":  "application/json",
            "cal-api-version": "2024-08-13"
        }
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post("https://api.cal.com/v2/bookings", json=payload, headers=headers)
            data = r.json()
            if r.status_code in (200, 201):
                return {"success": True, "booking_id": data.get("data", {}).get("uid", "N/A")}
            else:
                logger.error(f"Cal.com error: {data}")
                return {"success": False, "error": str(data)}
    except Exception as e:
        logger.error(f"book_cal_appointment exception: {e}")
        return {"success": False, "error": str(e)}


class ClinicTools:
    def __init__(self):
        pass

    @llm.ai_callable(description="Patient ka appointment book karo. date YYYY-MM-DD format mein do, time HH:MM 24-hour IST mein do.")
    async def book_appointment(
        self,
        patient_name: str,
        date: str,
        time: str,
        reason: str
    ) -> str:
        logger.info(f"Booking: {patient_name} on {date} at {time} for {reason}")
        result = await book_cal_appointment(patient_name, date, time, reason)
        if result["success"]:
            msg = (
                "Appointment Booked!\n"
                f"Patient: {patient_name}\nDate: {date}  Time: {time} IST\n"
                f"Reason: {reason}\nBooking ID: {result['booking_id']}"
            )
            await send_telegram(msg)
            return f"Appointment book ho gaya hai {patient_name} ke liye {date} ko {time} baje. Booking ID: {result['booking_id']}"
        else:
            return "Maafi chahti hoon, abhi booking mein thodi problem aa rahi hai. Kripya clinic mein directly call karein."

    @llm.ai_callable(description="Call khatam karo jab patient ki baat ho jaye ya woh phone rakhna chahein.")
    async def end_call(self) -> str:
        await send_telegram(f"Call ended - {CLINIC_NAME}")
        return f"Bahut shukriya! {CLINIC_NAME} mein aapka swagat hai. Khyal rakhein, namaskar!"


class PriyaAgent(Agent):
    def __init__(self, tools: ClinicTools):
        super().__init__(
            instructions=(
                f"Aap {CLINIC_NAME} ki AI receptionist Priya hain. "
                f"Aap Hindi aur Hinglish mein baat karti hain. "
                f"Doctor ka naam {DOCTOR_NAME} hai. Clinic ka time: {CLINIC_TIMING}. "
                "Hamesha 1-2 chhote sentences mein jawab den. "
                "Agar patient appointment lena chahein toh unka naam, date, time aur reason poochh kar book_appointment tool use karen. "
                "Agar patient bye kare ya baat khatam ho toh end_call tool use karen."
            ),
            tools=[tools.book_appointment, tools.end_call],
        )
        self._tools = tools

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=f"Namaskar! Aap {CLINIC_NAME} mein aa gaye hain. Main Priya hoon. Aap ka kya kaam ho sakta hai?"
        )


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    logger.info("Room connected")

    caller_id = "Unknown"
    for pid, p in ctx.room.remote_participants.items():
        caller_id = p.identity or pid
        break

    await send_telegram(f"Incoming Call\nCaller: {caller_id}\nClinic: {CLINIC_NAME}")

    if DEEPGRAM_API_KEY:
        stt_plugin = deepgram.STT(model="nova-2-general", language="hi")
    else:
        stt_plugin = sarvam.STT(
            language="unknown",
            model="saaras:v3",
            mode="translate",
            sample_rate=16000
        )

    llm_plugin = openai.LLM(
        model="llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
        max_completion_tokens=120,
    )

    tts_plugin = sarvam.TTS(
        target_language_code="hi-IN",
        model="bulbul:v2",
        speaker="priya",
        speech_sample_rate=24000,
    )

    vad_plugin = silero.VAD.load()

    tools = ClinicTools()
    agent = PriyaAgent(tools)

    session = AgentSession(
        stt=stt_plugin,
        llm=llm_plugin,
        tts=tts_plugin,
        vad=vad_plugin,
    )

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(close_on_disconnect=False),
    )
    logger.info("AgentSession started for caller: %s", caller_id)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="priya-clinic",
        )
    )
