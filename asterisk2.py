import os
import sys
import wave
import time
import asyncio
import struct
import random
import datetime
from dataclasses import dataclass
from typing import Dict, Optional

# import aiofiles
import numpy as np
# import stomp
from dotenv import load_dotenv
from loguru import logger
from get_credential import get_file_name
# from google.cloud import speech_v1p1beta1 as speech
# from deepgram import LiveOptions

# --- PIPECAT IMPORTS ---
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams

from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver

import socket

from pipecat.frames.frames import (
    CancelFrame,
    Frame,
    STTMuteFrame,
    StartFrame,
    EndFrame,
    # AudioRawFrame,
    InputAudioRawFrame,
    # InputDTMFFrame,
    TTSAudioRawFrame,
    # LLMContextFrame,
    LLMMessagesAppendFrame,
    TTSSpeakFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    TextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)

from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
# from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
# from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.processors.filters.stt_mute_filter import (
    STTMuteConfig,
    STTMuteFilter,
    STTMuteStrategy,
)

from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair
)

# from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.llm_service import LLMContext
# from pipecat.processors.aggregators.dtmf_aggregator import DTMFAggregator

from pipecat.services.llm_service import LLMContext
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.services.cartesia.tts import CartesiaTTSService, GenerationConfig
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.soniox.stt import SonioxSTTService

from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport

from pipecat.transcriptions.language import Language

from pipecat.adapters.schemas.tools_schema import ToolsSchema
# from pipecat.adapters.schemas.function_schema import FunctionSchema

from pipecat.services.google.vertex.llm import GoogleVertexLLMService
# --- LOCAL IMPORTS ---
from voices import get_voice_code
# from call_end_reasons import CallEndReason
from utils.uploadAudio import upload_to_s3_and_notify
# from utils.steriorecorder import StereoRecorder

from tools.whatsappResend import resend_whatsapp_message_handler, RESEND_WHATSAPP_DECL
from tools.endCallNew import END_CONVERSATION_TOOL
from tools.transferAgent import TRANSFER_AGENT_TOOL

# from tools.waitAndTransferCall import wait_and_transfer_call_handler, WAIT_AND_TRANSFER_DECL
# from tools.endCall import set_serializer_for_end_conversation
# from tools.duluxCROAPI import (
#     set_serializer_for_cro_tool,
#     get_cro_number_handler,
#     GET_CRO_NUMBER_FUNCTION,
# )

from max_duration_processor import MaxDurationProcessor
# from speech_timeout_processor import SpeechTimeoutProcessor
from datetime import datetime, timezone

from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    Frame, 
    TranscriptionFrame, 
    CancelFrame, 
    BotStartedSpeakingFrame,
    UserStartedSpeakingFrame
)
from loguru import logger

from datetime import datetime, timezone
from pipecat.frames.frames import Frame, TranscriptionFrame, TextFrame, TTSSpeakFrame, InterimTranscriptionFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
import re
import asyncio
from pipecat.services.elevenlabs import ElevenLabsTTSService

import zoneinfo

# 1. Clear any default loguru handlers
logger.remove()

# 2. Add a single handler with BOTH your filter and the DEBUG level
# logger.add(
#     sys.stderr,
#     level="DEBUG",
#     filter=lambda record: "destination [None] not registered" not in record["message"]
# )

logger.add(sys.stderr, level="DEBUG")

# logger.add(
#     sys.stderr,
#     level="DEBUG",
#     filter=lambda record: "STT" in record["message"] or 
#                           "soniox" in record["message"].lower() or
#                           "google" in record["message"].lower() or
#                           "stream" in record["message"].lower() or
#                           "grpc" in record["message"].lower() or
#                           "websocket" in record["message"].lower() or
#                           "Soniox" in record["message"] or
#                           "Google" in record["message"]
# )

load_dotenv(override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SONIOX_API_KEY = os.getenv("SONIOX_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")

SAMPLE_RATE = 8000
SESSION_PIPELINES = {}

BASE_SYSTEM_INSTRUCTION = """
    <ROLE_AND_PERSONA>
    You are the Renault Care Assistant, a professional, warm, and highly capable conversational voice AI representing Renault Care. Your core responsibility is to handle inbound customer calls, resolve technical and product inquiries comprehensively using the provided knowledge base, or route the call seamlessly to the correct department.
    </ROLE_AND_PERSONA>

    <VOICE_CHANNEL_CONSTRAINTS>
    - STRICTLY NO MARKDOWN: You must never output asterisks (*), hashtags (#), bullet points, bolding, dashes, or any text formatting. Your spoken output must be entirely raw, plain, and seamless conversational text.
    - NO COGNITIVE BLEED OR METADATA: You are FORBIDDEN from outputting any language analysis, state evaluations, decision-making logic, reasoning logs, tags, or internal notes. Do not echo back "Latest User Utterance", "Detected Language", or "Target State". Your output must contain absolutely nothing except the direct spoken response meant for the user.
    - DETAILED AND INFORMATIONAL: When responding to product queries, vehicle features, or specifications, do not give overly brief or lazy answers. Provide rich, descriptive, and comprehensive explanations utilizing actual technical metrics, comfort details, and engine options from the knowledge base. Speak in fluid, natural paragraphs.
    - INTERRUPTIBILITY: Structure sentences with natural pauses and spoken rhythm so that they are pleasant to listen to and easy for a human to politely interrupt.
    </VOICE_CHANNEL_CONSTRAINTS>

    <LANGUAGE_SELECTION_OVERRIDE>
    This is your single most important rule and it overrides everything else about language.

    The conversation history will contain a MIX of Hinglish and English turns, including your own earlier replies and the opening greeting. You MUST completely ignore the language of every previous turn. The language you used before has ZERO influence on this turn.

    Decide the language of THIS reply using ONLY the customer's most recent utterance. Re-decide this fresh on every single turn, for the entire call. Do not carry a language forward out of habit or momentum.
    </LANGUAGE_SELECTION_OVERRIDE>

    <LANGUAGE_STATE_MACHINE>
    You speak in exactly one of two languages per turn: ENGLISH or HINGLISH. Switching is instant, bidirectional, and may happen on any turn, any number of times.

    STATE: ENGLISH
    - USE WHEN: the customer's latest utterance is primarily in English.
    - RULES: Your reply must be 100% pure English. Do not use any Hindi or Hinglish tokens (no "ji", "achha", "haan", "namaskaar").

    STATE: HINGLISH
    - USE WHEN: the customer's latest utterance contains Hindi words, Hinglish phrasing, or Hindi syntax.
    - RULES: Speak in natural, colloquial Hinglish (Hindi grammar in Latin script, blended with common English nouns). Use respectful Hindi pronouns ("aap", "batayein", "kijiye").

    CRITICAL SNAP-BACK: If your previous reply was in Hinglish but the customer's latest utterance is in English, you MUST reply in pure English immediately. The reverse applies equally. Never let the previous language "leak" into a turn where the customer has switched.

    RAW-DATA EXCEPTION (narrow):
    - This applies ONLY when the customer's latest turn is nothing but raw data with no sentence around it — an isolated name ("Umesh"), a city ("Delhi"), a location ("Wazirpur"), a registration plate, or bare numbers.
    - In that single case, do not switch. Continue in the language you used on your previous turn.
    - If there are ANY real words, questions, or phrasing alongside the data, this exception does NOT apply — select the language normally from those words.
    </LANGUAGE_STATE_MACHINE>

    <GREETING_RULE>
    - CONVERSATION INITIATION (NO PRIOR HISTORY): If the conversation has just started (the history contains no previous assistant messages), 
        you MUST output exactly this warm Hinglish opening greeting, regardless of anything else:
      "Namaskaar! Main Renault ki taraf se bol rahi hoon. Kripya batayein main aapki kis prakar sahayata kar sakti hoon?"
    - THE GREETING DOES NOT LOCK THE LANGUAGE: This opening line is fixed company policy and is always in Hinglish. It does NOT set or lock the conversation language.
        The instant the customer speaks, apply the LANGUAGE_SELECTION_OVERRIDE and pick the language from their words alone. 
        If their first utterance is English, your very next reply must be fully English even though you greeted in Hinglish.
    - CONVERSATION IN-PROGRESS (HISTORY EXISTS): If there is at least one assistant message in the history, you are FORBIDDEN from repeating the greeting or any variation of it.  
    Immediately address the customer's active query in the correct language.
    </GREETING_RULE>
"""

async def prewarm_pipeline(call_id, stt_config, tts_config, assistant_config):

    credential_file_name, project_id = get_file_name()
    credentials_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), credential_file_name)

    # 2. Set environment variable so the underlying Google Auth finds it
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    logger.info(f"PROJECT ID : {project_id}")


    llm = GoogleVertexLLMService(
        project_id=project_id,
        location="asia-south1",
        model="gemini-2.5-flash",
        settings=GoogleVertexLLMService.Settings(
            temperature=0.1,
            top_p=0.8,
            top_k=10,
            system_instruction=BASE_SYSTEM_INSTRUCTION,   
        )
    )

    stt = None
    if stt_config and stt_config.get("engine").lower() == "deepgram":
        stt = DeepgramSTTService(
            api_key=stt_config.get("apiKey"),
            # Use the new Pipecat Settings class instead of LiveOptions
            settings=DeepgramSTTService.Settings(
            model="nova-3",
            language=stt_config.get("language"),            interim_results=True,
            punctuate=False,
            keyterm=[                                                      ## updated part for keyterm detection
            "Renault",
            "Duster",
            "Kiger",
            "Triber",
            "Kwid",
            ],
    ),
        )
    elif stt_config and stt_config.get("engine").lower() == "google":
        stt = GoogleSTTService(
            api_key=os.getenv("GOOGLE_API_KEY"),
            params=GoogleSTTService.InputParams(
                languages=[stt_config.get("language"),], 
                model=stt_config.get("model","latest_short"), 
                use_separate_recognition_per_channel=False,
                enable_automatic_punctuation=False,
                #enable_spoken_punctuation=False,
                # enable_spoken_emojis=False,
                profanity_filter=True,
                # enable_word_time_offsets=True,
                # enable_word_confidence=True,
                # enable_interim_results=True,
                enable_voice_activity_events=False,   # ← prevents auto end after first turn
                single_utterance=False,   # don't close stream after first result
                interim_results=True,     # keep stream alive between utterances

            ),
            
        )

    elif stt_config and stt_config.get("engine").lower() == "soniox":
        stt = SonioxSTTService(api_key=os.getenv("SONIOX_API_KEY"))
    else:
        raise ValueError(f"Invalid STT engine: {stt_config.get('engine')}")
    
    # TTS
    tts = None
    if tts_config and tts_config.get("engine").lower() == "cartesia":
        print("using tts config cartesia")
        voice_id=None
        voice = assistant_config.get("voice", {})

        if voice.get("provider").lower() == "cartesia":
            voice_id = voice.get("voiceId")
        else:
            language = voice.get("language", "")
            voice_id = get_voice_code(language.replace("_", "-"))

        tts = CartesiaTTSService(
            api_key=tts_config.get("apiKey"),
            voice_id=voice_id,
            params=CartesiaTTSService.InputParams(
                language=get_language_enum(assistant_config["voice"]["language"].replace("_", "-")),
                sample_rate=assistant_config["voice"].get("sampleRate", SAMPLE_RATE),
                encoding=assistant_config["voice"].get("encoding", "pcm_mulaw"),
                generation_config=GenerationConfig(speed=os.getenv('CARTESIA_SPEED'))
            ),
        )
    
    elif tts_config and tts_config.get("engine").lower() == "elevenlabs":
        tts = ElevenLabsTTSService(
        api_key=tts_config.get("apiKey"),
        settings=ElevenLabsTTSService.Settings(
            voice=tts_config.get("voiceId"),
           # model=tts_config.get("model"),        
            stability=0.7,
            similarity_boost=0.8,
            speed=1.1,
        ),
    )

    else:
        print(f"using tts config google - voiceId and language ","",assistant_config["voice"]["voiceId"],"  ",assistant_config["voice"]["language"].replace("_", "-"))
        tts = GoogleTTSService(
            voice_id=assistant_config["voice"]["voiceId"],
            params=GoogleTTSService.InputParams(
                language=get_language_enum(assistant_config["voice"]["language"].replace("_", "-"))
            ),
            credentials=None,
        )

    SESSION_PIPELINES[call_id] = {
        "stt": stt,
        "llm": llm,
        "tts": tts,
        "status": "PREWARMED"
    }

    asyncio.create_task(cleanup_if_not_connected(call_id))

async def cleanup_if_not_connected(call_id, timeout=40):
    await asyncio.sleep(timeout)

    session = SESSION_PIPELINES.get(call_id)

    if session and session["status"] == "PREWARMED":
        print(f"Cleaning unused pipeline {call_id}")
        await destroy_pipeline(call_id)


async def destroy_pipeline(call_id):
    session = SESSION_PIPELINES.pop(call_id, None)

    if not session:
        return

    runner = session.get("runner")
    if runner:
        # await destroy_pipeline(call_id)
        await runner.stop()

    print(f"Destroyed pipeline {call_id}")

def get_language_enum(lang_code: str) -> Language:
    normalized = lang_code.strip().lower()
    for lang in Language:
        if lang.value.lower() == normalized:
            return lang
    raise ValueError(f"Invalid language code: {lang_code}")


load_dotenv(override=True)

# from google.cloud import speech

def extract_initial_greeting(script_text: str, default_fallback: str = "Hello") -> str:
    """Extracts the greeting explicitly defined in the script."""
    if not script_text:
        return default_fallback
        
    # Matches: Greeting > " Any text inside quotes " (case-insensitive)
    match = re.search(r'Greeting\s*>\s*"([^"]+)"', script_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
        
    return default_fallback

ACTIVEMQ_DESTINATION = 'q.call.transcriptions'

async def addToQueue(params, activemq_conn, delay=0):
    logger.info(f"Pushing data to queue")
    try:
        logger.info("Pushing data to ActiveMQ queue")
        # Offload the blocking ActiveMQ network call to a background thread
        await asyncio.to_thread(
            activemq_conn.send_message, 
            params, 
            ACTIVEMQ_DESTINATION, 
            delay
        )
    except Exception as e:
        logger.exception(f"Error in push data to queue: {e}")

async def next_seq_for_call(redis_instance, call_id: str) -> int:
    if redis_instance:
        try:
            key = f"seq:{call_id}"
            seq = await redis_instance.incr(key)
            if seq == 1:
                try:
                    await redis_instance.expire(key, 60 * 60 * 2)
                except Exception:
                    pass
            print(f"the seq is key {key} seq {seq}")
            return int(seq)
        except Exception as e:
            logger.exception(f"Redis INCR failed for {call_id}: {e}")

GREETING_TEXT = "Hello"



AMI_HOST = os.getenv("AMI_HOST", "127.0.0.1")
AMI_PORT = int(os.getenv("AMI_PORT", "5038"))
AMI_USER = os.getenv("AMI_USER", "apiuser")
AMI_PASS = os.getenv("AMI_PASS", "your_secure_password")
AUDIOSOCKET_PORT = int(os.getenv("AUDIOSOCKET_PORT", "9092"))


async def force_ami_transfer(call_id: str, target_number: str):
    try:
        reader, writer = await asyncio.open_connection(AMI_HOST, AMI_PORT)
        login_cmd = f"Action: Login\r\nUsername: {AMI_USER}\r\nSecret: {AMI_PASS}\r\n\r\n"
        writer.write(login_cmd.encode())
        await writer.drain()
        try:
            while True:
                await asyncio.wait_for(reader.readline(), timeout=0.5)
        except asyncio.TimeoutError:
            pass
        cmd = f"Action: Command\r\nCommand: database get active_bot {call_id}\r\n\r\n"
        writer.write(cmd.encode())
        await writer.drain()
        resp = ""
        channel_name = None
        try:
            while True:
                line_bytes = await asyncio.wait_for(reader.readline(), timeout=1.0)
                if not line_bytes:
                    break
                line = line_bytes.decode()
                resp += line
                if "Value:" in line:
                    match = re.search(r'Value:\s*(\S+)', line)
                    if match:
                        channel_name = match.group(1)
                if "--END COMMAND--" in line or "not found" in line.lower():
                    break
        except asyncio.TimeoutError:
            pass
        logger.info(f"[AMI Debug] Raw DB Response for {call_id}:\n{resp}")
        if not channel_name:
            logger.error(f"Could not find active channel for {call_id}.")
            writer.write(b"Action: Logoff\r\n\r\n")
            writer.close()
            return
        logger.info(f"AMI found channel: {channel_name}. Yanking to {target_number}...")
        redirect_cmd = (
            f"Action: Redirect\r\n"
            f"Channel: {channel_name}\r\n"
            f"Context: transfer-to-human\r\n"
            f"Exten: {target_number}\r\n"
            f"Priority: 1\r\n\r\n"
        )
        writer.write(redirect_cmd.encode())
        await writer.drain()
        writer.write(f"Action: Command\r\nCommand: database del active_bot {call_id}\r\n\r\n".encode())
        await writer.drain()
        writer.write(b"Action: Logoff\r\n\r\n")
        writer.close()
    except Exception as e:
        logger.error(f"AMI Transfer Failed: {e}")


async def force_ami_hangup(call_id: str):
    try:
        reader, writer = await asyncio.open_connection(AMI_HOST, AMI_PORT)
        login_cmd = f"Action: Login\r\nUsername: {AMI_USER}\r\nSecret: {AMI_PASS}\r\n\r\n"
        writer.write(login_cmd.encode())
        await writer.drain()
        try:
            while True:
                await asyncio.wait_for(reader.readline(), timeout=0.5)
        except asyncio.TimeoutError:
            pass
        cmd = f"Action: Command\r\nCommand: database get active_bot {call_id}\r\n\r\n"
        writer.write(cmd.encode())
        await writer.drain()
        resp = ""
        channel_name = None
        try:
            while True:
                line_bytes = await asyncio.wait_for(reader.readline(), timeout=1.0)
                if not line_bytes:
                    break
                line = line_bytes.decode()
                resp += line
                if "Value:" in line:
                    match = re.search(r'Value:\s*(\S+)', line)
                    if match:
                        channel_name = match.group(1)
                if "--END COMMAND--" in line or "not found" in line.lower():
                    break
        except asyncio.TimeoutError:
            pass
        if not channel_name:
            logger.warning(f"[AMI] Could not find active channel for {call_id}.")
        else:
            logger.info(f"[AMI] Found channel: {channel_name}. Sending HARD HANGUP...")
            # Injecting Cause: 16 for normal clearing
            hangup_cmd = f"Action: Hangup\r\nChannel: {channel_name}\r\nCause: 16\r\n\r\n"
            writer.write(hangup_cmd.encode())
            await writer.drain()
        writer.write(f"Action: Command\r\nCommand: database del active_bot {call_id}\r\n\r\n".encode())
        await writer.drain()
        writer.write(b"Action: Logoff\r\n\r\n")
        writer.close()
    except Exception as e:
        logger.error(f"[AMI] Hangup Failed for {call_id}: {e}")


@dataclass
class RecordingState:
    server_name: Optional[str] = None
    filename: Optional[str] = None
    wav: Optional[wave.Wave_write] = None
    num_channels: int = 2
    sample_rate: int = SAMPLE_RATE
    user_turns: list = None
    bot_audio_chunks: list = None
    recording_start_time: float = 0.0
    last_timestamp: float = 0.0
    bot_cursor: float = 0.0

    def __post_init__(self):
        if self.user_turns is None:
            self.user_turns = []
        if self.bot_audio_chunks is None:
            self.bot_audio_chunks = []


async def reconstruct_stereo_from_continuous(st: RecordingState):
    if not st.user_turns and not st.bot_audio_chunks:
        logger.warning("No audio recorded, skipping reconstruction")
        return

    # Calculate total duration needed
    all_timestamps = []
    all_timestamps.extend(ts for ts, _ in st.user_turns)
    all_timestamps.extend(ts for ts, _ in st.bot_audio_chunks)
    if st.last_timestamp > 0:
        all_timestamps.append(st.last_timestamp)

    max_time = max(all_timestamps) if all_timestamps else 0
    total_duration = max_time

    if total_duration == 0:
        logger.warning("Zero duration recording, skipping")
        return

    # Add a small buffer to the end to prevent cutoff
    total_samples = int(total_duration * st.sample_rate) + (st.sample_rate * 2) 

    left_channel = np.zeros(total_samples, dtype=np.int16)
    right_channel = np.zeros(total_samples, dtype=np.int16)

    def process_channel_smoothly(chunks, channel_array):
        write_cursor = 0
        
        for timestamp, audio in chunks:
            audio_samples = np.frombuffer(audio, dtype=np.int16)
            start_sample = int(timestamp * st.sample_rate)
            
            # Jitter Fix: Check the gap between where the last chunk ended and this one begins
            gap = start_sample - write_cursor
            
            # If the gap is less than 150ms, it's likely jitter in a continuous stream. 
            # Snap the audio exactly to the write cursor to eliminate silent micro-gaps.
            jitter_tolerance_samples = int(st.sample_rate * 0.15)
            
            if abs(gap) < jitter_tolerance_samples and write_cursor > 0:
                start_sample = write_cursor
                
            end_sample = start_sample + len(audio_samples)
            
            # Safety bounds
            actual_end = min(end_sample, total_samples)
            actual_len = actual_end - start_sample
            
            if actual_len > 0:
                channel_array[start_sample:actual_end] = audio_samples[:actual_len]
                
            write_cursor = actual_end

    # Process both channels using the smoothing logic
    process_channel_smoothly(st.user_turns, left_channel)
    process_channel_smoothly(st.bot_audio_chunks, right_channel)

    # Combine into stereo
    stereo = np.column_stack((left_channel, right_channel))
    stereo_bytes = stereo.astype(np.int16).tobytes()

    with wave.open(st.filename, "wb") as wf:
        wf.setsampwidth(2)
        wf.setnchannels(2)
        wf.setframerate(st.sample_rate)
        wf.writeframes(stereo_bytes)

    logger.info(f"Temporal stereo recording saved: {st.filename}")

import time

class VADSpeedHackProcessor(FrameProcessor):
    """
    Tricks Pipecat's default aggregator into firing the LLM instantly.
    Also protects the LLM from phantom cancel frames caused by trailing VAD noise.
    """
    def __init__(self):
        super().__init__()
        self._last_hack_time = 0.0
        # 800ms protection window to let the LLM generate safely
        self._cooldown_sec = 0.8 

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # 1. UPSTREAM: Block phantom interruptions right after we forced the LLM
        # 1. UPSTREAM: Block phantom interruptions right after we forced the LLM
        if direction == FrameDirection.UPSTREAM:
            if isinstance(frame, CancelFrame) or type(frame).__name__ == "InterruptionFrame":
                if time.time() - self._last_hack_time < self._cooldown_sec:
                    logger.debug("🛡️ [SPEED HACK] Swallowing phantom CancelFrame to protect LLM generation.")
                    return  # Drop the frame; do NOT pass it upstream to the LLM

        # Normal frame passing
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        
        # 2. DOWNSTREAM: Inject the fake stop signal and start the cooldown clock
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if text:
                logger.success(f"⚡ [SPEED HACK] Forcing LLM trigger for: '{text}'")
                self._last_hack_time = time.time()
                await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

class TranscriptLogger(FrameProcessor):
    def __init__(self, label: str, role: str, redis_conn, organisation_id, campaign_id, activemq_conn, call_id, session_id):
        super().__init__()
        self.label = label
        self.role = role
        self.redis_conn = redis_conn
        self.organisation_id = organisation_id
        self.campaign_id = campaign_id
        self.activemq_conn = activemq_conn
        
        # Fixed: Removed the trailing comma so this is a string, not a tuple
        self.call_id = call_id 
        self.session_id = session_id

        # 1. Initialize a FIFO queue for sequential processing
        self._log_queue = asyncio.Queue()
        
        # 2. Start a single background worker task
        self._worker_task = asyncio.create_task(self._logging_worker())

    async def _logging_worker(self):
        """
        Runs in the background, consuming items from the queue one by one.
        This guarantees perfect chronological order for ActiveMQ.
        """
        while True:
            try:
                # Wait for the next item in the queue
                text_to_queue, speaker = await self._log_queue.get()
                
                # Redis INCR and ActiveMQ push now happen sequentially per call
                seq = await next_seq_for_call(self.redis_conn, self.call_id)
                msg = {
                    "callSid": self.call_id,
                    "callId": self.session_id,
                    "callStatus": "IN_PROGRESS",
                    "organisationId": self.organisation_id,
                    "campaignId": self.campaign_id,
                    "transcriptions": [{
                        "trans": text_to_queue,
                        "timestamp": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                        "speaker": speaker
                    }],
                    "seq": seq
                }
                await addToQueue(msg, self.activemq_conn)
                
                # Mark the item as processed
                self._log_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{self.label}] Background worker logging failed: {e}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        text_to_queue = None
        speaker = None

        if direction == FrameDirection.DOWNSTREAM:
            if self.role == "user" and isinstance(frame, TranscriptionFrame):
                logger.info(f"[{self.label}] USER (Final): {frame.text}")
                text_to_queue = frame.text
                speaker = "user"
            elif self.role == "bot":
                if isinstance(frame, TextFrame) and not isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
                    logger.info(f"[{self.label}] BOT SAID: {frame.text}")
                    text_to_queue = frame.text
                    speaker = "assistant"
                elif isinstance(frame, TTSSpeakFrame):
                    logger.info(f"[{self.label}] BOT SYSTEM PROMPT: {frame.text}")
                    text_to_queue = frame.text
                    speaker = "assistant"

        if text_to_queue:
            # INSTANTLY put data in the queue without blocking the Pipecat pipeline.
            # No parallel tasks created -> strict ordering maintained.
            self._log_queue.put_nowait((text_to_queue, speaker))

        # The frame moves to the next processor INSTANTLY
        await self.push_frame(frame, direction)

    async def stop(self, *args, **kwargs):
        """Clean up the worker task when the pipeline shuts down."""
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        # Do NOT call super().stop() here!


# ---------------------------------------------------------------------------
# FIX 1: BotSpeakingStateTracker
#
# Problem: STTMuteFilter (which sits UPSTREAM) needs BotStartedSpeakingFrame /
# BotStoppedSpeakingFrame to know when to mute the STT.  Those frames are
# produced INSIDE the output-transport's async playback queue – they are pushed
# upstream *from the transport* but that happens outside the normal pipeline
# frame-flow, so the STTMuteFilter never sees them reliably.
#
# Fix: Insert this lightweight processor BETWEEN the TTS service and the
# output transport.  It watches for TTSStartedFrame / TTSStoppedFrame
# (which flow DOWNSTREAM from TTS) and immediately re-emits
# BotStartedSpeakingFrame / BotStoppedSpeakingFrame UPSTREAM so the
# STTMuteFilter – which is upstream of the LLM – actually receives them.
#
# The shared_state dict is still updated here so the idle processor check
# remains race-condition-free (no async queue latency).
# ---------------------------------------------------------------------------
class BotSpeakingStateTracker(FrameProcessor):
    """
    Sits downstream of TTS, upstream of the output transport.

    DOWNSTREAM events:
      TTSStartedFrame  → bot_is_speaking=True, pushes BotStartedSpeakingFrame UPSTREAM
      TTSStoppedFrame  → ignored here. BotStoppedSpeakingFrame is pushed by the
                         output transport's _playback_loop AFTER the last audio
                         chunk is written to the socket — not when TTS generation
                         finishes. This is the correct 'audio done playing' signal.

    UPSTREAM events (barge-in path, only when always_mute=False):
      UserStartedSpeakingFrame → clears bot_is_speaking immediately so STTMuteFilter
                                  unlocks; pushes BotStoppedSpeakingFrame upstream.
                                  Skipped when always_mute=True (bot never interrupted).
      BotStoppedSpeakingFrame  → clears bot_is_speaking (echoed by output transport
                                  after last chunk plays, or after interruption drain).
    """

    def __init__(self, shared_state: dict, always_mute: bool = False):
        super().__init__()
        self._shared_state = shared_state
        self._always_mute = always_mute  # True = ALWAYS strategy active, no barge-in

    def _clear_speaking(self):
        if self._shared_state.get("bot_is_speaking", False):
            logger.debug("[BotSpeakingTracker] Clearing bot_is_speaking flag")
            self._shared_state["bot_is_speaking"] = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # ---- Downstream: TTS lifecycle ----
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, TTSStartedFrame):
                logger.debug("[BotSpeakingTracker] TTSStartedFrame → bot IS speaking")
                self._shared_state["bot_is_speaking"] = True
                await self.push_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)

            # NOTE: BotStoppedSpeakingFrame is NOT emitted here on TTSStoppedFrame.
            # TTSStoppedFrame fires when TTS *generation* finishes, but the audio
            # may still be queued in the output transport's playback loop and won't
            # finish playing for several more seconds. Emitting BotStopped here
            # would start the idle timer too early. The output transport's
            # _playback_loop now owns BotStoppedSpeakingFrame — it pushes it
            # upstream only after the last audio chunk has actually been written.

        # ---- Upstream: barge-in / interruption path ----
        elif direction == FrameDirection.UPSTREAM:
            if isinstance(frame, UserStartedSpeakingFrame):
                if not self._always_mute and self._shared_state.get("bot_is_speaking", False):
                    logger.info("[BotSpeakingTracker] User barge-in → clearing bot_is_speaking")
                    self._clear_speaking()
                    await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
                elif self._always_mute:
                    logger.debug("[BotSpeakingTracker] ALWAYS mode: ignoring UserStartedSpeaking upstream")

            elif isinstance(frame, BotStoppedSpeakingFrame):
                # Echoed by output transport after last audio chunk is played,
                # or after queue drain on interruption.
                self._clear_speaking()

        # Always pass the original frame in its original direction
        await self.push_frame(frame, direction)


class BargeInContextProcessor(FrameProcessor):
    """
    Detects if the bot was interrupted and injects a context note into the 
    user's transcription so the LLM knows the response is to a partial sentence.
    """
    def __init__(self):
        super().__init__()
        self._was_interrupted = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # 1. Catch the interruption signal flowing UPSTREAM from the transport/VAD
        if direction == FrameDirection.UPSTREAM:
            if isinstance(frame, CancelFrame) or type(frame).__name__ == "InterruptionFrame":
                logger.warning("[BargeIn] Interruption detected! Flagging next user input.")
                self._was_interrupted = True

        # 2. Catch the user's text flowing DOWNSTREAM from the STT
        elif direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, TranscriptionFrame) and self._was_interrupted:
                
                original_text = frame.text.strip()
                
                # Only inject if the user actually said something (avoid empty frames)
                if original_text:
                    
                    # Modify the text the LLM sees without breaking the Pipecat schema
                    # Modify the text the LLM sees to handle backchannels vs real interruptions
                    frame.text = (                                                                                                        ## updated part for backchannel handling
                        f"{original_text}\n\n"
                        f"(The customer said this while you were still speaking. "
                        f"If it is only a short acknowledgement such as 'ok', 'haan', 'hmm', 'yes', 'achha', "
                        f"treat it as agreement and continue naturally. Otherwise, treat it as their new request. "
                        f"Reply in the same language as the customer's words above.)"
                    )
                    logger.info(f"[BargeIn] Injecting context to stale user input: '{frame.text}'")

                    
                # Reset the flag for the next turn
                self._was_interrupted = False

        # 3. Always pass the frame along
        await self.push_frame(frame, direction)

# #----------------------------------------updated part----------------------------------------------
# class ContextDebugProcessor(FrameProcessor):
#     def __init__(self, context):
#         super().__init__()
#         self._context = context

#     async def process_frame(self, frame: Frame, direction: FrameDirection):
#         await super().process_frame(frame, direction)
#         if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TranscriptionFrame):
#             try:
#                 logger.warning(f"[CTX] {self._context.messages}")
#             except Exception as e:
#                 logger.warning(f"[CTX] could not read context: {e}")
#         await self.push_frame(frame, direction)
# # ---------------------------------------------------------------------------
# TranscriptionGate
#
# Problem: STTMuteFilter drops InputAudioRawFrame going INTO the STT service,
# but streaming STT engines (Deepgram, Google) have their own internal buffers.
# Audio that slipped through before the mute activated can still produce a
# TranscriptionFrame AFTER the bot started speaking. That transcription then
# reaches the LLM and triggers an unwanted response while the bot is mid-sentence.
#
# Fix: In ALWAYS mode, gate TranscriptionFrame / InterimTranscriptionFrame at
# the OUTPUT of STT. Any transcription that arrives while bot_is_speaking=True
# is silently dropped before it can reach the LLM.
#
# In barge-in mode (ALWAYS off) this gate is a no-op — all transcriptions pass.
# ---------------------------------------------------------------------------
class TranscriptionGate(FrameProcessor):
    """
    In ALWAYS mode: drops TranscriptionFrame and InterimTranscriptionFrame
    while the bot is speaking AND for a short grace window after it stops.

    Why the grace window?
      Streaming STT engines (Deepgram, Google) buffer audio internally.
      When the bot finishes speaking, the engine flushes buffered audio that
      was captured *during* bot speech and emits a final TranscriptionFrame.
      Without a grace window, that stale transcription arrives milliseconds
      after bot_is_speaking flips to False and passes the gate — reaching the
      LLM and triggering an unwanted response.

    The gate also tracks bot speaking state itself (via upstream frames) so
    it can make the drop decision synchronously without relying solely on the
    shared_state dict timing.

    In barge-in mode (always_mute=False) this processor is a complete no-op.
    """

    # Seconds to keep dropping transcriptions after bot stops speaking.
    # 600ms covers the STT engine's internal flush delay reliably.
    GRACE_WINDOW_SEC = 0.6

    def __init__(self, shared_state: dict, always_mute: bool = False):
        super().__init__()
        self._shared_state = shared_state
        self._always_mute = always_mute
        self._bot_speaking = False       # local mirror of speaking state
        self._grace_until: float = 0.0  # monotonic time until which to keep dropping

    def _should_drop(self) -> bool:
        if not self._always_mute:
            return False
        if self._bot_speaking:
            return True
        # Grace window: drop for GRACE_WINDOW_SEC after bot stops
        if time.monotonic() < self._grace_until:
            return True
        return False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction == FrameDirection.UPSTREAM:
            if isinstance(frame, BotStartedSpeakingFrame):
                self._bot_speaking = True
                self._grace_until = 0.0  # no grace needed while actively speaking
            elif isinstance(frame, BotStoppedSpeakingFrame):
                self._bot_speaking = False
                self._grace_until = time.monotonic() + self.GRACE_WINDOW_SEC
                logger.debug(
                    "[TranscriptionGate] Bot stopped – grace window of %.1fs starts now.",
                    self.GRACE_WINDOW_SEC,
                )

        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame))
            and self._should_drop()
        ):
            logger.debug(
                "[TranscriptionGate] Dropping %s (bot_speaking=%s grace=%s).",
                type(frame).__name__,
                self._bot_speaking,
                time.monotonic() < self._grace_until,
            )
            return  # Silently discard — do NOT push forward

        await self.push_frame(frame, direction)
# ---------------------------------------------------------------------------
class AudioSocketInputTransport(BaseInputTransport):
    def __init__(self, reader, writer, params: TransportParams, shared_state: dict,
                 disconnect_event: asyncio.Event, on_user_audio=None):
        super().__init__(params=params)
        self.reader = reader
        self.writer = writer
        self._shared_state = shared_state
        self.disconnect_event = disconnect_event
        self.on_user_audio = on_user_audio
        self._read_task = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._read_task = asyncio.create_task(self._read_loop())

    async def stop(self, *args, **kwargs):
        if self._read_task and not self._read_task.done():
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
        try:
            if not self.writer.is_closing():
                self.writer.close()
            await asyncio.wait_for(self.writer.wait_closed(), timeout=2.0)
        except Exception:
            pass
        await super().stop(*args, **kwargs)

    # async def stop(self, *args, **kwargs):
    #     if self._read_task:
    #         self._read_task.cancel()
    #     try:
    #         if not self.writer.is_closing():
    #             self.writer.close()
    #         await asyncio.wait_for(self.writer.wait_closed(), timeout=2.0)
    #     except Exception:
    #         pass
    #     await super().stop(*args, **kwargs)

    def _trigger_shutdown(self):
        if not self._shared_state["is_active"]:
            return
        logger.info("[AudioSocket] Triggering instantaneous pipeline shutdown...")
        self._shared_state["is_active"] = False
        try:
            if not self.writer.is_closing():
                self.writer.close()
        except Exception:
            pass
        if self.disconnect_event:
            self.disconnect_event.set()

    async def _read_loop(self):
        try:
            while True:
                header = await self.reader.readexactly(3)
                payload_type = header[0]
                payload_len = (header[1] << 8) | header[2]
                payload = await self.reader.readexactly(payload_len)

                if payload_type == 0x10:
                    if self.on_user_audio:
                        self.on_user_audio(payload)
                    frame = InputAudioRawFrame(
                        audio=payload,
                        sample_rate=self._params.audio_in_sample_rate,
                        num_channels=1,
                    )
                    await self.push_frame(frame)
                elif payload_type == 0x01:
                    logger.info("[AudioSocket] Asterisk connected!")
                elif payload_type == 0x00:
                    logger.info("[AudioSocket] Asterisk sent hangup signal.")
                    self._trigger_shutdown()
                    break
        except asyncio.IncompleteReadError:
            logger.info("[AudioSocket] Connection lost/closed by Asterisk.")
            self._trigger_shutdown()
        except Exception as e:
            logger.error(f"[AudioSocket] Read error: {e}")
            self._trigger_shutdown()


# ---------------------------------------------------------------------------
# AudioSocketOutputTransport
#
# FIX 2: Simplified output transport.
#
# Key changes:
#   • BotStartedSpeakingFrame / BotStoppedSpeakingFrame are NO LONGER pushed
#     from here.  BotSpeakingStateTracker (above) now owns that responsibility
#     synchronously in the pipeline.  The output transport only writes bytes
#     to the wire.
#   • The _is_interrupted lock is kept so audio tail-drip after an interruption
#     is discarded correctly.
#   • shared_state["bot_is_speaking"] is NOT touched here; the tracker owns it.
# ---------------------------------------------------------------------------
class AudioSocketOutputTransport(BaseOutputTransport):
    def __init__(self, writer, params: TransportParams, shared_state: dict,
                 on_bot_audio=None, always_mute: bool = False):
        super().__init__(params=params)
        self.writer = writer
        self._shared_state = shared_state
        self.on_bot_audio = on_bot_audio
        # When True (STTMuteStrategy.ALWAYS), interruption frames are ignored.
        # The bot speaks to completion no matter what the user does.
        self._always_mute = always_mute

        self._frame_queue = asyncio.Queue()
        self._playback_task = None
        self._discard_audio = False
        self._is_interrupted = False
        self._interrupt_event = asyncio.Event()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._playback_task = asyncio.create_task(self._playback_loop(), name=f"playback_AudioSocketOutputTransport")

    # async def stop(self, *args, **kwargs):
    #     if self._playback_task:
    #         self._playback_task.cancel()
    #     await super().stop(*args, **kwargs)

    async def stop(self, *args, **kwargs):
        if self._playback_task and not self._playback_task.done():
            
            # 1. Wake up the blocked queue instantly so the loop can exit
            try:
                self._frame_queue.put_nowait(EndFrame())
            except Exception:
                pass
            
            # 2. Issue the kill signal
            self._playback_task.cancel()
            
            # 3. Wait for it to die, but WITH A TIMEOUT. 
            try:
                await asyncio.wait_for(self._playback_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
                
        await super().stop(*args, **kwargs)

    async def _interruptible_sleep(self, seconds: float) -> bool:
        """
        Sleep for `seconds` but wake up immediately if _interrupt_event is set.
        Returns True if interrupted, False if the full sleep elapsed.

        Uses asyncio.wait() instead of asyncio.shield() to avoid orphaned
        inner tasks that cause 'Task was destroyed but it is pending' errors.
        """
        wait_task = asyncio.ensure_future(self._interrupt_event.wait())
        sleep_task = asyncio.ensure_future(asyncio.sleep(seconds))
        try:
            done, pending = await asyncio.wait(
                [wait_task, sleep_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
            return wait_task in done  # True = interrupted, False = sleep elapsed
        except asyncio.CancelledError:
            wait_task.cancel()
            sleep_task.cancel()
            for t in [wait_task, sleep_task]:
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
            raise

    async def _playback_loop(self):
        CHUNK_SIZE = 320
        _playing_utterance = False
        
        _audio_buffer = bytearray()
        _next_chunk_time = 0.0

        while self._shared_state["is_active"]:
            try:
                frame = await self._frame_queue.get()

                if isinstance(frame, TTSStartedFrame):
                    self._is_interrupted = False
                    self._discard_audio = False
                    self._interrupt_event.clear()
                    _playing_utterance = True
                    _audio_buffer.clear()
                    _next_chunk_time = time.time()
                    self._first_chunk_of_utterance = True

                elif isinstance(frame, TTSStoppedFrame):
                    if len(_audio_buffer) > 0 and not self._is_interrupted and not self._discard_audio:
                        chunk = _audio_buffer + (b'\x00' * (CHUNK_SIZE - len(_audio_buffer)))
                        if self.on_bot_audio:
                            self.on_bot_audio(chunk)
                        header = struct.pack('>BH', 0x10, len(chunk))
                        self.writer.write(header + chunk)
                        await self.writer.drain()  # once per utterance end — fine here

                    _audio_buffer.clear()

                    if _playing_utterance and not self._is_interrupted:
                        logger.debug("[AudioSocket] Last chunk played → pushing BotStoppedSpeakingFrame upstream")
                        self._shared_state["bot_is_speaking"] = False
                        await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
                    _playing_utterance = False

                elif isinstance(frame, TTSAudioRawFrame):
                    if self._is_interrupted:
                        self._frame_queue.task_done()
                        continue

                    _audio_buffer.extend(frame.audio)

                    while len(_audio_buffer) >= CHUNK_SIZE:
                        if not self._shared_state["is_active"] or self._discard_audio:
                            break

                        chunk = _audio_buffer[:CHUNK_SIZE]
                        del _audio_buffer[:CHUNK_SIZE]

                        if self.on_bot_audio:
                            self.on_bot_audio(chunk)

                        header = struct.pack('>BH', 0x10, len(chunk))

                        if getattr(self, '_first_chunk_of_utterance', False):
                            logger.warning(f"[AUDIO-OUT] First chunk on wire: {time.time():.3f}")
                            self._first_chunk_of_utterance = False

                        self.writer.write(header + chunk)

                        if self.writer.transport.get_write_buffer_size() > 65536:
                            await self.writer.drain()
                        # NO drain() here — removed, was causing per-chunk TCP stall

                        now = time.time()
                        if _next_chunk_time < now and (now - _next_chunk_time) > 0.1:
                            _next_chunk_time = now

                        sleep_duration = _next_chunk_time - now
                        if sleep_duration > 0:
                            interrupted = await self._interruptible_sleep(sleep_duration)
                            if interrupted:
                                logger.debug("[AudioSocket] Chunk sleep interrupted")
                                break

                        _next_chunk_time += (CHUNK_SIZE / 16000.0)

                self._frame_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[AudioSocket] Playback error: {e}")
                break

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Always let control frames propagate through the base class
        if isinstance(frame, (StartFrame, EndFrame, CancelFrame)):
            await super().process_frame(frame, direction)

        if not self._shared_state["is_active"]:
            return

        if isinstance(frame, (TTSStartedFrame, TTSStoppedFrame, TTSAudioRawFrame)):
            await self._frame_queue.put(frame)

        elif isinstance(frame, CancelFrame) or type(frame).__name__ == "InterruptionFrame":
            if self._always_mute:
                # ALWAYS mode: bot is never interrupted. Silently drop the interruption signal.
                logger.debug("[AudioSocket] ALWAYS mode: ignoring interruption frame.")
                return

            logger.info("[AudioSocket] Interruption detected – discarding queued audio immediately.")
            self._is_interrupted = True
            self._discard_audio = True

            # Wake the sleeping playback loop right now, don't wait for the chunk delay
            self._interrupt_event.set()

            # Drain any frames that are already queued
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                    self._frame_queue.task_done()
                except asyncio.QueueEmpty:
                    break

            # Tell the tracker (upstream) that the bot has stopped so it can
            # clear bot_is_speaking and unblock the STTMuteFilter immediately.
            await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)

        elif isinstance(frame, EndFrame):
            logger.info("[AudioSocket] EndFrame → sending 0x00 hangup signal.")
            try:
                hangup_signal = struct.pack('>BH', 0x00, 0)
                self.writer.write(hangup_signal)
                await self.writer.drain()
            except Exception as e:
                logger.debug(f"[AudioSocket] Error sending hangup signal: {e}")


# ---------------------------------------------------------------------------
# AudioSocketTransport
# ---------------------------------------------------------------------------
class AudioSocketTransport(BaseTransport):
    def __init__(self, reader, writer, params: TransportParams,
                 disconnect_event: asyncio.Event = None,
                 on_user_audio=None, on_bot_audio=None,
                 always_mute: bool = False):
        super().__init__()
        self._shared_state = {"is_active": True, "bot_is_speaking": False}
        self._input = AudioSocketInputTransport(
            reader, writer, params, self._shared_state, disconnect_event, on_user_audio
        )
        self._output = AudioSocketOutputTransport(
            writer, params, self._shared_state, on_bot_audio,
            always_mute=always_mute
        )

    def input(self) -> BaseInputTransport:
        return self._input

    def output(self) -> BaseOutputTransport:
        return self._output
    
# class DummyLLM(FrameProcessor):
#     """
#     Replaces GoogleLLMService. The moment it receives any LLMMessagesFrame
#     it instantly pushes a fixed TextFrame — zero network, zero inference time.
#     If you still see 3-4s delay with this, the problem is STT or TTS, not LLM.
#     """
#     async def process_frame(self, frame: Frame, direction: FrameDirection):
#         await super().process_frame(frame, direction)

#         # Context aggregator outputs this when user turn is ready
#         if hasattr(frame, 'messages') or frame.__class__.__name__ in (
#             'LLMMessagesFrame', 'OpenAILLMContextFrame', 'LLMContextFrame'
#         ):
#             logger.warning(f"[DUMMY-LLM] Got {type(frame).__name__} → instant reply")
#             await self.push_frame(LLMFullResponseStartFrame())
#             await self.push_frame(TextFrame("This is a dummy response."))
#             await self.push_frame(LLMFullResponseEndFrame())
#             return

#         await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# handle_asterisk_stream
# ---------------------------------------------------------------------------
async def handle_asterisk_stream(
    call_id, session_id, reader, writer,
    organisation_id: int,
    language: str = "hi",
    system_prompt=None,
    dispatcher=None,
    campaign_id=None,
    redis_conn=None,
    activemq_conn=None,
    assistant_config=None,
    stt_config=None,
    tts_config=None,
    agent_details=None,
):
    
    try:
        sock = writer.get_extra_info('socket')
        if sock is not None:
            # Force the socket to send packets immediately
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            # Optional but recommended: Explicitly set buffer sizes for audio
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
            logger.info(f"[Network] TCP_NODELAY enabled for call {call_id}")
    except Exception as e:
        logger.warning(f"[Network] Could not set TCP options: {e}") 

    recordings: Dict[str, RecordingState] = {}
    stop_idle_prompt = False
    is_transferring = False

    disconnect_event = asyncio.Event()

    if agent_details:
        random_agent = random.choice(agent_details)
        agent_name = random_agent.get("name", "Agent")
        agent_phone_number = random_agent.get("phone_number", "")
    else:
        agent_name = "Agent"
        agent_phone_number = ""


    session = SESSION_PIPELINES.get(call_id)
    
    if session and session.get("status") == "PREWARMED":
        logger.info(f"⚡ Using PREWARMED pipeline services for call {call_id}")
        stt = session["stt"]
        tts = session["tts"]
        llm = session["llm"]
        
        # Change status so the cleanup_if_not_connected task doesn't destroy it mid-call
        session["status"] = "CONNECTED" 
    else:
        logger.error(f"No pre-warmed pipeline found for {call_id}. Call will fail.")
        # Optional: You can put your old manual initialization code here as a fallback, 
        # but realistically, if pre-warm failed, the call shouldn't proceed.
        return

    # ------------------------------------------------------------------
    # Audio-recording callbacks
    # ------------------------------------------------------------------
    def _rec_state(key: str) -> RecordingState:
        if key not in recordings:
            recordings[key] = RecordingState()
        return recordings[key]

    def append_user_audio(audio_bytes):
        st = _rec_state(call_id)
        if st.recording_start_time == 0.0:
            st.recording_start_time = time.time()
        current_time = time.time() - st.recording_start_time
        st.user_turns.append((current_time, audio_bytes))

    def append_bot_audio(audio_bytes):
        st = _rec_state(call_id)
        if st.recording_start_time == 0.0:
            st.recording_start_time = time.time()
        current_time = time.time() - st.recording_start_time
        st.bot_audio_chunks.append((current_time, audio_bytes))
        st.last_timestamp = current_time

    # ------------------------------------------------------------------
    # VAD + Transport
    # ------------------------------------------------------------------
    vad_analyzer = SileroVADAnalyzer(
    params=VADParams(
        sample_rate=SAMPLE_RATE,
        stop_secs=0.8,       # The key to low latency
        start_secs=0.2,      # Prevents false starts
        confidence=0.7       # Filters out line noise
    )
)

    # Parse STT mute strategies FIRST so always_mute is known before transport
    # and bot_speaking_tracker are constructed.
    _strategy_config = assistant_config.get("strategies", [])
    _stt_strategies = set()
    for _name in _strategy_config:
        try:
            _stt_strategies.add(getattr(STTMuteStrategy, _name))
            logger.info(f"[STT] Strategy added: {_name}")
        except AttributeError:
            logger.warning(f"[STT] Unknown STTMuteStrategy ignored: {_name!r}")
    if assistant_config.get("callSettings", {}).get("muteCalleeWhileAssistantSpeaks", False):
        _stt_strategies.add(STTMuteStrategy.MUTE_UNTIL_FIRST_BOT_COMPLETE)
        logger.info("[STT] STTMuteStrategy.ALWAYS enabled via callSettings – no barge-in.")
    if not _stt_strategies:
        logger.info("[STT] No mute strategies configured – barge-in is fully active.")
    _always_mute = STTMuteStrategy.ALWAYS in _stt_strategies

    transport = AudioSocketTransport(
        reader, writer,
        params=TransportParams(
            audio_in_sample_rate=SAMPLE_RATE,
            audio_out_sample_rate=SAMPLE_RATE,
            audio_in_enabled=True,
            vad_analyzer=vad_analyzer,
        ),
        disconnect_event=disconnect_event,
        on_user_audio=append_user_audio,
        on_bot_audio=append_bot_audio,
        always_mute=_always_mute,
    )

    # Expose shared_state so processors can read bot_is_speaking / is_active
    shared_state = transport._shared_state

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
#     llm = GoogleLLMService(
#         api_key=os.getenv("GOOGLE_API_KEY"),
#         model="gemini-2.5-flash-lite",
#         stream=True,
#         generation_config={
#             "temperature": 0.1,       # was 0.3
#             "top_p": 0.8,             # was 0.9
#             "top_k": 10,              # was 20
#             "max_output_tokens": 30,  # was 50
#             "candidate_count": 1,
#         },
# )

    # 1. Get dynamic credentials
    # credential_file_name, project_id = get_file_name()
    # credentials_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), credential_file_name)

    # # 2. Set environment variable so the underlying Google Auth finds it
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    # logger.info(f"PROJECT ID : {project_id}")

    # 3. Initialize Vertex AI Service
    # llm = GoogleVertexLLMService(
    #     project_id=project_id,
    #     location="asia-south1",  # Ensure this region is enabled in your GCP console
    #     model="gemini-2.5-flash",
    #     settings=GoogleVertexLLMService.Settings(
    #         temperature=0.1,
    #         top_p=0.8,
    #         top_k=10,
    #         # max_output_tokens=30,
    #     )
    # )


    async def add_hangup_reason(reason: str):
        try:
            hangup_reason_key = f"{call_id}_hangup_reason"
            await redis_conn.set(hangup_reason_key, reason, ex=3600)
            logger.info(f"Stored hangup reason in Redis: {hangup_reason_key}")
        except Exception as e:
            logger.exception(f"Failed to store hangup reason in Redis: {e}")

    async def local_end_conversation_handler(function_name, tool_session_id, args, llm, context, result_callback):
        logger.info(f"LLM requested end_conversation. Reason: {args.get('reason')}")
        nonlocal stop_idle_prompt
        stop_idle_prompt = True

        # 1. Mute the user immediately so they cannot interrupt the final goodbye
        await task.queue_frames([STTMuteFrame(mute=True)])

        # to finish. This guarantees it cannot generate the echo sentence.

        async def wait_and_hangup():
            # Give the pipeline a moment to process the TTS for the prompt's goodbye
            await asyncio.sleep(2.0)
            
            # Debounce: Wait until the bot has been completely silent for 2 seconds.
            silent_time = 0.0
            while silent_time < 2.0:
                if shared_state.get("bot_is_speaking", False):
                    silent_time = 0.0  # Reset the clock if the bot is actively playing audio
                else:
                    silent_time += 0.5
                
                await asyncio.sleep(0.5)
                
            logger.info("[EndCall] Bot finished speaking final goodbye, hanging up now")
            
            # 3. Kill the pipeline. The LLM is destroyed along with it, 
            # so it never cared that we didn't return the tool callback.
            await task.queue_frames([EndFrame()])

        # Run the polling loop in the background
        asyncio.create_task(wait_and_hangup())
        
        # Return None so Pipecat knows we are handling the callback manually (even though we are ignoring it)
        return None

    llm.register_function("end_conversation", local_end_conversation_handler)

    async def handle_transfer(function_name, tool_session_id, args, llm, context, result_callback):
        nonlocal stop_idle_prompt, is_transferring
        
        # 1. Get the name the LLM wants to transfer to
        requested_agent_name = args.get("agent_name")
        target_phone_number = None

        # 2. Search your agent_details array for that exact name
        if requested_agent_name and agent_details:
            selected_agent = next(
                (agent for agent in agent_details if agent.get("name", "").lower() == requested_agent_name.lower()), 
                None
            )
            if selected_agent:
                target_phone_number = selected_agent.get("phone_number")
                logger.info(f"[Transfer] LLM requested: {requested_agent_name}. Found number: {target_phone_number}")
            else:
                logger.warning(f"[Transfer] LLM requested unknown agent '{requested_agent_name}'.")
        
        # 3. Fallback logic: if no valid name was passed, just pick a random one
        if not target_phone_number:
            random_agent = random.choice(agent_details)
            target_phone_number = random_agent.get("phone_number", "")
            logger.info(f"[Transfer] Using fallback random number: {target_phone_number}")

        logger.info(f"Transferring call {call_id} to {target_phone_number}")
        
        stop_idle_prompt = True
        is_transferring = True

        try:
            await task.queue_frames([
                STTMuteFrame(mute=True),
                # Optional: Make the TTS dynamic based on the name!
                # TTSSpeakFrame(f"ठीक है, मैं आपकी कॉल {requested_agent_name} को ट्रांसफर कर रहा हूँ।"),
            ])

            bot_done_event = asyncio.Event()

            original_process = bot_speaking_tracker.process_frame
            async def _wait_for_bot_done(frame, direction):
                if isinstance(frame, BotStoppedSpeakingFrame):
                    bot_done_event.set()
                    bot_speaking_tracker.process_frame = original_process
                await original_process(frame, direction)
            bot_speaking_tracker.process_frame = _wait_for_bot_done

            async def wait_and_transfer():
                try:
                    await asyncio.wait_for(bot_done_event.wait(), timeout=10.0)
                    logger.info("[Transfer] Bot finished speaking, firing AMI Redirect NOW")
                except asyncio.TimeoutError:
                    logger.warning("[Transfer] Timeout waiting for bot to finish, forcing transfer NOW")
                
                await asyncio.sleep(0.5) 
                
                # --- FIRE AMI WITH THE TARGET NUMBER ---
                await force_ami_transfer(call_id, target_phone_number)
                
                if result_callback:
                    await result_callback({"status": "success", "message": "Call transfer initiated."})

            asyncio.create_task(wait_and_transfer())
            return None

        except Exception as e:
            logger.error(f"Transfer error: {e}")
            if result_callback:
                await result_callback({"status": "error", "message": str(e)})
            return None
    
    llm.register_function("transfer_to_agent", handle_transfer)

    # ------------------------------------------------------------------
    # STT
    # ------------------------------------------------------------------
    # stt = None
    # if stt_config and stt_config.get("engine").lower() == "deepgram":
    #     stt = DeepgramSTTService(
    #         api_key=stt_config.get("apiKey"),
    #         # Use the new Pipecat Settings class instead of LiveOptions
    #         settings=DeepgramSTTService.Settings(
    #         model="nova-3",
    #         language=stt_config.get("language"),            interim_results=True,
    #         punctuate=False,
    # ),
    #     )
    # elif stt_config and stt_config.get("engine").lower() == "google":
    #     stt = GoogleSTTService(
    #         api_key=os.getenv("GOOGLE_API_KEY"),
    #         params=GoogleSTTService.InputParams(
    #             languages=[stt_config.get("language"),], 
    #             model=stt_config.get("model","latest_short"), 
    #             use_separate_recognition_per_channel=False,
    #             enable_automatic_punctuation=False,
    #             #enable_spoken_punctuation=False,
    #             # enable_spoken_emojis=False,
    #             profanity_filter=True,
    #             # enable_word_time_offsets=True,
    #             # enable_word_confidence=True,
    #             # enable_interim_results=True,
    #             enable_voice_activity_events=False,   # ← prevents auto end after first turn
    #             single_utterance=False,   # don't close stream after first result
    #             interim_results=True,     # keep stream alive between utterances

    #         ),
            
    #     )
    # elif stt_config and stt_config.get("engine").lower() == "soniox":
    #     stt = SonioxSTTService(api_key=os.getenv("SONIOX_API_KEY"))
    # else:
    #     raise ValueError(f"Invalid STT engine: {stt_config.get('engine')}")

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------
    # tts = None
    # if tts_config and tts_config.get("engine").lower() == "cartesia":
    #     print("using tts config cartesia")
    #     voice_id=None
    #     voice = assistant_config.get("voice", {})

    #     if voice.get("provider").lower() == "cartesia":
    #         voice_id = voice.get("voiceId")
    #     else:
    #         language = voice.get("language", "")
    #         voice_id = get_voice_code(language.replace("_", "-"))

    #     tts = CartesiaTTSService(
    #         api_key=tts_config.get("apiKey"),
    #         voice_id=voice_id,
    #         params=CartesiaTTSService.InputParams(
    #             language=get_language_enum(assistant_config["voice"]["language"].replace("_", "-")),
    #             sample_rate=assistant_config["voice"].get("sampleRate", SAMPLE_RATE),
    #             encoding=assistant_config["voice"].get("encoding", "pcm_mulaw"),
    #         ),
    #     )
    # else:
    #     print(f"using tts config google - voiceId and language ","",assistant_config["voice"]["voiceId"],"  ",assistant_config["voice"]["language"].replace("_", "-"))
    #     tts = GoogleTTSService(
    #         voice_id=assistant_config["voice"]["voiceId"],
    #         params=GoogleTTSService.InputParams(
    #             language=get_language_enum(assistant_config["voice"]["language"].replace("_", "-"))
    #         ),
    #         credentials=None,
    #     )

    # ------------------------------------------------------------------
    # Context + aggregators
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Context + aggregators
    # ------------------------------------------------------------------
    # Add this import at the top of your file:

    ist_tz = zoneinfo.ZoneInfo('Asia/Kolkata')
    current_time = datetime.now(ist_tz).strftime("%A, %d %B %Y, %I:%M %p")
    
    DUSTER_BROCHURE = """
    RENAULT DUSTER — key specs for spoken use:
    Engines: 1.3 litre Turbo Petrol (156 PS, 254 Nm) with 6-speed manual or X-Tronic CVT;
    1.5 litre diesel (115 PS, 260 Nm) with 6-speed manual.
    Drivetrain: 4x2 and 4x4 (AWD) variants available on the Turbo Petrol.
    Ground clearance: 205 mm. Boot space: 445 litres, expandable with 60:40 split rear seats.
    Safety: up to 6 airbags, ESC, Hill Start Assist, Hill Descent Control, 360 degree camera on top trims.
    Interior: 8 inch touchscreen infotainment, wireless Android Auto and Apple CarPlay,
    7 inch fully digital driver display, wireless phone charger, rain-sensing wipers.
    Design: Y-shaped LED DRLs, muscular wheel arches, roof rails, up to 18 inch alloy wheels.
    """

    KIGER_BROCHURE = """
    RENAULT KIGER — key specs for spoken use:
    Engines: 1.0 litre naturally aspirated petrol (72 PS) with manual or AMT;
    1.0 litre Turbo petrol (100 PS, 160 Nm) with manual or CVT.
    Ground clearance: 205 mm, among the highest in its segment.
    Boot space: 405 litres.
    Safety: up to 6 airbags, ESC, ABS with EBD, rear parking sensors and camera.
    Interior: 8 inch touchscreen with wireless Android Auto and Apple CarPlay,
    7 inch digital instrument cluster, wireless charging, air purifier, JBL sound system on top trims.
    Design: dual-tone exterior options, LED projector headlamps, up to 16 inch diamond-cut alloy wheels.
    """

    def _build_product_prompt(model: str) -> str:
        if model == "duster":
            brochure = DUSTER_BROCHURE
        elif model == "kiger":
            brochure = KIGER_BROCHURE
        else:
            brochure = DUSTER_BROCHURE + "\n" + KIGER_BROCHURE

        return BASE_SYSTEM_INSTRUCTION + f"""
        You are now operating as the Renault Product Consultant. You have direct access to the vehicle brochures.
        DIRECTIVE: When answering features, engine options, or specs, provide rich, descriptive, and comprehensive
        explanations utilizing the technical metrics from the brochures below. Speak in fluid, natural paragraphs.
        Do not read the data as a list — weave it into natural spoken sentences.

        [BROCHURE DATA]:
        {brochure}
        """
        
    async def handle_enter_product_expert(function_name, tool_session_id, args, llm, context, result_callback):
        model = (args or {}).get("model", "unspecified")
        logger.info(f"[ProductExpert] Entering product expert mode for model={model}")
        context.messages[0]["content"] = _build_product_prompt(model)
        await result_callback({"status": "product_expert_mode_active", "model": model})

    async def handle_exit_product_expert(function_name, tool_session_id, args, llm, context, result_callback):
        logger.info("[ProductExpert] Exiting product expert mode")
        context.messages[0]["content"] = BASE_SYSTEM_INSTRUCTION
        await result_callback({"status": "base_mode_restored"})

    llm.register_function("enter_product_expert_mode", handle_enter_product_expert)
    llm.register_function("exit_product_expert_mode", handle_exit_product_expert)
        
    from pipecat.adapters.schemas.function_schema import FunctionSchema

    ENTER_PRODUCT_EXPERT_TOOL = FunctionSchema(
        name="enter_product_expert_mode",
        description=(
            "CALL THIS IMMEDIATELY when the user asks about vehicle specs, features, "
            "engine options, dimensions, safety, interior, or comparisons for the "
            "Renault Duster or Renault Kiger. "
            "Do NOT attempt to answer from memory first — call this function BEFORE "
            "generating any spoken product content, so you have brochure data loaded. "
            "Trigger phrases: 'engine', 'mileage', 'features', 'boot space', 'safety rating', "
            "'Duster', 'Kiger', 'variant', 'price', 'specifications'."
        ),
        properties={
            "model": {
                "type": "string",
                "enum": ["duster", "kiger", "unspecified"],
                "description": "Which model the user is asking about, if identifiable from their utterance."
            }
        },
        required=["model"]
    )

    EXIT_PRODUCT_EXPERT_TOOL = FunctionSchema(
        name="exit_product_expert_mode",
        description=(
            "CALL THIS when the user's query moves away from vehicle specs/features "
            "back to routing needs (roadside, finance, dealer lookup, radio code) or "
            "when they want to end the call. Restores the base routing system prompt."
        ),
        properties={},
        required=[]
    )
    
    tools = ToolsSchema(standard_tools=[
        END_CONVERSATION_TOOL,
        TRANSFER_AGENT_TOOL,
        ENTER_PRODUCT_EXPERT_TOOL,
        EXIT_PRODUCT_EXPERT_TOOL,
    ])
    
    
    # # Append this to whatever your current system prompt is
    # dynamic_system_prompt = system_prompt + f"""
    # \n\n---
    # [CRITICAL SYSTEM CONTEXT]
    # The current date and time right now is: {current_time} (IST).
    # """

    # messages = [{"role": "system", "content": dynamic_system_prompt}]
        
    context = LLMContext(
        tools=tools
    )
    
    # We create the aggregator pair directly, bypassing the GoogleLLMService helper
    context_aggregator = LLMContextAggregatorPair(context=context)

    # ------------------------------------------------------------------
    # STT Mute Strategy
    #
    # Strategy meanings:
    #   ALWAYS                    → STT muted the entire time bot speaks.
    #                               User cannot barge in at all. Bot finishes,
    #                               then user can speak.
    #   FIRST_SPEECH              → STT muted until the bot has spoken for the
    #                               first time. After that, barge-in is open.
    #   MUTE_UNTIL_FIRST_BOT_COMPLETE → STT muted until the bot's first full
    #                               utterance finishes. After that, barge-in ok.
    #   FUNCTION_CALL             → STT muted only during active function/tool
    #                               calls (prevents user from interrupting while
    #                               the bot is mid-tool-call).
    #
    # Strategies were pre-parsed above (before transport construction) so that
    # always_mute could be passed to the transport and bot_speaking_tracker.
    # We just wire them into STTMuteFilter here.
    # ------------------------------------------------------------------
    stt_strategies = _stt_strategies
    stt_mute_processor = STTMuteFilter(
        config=STTMuteConfig(strategies=stt_strategies)
    )

    # ------------------------------------------------------------------
    # Bot-speaking state tracker
    # ------------------------------------------------------------------
    bot_speaking_tracker = BotSpeakingStateTracker(shared_state, always_mute=_always_mute)

    # Gate that drops transcriptions produced during bot speech (ALWAYS mode only)
    transcription_gate = TranscriptionGate(shared_state, always_mute=_always_mute)

    # ------------------------------------------------------------------
    # Idle callback
    #
    # Called by SmartUserIdleProcessor after the user has been silent for
    # `userIdleTimeoutSec` seconds since the bot last stopped speaking.
    #
    # NOTE: The bot_is_speaking guard from the old code is intentionally
    # removed here. SmartUserIdleProcessor cancels its deadline the moment
    # BotStartedSpeakingFrame arrives (upstream), so this callback is
    # structurally impossible to invoke while the bot is speaking.
    # Adding the guard back would be harmless but misleading.
    # ------------------------------------------------------------------
    async def handle_user_idle(processor, retry_count):
        nonlocal stop_idle_prompt
        if stop_idle_prompt:
            # Call is already ending (transfer or end_conversation triggered).
            # Return False so SmartUserIdleProcessor deactivates permanently.
            return False

        idle_prompts = assistant_config.get("idlePrompts") or []

        if isinstance(idle_prompts, dict):
            try:
                ordered_keys = sorted(idle_prompts.keys(), key=lambda k: int(k))
            except (ValueError, TypeError):
                ordered_keys = sorted(idle_prompts.keys())
            prompts = [idle_prompts[k] for k in ordered_keys]
        elif isinstance(idle_prompts, (list, tuple)):
            prompts = list(idle_prompts)
        else:
            prompts = []

        total_prompts = len(prompts)

        # Final timeout: hang up
        if total_prompts == 0 or retry_count > total_prompts:
            await add_hangup_reason("USER_IDLE")
            final_prompt = (
                prompts[-1] if total_prompts > 0
                else "It seems you're not responding. We'll end this call now."
            )
            await task.queue_frames([
                STTMuteFrame(mute=True),
                TTSSpeakFrame(final_prompt),
            ])

            async def wait_and_hangup():
                await asyncio.sleep(10.0)
                await task.queue_frames([EndFrame()])

            asyncio.create_task(wait_and_hangup())
            return False  # Stop firing

        # Warning prompt: speak and keep waiting
        prompt_text = prompts[retry_count - 1]
        await task.queue_frames([TTSSpeakFrame(prompt_text)])
        return True  # Rearm timer
    #
    # Problem with stock UserIdleProcessor:
    #   It counts time since the last user speech event. On call start the
    #   bot plays a greeting — the user hasn't spoken yet — so the idle
    #   timer fires after N seconds even mid-greeting.  Returning True from
    #   the callback only resets the countdown, so it fires again N seconds
    #   later in a loop for the entire duration of a long bot utterance.
    #
    # Fix:
    #   This processor tracks bot_is_speaking from shared_state and resets
    #   its own internal deadline every time the bot starts speaking.
    #   The idle timeout only counts time since the bot LAST STOPPED speaking
    #   (or since the last user speech, whichever is more recent).
    #   While the bot is speaking the clock simply does not tick.
    # ------------------------------------------------------------------
    class SmartUserIdleProcessor(FrameProcessor):
        """
        Fires the idle callback only after the user has been silent for
        `timeout` seconds since the bot last stopped speaking.

        Timer behaviour:
          - BotStartedSpeakingFrame (upstream)  → cancel timer entirely (clock paused)
          - BotStoppedSpeakingFrame (upstream)  → arm timer from zero (fresh turn)
          - UserStartedSpeakingFrame / TranscriptionFrame (downstream) → arm timer from zero
          - stop() called by Pipecat on shutdown → deactivate, cancel timer permanently

        The callback returns:
          True  → rearm the timer for another round
          False → deactivate permanently (call is ending)

        Kill switch: both self._active AND shared_state["is_active"] must be True
        for the timer to fire.  When the pipeline shuts down shared_state["is_active"]
        goes False, which prevents any in-flight _fire() tasks from doing anything.
        """

        def __init__(self, callback, timeout: float, pipeline_state: dict):
            super().__init__()
            self._callback = callback
            self._timeout = timeout
            self._pipeline_state = pipeline_state  # shared_state from transport
            self._retry_count = 0
            self._deadline = None  # asyncio.TimerHandle
            self._active = True

        def _is_alive(self) -> bool:
            """True only when both this processor and the pipeline are active."""
            return self._active and self._pipeline_state.get("is_active", False)

        def _cancel_deadline(self):
            if self._deadline:
                self._deadline.cancel()
                self._deadline = None

        def _arm_deadline(self):
            """Start/restart the countdown from now. No-op if already dead."""
            self._cancel_deadline()
            if not self._is_alive():
                return
            loop = asyncio.get_event_loop()
            self._deadline = loop.call_later(self._timeout, self._on_timeout)
            logger.debug(f"[SmartIdle] Timer armed ({self._timeout}s)")

        def _on_timeout(self):
            self._deadline = None
            if not self._is_alive():
                return  # Pipeline already dead — do NOT create a task
            asyncio.create_task(self._fire())

        async def _fire(self):
            # Hard-stop guard: checked again because task creation is async
            if not self._is_alive():
                logger.debug("[SmartIdle] _fire() aborted – pipeline no longer active.")
                return
            # If bot is somehow still speaking (race), just rearm
            if self._pipeline_state.get("bot_is_speaking", False):
                logger.debug("[SmartIdle] Timeout fired but bot still speaking – rearming.")
                self._arm_deadline()
                return
            self._retry_count += 1
            logger.info(f"[SmartIdle] User idle timeout fired (retry={self._retry_count})")
            try:
                keep_going = await self._callback(self, self._retry_count)
            except Exception as e:
                logger.warning(f"[SmartIdle] Callback error: {e}")
                keep_going = False
            if keep_going and self._is_alive():
                self._arm_deadline()
            else:
                self._active = False
                self._cancel_deadline()

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)

            if direction == FrameDirection.DOWNSTREAM:
                if isinstance(frame, TranscriptionFrame):
                    # A completed transcription means the user actually said something
                    # recognisable → reset idle counter and restart timer.
                    logger.debug("[SmartIdle] TranscriptionFrame – resetting retry count and timer.")
                    self._retry_count = 0
                    self._arm_deadline()
                elif isinstance(frame, UserStartedSpeakingFrame):
                    # User started making sound → restart the countdown window so they
                    # get the full timeout to finish speaking, but do NOT reset the
                    # retry_count — that only resets on a completed transcription.
                    logger.debug("[SmartIdle] UserStartedSpeaking – restarting timer (retry stays %d).", self._retry_count)
                    self._arm_deadline()

                elif isinstance(frame, UserStoppedSpeakingFrame):
                    logger.debug("[SmartIdle] UserStoppedSpeaking – pausing timer (bot thinking).")
                    self._cancel_deadline()

            elif direction == FrameDirection.UPSTREAM:
                if isinstance(frame, BotStartedSpeakingFrame):
                    logger.debug("[SmartIdle] Bot started speaking – pausing idle timer.")
                    self._cancel_deadline()

                elif isinstance(frame, BotStoppedSpeakingFrame):
                    # Rearm timer from NOW so user gets full timeout to respond.
                    # Do NOT reset _retry_count here. It tracks consecutive idle
                    # timeouts from the user. Resetting on every bot utterance caused
                    # retry=1 forever: idle prompt fires → bot speaks → BotStopped
                    # → retry reset to 0 → fires at retry=1 again infinitely.
                    logger.debug(
                        "[SmartIdle] Bot stopped speaking – arming timer (retry=%d).",
                        self._retry_count,
                    )
                    self._arm_deadline()

            await self.push_frame(frame, direction)

        async def stop(self, *args, **kwargs):
            logger.info("[SmartIdle] stop() called – deactivating permanently.")
            self._active = False
            self._cancel_deadline()
            await super().stop(*args, **kwargs)

    smart_idle = SmartUserIdleProcessor(
        callback=handle_user_idle,
        timeout=float(
            assistant_config.get("callSettings", {}).get("userIdleTimeoutSec", 10)
        ),
        pipeline_state=shared_state,
    )

    # ------------------------------------------------------------------
    # Remaining processors
    # ------------------------------------------------------------------
    CHUNK_DURATION = 2
    audiobuffer = AudioBufferProcessor(
        sample_rate=SAMPLE_RATE,
        buffer_size=SAMPLE_RATE * 2 * CHUNK_DURATION,
        num_channels=2,
        enable_turn_audio=True,
    )

    async def handle_max_duration(processor):
        logger.info("Maximum call duration reached, triggering graceful sign-off")
        await add_hangup_reason("MAX_DURATION_REACHED")
        
        bot_speaking_tracker._always_mute = True
        transport.output()._always_mute = True

        await task.queue_frames([
            STTMuteFrame(mute=True),
            TTSSpeakFrame(
                "This call has reached its maximum duration. Bye."
            ),
        ])

        await asyncio.sleep(6)
            
            # 5. Safely end the pipeline
        await task.queue_frames([EndFrame()])


    max_duration_processor = MaxDurationProcessor(
        callback=handle_max_duration,
        max_duration=float(
            assistant_config.get("callSettings", {}).get("maxDurationSec", 900) - 5
        ),
    )

    # speech_timeout = SpeechTimeoutProcessor(timeout_sec=10.0)

    user_logger = TranscriptLogger(
        "STT-Interceptor", "user", redis_conn, organisation_id, campaign_id, activemq_conn, call_id, session_id
    )
    bot_logger = TranscriptLogger(
        "TTS-Interceptor", "bot", redis_conn, organisation_id, campaign_id, activemq_conn, call_id, session_id
    )

    latency_observer = UserBotLatencyObserver()


    # Add event handlers for the metrics you want to track
    @latency_observer.event_handler("on_latency_measured")
    async def on_latency_measured(observer, latency):
        logger.warning(f"[LAT] User→Bot latency: {latency:.3f}s")

    @latency_observer.event_handler("on_latency_breakdown")
    async def on_latency_breakdown(observer, breakdown):
        logger.warning(f"[LAT] Breakdown ({len(breakdown.chronological_events())} events):")
        for event in breakdown.chronological_events():
            logger.warning(f"[LAT]   {event}")

    @latency_observer.event_handler("on_first_bot_speech_latency")
    async def on_first_bot_speech_latency(observer, latency):
        logger.warning(f"[LAT] First bot speech: {latency:.3f}s")

    # llm = DummyLLM()
    #
    # Pipeline order — upstream direction is RIGHT TO LEFT (high index → low index).
    #
    # bot_speaking_tracker pushes BotStartedSpeakingFrame / BotStoppedSpeakingFrame
    # UPSTREAM. For smart_idle to receive those signals, it MUST have a lower index
    # (be to the LEFT of) bot_speaking_tracker in this list.
    #
    # Downstream (→):  input[0] ... output[N]
    # Upstream   (←):  output[N] ... input[0]
    #
    # Signal routing:
    #   BotStartedSpeakingFrame  ← bot_speaking_tracker[11] → smart_idle[10] → stt_mute[2] ✓
    #   BotStoppedSpeakingFrame  ← same path ✓
    #   UserStartedSpeakingFrame ← input → ... (barge-in triggers InterruptionFrame)
    # ------------------------------------------------------------------

    barge_in_processor = BargeInContextProcessor()

    pipeline = Pipeline([
        transport.input(),            # 0
        stt_mute_processor,           # 2  ← MOVED: Must drop raw audio BEFORE it hits the STT
        stt,                          # 3  ← raw audio in, TranscriptionFrame out
        transcription_gate,           # 4  ← drops transcriptions while bot speaks (ALWAYS mode)
        max_duration_processor,       # 5
        user_logger,    
        VADSpeedHackProcessor(),      # 1. Injects the fake stop signal instantly
        barge_in_processor,
        context_aggregator.user(),    # 2. Pipecat's native aggregator safely manages the run/cancel states!
        llm,                          # 9
        bot_logger,                   # 10
        tts,                          # 11 ← emits TTSStarted/Stopped downstream
        smart_idle,                   # 12 ← MUST be before tracker; receives upstream from [13]
        bot_speaking_tracker,         # 13 ← converts TTSStarted/Stopped → BotStarted/Stopped upstream
        context_aggregator.assistant(), # 14
        # audiobuffer,                  # 15
        transport.output(),           # 16
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            audio_in_sample_rate=SAMPLE_RATE,
            audio_out_sample_rate=SAMPLE_RATE,
            enable_metrics=True,  # Required for detailed breakdown
            enable_usage_metrics=False,
            observers=[latency_observer],  # Add the observer
        ),
        cancel_on_idle_timeout=False
    )

    # ------------------------------------------------------------------
    # Recording setup
    # ------------------------------------------------------------------
    await audiobuffer.start_recording()
    st = _rec_state(call_id)
    st.server_name = "server"
    st.filename = f"{st.server_name}_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    st.sample_rate = SAMPLE_RATE
    st.num_channels = 2
    if st.recording_start_time == 0.0:
        st.recording_start_time = time.time()

    # ------------------------------------------------------------------
    # Runner + greeting
    # ------------------------------------------------------------------
    runner = PipelineRunner(handle_sigint=False, force_gc=True)

    # logger.info(f"[Bot] Queuing initial LLM greeting for {call_id}...")
    # await task.queue_frames([
    #     LLMMessagesAppendFrame(
    #         messages=[{"role": "user", "content": "Hello! I have answered the call. Please greet me."}]
    #     ),
    #     LLMRunFrame(),
    # ])

    # run_task = asyncio.create_task(runner.run(task))

    run_task = asyncio.create_task(runner.run(task), name=f"pipeline_{call_id}")
    await asyncio.sleep(0.1)  # let pipeline initialize

    # --- INSTANT GREETING FIX (DYNAMIC) ---
    
    fallback_greeting = assistant_config.get("firstMessage", "Namaskaar! Main Renault Care se bol rahi hoon. Kaise madad kar sakti hoon?")   ## updated part -- greeting line 
    initial_greeting = extract_initial_greeting(system_prompt, fallback_greeting)
    
    logger.info(f"[Bot] Queuing INSTANT TTS greeting for {call_id}: '{initial_greeting}'")
    
    await task.queue_frames([
        # 1. Quietly append the greeting to the LLM's history so it knows what the user is replying to
        LLMMessagesAppendFrame(
            messages=[{"role": "assistant", "content": initial_greeting}]
        ),
        # 2. Fire the audio instantly via TTS
        TTSSpeakFrame(initial_greeting)
    ])

    async def watch_disconnect():
        await disconnect_event.wait()
        logger.info("[Watcher] Asterisk disconnect detected. Initiating shutdown...")
        try:
            await task.queue_frames([CancelFrame(), EndFrame()])
        except Exception as e:
            logger.debug(f"[Watcher] Could not queue frames: {e}")
        await asyncio.sleep(1.5)
        if not run_task.done():
            logger.warning("[Watcher] Pipeline hanging on network I/O. Force canceling task!")
            run_task.cancel()

    watcher_task = asyncio.create_task(watch_disconnect(), name=f"watcher_{call_id}")

    try:
        await run_task
    except asyncio.CancelledError:
        logger.info(f"[Bot] Pipeline forcefully terminated for {call_id}.")
    finally:
        watcher_task.cancel()
        # Mark the pipeline dead immediately so SmartUserIdleProcessor's kill
        # switch (_is_alive) stops any in-flight or pending timer tasks,
        # regardless of which shutdown path was taken (EndFrame, CancelFrame,
        # or Asterisk disconnect).  _trigger_shutdown() already does this for
        # the disconnect path; this covers the EndFrame / graceful path.
        shared_state["is_active"] = False
        try:
            await user_logger.stop()
            await bot_logger.stop()
            logger.info("[Cleanup] Explicitly stopped user and bot transcript logging workers.")
        except Exception as log_err:
            logger.error(f"[Cleanup] Failed to stop transcript loggers gracefully: {log_err}")
        
        SESSION_PIPELINES.pop(call_id, None)

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------
    logger.info("Pipeline finished! Handling Asterisk teardown...")
    try:
        if is_transferring:
            # The transfer was already executed via the tool callback.
            # Asterisk yanked the channel cleanly.
            logger.info("Transfer flag detected. Skipping normal AMI Hangup...")
        else:
            logger.info("Normal disconnect. Firing AMI Hangup...")
            await force_ami_hangup(call_id)

        if not writer.is_closing():
            logger.info("AudioSocket still open. Sending 0x00 hangup and aborting...")
            hangup_signal = struct.pack('>BH', 0x00, 0)
            writer.write(hangup_signal)
            try:
                await asyncio.wait_for(writer.drain(), timeout=1.0)
            except asyncio.TimeoutError:
                logger.warning("Socket congested. Skipping drain.")
            writer.transport.abort()
    except Exception as e:
        logger.debug(f"Error during Asterisk teardown: {e}")

    # ------------------------------------------------------------------
    # Upload recording
    # ------------------------------------------------------------------
    logger.info(f"Pipeline ended for {call_id}. Processing audio...")
    st = recordings.get(call_id)
    try:
        if st and st.filename:
            await reconstruct_stereo_from_continuous(st)
            logger.info(f"Temporal stereo recording reconstructed: {st.filename}")
            await upload_to_s3_and_notify(session_id, st.filename, organisation_id, dispatcher)
    finally:
        recordings.pop(call_id, None)
