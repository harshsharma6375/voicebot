
import argparse
import asyncio
from http.client import HTTPException
import io
import json
import logging
import uuid
from fastapi import BackgroundTasks, Request, Response
import os
from datetime import datetime, timedelta, timezone
import threading
import time
from typing import Optional
import wave

import aiofiles
import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from twilio.rest import Client

from emversityBot import run_emversity_bot
# from neetPrepBot import run_neet_prep_bot
from duluxBot import run_twillio_bot

from call_end_reasons import CallEndReason

from events.events import EventDispatcher
from utils.metautils import make_meta
from fastapi import Form
import stomp
from redis.asyncio import Redis

from pydantic import BaseModel

# from daily_sip import create_daily_room, create_bot_token, stop_sip_dialout
# from bot_with_daily import bot_with_daily
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from asterisk_bot import handle_asterisk_stream, prewarm_pipeline, cleanup_if_not_connected, destroy_pipeline
from asterisk.ami import AMIClient, SimpleAction
from typing import Optional 
from mongo import _connect as connect_mongo, _close as close_mongo, get_config_by_assistant_id, get_telephony_provider
import os
from mariadb import get_workflow_node_by_id, _connect, _close



load_dotenv(override=True)


AMI_HOST = os.getenv("AMI_HOST", "127.0.0.1")
AMI_PORT = int(os.getenv("AMI_PORT", "5038"))
AMI_USER = os.getenv("AMI_USER", "apiuser")
AMI_PASS = os.getenv("AMI_PASS", "your_secure_password")
AUDIOSOCKET_PORT = int(os.getenv("AUDIOSOCKET_PORT", "9092"))

class DialRequest(BaseModel):
    phone_number: str
from workflow_processor import WorkflowProcessor

# configure logging once at module import time
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

RETRYABLE_STATUSES = {"FAILED", "NO_ANSWER", "USER_BUSY"}
MAX_RETRIES = int(os.getenv("MAX_RETRIES")) # max retries for a call starts from 0
RETRY_DELAY_MS = int(os.getenv("RETRY_DELAY_MS"))
PERMIT_HIT_RETRY_MS = int(os.getenv("PERMIT_HIT_RETRY_MS"))

logger = logging.getLogger(__name__)

MAIN_LOOP = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ==========================================
    # 1. STARTUP PHASE (Before Yield)
    # ==========================================
    
    # Start the TCP AudioSocket server
    server = await asyncio.start_server(spawn_asterisk_bot, '0.0.0.0', AUDIOSOCKET_PORT)
    logger.info(f"[*] AudioSocket Server listening on Port {AUDIOSOCKET_PORT}")

    global MAIN_LOOP, activemq_client, redis_instance
    MAIN_LOOP = asyncio.get_event_loop()

    # Connect to Redis
    try:
        redis_instance = Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        await redis_instance.ping()
        
        global check_and_incr_all
        global safe_decr_all
        
        check_and_incr_all = redis_instance.register_script(INCR_ALL_IF_BELOW_LIMITS_SCRIPT)
        safe_decr_all = redis_instance.register_script(SAFE_DECR_ALL_SCRIPT)
        
        logger.info("Redis connection established successfully.")
    except Exception as e:
        logger.error(f"Redis unavailable: {e}")
        redis_instance = None

    # Connect to ActiveMQ
    activemq_client = ActiveMQClient(
        ACTIVEMQ_HOST, ACTIVEMQ_PORT, ACTIVEMQ_USERNAME, ACTIVEMQ_PASSWORD, ACTIVEMQ_DESTINATION
    )

    threading.Thread(target=subscribeToQueue, daemon=True).start()
    logger.info("ActiveMQ consumer thread started.")

    # Setup MongoDB (Called from mongo.py)
    try:
        await connect_mongo(app)
        logger.info("MongoDB connected successfully.")
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")

    # --- ADDED: Setup MariaDB ---
    # try:
    #     await _connect(app)
    #     logger.info("MariaDB (workflow_builder) connected successfully.")
    # except Exception as e:
    #     logger.error(f"MariaDB connection failed: {e}")

    # ==========================================
    # 2. RUNNING & SHUTDOWN PHASE (Yield & Finally)
    # ==========================================
    
    
    try:
        # FastAPI pauses here and handles user requests
        yield
        
    except Exception as e:
        # If the FastAPI app experiences a fatal crash, it gets caught here
        logger.error(f"Critical error while the app was running: {e}")
        
    finally:
        # The 'finally' block ALWAYS runs, whether you pressed Ctrl+C or the app crashed
        logger.info("Shutting down services gracefully...")

        # --- ADDED: Close MariaDB Pool ---
        # await _close(app)
        
        await dispatcher.close()
        
        server.close()
        await server.wait_closed()
        
        logger.info("Cleanup complete. Goodbye!")

# ---------------------------
# FastAPI App & Middleware
# ---------------------------
app = FastAPI(lifespan=lifespan)
app.state.testing = False
dispatcher = EventDispatcher()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# In-memory Session Storage
# ---------------------------
SESSION_PROMPTS = {}
SESSION_CLIENT = {}
SESSION_ORG = {}
SESSION_CALLEE_NAME = {}
LANGUAGE_PREFERENCES = {}
CUSTOMER_NUMBERS = {}
SESSION_CAMPAIGN_ID = {}
SESSION_WORKFLOW_VARIABLES = {}
SESSION_WORKFLOW_NODES = {}
PERSONA_PROMPT = "You are a helpful AI assistant."
connected_clients = set()

# ---------------------------
# SAN PBX Config
# ---------------------------
SAN_API_BASE = "https://clouduat28.sansoftwares.com/pbxadmin/sanpbxapi"
ACCESS_TOKEN = "15f5924dc6778b97212085051cc97856"
ACCESS_KEY = "odiqie"
API_TOKEN = None
TOKEN_EXPIRY = None
TOKEN_LOCK = asyncio.Lock()


TWILIO_STATUS_MAP = {
    "initiated":    ("QUEUED", {"message": "Call queued"}),
    "queued":      ("QUEUED", {"message": "Call queued"}),     # or eventType="QUEUED" if you want distinct
    "ringing":     ("RINGING",   {"message": "Call ringing"}),
    "in-progress": ("ANSWERED",  {"message": "Call answered"}),    # Twilio uses in-progress; you want ANSWERED
    "answered":    ("ANSWERED",  {"message": "Call answered"}),
    "busy":        ("USER_BUSY", {"message": "Call busy", "reason": "User busy"}),
    "no-answer":   ("NO_ANSWER", {"message": "No answer", "reason": "User no-answer"}),
    "failed":      ("FAILED",    {"message": "Failed", "reason": "call failed"}),
    "completed":   ("COMPLETED", {"message": "Call completed successfully"}),
}

REQUIRED_KEYS =  ["system_message", "organization_id", "target_phone_number", "call_to", "assistant_id", "session_id", "twilio_phone_number"]
SANS_STATUS_MAP = {
    "initiated":    ("QUEUED", {"message": "Call queued"}),
    "queued":      ("QUEUED", {"message": "Call queued"}),     # or eventType="QUEUED" if you want distinct
    "ringing":     ("RINGING",   {"message": "Call ringing"}),
    "answer": ("COMPLETED",  {"message": "Call completed successfully"}),    # sans uses in-progress; you want ANSWERED
    "answered":    ("ANSWERED",  {"message": "Call answered"}),
    "busy":        ("USER_BUSY", {"message": "Call busy", "reason": "User busy"}),
    "noanswer":   ("NO_ANSWER", {"message": "No answer", "reason": "User no-answer"}),
    "failed":      ("FAILED",    {"message": "Failed", "reason": "call failed"}),
    "completed":   ("COMPLETED", {"message": "Call completed successfully"}),
}

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")


ACTIVEMQ_HOST = os.getenv("ACTIVEMQ_HOST")
ACTIVEMQ_PORT = 61613
ACTIVEMQ_USERNAME = 'admin'
ACTIVEMQ_PASSWORD = 'admin'
ACTIVEMQ_DESTINATION = 'q.call.jobs'
ACTIVEMQ_CALL_COMPLETED = 'q.call.completed'

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = 6379

redis_instance = None
activemq_client = None

CALL_CONTEXT_TTL_BUFFER_SECONDS = 15 * 60

check_and_incr_all = None
safe_decr_all = None 

# @app.on_event("shutdown")
# async def _close_dispatcher():
#     await dispatcher.close()
  

# @app.on_event("startup")
# async def startup_event():
    # global MAIN_LOOP, activemq_client, redis_instance
    # MAIN_LOOP = asyncio.get_event_loop()

    # # Redis
    # try:
    #     redis_instance = Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    #     await redis_instance.ping()
        
    #     global check_and_incr_all
    #     global safe_decr_all
        
    #     check_and_incr_all =  redis_instance.register_script(INCR_ALL_IF_BELOW_LIMITS_SCRIPT)
    #     safe_decr_all =  redis_instance.register_script(SAFE_DECR_ALL_SCRIPT)
        
    #     logger.info("Redis connection established successfully.")
    # except Exception as e:
    #     logger.error(f"Redis unavailable: {e}")
    #     redis_instance = None

    # # ActiveMQ
    # activemq_client = ActiveMQClient(
    #     ACTIVEMQ_HOST, ACTIVEMQ_PORT, ACTIVEMQ_USERNAME, ACTIVEMQ_PASSWORD, ACTIVEMQ_DESTINATION
    # )

    # threading.Thread(target=subscribeToQueue, daemon=True).start()
    # logger.info("ActiveMQ consumer thread started.")




# app = FastAPI(lifespan=lifespan)

INCR_ALL_IF_BELOW_LIMITS_SCRIPT = """
local keys = KEYS
local limits = ARGV
local values = {}

-- Check all keys first
for i = 1, #keys do
  local current = tonumber(redis.call('GET', keys[i]) or '0')
  local limit = tonumber(limits[i])
  if current >= limit then
    return 0  -- One limit exceeded, abort
  end
  values[i] = current
end

-- All passed, increment all
for i = 1, #keys do
  redis.call('INCR', keys[i])
end

return 1
"""

SAFE_DECR_ALL_SCRIPT = """
local keys = KEYS

for i = 1, #keys do
  local current = tonumber(redis.call('GET', keys[i]) or '0')
  if current > 0 then
    redis.call('DECR', keys[i])
  else
    redis.call('SET', keys[i], 0)
  end
end

return 1
"""

    
    
async def add_to_redis(key: str, value, ttl_seconds: Optional[int] = 0):
    """
    Stores a value in Redis and optionally sets a TTL.
    """
    try:
        if redis_instance is None:
            logger.warning("Redis connection not available; skipping cache set.")
            return

        # logger.info(f"Data added for key: {key} (ttl={ttl_seconds} )")

        set_kwargs = {}
        if ttl_seconds and ttl_seconds > 0:
            set_kwargs["ex"] = ttl_seconds
        if value is None:
            serialized_value = ""
        elif isinstance(value, (bytes, bytearray, memoryview)):
            serialized_value = value
        else:
            serialized_value = str(value)

        await redis_instance.set(key, serialized_value, **set_kwargs)
    except Exception as e:
        logger.exception(f"Error adding data to Redis: {e}")


async def get_from_redis(key: str):
    """
    Retrieves and parses JSON data from Redis.
    """
    try:
        value = await redis_instance.get(key)
        if value is None:
            logger.info(f"No data found for key: {key}")
            return None
        return value
    except Exception as e:
        logger.exception(f"Error getting data from Redis: {e}")
        return None
    
async def delete_from_redis(key: str):
    """
    Deletes a key-value pair from Redis.
    """
    try:
        result = await redis_instance.delete(key)
        if result == 1:
            logger.info(f"Key '{key}' deleted successfully.")
        else:
            logger.info(f"Key '{key}' not found.")
    except Exception as e:
        logger.exception(f"Error deleting key from Redis: {e}")

async def check_usage_eligibility(org_id) -> bool:
    """
    Calls the billing API to check if the organization is eligible to make a call.
    Returns True if usageEligibility is true, False otherwise.
    """
    api_url = os.getenv("METRIC_ELIGIBILITY_URL")
    
    headers = {
        'Accept': 'application/json',
        'X-MASTER-TOKEN': 'gCb6ksoGTQX+V0YGcTT6F82dqf0e2c9em+Q4s8inWR/vj7p/ZMSj7QdDJPJSMGBJ',
    }

    payload = {
        "clientBotExternalId": org_id,
        "productCode": "ODIO_BOT_VOICE"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.request(
                "GET",
                api_url,
                headers=headers,
                json=payload,
                timeout=10.0
            )
            response.raise_for_status()
            api_response = response.json()
            
            if api_response.get("status") != 200:
                logger.error(f"Eligibility API returned non-200 status: {api_response.get('message')}")
                return False
                
            config_data = api_response.get("data", {})
            is_eligible = config_data.get("usageEligibility", False)
            
            print(f"Eligibility for org {org_id}: {is_eligible}")
            return is_eligible
            
    except Exception as e:
        print(f"Failed to fetch eligibility config from API: {e}")
        # Defaulting to False (deny access) if the API fails, to prevent unbilled usage
        return False

class MyQueueListener(stomp.ConnectionListener):

    def on_error(self, frame):
        logger.info(f'Received an error: "{frame.body}"')

    def on_message(self, frame):
        try:
            data = json.loads(frame.body)
            # logger.info(f"Data from queue: {data}")

            asyncio.run_coroutine_threadsafe(
                self.handle_message_async(data),
                MAIN_LOOP
            )

        except Exception as e:
            logger.exception(f"Error in consume queue {e}")

    async def handle_message_async(self, data):
        try:
            org_id = data.get("organization_id")

            # Create a dictionary to easily print the specific call's context

            
            # Note: if check_usage_eligibility is inside the class, use self.check_usage_eligibility
            # is_eligible = await check_usage_eligibility(org_id)
            is_eligible = True
            
            if not is_eligible:
                print(f"Call blocked: Organization {org_id} is not eligible for usage.=================================")

                meta = make_meta(
                    call_id=data.get("session_id"),
                    from_number=data.get("twilio_phone_number"),
                    to_number=data.get("target_phone_number"),
                    status='FAILED',
                    reason='CALL LIMIT',
                    recording_url=None,
                )

                await dispatcher.send(
                    meta=meta,
                    event_type="FAILED",
                    event_data={"message": "FAILED"},
                    leg=None
                )

                return
            print(f"we are eligible to make the calls -------------------------------------=========================")

            # Use SESSION_PROMPTS as the base to find all active call IDs
            active_call_sids = list(SESSION_PROMPTS.keys())

            print(f"==================================================")
            print(f"=== TOTAL ACTIVE CALLS IN MEMORY: {len(active_call_sids)} ===")
            print(f"==================================================")

            # 1. Quick Leak Check: Print the size of every dictionary
            # If these numbers don't match, you have a memory leak!
            print(f"Dictionary Sizes -> PROMPTS: {len(SESSION_PROMPTS)}, CLIENT: {len(SESSION_CLIENT)}, "
                        f"ORG: {len(SESSION_ORG)}, CALLEE: {len(SESSION_CALLEE_NAME)}, "
                        f"NUMBERS: {len(CUSTOMER_NUMBERS)}, CAMPAIGN: {len(SESSION_CAMPAIGN_ID)}, "
                        f"WORKFLOW_VARS: {len(SESSION_WORKFLOW_VARIABLES)}, NODES: {len(SESSION_WORKFLOW_NODES)}")

            # 2. Detailed dump for each specific call
            for call_sid in active_call_sids:
                call_state = {
                    "prompt": SESSION_PROMPTS.get(call_sid),
                    "client": SESSION_CLIENT.get(call_sid),
                    "org": SESSION_ORG.get(call_sid),
                    "callee_name": SESSION_CALLEE_NAME.get(call_sid),
                    "customer_number": CUSTOMER_NUMBERS.get(call_sid),
                    "campaign_id": SESSION_CAMPAIGN_ID.get(call_sid),
                    "workflow_variables": SESSION_WORKFLOW_VARIABLES.get(call_sid),
                    "workflow_nodes": SESSION_WORKFLOW_NODES.get(call_sid),
                }

                print(f"--- STATE FOR CALL ID: {call_sid} ---")
                print(call_state)

            global check_and_incr_all

            async def try_increment_multiple(keys: list[str], limits: list[int]) -> bool:
                result = await check_and_incr_all(keys=keys, args=limits)
                return result == 1
            
            #global level
            global_count_env = int(os.getenv("GLOBAL_CALLS_COUNT", 3))
            
            #provider level
            provider_config = await get_telephony_provider(app, data.get("organization_id"))
            provider = provider_config["phoneAi"]["telephony"]["provider"]

            # logger.info(f"the provider is {provider}")

            stt_config = provider_config["phoneAi"]["stt"]
            tts_config = provider_config['phoneAi']['tts']

            callType = CALL_PROVIDER_MAP.get(provider, "")
            logger.info(f"the call type is {callType}")
            provider_calls_key =  f"{callType}_CALLS_COUNT"
            provider_count_env = int(os.getenv(provider_calls_key, 3))
           
            #campaign level
            campaign_key = None
            if(data.get("campaign_id")):
                campaign_key = f"CAMPAIGN_{data['campaign_id']}_CALLS_COUNT"
            else:
                campaign_key = f"CAMPAIGN_DEFAULT_CALLS_COUNT"
            # logger.info(f"keys provider {provider_calls_key} campaign {campaign_key}")
            campaign_count_env = int(os.getenv(campaign_key, 3))
            
            #organisation level
            org_key = f"org_"+ str(data['organization_id']) + "_CALLS_COUNT"
            org_count_env = int(os.getenv(org_key, 3))
            
            logger.info(f"trying to increment the multiple")
            success = await try_increment_multiple(
                ["GLOBAL_CALLS_COUNT",org_key,provider_calls_key, campaign_key],
                [global_count_env, org_count_env , provider_count_env, campaign_count_env]
            )

            if success:
                logger.info("All increments done")
                msg_key = f"prompt_{data.get('session_id')}"
                data["system_message"] = await get_from_redis(msg_key)
                
                counter_keys = ["GLOBAL_CALLS_COUNT", provider_calls_key, campaign_key, org_key]
                try:
                    data["counter_keys"] = counter_keys
                    response = await make_call(data, provider, counter_keys)
                    raw_body = response.body.decode()
                    
                    try:
                        decoded = json.loads(raw_body)
                    except json.JSONDecodeError:
                        decoded = {"status": "error", "detail": raw_body}

                    if decoded.get("status") != "success":
                        logger.error(f"Call init failed with payload: {decoded}")
                        return

                    if decoded.get("workflow_variables"):
                        print(f"check -> workflow_variables before save {decoded.get('workflow_variables')}")
                        data["workflow_variables"] = decoded["workflow_variables"]

                    await add_to_redis(str("org_id_stt_config_"+ str(data.get("organization_id"))), json.dumps(stt_config, default=str))
                    await add_to_redis(str("org_id_tts_config_"+ str(data.get("organization_id"))), json.dumps(tts_config, default=str))

                    # 2. ONLY SAVE HERE IF IT IS NOT ASTERISK, because pre-warmup we saved these configs initially only
                    if callType != "ASTERISK":
                        await add_to_redis(decoded["call_sid"]+"_extra_keys", json.dumps(counter_keys, default=str))
                        if 'system_message' in data:
                            del data['system_message']
                        await add_to_redis(decoded["call_sid"]+"_extra_data", json.dumps(data, default=str))

                except Exception as call_error:
                    logger.exception(f"Error in make_call or response processing: {call_error}")
                    await decrease_redis_permits(counter_keys)
                    return
            else:
                await addToQueue(data, PERMIT_HIT_RETRY_MS)
                logger.info("Concurrent calls redis permit hit")

        except Exception as e:
            logger.exception(f"Error in handle_message_async: {e}")

# ---------------------------
# Call Handlers
# ---------------------------
async def handle_twilio(data, system_message,workflow_variables=None):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    call = client.calls.create(
        to=data["target_phone_number"],
        from_=data["twilio_phone_number"],
        url="https://vbot.odioiq.com/",
        record=False,
        status_callback="https://vbot.odioiq.com/twilio/status",
        status_callback_event=["initiated", "queued", "ringing", "answered", "completed"]
    )
    assistant_config = data.get("assistant_config")
    if assistant_config:
        await add_to_redis(f"callctx:{call.sid}", json.dumps(assistant_config, default=str), CALL_CONTEXT_TTL_BUFFER_SECONDS)
    else:
        await add_to_redis(f"callctx:{call.sid}", None, CALL_CONTEXT_TTL_BUFFER_SECONDS)
    SESSION_PROMPTS[call.sid] = system_message
    SESSION_CLIENT[call.sid] = data["client_name"]
    SESSION_ORG[call.sid] = data["organization_id"]
    CUSTOMER_NUMBERS[call.sid] = data["target_phone_number"]
    SESSION_CAMPAIGN_ID[call.sid] = data["campaign_id"]
    if workflow_variables is not None:
        SESSION_WORKFLOW_VARIABLES[call.sid] = workflow_variables
    if data and data.get("nodes"):
        SESSION_WORKFLOW_NODES[call.sid] = data["nodes"]
    return {"status": "success", "call_sid": call.sid, "workflow_variables": workflow_variables}


async def handle_twilio_sip(data, system_message,workflow_variables=None):
    """
    Place a SIP call via Twilio and connect to the Dulux Bot WebSocket.
    """
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    from_number = data["twilio_phone_number"]

    # Build SIP URI from phone number and gateway
    phone_number = data["target_phone_number"]   # e.g. +919354218785
    sip_gateway_host = os.getenv("SIP_GATEWAY_HOST", "114.143.73.93")
    transport = data.get("transport", "udp")

    to_sip_uri = f"sip:{phone_number}@{sip_gateway_host};transport={transport}"

    call = client.calls.create(
        to=to_sip_uri,
        from_=from_number,
        url="https://vbot.odioiq.com/",
        record=False,
        status_callback="https://vbot.odioiq.com/twilio/status",
        status_callback_event=["initiated", "queued", "ringing", "answered", "completed"]
    )

    assistant_config = data.get("assistant_config")
    if assistant_config:
        await add_to_redis(f"callctx:{call.sid}", json.dumps(assistant_config, default=str),CALL_CONTEXT_TTL_BUFFER_SECONDS)
    else:
        await add_to_redis(f"callctx:{call.sid}", None, CALL_CONTEXT_TTL_BUFFER_SECONDS)

    SESSION_PROMPTS[call.sid] = system_message
    SESSION_CLIENT[call.sid] = data["client_name"]
    SESSION_ORG[call.sid] = data["organization_id"]
    CUSTOMER_NUMBERS[call.sid] = data["target_phone_number"]
    SESSION_CAMPAIGN_ID[call.sid] = data["campaign_id"]
    if workflow_variables is not None:
        SESSION_WORKFLOW_VARIABLES[call.sid] = workflow_variables
    if data and data.get("nodes"):
        SESSION_WORKFLOW_NODES[call.sid] = data["nodes"]
    return {"status": "success", "call_sid": call.sid, "workflow_variables": workflow_variables}

#NOT USED
async def handle_emversity(data, system_message,workflow_variables=None):
    api_token = await get_san_api_token()
    san_url = f"{SAN_API_BASE}/dialcall"
    payload = {
        "appid": 3,
        "call_to": data["target_phone_number"],
        "caller_id": data["caller_id"],
        "status_callback": "https://vbot.odioiq.com/",
        "custom_field": data.get("custom_field", {}),
    }
    headers = {"Apitoken": api_token, "Content-Type": "application/json"}
    async with httpx.AsyncClient() as client:
        resp = await client.post(san_url, json=payload, headers=headers)
        if resp.status_code == 401:
            api_token = await get_san_api_token()
            headers["Apitoken"] = api_token
            resp = await client.post(san_url, json=payload, headers=headers)
        resp.raise_for_status()
        call_id = resp.json()["data"]["msg"]["callid"]
    
    assistant_config = data.get("assistant_config")
    if assistant_config:
        await add_to_redis(f"callctx:{call_id}", json.dumps(assistant_config, default=str),CALL_CONTEXT_TTL_BUFFER_SECONDS)
    else:
        await add_to_redis(f"callctx:{call_id}", None, CALL_CONTEXT_TTL_BUFFER_SECONDS)

    SESSION_PROMPTS[call_id] = system_message
    SESSION_CLIENT[call_id] = data["client_name"]
    SESSION_ORG[call_id] = data["organisation_id"]
    CUSTOMER_NUMBERS[call_id] = data["target_phone_number"]
    SESSION_CAMPAIGN_ID[call_id] = data["campaign_id"]
    if workflow_variables is not None:
        SESSION_WORKFLOW_VARIABLES[call_id] = workflow_variables
    if data and data.get("nodes"):
        SESSION_WORKFLOW_NODES[call_id] = data["nodes"]
    return {"status": "success", "call_sid": call_id,"workflow_variables": workflow_variables}

async def handle_sans(data, system_message,workflow_variables=None):
    api_token = await get_san_api_token()
    san_url = f"{SAN_API_BASE}/dialcall"
    payload = {
        "appid": 3,
        "call_to": data["call_to"],
       # "caller_id": data["caller_id"],
       "caller_id": "8062810016",
        "status_callback": "https://vbot.odioiq.com/sans/ws",
        "custom_field": data.get("custom_field", {}),
    }
    headers = {"Apitoken": api_token, "Content-Type": "application/json"}
    async with httpx.AsyncClient() as client:
        resp = await client.post(san_url, json=payload, headers=headers)
        if resp.status_code == 401:
            api_token = await get_san_api_token()
            headers["Apitoken"] = api_token
            resp = await client.post(san_url, json=payload, headers=headers)
        resp.raise_for_status()
        call_id = resp.json()["data"]["msg"]["callid"]
    
    assistant_config = data.get("assistant_config")
    if assistant_config:
        await add_to_redis(f"callctx:{call_id}", json.dumps(assistant_config, default=str),CALL_CONTEXT_TTL_BUFFER_SECONDS)
    else:
        await add_to_redis(f"callctx:{call_id}", None, CALL_CONTEXT_TTL_BUFFER_SECONDS)

    SESSION_PROMPTS[call_id] = system_message
    SESSION_CLIENT[call_id] = data["client_name"]
    SESSION_ORG[call_id] = data["organization_id"]
    CUSTOMER_NUMBERS[call_id] = data["target_phone_number"]
    SESSION_CAMPAIGN_ID[call_id] = data["campaign_id"]
    if workflow_variables is not None:
        SESSION_WORKFLOW_VARIABLES[call_id] = workflow_variables
    if data and data.get("nodes"):
        SESSION_WORKFLOW_NODES[call_id] = data["nodes"]
    return {"status": "success", "call_sid": call_id,"workflow_variables": workflow_variables}

# async def handle_greeter(data, system_message,workflow_variables=None):

#         room_url, room_name = await create_daily_room()
#         token = await create_bot_token(room_name)

#         logger.info(f"room_url is {room_url}")

#         assistant_config = data.get("assistant_config")
#         if assistant_config:
#             await add_to_redis(f"callctx:{room_name}", json.dumps(assistant_config, default=str), CALL_CONTEXT_TTL_BUFFER_SECONDS)
#         else:
#             await add_to_redis(f"callctx:{room_name}", None, CALL_CONTEXT_TTL_BUFFER_SECONDS)


#         SESSION_PROMPTS[room_name] = system_message
#         SESSION_CLIENT[room_name] = data["client_name"]
#         SESSION_ORG[room_name] = data["organization_id"]
#         CUSTOMER_NUMBERS[room_name] = data["target_phone_number"]
#         SESSION_CAMPAIGN_ID[room_name] = data["campaign_id"]
#         if workflow_variables is not None:
#             SESSION_WORKFLOW_VARIABLES[room_name] = workflow_variables
#         if data and data.get("nodes"):
#             SESSION_WORKFLOW_NODES[room_name] = data["nodes"]
#         return {"status": "success", "call_sid": room_name, "token": token, "room_name": room_name, "workflow_variables": workflow_variables}

# async def handle_bot_with_daily(room_url, token, room_name, phone_number, counter_keys):

#     prompt = SESSION_PROMPTS.pop(room_url, None) or PERSONA_PROMPT
#     organisation_id = SESSION_ORG.pop(room_url, None)
#     campaign_id = SESSION_CAMPAIGN_ID.pop(room_url, None)

#     assistant_config = await get_from_redis(f"callctx:{room_url}")

#     # logger.info(f"the assistant config from redis is {assistant_config}")
#     if assistant_config:
#         assistant_config = json.loads(assistant_config)
#     else:
#         assistant_config = None
    
#     stt_config = await get_from_redis(str("org_id_stt_config_"+ str(organisation_id)))
#     tts_config = await get_from_redis(str("org_id_tts_config_"+ str(organisation_id)))

#     raw = await get_from_redis(room_url+"_extra_data")
#     extra_data = json.loads(raw.decode())

#     await bot_with_daily(room_url, token, room_name,
#                           organisation_id, system_prompt=prompt, dispatcher=dispatcher,
#                           campaign_id = campaign_id,
#                           redis_conn=redis_instance, activemq_conn=activemq_client,
#                           assistant_config=assistant_config, stt_config=json.loads(stt_config), tts_config=json.loads(tts_config),
#                           call_id=extra_data.get("session_id"),agent_details=extra_data.get("agent_details",{}),phone_number=phone_number,counter_keys=counter_keys)

class CallRequest(BaseModel):
    phone_number: str

async def handle_asterisk(data, system_message, workflow_variables=None):

    client = AMIClient(address=AMI_HOST, port=AMI_PORT)

    my_call_id = str(uuid.uuid4())

    provider_config = await get_telephony_provider(
        app,
        data.get("organization_id")
    )

    stt_config = provider_config["phoneAi"]["stt"]
    tts_config = provider_config["phoneAi"]["tts"]

    assistant_config = data.get("assistant_config", {})

    await prewarm_pipeline(
        my_call_id,
        stt_config,
        tts_config,
        assistant_config
    )

    print(f"[PREWARM DONE] {my_call_id}")
    print(f"Dialing via Local channel to ensure hangup handler attachment. ID: {my_call_id}")

    counter_keys = data.get("counter_keys", [])
    
    # Clean up the payload to save space in Redis
    if "system_message" in data:
        del data["system_message"]
        
    # Pre-warm Redis before Asterisk even attempts the call
    await add_to_redis(f"{my_call_id}_extra_keys", json.dumps(counter_keys, default=str))

    data["workflow_variables"] = workflow_variables

    await add_to_redis(f"{my_call_id}_extra_data", json.dumps(data, default=str))
    
    try:
        future = client.login(username=AMI_USER, secret=AMI_PASS)
        if future.response.is_error():
            raise HTTPException(status_code=500, detail="AMI Login Failed")

        # --- THE FIX: Route through the Local channel dialplan ---
        # Format: Local/PhoneNumber*UUID@outbound-dialer
        target_channel = f"Local/{data['target_phone_number']}*{my_call_id}@outbound-dialer"

        action = SimpleAction(
            'Originate',
            Channel=target_channel, 
            Context='from-internal',
            Exten=f"100*{my_call_id}",
            Priority=1,
            CallerID=data['twilio_phone_number'],
            Async='true'
        )

        # --- INSTANT DIALING WEBHOOK ---
        try:
            # 1. Construct the exact Pydantic payload your webhook expects
            dialing_payload = WebhookPayload(
                call_id=my_call_id, 
                status="DIALING",
                dial_status=None,
                cause=None
            )
            
            # 2. Fire it off in the background so it doesn't block the dialer!
            asyncio.create_task(call_status_webhook(dialing_payload))
            
        except Exception as fn_err:
            print(f"Warning: Failed to execute internal DIALING function: {fn_err}")
        # -----------------------------------------------

        future_action = client.send_action(action)
        response = future_action.response
        client.logoff()
        
        if response.is_error():
            raise HTTPException(status_code=500, detail="Asterisk Rejected Call")
        
        assistant_config = data.get("assistant_config")
        if assistant_config:
            await add_to_redis(f"callctx:{my_call_id}", json.dumps(assistant_config, default=str), CALL_CONTEXT_TTL_BUFFER_SECONDS)
        else:
            await add_to_redis(f"callctx:{my_call_id}", None, CALL_CONTEXT_TTL_BUFFER_SECONDS)

        SESSION_PROMPTS[my_call_id] = system_message
        SESSION_CLIENT[my_call_id] = data["client_name"]
        SESSION_ORG[my_call_id] = data["organization_id"]
        CUSTOMER_NUMBERS[my_call_id] = data["target_phone_number"]
        SESSION_CAMPAIGN_ID[my_call_id] = data["campaign_id"]


        
        if workflow_variables is not None:
            SESSION_WORKFLOW_VARIABLES[my_call_id] = workflow_variables
        if data and data.get("nodes"):
            SESSION_WORKFLOW_NODES[my_call_id] = data["nodes"]
            
        return {"status": "success", "call_sid": my_call_id, "workflow_variables": workflow_variables}
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

CALL_HANDLER_MAP = {
    "TWILIO": handle_twilio,
    "TWILIO_SIP": handle_twilio_sip,
    "SAN_SOFTWARE": handle_sans,
    # "GREETER":handle_greeter,
    "ASTERISK":handle_asterisk
}

CALL_PROVIDER_MAP = {
    "TWILIO": "TWILIO",
    "TWILIO_SIP": "TWILIO",
    "SAN_SOFTWARE": "SANS",
    "GREETER": "GREETER",
    "ASTERISK":"ASTERISK"
}

def get_call_handler(telephoney: int):
    if telephoney:
        return CALL_HANDLER_MAP[telephoney]

def call_id_to_mobile_number(call_id: str) -> str:
    return CUSTOMER_NUMBERS.get(call_id)


# ---------------------------
# SAN API Token Management
# ---------------------------
async def get_san_api_token() -> str:
    global API_TOKEN, TOKEN_EXPIRY
    async with TOKEN_LOCK:
        if API_TOKEN and TOKEN_EXPIRY and datetime.utcnow() < TOKEN_EXPIRY:
            return API_TOKEN

        url = f"{SAN_API_BASE}/gentoken"
        headers = {"Content-Type": "application/json", "Accesstoken": os.getenv("SAN_ACCESS_TOKEN", ACCESS_TOKEN)}
        payload = {"access_key": os.getenv("SAN_ACCESS_KEY", ACCESS_KEY)}
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        if data.get("status") != "success":
            raise RuntimeError(f"Failed to get SAN API token: {data}")

        API_TOKEN = data["Apitoken"]
        expiry_str = data.get("expiry_time")
        TOKEN_EXPIRY = datetime.strptime(expiry_str, "%Y-%m-%d %H:%M:%S") if expiry_str else datetime.utcnow() + timedelta(minutes=20)
        return API_TOKEN

# def subscribeToQueue():
#     try:
#         conn = stomp.Connection([(ACTIVEMQ_HOST, ACTIVEMQ_PORT)])
#         conn.set_listener('', MyQueueListener())
#         conn.connect(ACTIVEMQ_USERNAME, ACTIVEMQ_PASSWORD, wait=True)
#         conn.subscribe(destination=ACTIVEMQ_DESTINATION, id=1, ack='auto')
#         logger.info(f"Subscribed to {ACTIVEMQ_DESTINATION}")
#     except Exception as e:
#         logger.exception(f"Not able to establish connection with ActiveMq - phone calls ",str(e))
#         return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)

def subscribeToQueue():
    # 1. Loop forever so the thread never dies
    while True:
        conn = None
        try:
            logger.info(f"Attempting to connect to ActiveMQ at {ACTIVEMQ_HOST}:{ACTIVEMQ_PORT}...")
            
            # Initialize Connection
            conn = stomp.Connection([(ACTIVEMQ_HOST, ACTIVEMQ_PORT)])
            conn.set_listener('', MyQueueListener())
            
            # Connect and Subscribe
            conn.connect(ACTIVEMQ_USERNAME, ACTIVEMQ_PASSWORD, wait=True)
            conn.subscribe(destination=ACTIVEMQ_DESTINATION, id=1, ack='auto')
            logger.info(f"Successfully subscribed to {ACTIVEMQ_DESTINATION}")

            # 2. Monitor the connection
            # We stay in this inner loop as long as the connection is alive
            while conn.is_connected():
                time.sleep(2)  # Check status every 2 seconds

            logger.warning("ActiveMQ connection lost. Reconnecting...")

        except Exception as e:
            logger.error(f"ActiveMQ Listener Error: {e}")
            # If connection failed (e.g., Broker down), wait before retrying
            time.sleep(5) 
        
        finally:
            # 3. Cleanup before next loop iteration
            try:
                if conn and conn.is_connected():
                    conn.disconnect()
            except:
                pass
    
class ActiveMQClient:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, host, port, username, password, destination):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.host = host
                    cls._instance.port = port
                    cls._instance.username = username
                    cls._instance.password = password
                    cls._instance.destination = destination
                    cls._instance.conn = None
                    cls._instance.connect()
        return cls._instance

    def connect(self):
        try:
            self.conn = stomp.Connection([(self.host, self.port)])
            self.conn.connect(self.username, self.password, wait=True)
            logger.info("Connected to ActiveMQ successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to ActiveMQ: {e}")
            self.conn = None

    def ensure_connected(self):
        if not self.conn or not self.conn.is_connected():
            logger.warning("Reconnecting to ActiveMQ...")
            self.connect()

    # def send_message(self, body, destination ,delay=0):
    #     try:
    #         self.ensure_connected()
    #         headers = {'AMQ_SCHEDULED_DELAY': str(delay)}
    #         self.conn.send(destination=destination, body=json.dumps(body, default=str), headers=headers)
    #         logger.info("Message sent to ActiveMQ.")
    #     except Exception as e:
    #         logger.error(f"Error sending message: {e}")
    #         self.connect()  # try reconnecting next time

    # def send_message(self, body, destination, delay=0, attempts=3):
    #     for i in range(attempts):
    #         try:
    #             self.ensure_connected()
    #             headers = {'AMQ_SCHEDULED_DELAY': str(delay)}
    #             self.conn.send(destination=destination, body=json.dumps(body, default=str), headers=headers)
    #             logger.info("Message sent to ActiveMQ.")
    #             return # Success! Exit the function
            
    #         except Exception as e:
    #             logger.error(f"Attempt {i+1} failed: {e}")
    #             self.disconnect() # Good practice to clean up old socket
    #             self.connect() 
    #             if i == attempts - 1:
    #                 logger.error("All retry attempts failed. Message lost.")
    #                 raise e

    def send_message(self, body, destination, delay_ms=0, attempts=3, persistent=True):
        """
        delay_ms: delay in milliseconds (ActiveMQ scheduler expects ms)
        destination examples:
        - "/queue/my.queue"
        - "/topic/my.topic"
        """
        delay_ms = int(delay_ms or 0)
        if delay_ms < 0:
            delay_ms = 0

        headers = {
            "AMQ_SCHEDULED_DELAY": str(delay_ms),
            "content-type": "application/json",
        }
        if persistent:
            headers["persistent"] = "true"

        payload = json.dumps(body, default=str)

        last_err = None
        for i in range(1, attempts + 1):
            try:
                self.ensure_connected()
                self.conn.send(destination=destination, body=payload, headers=headers)
                logger.info(f"Message sent to ActiveMQ (delay_ms={delay_ms}) to {destination}")
                return
            except Exception as e:
                last_err = e
                logger.error(f"Attempt {i} failed: {e}")
                try:
                    self.disconnect()
                except Exception:
                    pass
                try:
                    self.connect()
                except Exception as e2:
                    logger.error(f"Reconnect failed: {e2}")

        logger.error("All retry attempts failed. Message lost.")
        raise last_err

    def disconnect(self):
        try:
            if self.conn and self.conn.is_connected():
                self.conn.disconnect()
                logger.info("Disconnected from ActiveMQ.")
        except Exception as e:
            logger.error(f"Error disconnecting from ActiveMQ: {e}")
            
            

async def addToQueue(params, delay=0):
    logger.info(f"Pushing data to queue")
    try:
        global ACTIVEMQ_DESTINATION
        logger.info("Pushing data to ActiveMQ queue")
        activemq_client.send_message(params, ACTIVEMQ_DESTINATION ,delay)
    except Exception as e:
        logger.exception(f"Error in push data to queue: {e}")
        logger.exception("Error in push data to queue")
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500) 

# ---------------------------
# Unified Call Endpoint
# ---------------------------
# @app.post("/make-call")        
@app.post("/make-call")
async def add_to_queue(request: Request):
    
    try:
        data = await request.json()
        logger.info(f"request for placing call {data}")
        required_keys = REQUIRED_KEYS
        missing = [key for key in required_keys if key not in data or not data[key]]
        if missing:
            logger.info(f"missing keys for making api call, fail at first check")
            return JSONResponse(
                {"status": "error", "detail": f"Missing or empty keys: {', '.join(missing)}"},
                status_code=400
            )
        
        prompt_key= f"prompt_{data.get('session_id')}"
        prompt_value = data.get("system_message", "") 
        del data['system_message']
        await add_to_redis(prompt_key, prompt_value)
        
        await addToQueue(data)
        return JSONResponse({"status": "success", "call_sid": data.get("session_id")})

    except Exception as e:
        logger.exception("Error in add to queue")
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)

def replace_custom_variables(msg, custom_variables):
    # Convert bytes → str first
    if isinstance(msg, bytes):
        msg = msg.decode("utf-8", errors="ignore")

    for key, value in custom_variables.items():
        msg = msg.replace(key, str(value))

    return msg


async def make_call(data, provider, counter_list):
    client_name = data.get("client_name")
    system_message = data["system_message"]
    nodes = data.get("nodes", [])

    # Process workflow if nodes are present
    workflow_variables = None
    updated_prompt = system_message
    # logger.info(f"call make_call ->  {data}")
    if nodes and len(nodes) > 0:
        try:
            # logger.info("Processing workflow nodes...")
            processor = WorkflowProcessor(data)

            # Execute workflow and get both variables and processed prompt
            workflow_variables, processed_prompt = await processor.execute_workflow()

            # logger.info(f"Workflow processed successfully")
            # logger.info(f"Variables extracted: {workflow_variables}")
            logger.info(f"Processed prompt: {processed_prompt}")
            logger.info(f"workflow_variables : {workflow_variables}")

            # Use the processed prompt if available, otherwise keep original
            if processed_prompt:
                updated_prompt = processed_prompt
                system_message = updated_prompt
            else:
                logger.warning("No prompt returned from workflow, using original system_message")

        except Exception as e:
            logger.exception(f"Error processing workflow: {e}")
            logger.warning("Continuing with original system message due to workflow error")
            # Set workflow_variables to empty dict on error
            workflow_variables = {}

    # Get assistant config and apply custom variables
    assistant_config = await get_config_by_assistant_id(app, data.get("assistant_id"))
    if assistant_config:
        data["assistant_config"] = assistant_config
        custom_variables = data.get("custom_variable", {})

        print(f"the custom variable replacing {custom_variables}")
        system_message = replace_custom_variables(system_message, custom_variables)
    else:
        data["assistant_config"] = None
    
    # logger.info(f"Assistant config from mongo: {assistant_config}")
    # logger.info(f"Final system message: {system_message}")

    # Get the appropriate handler
    handler = get_call_handler(provider)
    if not handler:
        return JSONResponse(
            {"status": "error", "detail": f"Unknown provider: {provider}"},
            status_code=400
        )

    try:
        # Call the handler with workflow variables
        result = await handler(data, system_message, workflow_variables)
        return JSONResponse(result)

    except Exception as e:
        logger.exception(f"Error in make_call: {e}")

        # Send failure event
        event_type, event_data = SANS_STATUS_MAP.get("failed", ("FAILED", {"message": "failed"}))
    
        meta = make_meta(
            call_id=data.get("session_id"),
            from_number=data.get("twilio_phone_number"),
            to_number=data.get("target_phone_number"),
            status=event_type,
            reason="",
            recording_url=None,
        )

        await dispatcher.send(
                meta=meta,
                event_type=event_type,
                event_data=event_data,
                leg=None
            )
        await decrease_redis_permits(counter_list)
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)

# ---------------------------
# WebSocket Handlers (1:1 per client)
@app.websocket("/sans/ws")
async def sans_ws(websocket: WebSocket):
    """
    SAN PBX WebSocket for streaming audio and call events.
    Mirrors Twilio run_bot logic.
    """
    await websocket.accept()
    connected_clients.add(websocket)
    # logger.info("SAN PBX WebSocket client connected")

    try:
        # Receive the first message from SAN PBX
        start_data = await websocket.receive_text()
        call_data = json.loads(start_data)

        # Extract call and stream IDs from SAN PBX payload
        call_id = call_data.get("callId")
        stream_sid = call_data.get("streamId") or call_id

        # logger.info(f"Starting call pipeline for CallID: {call_id}, StreamID: {stream_sid}")

        # Retrieve session info
        prompt = SESSION_PROMPTS.pop(call_id, "Hello from bot!")
        api_token = await get_san_api_token()
        client_name = SESSION_CLIENT.pop(call_id, None)
        organisation_id = SESSION_ORG.pop(call_id, None)
        callee_name = SESSION_CALLEE_NAME.pop(call_id, None)
        campaign_id = SESSION_CAMPAIGN_ID.pop(call_id, None)

        assistant_config = await get_from_redis(f"callctx:{call_id}")

        # logger.info(f"the assistant config from redis is {assistant_config}")
        if assistant_config:
            assistant_config = json.loads(assistant_config)
        else:
            assistant_config = None
        
        stt_config = await get_from_redis(str("org_id_stt_config_"+ str(organisation_id)))
        tts_config = await get_from_redis(str("org_id_tts_config_"+ str(organisation_id)))

        raw = await get_from_redis(call_id+"_extra_data")
        extra_data = json.loads(raw.decode())

       # print(f"Using handler {handler.__name__} for client {client_name}")
        await run_emversity_bot(websocket, stream_sid, call_id,
                                app.state.testing, api_token, organisation_id, callee_name, system_prompt=prompt, 
                                dispatcher=dispatcher, campaign_id=campaign_id
                                ,redis_conn=redis_instance, activemq_conn=activemq_client,
                                 assistant_config=assistant_config, stt_config=json.loads(stt_config) , 
                                 tts_config=json.loads(tts_config) ,call_id=extra_data.get("session_id", None))

    except Exception as e:
        logger.exception("SAN PBX WebSocket disconnected or error:", e)
    finally:
        connected_clients.discard(websocket)
        logger.info("SAN PBX WebSocket client disconnected")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    start_data = websocket.iter_text()
    await start_data.__anext__()
    call_data = json.loads(await start_data.__anext__())
    # logger.info(call_data, flush=True)
    stream_sid = call_data["start"]["streamSid"]
    call_sid = call_data["start"]["callSid"]
    logger.info("WebSocket connection accepted")

    prompt = SESSION_PROMPTS.pop(call_sid, None) or PERSONA_PROMPT
    organisation_id = SESSION_ORG.pop(call_sid, None)
    campaign_id = SESSION_CAMPAIGN_ID.pop(call_sid, None)

    assistant_config = await get_from_redis(f"callctx:{call_sid}")

    logger.info(f"the assistant config from redis is {assistant_config}")
    if assistant_config:
        assistant_config = json.loads(assistant_config)
    else:
        assistant_config = None
    
    stt_config = await get_from_redis(str("org_id_stt_config_"+ str(organisation_id)))
    tts_config = await get_from_redis(str("org_id_tts_config_"+ str(organisation_id)))

    raw = await get_from_redis(call_sid+"_extra_data")
    extra_data = json.loads(raw.decode())

    await run_twillio_bot(websocket, stream_sid, call_sid, app.state.testing, 
                          organisation_id, system_prompt=prompt, dispatcher=dispatcher,
                          campaign_id = campaign_id,
                          redis_conn=redis_instance, activemq_conn=activemq_client,
                          assistant_config=assistant_config, stt_config=json.loads(stt_config), tts_config=json.loads(tts_config),
                          call_id=extra_data.get("session_id"))
    

async def spawn_asterisk_bot(reader, writer):
    call_id = 0
    try:
        header = await reader.readexactly(3)
        payload_type = header[0]
        # Calculate length (AudioSocket uses big-endian for the 16-bit length)
        payload_len = (header[1] << 8) | header[2]
        payload = await reader.readexactly(payload_len)
        
        # 0x01 is the AudioSocket code for "Here is the UUID"
        if payload_type == 0x01:  
            # Convert the 16 raw bytes into a standard UUID string
            call_id = str(uuid.UUID(bytes=payload))
            
    except asyncio.IncompleteReadError:
        logger.error("[Bot] Connection dropped before sending UUID.")
        return

    print(f"the call id is {call_id}")
    logger.info("WebSocket connection accepted")

    prompt = SESSION_PROMPTS.pop(call_id, None) or PERSONA_PROMPT
    organisation_id = SESSION_ORG.pop(call_id, None)
    campaign_id = SESSION_CAMPAIGN_ID.pop(call_id, None)

    assistant_config = await get_from_redis(f"callctx:{call_id}")

    if assistant_config:
        assistant_config = json.loads(assistant_config)
    else:
        assistant_config = None
    
    stt_config = await get_from_redis(str("org_id_stt_config_"+ str(organisation_id)))
    tts_config = await get_from_redis(str("org_id_tts_config_"+ str(organisation_id)))

    raw = await get_from_redis(call_id+"_extra_data")
    extra_data = json.loads(raw.decode())

    session_id = extra_data.get("session_id")

    try:
        await handle_asterisk_stream(call_id, session_id, reader, writer, 
                              organisation_id, system_prompt=prompt, dispatcher=dispatcher,
                              campaign_id = campaign_id,
                              redis_conn=redis_instance, activemq_conn=activemq_client,
                              assistant_config=assistant_config, stt_config=json.loads(stt_config), tts_config=json.loads(tts_config),
                              agent_details=extra_data.get("agent_details",{}))
    finally:
        # Guarantee memory is freed the exact millisecond the stream stops
        cleanup_call_session(call_id)

        for task in asyncio.all_tasks():
            if call_id in task.get_name():
                task.cancel()
   
    
# async def spawn_asterisk_bot(reader, writer):
#     call_id = 0
#     try:
#         header = await reader.readexactly(3)
#         payload_type = header[0]
#         # Calculate length (AudioSocket uses big-endian for the 16-bit length)
#         payload_len = (header[1] << 8) | header[2]
#         payload = await reader.readexactly(payload_len)
        
#         # 0x01 is the AudioSocket code for "Here is the UUID"
#         if payload_type == 0x01:  
#             # Convert the 16 raw bytes into a standard UUID string
#             call_id = str(uuid.UUID(bytes=payload))
            
#     except asyncio.IncompleteReadError:
#         logger.error("[Bot] Connection dropped before sending UUID.")
#         return

#     print(f"the call id is {call_id}")
#     logger.info("WebSocket connection accepted")

#     # --- DEFAULT INBOUND CONFIGURATIONS ---
#     DEFAULT_PROMPT = "you are a agent, talk in english"
    
#     DEFAULT_ASSISTANT_CONFIG = {
#         "_id": "692d8ba0e5a87846a613520d", 
#         "_class": "com.ezeia.ezibot.collections.phoneai.PhoneAiAssistantConfigCollection", 
#         "orgId": 26, 
#         "name": "Dharan Jampani", 
#         "description": "Voce assistant", 
#         "voice": {"provider": "GOOGLE", "language": "EN_IN", "voiceName": "Hindi Calm Man", "voiceId": "en-IN-Chirp3-HD-Kore", "sampleRate": 8000, "encoding": "pcm_mulaw"}, 
#         "customVariables": {"callee_name": "callee_name", "mobile_number": "mobile_number"}, 
#         "introMessage": "\"Hi there! This is your virtual assistant. How can I help you today?\"", 
#         "customAnalysis": {
#             "enabled": True, 
#             "prompt": "\"Analyze the call transcript and provide a structured JSON output. Identify the customer’s main intent, whether a callback is required, and any follow-up actions. Keep all fields factual with no assumptions. If the user requests a callback at any time, set callBackRequired to true and extract the callback time if mentioned. Keep all text short, clear, and machine-readable. Use the exact keys provided below:", 
#             "variables": ["summary", "customerIntent", "callOutcome", "callBackRequired", "callBackTime", "sentiment", "actionItems", "importantDetails"]
#         }, 
#         "callSettings": {"maxDurationSec": 900, "noiseFilter": False, "muteCalleeWhileAssistantSpeaks": True, "backgroundAudio": {}, "userIdleTimeoutSec": 10}, 
#         "dtmf": {"enabled": True}, 
#         "status": "ACTIVE", 
#         "createdBy": 149, 
#         "updatedBy": 149, 
#         "version": 1, 
#         "createdAt": "2025-12-01 12:35:44.898000", 
#         "updatedAt": "2025-12-02 07:30:38.901000", 
#         "strategies": ["FUNCTION_CALL", "MUTE_UNTIL_FIRST_BOT_COMPLETE"], 
#         "idlePrompts": {
#             "1": "“Are you still there? Please let me know how I can help.”", 
#             "2": "“Just checking in — do you still need assistance?”", 
#             "3": "“I haven’t heard from you. I may need to end the call if there’s no response.”"
#         }
#     }

#     DEFAULT_STT_CONFIG = {
#         "engine": "deepgram",
#         "language": "en-IN",
#         "apiKey": "fdf65c552ad2d2d09c591aa5569b6f9892fee852",
#         "encoding": "linear16",
#         "model": "latest_short",
#         "sampleRate": 8000
#     }

#     DEFAULT_TTS_CONFIG = {
#         "engine": "google",
#         "voiceId": "95d51f79-c397-46f9-b49a-23763d3eaa2d",
#         "apiKey": "sk_car_9GXQNhRFsUZQ7R5oGHMYYb",
#         "encoding": "pcm_mulaw",
#         "sampleRate": 8000
#     }

#     # Extract session data
#     prompt = SESSION_PROMPTS.pop(call_id, None) or DEFAULT_PROMPT
#     organisation_id = SESSION_ORG.pop(call_id, 26)
#     campaign_id = SESSION_CAMPAIGN_ID.pop(call_id, 37)

#     # 1. Safely handle missing Assistant Config
#     raw_config = await get_from_redis(f"callctx:{call_id}")
#     assistant_config = json.loads(raw_config) if raw_config else DEFAULT_ASSISTANT_CONFIG
    
#     # 2. Safely handle missing extra_data
#     raw = await get_from_redis(call_id+"_extra_data")
#     if raw:
#         extra_data = json.loads(raw.decode())
#     else:
#         logger.info(f"No Redis data for {call_id}. Treating as INBOUND call. Applying defaults.")
#         extra_data = {
#             "target_phone_number": "+917982409096", 
#             "twilio_phone_number": "+919484956707", 
#             "client_name": "OdioIq", 
#             "call_to": "7982409096", 
#             "session_id": call_id,  # Track dynamically using Asterisk's UUID
#             "organization_id": 26, 
#             "assistant_id": "692d8ba0e5a87846a613520d", 
#             "agent_details": [{"name": "max", "phone_number": "917701910361"}], 
#             "custom_variable": {"callee_name": "rishabh"}, 
#             "campaign_id": 37, 
#             "assistant_config": DEFAULT_ASSISTANT_CONFIG
#         }
        
#         # Save this inbound state to Redis so /webhook/status can clean it up later without crashing
#         await add_to_redis(call_id+"_extra_data", json.dumps(extra_data))
#         await add_to_redis(call_id+"_extra_keys", json.dumps(["CAMPAIGN_37_CALLS_COUNT", "org_26_CALLS_COUNT"]))

#     # 3. Safely handle missing STT/TTS configs
#     stt_raw = await get_from_redis(str("org_id_stt_config_"+ str(organisation_id)))
#     tts_raw = await get_from_redis(str("org_id_tts_config_"+ str(organisation_id)))
    
#     stt_config = json.loads(stt_raw) if stt_raw else DEFAULT_STT_CONFIG
#     tts_config = json.loads(tts_raw) if tts_raw else DEFAULT_TTS_CONFIG

#     session_id = extra_data.get("session_id", call_id)

#     # Launch Pipecat
#     await handle_asterisk_stream(
#         call_id, 
#         session_id, 
#         reader, 
#         writer, 
#         organisation_id, 
#         system_prompt=prompt, 
#         dispatcher=dispatcher,
#         campaign_id=campaign_id,
#         redis_conn=redis_instance, 
#         activemq_conn=activemq_client,
#         assistant_config=assistant_config, 
#         stt_config=stt_config, 
#         tts_config=tts_config,
#         agent_details=extra_data.get("agent_details", {})
#     )


# ---------------------------
# TwiML Endpoint
# ---------------------------

@app.post("/")
async def start_call_ws():
    return HTMLResponse(open("templates/twilio.xml").read(), media_type="application/xml")

@app.post("/sans")
async def start_call_sans():
    return HTMLResponse(open("templates/streams.xml").read(), media_type="application/xml")

@app.post("/sans/ws")
async def sans_webhook(request: Request):
    data = await request.json()
    # logger.info("=== Received Sans Call Request ===")
    # logger.info(json.dumps(data, indent=4, default=str))

    # Broadcast to all connected WebSocket clients

    disposition = data.get("disposition")
    call_id = data.get("Linkedid")
    from_number = data.get("did")
    call_to = data.get("call_to")

    status = disposition.lower()
    logger.info(f"call status received from SANS : {status}")

    event_type, event_data = SANS_STATUS_MAP.get(status, ("EVENT", {"message": status}))
   # event_type = status
    logger.info(f"call status to bot : {event_type}")

    raw = await get_from_redis(call_id+"_extra_data")
    extra_data = json.loads(raw.decode())
    
    # Check for custom hangup reason from bot
    custom_reason = None
    hangup_reason_key = f"{call_id}_hangup_reason"
    try:
        hangup_reason_raw = await get_from_redis(hangup_reason_key)
        if hangup_reason_raw:
            custom_reason = hangup_reason_raw.decode('utf-8')
            logger.info(f"Retrieved custom hangup reason: {custom_reason}")
            # Clean up the reason from Redis
            await delete_from_redis(hangup_reason_key)
    except Exception as e:
        logger.exception(f"Error retrieving hangup reason from Redis: {e}")
    
    if event_type in ('COMPLETED', 'USER_BUSY', 'NO_ANSWER', 'FAILED'):
        user_data = await get_from_redis(call_id+"_extra_keys")
        user_data = json.loads(user_data)
        decrease_list = [user_data[0], user_data[1], user_data[2], user_data[3]]
        await decrease_redis_permits(decrease_list)
        await delete_from_redis(call_id+"_extra_keys")
        await delete_from_redis(call_id+"_extra_data")

    final_call_status=None
    if event_type in ('COMPLETED'):
        
        ended_at_ts = int(time.time())   # or your ended value
        ended_at_iso = (
            datetime.fromtimestamp(ended_at_ts, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
        
        # Use custom reason if available, otherwise use default "COMPLETED"
        final_call_status = custom_reason if custom_reason else str(CallEndReason.COMPLETED)
        
        msg = {
            "callSid": call_id,
            "callId":extra_data.get("session_id"),
            "organisationId": extra_data.get("organization_id", ""),
            "departmentId": extra_data.get("department_id", ""),
            "campaignId": extra_data.get("campaign_id", ""),
            "assistantId": extra_data.get("assistant_id", ""),
            "callStatus": "COMPLETED",
            "reason": final_call_status,
            "endedAt": ended_at_iso,
            "nodes":extra_data.get("nodes", []), # All nodes from request. TODO: ENTER THIS
            "workflow_variables":extra_data.get("workflow_variables", {}), # variables in key and value pair.
        }
        logger.info(f" 1 msg senf to call complete {msg}")
        activemq_client.send_message(msg, ACTIVEMQ_CALL_COMPLETED ,0)
    
    if event_type in RETRYABLE_STATUSES:
        call_number = call_to
        retry_count = await increment_retry_count(call_number)
        msg_key = f"prompt_{extra_data.get('session_id')}"

        if retry_count <= MAX_RETRIES:
            system_message = await get_from_redis(msg_key)
            extra_data["system_message"] = system_message
            logger.info(
                f"Retrying call | call_number={call_number} | attempt={retry_count}"
            )
            await addToQueue(extra_data, delay=RETRY_DELAY_MS)
            logger.info(f"Call requeued | call_number={call_number} | delay={RETRY_DELAY_MS}ms")
        else:
            await delete_from_redis(msg_key)
            await delete_from_redis(f"retry:call:{call_number}")
            logger.info(f"Max retries reached | call_number={call_number} | attempts={retry_count}")

    recording_url = None
    meta = make_meta(
        call_id=extra_data.get("session_id"),
        from_number=from_number,
        to_number=call_to,
        status=event_type.upper(),
        reason=final_call_status,
        recording_url=recording_url,
    )

    logger.info(f"the session id after is {extra_data.get("session_id")}")

    await dispatcher.send(
        meta=meta,
        event_type=event_type.upper(),
        event_data=event_data,
        leg=None
    )


    for ws in connected_clients.copy():
        try:
            await ws.send_text(json.dumps(data, default=str))
        except Exception:
            connected_clients.remove(ws)

    return JSONResponse(content={"status": "received"})

@app.post("/twilio/status")
async def twilio_status_webhook(
    CallSid: str = Form(...),
    CallStatus: str = Form(...),
    From: str = Form(""),
    To: str = Form(""),
    RecordingUrl: str = Form(None),
    RecordingSid: str = Form(None),
):
    status = CallStatus.lower()
    logger.info(f"call status received from twilio : {status}")
    # if status == "in-progress":
    #     status="answered"

    event_type, event_data = TWILIO_STATUS_MAP.get(status, ("EVENT", {"message": status}))
   # event_type = status
    logger.info(f"call status to bot : {event_type}")

    raw = await get_from_redis(CallSid+"_extra_data")
    extra_data = json.loads(raw.decode())
    
    # Check for custom hangup reason from bot
    custom_reason = None
    hangup_reason_key = f"{CallSid}_hangup_reason"
    try:
        hangup_reason_raw = await get_from_redis(hangup_reason_key)
        if hangup_reason_raw:
            custom_reason = hangup_reason_raw.decode('utf-8')
            logger.info(f"Retrieved custom hangup reason: {custom_reason}")
            # Clean up the reason from Redis
            await delete_from_redis(hangup_reason_key)
    except Exception as e:
        logger.exception(f"Error retrieving hangup reason from Redis: {e}")
    
    if event_type in ('COMPLETED', 'USER_BUSY', 'NO_ANSWER', 'FAILED'):
        user_data = await get_from_redis(CallSid+"_extra_keys")
        user_data = json.loads(user_data)
        decrease_list = [user_data[0], user_data[1], user_data[2], user_data[3]]
        await decrease_redis_permits(decrease_list)
        await delete_from_redis(CallSid+"_extra_keys")
        await delete_from_redis(CallSid+"_extra_data")
    
    final_call_status=None
    if event_type in ('COMPLETED'):
        ended_at_ts = int(time.time())   # or your ended value
        ended_at_iso = (
            datetime.fromtimestamp(ended_at_ts, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
        
        # Use custom reason if available, otherwise use default "COMPLETED"
        final_call_status = custom_reason if custom_reason else str(CallEndReason.COMPLETED)
        
        msg = {
            "callSid": CallSid,
            "callId":extra_data.get("session_id"),
            "organisationId": extra_data.get("organization_id", ""),
            "departmentId": extra_data.get("department_id", ""),
            "campaignId": extra_data.get("campaign_id", ""),
            "assistantId": extra_data.get("assistant_id", ""),
            "callStatus": "COMPLETED",
            "reason": final_call_status,
            "endedAt":ended_at_iso,
            "nodes":extra_data.get("nodes", []), # All nodes from request. TODO: ENTER THIS
            "workflow_variables":extra_data.get("workflow_variables", {}), # variables in key and value pair.
        }
        logger.info(f"2. msg senf to call complete {msg}")
        activemq_client.send_message(msg, ACTIVEMQ_CALL_COMPLETED ,0)
    
    if event_type in RETRYABLE_STATUSES:
        call_number = To
        retry_count = await increment_retry_count(call_number)
        msg_key = f"prompt_{extra_data.get('session_id')}"

        if retry_count <= MAX_RETRIES:
            system_message = await get_from_redis(msg_key)
            extra_data["system_message"] = system_message
            logger.info(
                f"Retrying call | call_number={call_number} | attempt={retry_count}"
            )
            await addToQueue(extra_data, delay=RETRY_DELAY_MS)
            logger.info(f"Call requeued | call_number={call_number} | delay={RETRY_DELAY_MS}ms")
        else:
            await delete_from_redis(msg_key)
            await delete_from_redis(f"retry:call:{call_number}")
            logger.info(f"Max retries reached | call_number={call_number} | attempts={retry_count}")

    recording_url = None

    # logger.info(f"the session id after is {extra_data.get("session_id")}")

    meta = make_meta(
        call_id=extra_data.get("session_id"),
        from_number=From,
        to_number=To,
        status=event_type.upper(),
        reason=final_call_status,
        recording_url=recording_url,
    )

    await dispatcher.send(
        meta=meta,
        event_type=event_type.upper(),
        event_data=event_data,
        leg=None
    )
    return {"ok": True}


# @app.post("/greeter/status")
# async def webhook(request: Request):
#     content_type = request.headers.get("content-type", "")
#     if "application/json" in content_type:
#         data = await request.json()
#     else:
#         data = dict(await request.form())

#     logger.info(f"Greeter Webhook data: {data}")

#     event_type = data.get("type") 
#     room = data.get("payload", {}).get("room") or data.get("room") # Adjust based on actual payload structure

#     if not room:
#         logger.error("No room found in payload")
#         return {"status": "ignored"}
#     events_key = f"room_events:{room}"
    
#     raw_history = await get_from_redis(events_key)
#     history = json.loads(raw_history) if raw_history else []
    
#     history.append(event_type)
#     await add_to_redis(events_key, json.dumps(history), 3600) # 1 hour TTL
    
#     # logger.info(f"Updated event history for {room}: {history}")

#     mapped_status = None
    
#     if event_type == "dialout.connected":
#         mapped_status = "RINGING" 

#     elif event_type == "dialout.answered":
#         mapped_status = "ANSWERED"

#     elif event_type == "dialout.error":
#         mapped_status = "FAILED"

#     elif event_type == "dialout.stopped":
#         if "dialout.answered" in history:
#             mapped_status = "COMPLETED"
#         elif "dialout.connected" in history:
#             mapped_status = "USER_BUSY"
#         else:
#             mapped_status = "NO_ANSWER" # Fallback if it stopped before connecting

#     if not mapped_status:
#         return {"status": "ignored", "reason": "unknown event type"}
    
#     # if event_type == "dialout.stopped":
#     #     asyncio.create_task(stop_sip_dialout(room))


#     raw_extra = await get_from_redis(room + "_extra_data")
#     if not raw_extra:
#         logger.error(f"No extra_data found for room {room}")
#         return {"status": "error", "detail": "Session context lost"}
        
#     extra_data = json.loads(raw_extra.decode())
    
#     # 6. Check for Custom Hangup Reason (from Bot)
#     custom_reason = None
#     hangup_reason_key = f"{room}_hangup_reason"
#     try:
#         hangup_reason_raw = await get_from_redis(hangup_reason_key)
#         if hangup_reason_raw:
#             custom_reason = hangup_reason_raw.decode('utf-8')
#             # Clean up
#             await delete_from_redis(hangup_reason_key)
#     except Exception as e:
#         logger.exception(f"Error retrieving hangup reason: {e}")

#     # 7. Handle Terminal States (Cleanup Permits & Keys)
#     if mapped_status in ('COMPLETED', 'USER_BUSY', 'NO_ANSWER', 'FAILED'):
#         user_keys_raw = await get_from_redis(room + "_extra_keys")
#         if user_keys_raw:
#             user_keys = json.loads(user_keys_raw)
#             decrease_list = ['GLOBAL_CALLS_COUNT', 'GREETER_CALLS_COUNT', user_keys[0], user_keys[1]]
#             await decrease_redis_permits(decrease_list)
            
#         await delete_from_redis(room + "_extra_keys")
#         await delete_from_redis(room + "_extra_data")
#         await delete_from_redis(events_key) # Clean up the event history we created

#     final_call_status=None
#     if mapped_status == 'COMPLETED':
#         ended_at_ts = int(time.time())
#         ended_at_iso = (
#             datetime.fromtimestamp(ended_at_ts, tz=timezone.utc)
#             .isoformat()
#             .replace("+00:00", "Z")
#         )

#         final_call_status = custom_reason if custom_reason else str(CallEndReason.COMPLETED)
        
#         msg = {
#             "callSid": room,
#             "callId": extra_data.get("session_id"),
#             "organisationId": extra_data.get("organization_id", ""),
#             "departmentId": extra_data.get("department_id", ""),
#             "campaignId": extra_data.get("campaign_id", ""),
#             "assistantId": extra_data.get("assistant_id", ""),
#             "callStatus": "COMPLETED",
#             "reason": final_call_status,
#             "endedAt": ended_at_iso,
#             "nodes":extra_data.get("nodes", []), # All nodes from request. TODO: ENTER THIS
#             "workflow_variables":extra_data.get("workflow_variables", {}), # variables in key and value pair.
#         }
#         activemq_client.send_message(msg, ACTIVEMQ_CALL_COMPLETED, 0)

#     from_number = extra_data.get("twilio_phone_number", "") 
#     to_number = extra_data.get("target_phone_number", "")

#     if mapped_status in RETRYABLE_STATUSES:
#         call_number = to_number
#         retry_count = await increment_retry_count(call_number)
#         msg_key = f"prompt_{extra_data.get('session_id')}"

#         if retry_count <= MAX_RETRIES:
#             system_message = await get_from_redis(msg_key)
#             extra_data["system_message"] = system_message
#             logger.info(
#                 f"Retrying call | call_number={call_number} | attempt={retry_count}"
#             )
#             await addToQueue(extra_data, delay=RETRY_DELAY_MS)
#             logger.info(f"Call requeued | call_number={call_number} | delay={RETRY_DELAY_MS}ms")
#         else:
#             await delete_from_redis(msg_key)
#             await delete_from_redis(f"retry:call:{call_number}")
#             logger.info(f"Max retries reached | call_number={call_number} | attempts={retry_count}")

    
#     meta = make_meta(
#         call_id=extra_data.get("session_id"),
#         from_number=from_number,
#         to_number=to_number,
#         status=mapped_status,
#         reason=final_call_status,
#         recording_url=None,
#     )

#     await dispatcher.send(
#         meta=meta,
#         event_type=mapped_status,
#         event_data={"message": final_call_status},
#         leg=None
#     )

#     return {"status": "ok"}

from fastapi import APIRouter, HTTPException
import redis.asyncio as redis # Assuming you are using the async version

# ... your existing setup ...

@app.post("/redis/reset-counts")
async def reset_count_keys():
    try:
        # 1. Initialize the cursor and the pattern
        cursor = 0
        pattern = "*_COUNT"
        keys_to_reset = []

        # 2. Iteratively find all keys matching the pattern
        while True:
            cursor, keys = await redis_instance.scan(cursor=cursor, match=pattern)
            keys_to_reset.extend(keys)
            if cursor == 0:
                break

        if not keys_to_reset:
            return {"message": "No keys found matching the pattern", "count": 0}

        # 3. Use a pipeline to set all found keys to 0 efficiently
        async with redis_instance.pipeline(transaction=True) as pipe:
            for key in keys_to_reset:
                pipe.set(key, 0)
            await pipe.execute()

        return {
            "status": "success",
            "message": f"Reset {len(keys_to_reset)} keys to 0",
            "keys_affected": [k.decode() if isinstance(k, bytes) else k for k in keys_to_reset]
        }

    except Exception as e:
        logger.error(f"Error resetting redis keys: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/redis/call-counts")
async def get_call_counts():
    try:
        cursor = 0
        keys_list = []
        
        # Scan only relevant keys
        while True:
            cursor, keys = await redis_instance.scan(
                cursor=cursor,
                match="*_CALLS_COUNT"
            )
            keys_list.extend(keys)

            if cursor == 0:
                break

        if not keys_list:
            return {"status": "success", "total": 0, "data": {}}

        # Fetch all values in one go
        values = await redis_instance.mget(keys_list)

        result = {}
        for k, v in zip(keys_list, values):
            key = k.decode('utf-8') if isinstance(k, bytes) else k
            
            if v is not None:
                value = int(v)  # since you said it's integer
            else:
                value = 0  # or None, depending on your preference

            result[key] = value

        return {
            "status": "success",
            "total_keys": len(result),
            "data": result
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
    
from typing import List
from pydantic import BaseModel

class KeyList(BaseModel):
    keys: List[str]

@app.post("/redis/delete-keys")
async def delete_keys(data: KeyList):
    try:
        if not data.keys:
            return {"message": "No keys provided"}

        # .delete() returns the number of keys actually removed
        deleted_count = await redis_instance.delete(*data.keys)

        return {
            "status": "success",
            "requested_keys": len(data.keys),
            "actually_deleted": deleted_count
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    

async def increment_retry_count(session_id: str) -> int:
    key = f"retry:call:{session_id}"
    count = await redis_instance.incr(key)
    await redis_instance.expire(key, 24 * 60 * 60)
    return count


async def decrease_redis_permits(keys_list):
    # safe_decr_all = redis_instance.register_script(SAFE_DECR_ALL_SCRIPT)
    global safe_decr_all

    async def safe_decrement_multiple(keys: list[str]) -> None:
        await safe_decr_all(keys=keys)
        
    await safe_decrement_multiple(keys_list)
    logger.info(f"decreased redis permits count for one call")

async def save_audio(server_name: str, audio: bytes, sample_rate: int, num_channels: int):
    #upload this to s3 and make the completed meta call
    if len(audio) > 0:
        filename = (
            f"{server_name}_recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        )
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio)
            async with aiofiles.open(filename, "wb") as file:
                await file.write(buffer.getvalue())
        logger.info(f"Merged audio saved to {filename}")
    else:
        logger.info("No audio data to save")

@app.get("/send-delayed-message")
def send_delayed_message(msg: str = "Hello with delay", delay_ms: int = 10000):
    conn = stomp.Connection([(ACTIVEMQ_HOST, ACTIVEMQ_PORT)])
    
    try:
        conn.connect('admin', 'admin', wait=True)
        
        # This header is the magic part that requires schedulerSupport="true"
        headers = {
            'AMQ_SCHEDULED_DELAY': str(delay_ms)
        }
        
        conn.send(body=msg, destination=ACTIVEMQ_DESTINATION, headers=headers)
        
        return {
            "status": "sent",
            "message": msg,
            "delay_applied_ms": delay_ms,
            "destination": ACTIVEMQ_DESTINATION
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        if conn.is_connected():
            conn.disconnect()


HANGUP_CAUSES = {
    # --- SUCCESSFUL CALLS ---
    "16": "COMPLETED",     # Normal call clearing (Someone answered and eventually hung up)

    # --- NO ANSWER / IGNORED ---
    "0": "NO_ANSWER",      # No cause code set / SIP 487 Request Terminated (Caller hung up before answer)
    "18": "NO_ANSWER",     # No user responding (Phone rang until carrier timeout)
    "19": "NO_ANSWER",     # No answer from user (User alerted but didn't pick up)

    # --- BUSY / DECLINED ---
    "17": "USER_BUSY",     # User busy (They are on another call or hit the red "Decline" button)
    "21": "USER_BUSY",     # Call rejected (Carrier blocked it, or user's phone auto-rejected it)

    # --- ROUTING FAILURES & INVALID NUMBERS ---
    "1": "FAILED",         # Unallocated/Unassigned number (Number doesn't exist)
    "2": "FAILED",         # No route to specified transit network
    "3": "FAILED",         # No route to destination
    "22": "FAILED",        # Number changed
    "27": "FAILED",        # Destination out of order (Phone is switched off or out of coverage)
    "28": "FAILED",        # Invalid number format (e.g., missing country code)
    
    # --- NETWORK & CARRIER ERRORS ---
    "34": "FAILED",        # No circuit/channel available (SIP provider rate limit or congestion)
    "38": "FAILED",        # Network out of order
    "41": "FAILED",        # Temporary failure (Generic SIP 503 Service Unavailable)
    "42": "FAILED",        # Switching equipment congestion
    "47": "FAILED",        # Resource unavailable
    "58": "FAILED",        # Bearer capability not presently available (Codec mismatch)
    "88": "FAILED",        # Incompatible destination
}

class WebhookPayload(BaseModel):
    call_id: str
    status: str
    cause: Optional[str] = None
    dial_status: Optional[str] = None  # 1. Added dial_status

def get_call_status(cause_code: str, dial_status: str = "UNKNOWN", asterisk_status: str = "") -> str:
    """
    Helper function to determine the final business status of the call.
    Checks explicit Asterisk statuses, then DIALSTATUS, and finally falls back to cause codes.
    """
    # 1. Handle explicit Asterisk statuses (e.g., from [inbound-calls])
    if asterisk_status == "ANSWERED":
        return "ANSWERED"
    if asterisk_status == "DIALING":
        return "RINGING"
    if asterisk_status == "TRANSFERRING":
        return "COMPLETED"

    # 2. CRITICAL FIX: Check DIALSTATUS before checking cause code 16!
    # Because a timeout sends Cause 16 AND dial_status NOANSWER.
    if dial_status in ["NOANSWER", "CANCEL"]:
        return "NO_ANSWER"
    if dial_status == "BUSY":
        return "USER_BUSY"
    if dial_status == "CHANUNAVAIL":
        return "FAILED"

    # 3. Fall back to the ISDN Cause Code mapping
    status = HANGUP_CAUSES.get(str(cause_code))
    
    return status or "FAILED" # Default to FAILED if completely unknown

def cleanup_call_session(call_sid: str):
    """
    Safely removes all in-memory state for a given call to prevent memory leaks.
    Using .pop(key, None) ensures it doesn't crash if the key is already gone.
    """
    SESSION_PROMPTS.pop(call_sid, None)
    SESSION_CLIENT.pop(call_sid, None)
    SESSION_ORG.pop(call_sid, None)
    SESSION_CALLEE_NAME.pop(call_sid, None)
    LANGUAGE_PREFERENCES.pop(call_sid, None)
    CUSTOMER_NUMBERS.pop(call_sid, None)
    SESSION_CAMPAIGN_ID.pop(call_sid, None)
    SESSION_WORKFLOW_VARIABLES.pop(call_sid, None)
    SESSION_WORKFLOW_NODES.pop(call_sid, None)
    
    logger.debug(f"[Memory Management] Cleared RAM state for call: {call_sid}")

#asterisk webhook call data
@app.post("/webhook/status")
async def call_status_webhook(payload: WebhookPayload):
    logger.info(f"webhook response is {payload}")
    
    # 2. CRITICAL FIX: Pass dial_status and status correctly
    mapped_status = get_call_status(
        cause_code=payload.cause, 
        dial_status=payload.dial_status, 
        asterisk_status=payload.status
    )

    # --- Redis State Guard ---
    answered_flag_key = f"{payload.call_id}_was_answered"

    if mapped_status == "ANSWERED":
        # Flag the call as answered with a 2-hour TTL (7200 seconds) 
        # Adjust TTL based on your maximum expected call duration
        await add_to_redis(answered_flag_key, "true", 7200)

    elif mapped_status in ["NO_ANSWER", "FAILED"]:
        # Check if the call had previously connected
        was_answered = await get_from_redis(answered_flag_key)
        if was_answered:
            logger.info(f"Blocked phantom {mapped_status} event for {payload.call_id}. Call was already ANSWERED.")
            # Return 200 OK so Asterisk stops resending the webhook, but halt processing
            return {"status": "ignored", "detail": f"Phantom {mapped_status} suppressed"}
    # -------------------------


    raw_extra = await get_from_redis(payload.call_id + "_extra_data")
    if not raw_extra:
        logger.error(f"No extra_data found for call_id {payload.call_id}")
        return {"status": "error", "detail": "Session context lost"}
        
    extra_data = json.loads(raw_extra.decode())
    
    # 6. Check for Custom Hangup Reason (from Bot)
    custom_reason = None
    hangup_reason_key = f"{payload.call_id}_hangup_reason"
    try:
        hangup_reason_raw = await get_from_redis(hangup_reason_key)
        if hangup_reason_raw:
            custom_reason = hangup_reason_raw.decode('utf-8')
            # Clean up
            await delete_from_redis(hangup_reason_key)
    except Exception as e:
        logger.exception(f"Error retrieving hangup reason: {e}")

    # 7. Handle Terminal States (Cleanup Permits & Keys)
    if mapped_status in ('COMPLETED', 'USER_BUSY', 'NO_ANSWER', 'FAILED'):
        user_keys_raw = await get_from_redis(payload.call_id + "_extra_keys")
        if user_keys_raw:
            user_keys = json.loads(user_keys_raw)
            decrease_list = [user_keys[0], user_keys[1], user_keys[2], user_keys[3]]
            await decrease_redis_permits(decrease_list)
            
        await delete_from_redis(payload.call_id + "_extra_keys")
        await delete_from_redis(payload.call_id + "_extra_data")
        await delete_from_redis(answered_flag_key)
        cleanup_call_session(payload.call_id)

    final_call_status=None
    if mapped_status == 'COMPLETED':
        ended_at_ts = int(time.time())
        ended_at_iso = (
            datetime.fromtimestamp(ended_at_ts, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

        final_call_status = custom_reason if custom_reason else str(CallEndReason.COMPLETED)
        
        msg = {
            "callSid": payload.call_id,
            "callId": extra_data.get("session_id"),
            "organisationId": extra_data.get("organization_id", ""),
            "departmentId": extra_data.get("department_id", ""),
            "campaignId": extra_data.get("campaign_id", None),
            "assistantId": extra_data.get("assistant_id", ""),
            "callStatus": "COMPLETED",
            "reason": final_call_status,
            "endedAt": ended_at_iso,
            "nodes":extra_data.get("nodes", []), # All nodes from request. TODO: ENTER THIS
            "workflow_variables":extra_data.get("workflow_variables", {}), # variables in key and value pair.
        }
        print(f"Sending the message this------------------------------------------ {msg}")
        activemq_client.send_message(msg, ACTIVEMQ_CALL_COMPLETED, 0)
    from_number = extra_data.get("twilio_phone_number", "")
    to_number = extra_data.get("target_phone_number", "")

    if mapped_status in RETRYABLE_STATUSES:
        call_number = to_number
        retry_count = await increment_retry_count(extra_data.get('session_id'))
        msg_key = f"prompt_{extra_data.get('session_id')}"

        if retry_count <= MAX_RETRIES:
            system_message = await get_from_redis(msg_key)
            extra_data["system_message"] = system_message
            logger.info(
                f"Retrying call | call_number={call_number} | attempt={retry_count}"
            )
            await addToQueue(extra_data, delay=RETRY_DELAY_MS)
            logger.info(f"Call requeued | call_number={call_number} | delay={RETRY_DELAY_MS}ms")
        else:
            await delete_from_redis(msg_key)
            await delete_from_redis(f"retry:call:{extra_data.get('session_id')}")
            logger.info(f"Max retries reached | call_number={call_number} | attempts={retry_count}")

    
    meta = make_meta(
        call_id=extra_data.get("session_id"),
        from_number=from_number,
        to_number=to_number,
        status=mapped_status,
        reason=final_call_status,
        recording_url=None,
    )

    # print(f"Sending the message this-----------------------------------------2222222222222222222222222- {msg}")

    await dispatcher.send(
        meta=meta,
        event_type=mapped_status,
        event_data={"message": final_call_status},
        leg=None
    )

    return {"status": "ok"}

def extract_prompt_from_node(node_data: dict) -> str:
    try:
        data_str = node_data.get("data")
        if not data_str:
            return ""
        data_json = json.loads(data_str)
        inbound = data_json.get("inbound", {})
        prompt = inbound.get("prompt")
        if not prompt:
            prompt = inbound.get("formattedPrompt", {}).get("prompt", "")
        return prompt
    except Exception as e:
        print(f"Error parsing prompt: {e}")
        return ""


@app.post("/webhook/inbound")
async def handle_inbound_setup(request: Request):
    # 1. Generate the unique session ID natively in Python
    call_id = str(uuid.uuid4())
    
    try:
        # 2. Intercept the raw HTTP body
        raw_body = await request.body()
        body_str = raw_body.decode("utf-8")
        
        # 3. Clean Asterisk's literal backslash escaping
        clean_body = body_str.replace('\\,', ',')
        
        # 4. Safely parse the valid JSON
        data = json.loads(clean_body)
        
        caller_number = data.get("caller", "Unknown")
        callee_number = data.get("callee")
        
    except Exception as e:
        logger.error(f"Failed to parse Asterisk payload: {e}")
        caller_number = "Unknown"
        callee_number = None

    if not callee_number:
        logger.error("No callee number provided in payload.")
        raise HTTPException(status_code=400, detail="Missing callee number")

    logger.info(f"Incoming call from {caller_number} callee {callee_number}. Generating Call ID: {call_id}")

    # 5. Fetch dynamic configuration from the new API
    api_url = os.getenv("INBOUND_CONFIG_URL")
    
    # Ensure numbers are formatted with a '+' if your API requires it. 
    # httpx automatically handles URL encoding (e.g., turning '+' into '%2B')
    bot_num_param = callee_number if callee_number.startswith('+') else f"+{callee_number}"
    cust_num_param = caller_number if caller_number.startswith('+') else f"+{caller_number}"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                api_url, 
                params={"bot_number": bot_num_param, "customer_number": cust_num_param},
                timeout=10.0
            )
            response.raise_for_status()
            api_response = response.json()
            
            if api_response.get("status") != 200:
                logger.error(f"API returned non-200 status: {api_response.get('message')}")
                raise HTTPException(status_code=403, detail="Unauthorized Callee or Config Not Found")
                
            config_data = api_response.get("data", {})
            print(f"the config data is {config_data}")
            
    except Exception as e:
        logger.error(f"Failed to fetch config from API: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch bot configuration")

    # 6. Extract dynamic values from the API response
    ASSISTANT_ID = config_data.get("assistant_id")
    ORG_ID = int(config_data.get("organization_id"))
    WORKFLOW_ID = config_data.get("workflow_id")
    DEPARTMENT_ID = config_data.get("department_id")
    CLIENT_NAME = config_data.get("client_name", "OdioIq")
    AGENT_DETAILS = config_data.get("agent_details", [])
    PROMPT = config_data.get("system_message", "")
    NODES = config_data.get("nodes", "")

    PROMPT = replace_custom_variables(PROMPT, {'mobile_number': caller_number})

    print(f"the updated prompt is ---============================================== {PROMPT}")
    
    # Fallback to env variable if Campaign ID isn't provided by the API yet
    # CAMPAIGN_ID = int(os.getenv("CAMPAIGN_ID", 37)) 

    if not ASSISTANT_ID or not ORG_ID:
        logger.error("API response missing critical data (assistant_id or organization_id)")
        raise HTTPException(status_code=500, detail="Invalid bot configuration data")

    # 7. Fetch DB configurations using the dynamic IDs
    provider_config = await get_telephony_provider(app, ORG_ID)
    stt_config = provider_config["phoneAi"]["stt"]
    tts_config = provider_config['phoneAi']['tts']

    assistant_config = await get_config_by_assistant_id(app, ASSISTANT_ID)
    assistant_config.pop('createdAt', None)
    assistant_config.pop('updatedAt', None)

    await prewarm_pipeline(
        call_id,
        stt_config,
        tts_config,
        assistant_config,
        # project_id="your-gcp-project-id" # Pass this if you updated prewarm for Vertex
    )
        # 10. Prepare Metadata for Dispatcher
    meta = {
        "callId": call_id,
        "sessionId": call_id,
        "organizationId": ORG_ID, # Dynamic now
        "odioBotExternalProviderConfigId": str(provider_config["_id"]),
        "fromNumber": callee_number,
        "toNumber": caller_number, # Dynamic now
        "status": "INITIATED",
        "callDirection": "INBOUND",
        "workflowId": WORKFLOW_ID, # Dynamic now
    }

    # Schedules the tasks on the event loop and immediately moves to the next line
    # background_tasks.add_task(
    await dispatcher.send(
        meta=meta,
        event_type="INITIATED",
        event_data={"message": "Call initiated"},
        leg=None
    )

    time.sleep(1)
    # 8. Build extra_data with the new API payload
    extra_data = {
        "target_phone_number": caller_number, 
        "twilio_phone_number": callee_number, # Dynamic now
        "client_name": CLIENT_NAME, 
        "call_to": caller_number, 
        "session_id": call_id,
        "organization_id": ORG_ID, 
        "assistant_id": ASSISTANT_ID,
        "agent_details": AGENT_DETAILS, # Injected from API
        "campaign_id": None, 
        "department_id": DEPARTMENT_ID,
        "nodes":NODES,
        'custom_variable': {'mobile_number': caller_number},
        "workflow_variables": {"caller_number": caller_number}
    }

    process_message_case(extra_data)

    # 9. Pre-warm the Redis cache
    await add_to_redis(call_id + "_extra_data", json.dumps(extra_data))
    await add_to_redis(f"org_id_stt_config_{ORG_ID}", json.dumps(stt_config))
    await add_to_redis(f"org_id_tts_config_{ORG_ID}", json.dumps(tts_config))

    # Note: Since the API returns the prompt directly in 'system_message', 
    # you no longer need to fetch the node details from your DB unless required elsewhere.
    SESSION_PROMPTS[call_id] = PROMPT
    SESSION_ORG[call_id] = ORG_ID
    # SESSION_CAMPAIGN_ID[call_id] = CAMPAIGN_ID
    
    # Save callctx to Redis (Required by your original spawn_asterisk_bot)
    await add_to_redis(f"callctx:{call_id}", json.dumps(assistant_config), 900) 

    meta2 = {
        **meta,
        "status": "ANSWERED",
    }

    # background_tasks.add_task(
    await dispatcher.send(
        meta=meta2,
        event_type="ANSWERED",
        event_data={"message": "ANSWERED"},
        leg=None
    )

    print(f"sending the data new one ------------------------------------{meta}")

    # Return the UUID as plain text so Asterisk can read it instantly
    return Response(content=call_id, media_type="text/plain")

def process_message_case( payload):
    # 1. Normalize the nodes array to ensure backward compatibility
    normalized_nodes = []
    for node in payload.get("nodes", []):
        normalized_nodes.append({
            "nodeId": node.get("node_id", node.get("nodeId")),
            "nextNodeId": node.get("next_node_id", node.get("nextNodeId")),
            "nodeType": node.get("node_type", node.get("nodeType")),
            "nodeSubType": node.get("node_sub_type", node.get("nodeSubType")),
            "data": node.get("data") # 'data' contents (like sharedGoogleSheetUrl) remained the same
        })
    
    # 2. Overwrite the nodes in the payload with the corrected formatting
    payload["nodes"] = normalized_nodes

@app.get("/health/diagnostics/deep-dive")
async def deep_dive_diagnostics():
    """
    Identifies EXACTLY which call_ids are stuck in memory or hanging the event loop.
    """
    # 1. Measure Event Loop Lag
    start_time = time.monotonic()
    await asyncio.sleep(0)  
    loop_lag_ms = (time.monotonic() - start_time) * 1000

    # 2. Identify EXACT Ghost Sessions (Memory Leaks)
    # If a call_id is in NODES but got popped from PROMPTS, it's a ghost.
    active_sids = set(SESSION_PROMPTS.keys())
    node_sids = set(SESSION_WORKFLOW_NODES.keys())
    
    ghost_sids = list(node_sids - active_sids)
    
    # 3. Inspect Active Async Tasks
    tasks = asyncio.all_tasks()
    stuck_call_tasks = {}
    generic_running_tasks = []
    
    for t in tasks:
        task_name = t.get_name()
        
        # If the task name contains a UUID/call_id (because we named it!)
        if "call-" in task_name or "-" in task_name and len(task_name) > 20:
            # Group tasks by the call_id they belong to
            if task_name not in stuck_call_tasks:
                stuck_call_tasks[task_name] = "Running"
        else:
            coro_name = t.get_coro().__name__ if t.get_coro() else "Unknown"
            generic_running_tasks.append(coro_name)

    # Determine overall status
    if loop_lag_ms > 150 or len(ghost_sids) > 0:
        status = "DEGRADED"
    else:
        status = "HEALTHY"

    return {
        "status": status,
        "metrics": {
            "event_loop_lag_ms": round(loop_lag_ms, 2),
            "healthy_active_calls": len(active_sids),
            "ghost_sessions_leaking": len(ghost_sids),
        },
        "forensics": {
            "leaking_call_sids": ghost_sids, # Look these up in Asterisk logs!
            "hanging_named_tasks": stuck_call_tasks, # The exact tasks that refuse to close
        },
        "background_noise": {
            "total_unnamed_tasks": len(generic_running_tasks),
            # Count the generic tasks to make the output readable
            "unnamed_task_types": dict(__import__('collections').Counter(generic_running_tasks))
        }
    }

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Twilio Chatbot Server")
    parser.add_argument("-t", "--test", action="store_true", default=False, help="set the server in testing mode")
    args, _ = parser.parse_known_args()
    app.state.testing = args.test
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Command to run the server:
# uvicorn server:app --host 0.0.0.0 --port 8000 --reload
