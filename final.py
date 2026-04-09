import os
import re
import sys
import json
import time
import queue
import threading
import datetime
import urllib.request
import urllib.parse
import tempfile
import cv2
import asyncio
import edge_tts
from playsound3 import playsound
import pygame
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from dotenv import load_dotenv


# Our Config

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

TRIGGER_WORD = "computer"
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"
VIDEO_PATH = "Assets/background.mp4"
SAMPLE_RATE = 16000
CHANNELS = 1


# GOOGLE CALENDAR

SCOPES = ['https://www.googleapis.com/auth/calendar']

def get_calendar_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        try:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                from google_auth_oauthlib.flow import InstalledAppFlow
                flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
                creds = flow.run_local_server(port=0)
            with open("token.json", "w") as token:
                token.write(creds.to_json())
        except Exception as e:
            print("Calendar auth error:", e)
            return None
    try:
        return build("calendar", "v3", credentials=creds)
    except Exception as e:
        print("Google Calendar service error:", e)
        return None

service = get_calendar_service()


# Text Norminalization (This is for the mistakes our vosk model could mishear)

WORD_TO_NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50",
}

MISHEARING_FIXES = {
    "to the": "today",
    "to die": "today",
    "to day": "today",
    "to dei": "today",
    "to dey": "today",
    "to morrow": "tomorrow",
    "to more": "tomorrow",
    "to moro": "tomorrow",
    "next weak": "next week",
    "a m": "am",
    "p m": "pm",
}

def normalize_text(text: str) -> str:
    t = text.lower().strip()
    for wrong, right in MISHEARING_FIXES.items():
        t = t.replace(wrong, right)
    words = t.split()
    result = []
    i = 0
    while i < len(words):
        word = words[i]
        if word in ("twenty", "thirty", "forty", "fifty") and i + 1 < len(words):
            next_word = words[i + 1]
            if next_word in WORD_TO_NUM and int(WORD_TO_NUM[next_word]) < 10:
                result.append(str(int(WORD_TO_NUM[word]) + int(WORD_TO_NUM[next_word])))
                i += 2
                continue
        result.append(WORD_TO_NUM.get(word, word))
        i += 1
    normalized = " ".join(result)
    normalized = re.sub(r'(\d)(am|pm)', r'\1 \2', normalized)
    print(f"Normalized: '{text}' -> '{normalized}'")
    return normalized


# TIME PARSER

def parse_event_datetime(text: str):
    now = datetime.datetime.now(datetime.timezone.utc)
    t = text.lower().strip()
    base = now + datetime.timedelta(days=1) if "tomorrow" in t else now
    match = re.search(
        r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)|(\d{1,2}):(\d{2})|(\d{1,2})\s+o\'?clock',
        t
    )
    if not match:
        return None
    hour = None
    minute = 0
    if match.group(1) is not None:
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        meridiem = match.group(3)
        if meridiem == "pm" and hour != 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
    elif match.group(4) is not None:
        hour = int(match.group(4))
        minute = int(match.group(5))
    elif match.group(6) is not None:
        hour = int(match.group(6))
    if hour is None or hour > 23:
        return None
    result = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if result < now and "tomorrow" not in t:
        result += datetime.timedelta(days=1)
    return result


# Calendder Helpers

def format_event_time(raw_start: str) -> str:
    try:
        dt = datetime.datetime.fromisoformat(raw_start.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%A %d %B at %I:%M %p")
    except Exception:
        return raw_start

def fetch_upcoming_events(days=7, max_results=20):
    if not service:
        return []
    now = datetime.datetime.now(datetime.timezone.utc)
    until = now + datetime.timedelta(days=days)
    try:
        result = service.events().list(
            calendarId="primary",
            timeMin=now.isoformat(),
            timeMax=until.isoformat(),
            maxResults=max_results,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        return result.get("items", [])
    except Exception as e:
        print(f"Fetch error: {e}")
        return []

def fetch_events_for_range(start_dt: datetime.datetime, end_dt: datetime.datetime, max_results=20):
    """Fetch events strictly within [start_dt, end_dt)."""
    if not service:
        return []
    try:
        result = service.events().list(
            calendarId="primary",
            timeMin=start_dt.isoformat(),
            timeMax=end_dt.isoformat(),
            maxResults=max_results,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        return result.get("items", [])
    except Exception as e:
        print(f"Fetch error: {e}")
        return []

def resolve_period(period: str):
    now = datetime.datetime.now(datetime.timezone.utc)
    local_now = datetime.datetime.now()
    DAY_NAMES = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
    p = period.lower().strip()
    if "today" in p:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, start + datetime.timedelta(days=1), "today"
    if "tomorrow" in p:
        tomorrow = now + datetime.timedelta(days=1)
        start = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, start + datetime.timedelta(days=1), "tomorrow"
    if "next week" in p:
        days_until_monday = (7 - local_now.weekday()) % 7 or 7
        monday = now + datetime.timedelta(days=days_until_monday)
        start = monday.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, start + datetime.timedelta(days=7), "next week"
    if "this week" in p:
        monday = now - datetime.timedelta(days=local_now.weekday())
        start = monday.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, start + datetime.timedelta(days=7), "this week"
    if "weekend" in p:
        days_until_saturday = (5 - local_now.weekday()) % 7
        saturday = now + datetime.timedelta(days=days_until_saturday)
        start = saturday.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, start + datetime.timedelta(days=2), "the weekend"
    for i, day in enumerate(DAY_NAMES):
        if day in p:
            days_ahead = (i - local_now.weekday()) % 7 or 7
            target = now + datetime.timedelta(days=days_ahead)
            start = target.replace(hour=0, minute=0, second=0, microsecond=0)
            return start, start + datetime.timedelta(days=1), day.capitalize()
    return now, now + datetime.timedelta(days=7), "the next 7 days"


# Get Our Weather infomation the default is Abuja

DEFAULT_CITY = os.getenv("DEFAULT_CITY", "Abuja")

def geocode_city(city: str):
    """Return (lat, lon, display_name) for a city using Open-Meteo geocoding."""
    url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode({
        "name": city, "count": 1, "language": "en", "format": "json"
    })
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read())
        results = data.get("results")
        if not results:
            return None
        r = results[0]
        return r["latitude"], r["longitude"], r.get("name", city)
    except Exception as e:
        print(f"Geocode error: {e}")
        return None

def fetch_weather(lat: float, lon: float):
    """Fetch current weather from Open-Meteo (no API key needed)."""
    url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,weathercode,windspeed_10m,precipitation",
        "temperature_unit": "celsius",
        "windspeed_unit": "kmh",
        "timezone": "auto"
    })
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"Weather fetch error: {e}")
        return None

WMO_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "icy fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow",
    77: "snow grains",
    80: "slight showers", 81: "moderate showers", 82: "violent showers",
    85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}

def describe_weather(data: dict, city_name: str) -> str:
    cur = data.get("current", {})
    temp   = cur.get("temperature_2m", "?")
    humid  = cur.get("relative_humidity_2m", "?")
    wind   = cur.get("windspeed_10m", "?")
    precip = cur.get("precipitation", 0)
    code   = cur.get("weathercode", 0)
    condition = WMO_CODES.get(code, "unknown conditions")
    rain_note = f", with {precip} mm of precipitation" if precip and precip > 0 else ""
    return (
        f"In {city_name} it's currently {condition} — {temp}°C, "
        f"humidity {humid}%, wind {wind} km/h{rain_note}."
    )

def find_best_matching_event(query: str):
    events = fetch_upcoming_events()
    if not events:
        return None
    query_lower = query.lower()
    best = None
    best_score = 0
    for e in events:
        title = e.get("summary", "").lower()
        score = sum(1 for word in query_lower.split() if word in title)
        if score > best_score:
            best_score = score
            best = e
    return best if best_score > 0 else None


# Tools our ai agent uses like getting the current time, add events to our calender, list what is in our schedule, get weather information

@tool
def get_current_time() -> str:
    """Returns the current date and time."""
    return datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

@tool
def add_calendar_event(details: str) -> str:
    """Add a calendar event given a description like 'study at 9 pm today'."""
    if not service:
        return "Calendar service unavailable."
    parsed = parse_event_datetime(details)
    if not parsed:
        return "Couldn't understand the time. Try saying something like 'study at 9 pm today'."
    event = {
        "summary": details,
        "start": {"dateTime": parsed.isoformat(), "timeZone": "UTC"},
        "end": {"dateTime": (parsed + datetime.timedelta(hours=1)).isoformat(), "timeZone": "UTC"},
    }
    try:
        service.events().insert(calendarId="primary", body=event).execute()
        return f"Done, added: {details}"
    except Exception as e:
        return f"Failed to add event: {e}"

@tool
def list_schedule(period: str = "next 7 days") -> str:
    """Returns calendar events for today, tomorrow, next week, a weekday, or the next 7 days."""
    start_dt, end_dt, label = resolve_period(period)
    events = fetch_events_for_range(start_dt, end_dt)
    if not events:
        return f"You're free {label} — no events scheduled."
    lines = [f"Here's what you have {label}:"]
    for e in events:
        title = e.get("summary", "No title")
        raw_start = e.get("start", {}).get("dateTime", e.get("start", {}).get("date", ""))
        lines.append(f"  - {title} — {format_event_time(raw_start)}")
    return "\n".join(lines)

@tool
def get_weather(location: str = "") -> str:
    """Returns current weather for a city. Uses DEFAULT_CITY if no location given."""
    city = location.strip() or DEFAULT_CITY
    geo = geocode_city(city)
    if not geo:
        return f"Sorry, I couldn't find weather data for '{city}'."
    lat, lon, city_name = geo
    data = fetch_weather(lat, lon)
    if not data:
        return "Sorry, I couldn't fetch the weather right now."
    return describe_weather(data, city_name)

@tool
def delete_all_events() -> str:
    """Deletes ALL upcoming calendar events for the next 7 days."""
    events = fetch_upcoming_events()
    if not events:
        return "No upcoming events to delete."
    deleted = 0
    for e in events:
        try:
            service.events().delete(calendarId="primary", eventId=e["id"]).execute()
            deleted += 1
        except Exception:
            pass
    return f"Deleted {deleted} event{'s' if deleted != 1 else ''} from your schedule."

@tool
def delete_single_event(event_name: str) -> str:
    """Delete a specific event by name, e.g. 'study' or 'meeting'."""
    if not service:
        return "Calendar service unavailable."
    event = find_best_matching_event(event_name)
    if not event:
        return f"Couldn't find an event matching '{event_name}'."
    title = event.get("summary", "that event")
    raw_start = event.get("start", {}).get("dateTime", event.get("start", {}).get("date", ""))
    try:
        service.events().delete(calendarId="primary", eventId=event["id"]).execute()
        return f"Deleted '{title}' scheduled for {format_event_time(raw_start)}."
    except Exception as e:
        return f"Failed to delete event: {e}"

@tool
def modify_event(instruction: str) -> str:
    """Modify an existing event."""
    if not service:
        return "Calendar service unavailable."

    try:
        parse_response = llm.invoke([
            {"role": "system", "content":
                "Parse a calendar modification instruction. "
                "Return JSON only with keys: "
                "'event_name' (the event to find), "
                "'new_title' (new name if renaming, else null), "
                "'new_time' (new time phrase if rescheduling, else null). "
                "Example: 'change study to 10 pm tomorrow' -> "
                "{\"event_name\": \"study\", \"new_title\": null, \"new_time\": \"10 pm tomorrow\"}"
            },
            {"role": "user", "content": instruction}
        ])
        raw = parse_response.content.strip()
        raw = re.sub(r'```json|```', '', raw).strip()
        parsed = json.loads(raw)
    except Exception as e:
        return f"Couldn't understand the modification: {e}"

    event_name = parsed.get("event_name", "")
    new_title = parsed.get("new_title")
    new_time = parsed.get("new_time")

    event = find_best_matching_event(event_name)
    if not event:
        return f"Couldn't find an event matching '{event_name}'."

    old_title = event.get("summary", "that event")
    update = {}

    if new_title:
        update["summary"] = new_title

    if new_time:
        new_time_normalized = normalize_text(new_time)
        new_dt = parse_event_datetime(new_time_normalized)
        if not new_dt:
            return f"Found '{old_title}' but couldn't understand the new time '{new_time}'."
        update["start"] = {"dateTime": new_dt.isoformat(), "timeZone": "UTC"}
        update["end"] = {"dateTime": (new_dt + datetime.timedelta(hours=1)).isoformat(), "timeZone": "UTC"}

    if not update:
        return "Nothing to update — please say what to change."

    try:
        service.events().patch(calendarId="primary", eventId=event["id"], body=update).execute()
        changes = []
        if new_title:
            changes.append(f"renamed to '{new_title}'")
        if new_time:
            changes.append(f"moved to {format_event_time(update['start']['dateTime'])}")
        return f"Updated '{old_title}': {' and '.join(changes)}."
    except Exception as e:
        return f"Failed to update event: {e}"


# AI ROUTER

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
chat_history = []

INTENT_SYSTEM = """You are an intent classifier for a voice assistant.
Given a user message, respond with EXACTLY one of these labels and nothing else:
GET_TIME        - user asks what time or date it is
LIST_SCHEDULE   - user asks about their plans, schedule, agenda, events, or calendar
ADD_EVENT       - user wants to add, set, create, or schedule a reminder or event
DELETE_ONE      - user wants to delete or remove a specific single event by name
DELETE_ALL      - user wants to clear or delete everything / all events / their whole schedule
MODIFY_EVENT    - user wants to change, reschedule, rename, or update a specific event
GET_WEATHER     - user asks about weather, temperature, rain, forecast, or climate
CHAT            - anything else

Reply with only the label. No punctuation, no explanation."""

def detect_intent(text: str) -> str:
    response = llm.invoke([
        {"role": "system", "content": INTENT_SYSTEM},
        {"role": "user", "content": text}
    ])
    raw = response.content.strip().upper()
    for label in ["GET_TIME", "LIST_SCHEDULE", "ADD_EVENT", "DELETE_ONE", "DELETE_ALL", "MODIFY_EVENT", "GET_WEATHER", "CHAT"]:
        if label in raw:
            return label
    return "CHAT"

def chat_response(text: str) -> str:
    global chat_history
    chat_history.append({"role": "user", "content": text})
    messages = [
        {"role": "system", "content": "You are a concise voice assistant. Reply in 1-2 sentences."}
    ] + chat_history[-10:]
    response = llm.invoke(messages)
    reply = response.content.strip()
    chat_history.append({"role": "assistant", "content": reply})
    return reply

def extract_event_name(text: str) -> str:
    response = llm.invoke([
        {"role": "system", "content":
            "Extract only the event name from the user's message. "
            "Return just the name, nothing else. "
            "Examples: 'delete my study session' -> 'study', "
            "'remove the meeting' -> 'meeting', "
            "'cancel gym at 6pm' -> 'gym'"},
        {"role": "user", "content": text}
    ])
    return response.content.strip().lower()

def handle_command(text: str) -> str:
    text = normalize_text(text)
    try:
        intent = detect_intent(text)
        print(f"Intent: {intent}")
    except Exception as e:
        print(f"Intent detection error: {e}")
        intent = "CHAT"

    if intent == "GET_TIME":
        return get_current_time.func()

    elif intent == "LIST_SCHEDULE":
        try:
            period_response = llm.invoke([
                {"role": "system", "content":
                    "Extract the time period from the user's message. "
                    "Return ONLY one of: today, tomorrow, next week, this week, weekend, "
                    "or a weekday name like monday/tuesday/etc. "
                    "If no specific period is mentioned, return: next 7 days. "
                    "No punctuation or extra words."},
                {"role": "user", "content": text}
            ])
            period = period_response.content.strip().lower()
        except Exception:
            period = "next 7 days"
        return list_schedule.func(period)

    elif intent == "GET_WEATHER":
        try:
            loc_response = llm.invoke([
                {"role": "system", "content":
                    "Extract the city or location from the user's message. "
                    "Return ONLY the city name, nothing else. "
                    "If no location is mentioned, return an empty string."},
                {"role": "user", "content": text}
            ])
            location = loc_response.content.strip()
        except Exception:
            location = ""
        return get_weather.func(location)


    elif intent == "ADD_EVENT":
        try:
            extraction = llm.invoke([
                {"role": "system", "content":
                    "Extract the event name and time from the user's message. "
                    "Return only a short phrase like 'study at 9 pm today' or 'meeting at 3 pm tomorrow'. "
                    "No extra words."},
                {"role": "user", "content": text}
            ])
            details = normalize_text(extraction.content.strip())
            return add_calendar_event.func(details)
        except Exception as e:
            return f"Sorry, I couldn't add that event: {e}"

    elif intent == "DELETE_ONE":
        event_name = extract_event_name(text)
        return delete_single_event.func(event_name)

    elif intent == "DELETE_ALL":
        return delete_all_events.func()

    elif intent == "MODIFY_EVENT":
        return modify_event.func(text)

    else:
        return chat_response(text)


# VISUALIZER

def run_visualizer():
    os.environ['SDL_VIDEO_WINDOW_POS'] = f'{1240}, {510}'
    pygame.init()
    screen = pygame.display.set_mode((300, 200), pygame.NOFRAME)
    video = cv2.VideoCapture(VIDEO_PATH)
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
        ret, frame = video.read()
        if not ret:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = cv2.resize(frame, (300, 200))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        screen.blit(surface, (0, 0))
        pygame.display.update()
        clock.tick(60)


# VOICE ASSISTANT

class VoiceAssistant:
    def __init__(self):
        self.model = Model(VOSK_MODEL_PATH)
        self.recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)
        self.audio_queue = queue.Queue()
        pygame.mixer.init()

    def callback(self, indata, frames, time_info, status):
        if self.audio_queue.qsize() < 10:
            self.audio_queue.put(bytes(indata))

    def speak(self, text):
        def play():
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    path = fp.name
                
                async def get_audio():
                    #A natural, clear female voice
                    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
                    await communicate.save(path)
                
                asyncio.run(get_audio())

                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                pygame.mixer.music.unload()
                os.remove(path)
                
            except Exception as e:
                print("TTS playback error:", e)

        # Run in a daemon thread so it doesn't block your visualizer or listener
        threading.Thread(target=play, daemon=True).start()

    def process_command(self, text):
        print("You:", text)
        try:
            reply = handle_command(text)
            print("AI:", reply)
            self.speak(reply)
        except Exception as e:
            print(f"Error processing command: {e}")
            self.speak("Sorry, I had trouble with that. Please try again.")

    def start(self):
        print(f"Say '{TRIGGER_WORD}' to wake me up")
        listening = False
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=8000,
            dtype="int16",
            channels=CHANNELS,
            callback=self.callback
        ):
            while True:
                data = self.audio_queue.get()
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "").strip().lower()
                    if not text:
                        continue
                    if not listening:
                        if TRIGGER_WORD in text:
                            playsound('Assets\\mixit.wav')
                            print("Listening...")
                            listening = True
                        continue
                    if listening:
                        self.process_command(text)
                        listening = False
                        print("Sleeping...")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    threading.Thread(target=run_visualizer, daemon=True).start()
    VoiceAssistant().start()