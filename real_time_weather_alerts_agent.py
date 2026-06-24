"""
==============================================================================
  Real_Time_Weather_Alerts_Agent
  ReAct Loop  |  Google GenAI SDK  |  Gemini 2.0 Flash
  No paid API needed for weather data (Nominatim + Open-Meteo are free)
==============================================================================

SETUP (one-time):
    pip install google-genai python-dotenv

Create a .env file next to this script with:
    GEMINI_API_KEY=your_key_here

Get a free Gemini API key at: https://aistudio.google.com/app/apikey

USAGE:
    python real_time_weather_alerts_agent.py
    Then type any weather question, e.g.:
        "Are there any severe weather alerts for Miami, Florida?"
        "What is the weather forecast for Chicago, Illinois?"
        "Should I be worried about weather in Seattle, Washington?"
==============================================================================
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types

# ─────────────────────────────────────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY not found. "
        "Add it to a .env file or set it as an environment variable."
    )

# ⚠  Replace with your real e-mail — Nominatim requires a valid contact
CONTACT_EMAIL = "krprls@gmail.com"

NOMINATIM_SEARCH_URL   = "https://nominatim.openstreetmap.org/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
USER_AGENT = f"WeatherAlertAgent/1.0 (contact: {CONTACT_EMAIL})"

NOMINATIM_MIN_INTERVAL = 1.1   # seconds between Nominatim calls (be polite)
_last_nominatim_ts: float = 0.0

MAX_REACT_ITERATIONS = 10      # safety ceiling on the agent loop

# Thresholds that trigger a severe-weather alert
THRESHOLDS: Dict[str, Any] = {
    "wind_kph_warning":           50.0,
    "wind_kph_critical":          80.0,
    "precip_mm_next_hour_warning":10.0,
    "precip_mm_next_hour_critical":25.0,
    "precip_mm_daily_warning":    30.0,
    "precip_mm_daily_critical":   60.0,
    "temp_c_high_warning":        35.0,
    "temp_c_high_critical":       38.0,
    "temp_c_low_warning":         -5.0,
    "temp_c_low_critical":        -10.0,
    "lookahead_hours":            24,
    "lookahead_days":             3,
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SevereAlert:
    severity: str        # "warning" | "critical"
    kind: str            # "wind" | "precip" | "heat" | "cold"
    when: str
    message: str
    facts: Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# 2. HTTP HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _http_get_json(
    url: str,
    params: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    timeout_s: int = 30,
    retries: int = 2,
) -> Any:
    """Simple HTTP GET that returns parsed JSON with retry logic."""
    qs       = urllib.parse.urlencode(params)
    full_url = f"{url}?{qs}"
    req      = urllib.request.Request(full_url, headers=headers or {}, method="GET")

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(1.0 + attempt * 1.5)
                continue
            raise
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.0 + attempt * 1.5)
                continue
            raise
    raise RuntimeError(f"HTTP request failed after retries: {last_err!r}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. TOOL IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _get_lat_long(city: str, state: str, country: str = "USA") -> Tuple[float, float]:
    """Geocode a city/state to (lat, lon) via Nominatim (OpenStreetMap). Free."""
    global _last_nominatim_ts

    if not CONTACT_EMAIL or "@" not in CONTACT_EMAIL:
        raise ValueError("Set CONTACT_EMAIL to a real e-mail (Nominatim policy).")

    # Politely rate-limit Nominatim
    gap = NOMINATIM_MIN_INTERVAL - (time.time() - _last_nominatim_ts)
    if gap > 0:
        time.sleep(gap)

    q = f"{city}, {state}, {country}".strip(", ")
    data = _http_get_json(
        NOMINATIM_SEARCH_URL,
        params={"q": q, "format": "jsonv2", "limit": 1, "email": CONTACT_EMAIL},
        headers={"User-Agent": USER_AGENT, "From": CONTACT_EMAIL, "Accept": "application/json"},
        timeout_s=30,
        retries=1,
    )
    _last_nominatim_ts = time.time()

    if not data:
        raise ValueError(f"Location not found: {q!r}")
    return float(data[0]["lat"]), float(data[0]["lon"])


def _get_extended_weather_forecast(lat: float, lon: float, forecast_days: int = 7) -> Dict[str, Any]:
    """Fetch hourly + daily forecast from Open-Meteo (100 % free, no key needed)."""
    return _http_get_json(
        OPEN_METEO_FORECAST_URL,
        params={
            "latitude":        lat,
            "longitude":       lon,
            "timezone":        "auto",
            "forecast_days":   int(forecast_days),
            "daily":           ",".join([
                "temperature_2m_max", "temperature_2m_min",
                "precipitation_sum",  "wind_speed_10m_max",
            ]),
            "hourly":          ",".join([
                "temperature_2m", "precipitation", "wind_speed_10m",
            ]),
            "current_weather": "true",
        },
        headers={"Accept": "application/json"},
        timeout_s=30,
        retries=2,
    )


def _severity(value: float, warn: float, crit: float) -> Optional[str]:
    if value >= crit:
        return "critical"
    if value >= warn:
        return "warning"
    return None


def _evaluate_severe_weather(
    payload: Dict[str, Any],
    thresholds: Dict[str, Any],
) -> List[SevereAlert]:
    """Compare forecast data against thresholds and return a list of SevereAlerts."""
    alerts: List[SevereAlert] = []
    hourly  = payload.get("hourly")  or {}
    daily   = payload.get("daily")   or {}
    current = payload.get("current_weather") or {}

    cur_time = current.get("time") or datetime.now(timezone.utc).isoformat()

    # ── Current conditions ──────────────────────────────────────────────────
    cur_wind = current.get("windspeed")
    if isinstance(cur_wind, (int, float)):
        sev = _severity(float(cur_wind), thresholds["wind_kph_warning"], thresholds["wind_kph_critical"])
        if sev:
            alerts.append(SevereAlert(sev, "wind", str(cur_time),
                f"High wind right now: {cur_wind:.1f} km/h",
                {"wind_kph": float(cur_wind), "source": "current"}))

    cur_temp = current.get("temperature")
    if isinstance(cur_temp, (int, float)):
        sev = _severity(float(cur_temp), thresholds["temp_c_high_warning"], thresholds["temp_c_high_critical"])
        if sev:
            alerts.append(SevereAlert(sev, "heat", str(cur_time),
                f"High temperature right now: {cur_temp:.1f}°C",
                {"temp_c": float(cur_temp), "source": "current"}))
        if float(cur_temp) <= thresholds["temp_c_low_critical"]:
            alerts.append(SevereAlert("critical", "cold", str(cur_time),
                f"Extreme cold right now: {cur_temp:.1f}°C",
                {"temp_c": float(cur_temp), "source": "current"}))
        elif float(cur_temp) <= thresholds["temp_c_low_warning"]:
            alerts.append(SevereAlert("warning", "cold", str(cur_time),
                f"Low temperature right now: {cur_temp:.1f}°C",
                {"temp_c": float(cur_temp), "source": "current"}))

    # ── Hourly lookahead ────────────────────────────────────────────────────
    h_times = hourly.get("time") or []
    h_prec  = hourly.get("precipitation") or []
    h_wind  = hourly.get("wind_speed_10m") or []
    h_temp  = hourly.get("temperature_2m") or []
    n_h = min(thresholds["lookahead_hours"], len(h_times), len(h_prec), len(h_wind), len(h_temp))

    for i in range(n_h):
        when = str(h_times[i])
        try:
            p   = float(h_prec[i])
            sev = _severity(p, thresholds["precip_mm_next_hour_warning"], thresholds["precip_mm_next_hour_critical"])
            if sev:
                alerts.append(SevereAlert(sev, "precip", when,
                    f"Heavy precipitation: {p:.1f} mm/h",
                    {"precip_mm": p, "window": "1h", "source": "hourly"}))
        except Exception:
            pass
        try:
            w   = float(h_wind[i])
            sev = _severity(w, thresholds["wind_kph_warning"], thresholds["wind_kph_critical"])
            if sev:
                alerts.append(SevereAlert(sev, "wind", when,
                    f"High wind: {w:.1f} km/h",
                    {"wind_kph": w, "source": "hourly"}))
        except Exception:
            pass
        try:
            t    = float(h_temp[i])
            sev  = _severity(t, thresholds["temp_c_high_warning"], thresholds["temp_c_high_critical"])
            if sev:
                alerts.append(SevereAlert(sev, "heat", when,
                    f"High temperature: {t:.1f}°C", {"temp_c": t, "source": "hourly"}))
            if t <= thresholds["temp_c_low_critical"]:
                alerts.append(SevereAlert("critical", "cold", when,
                    f"Extreme cold: {t:.1f}°C", {"temp_c": t, "source": "hourly"}))
            elif t <= thresholds["temp_c_low_warning"]:
                alerts.append(SevereAlert("warning", "cold", when,
                    f"Low temperature: {t:.1f}°C", {"temp_c": t, "source": "hourly"}))
        except Exception:
            pass

    # ── Daily lookahead ─────────────────────────────────────────────────────
    d_times = daily.get("time") or []
    d_prec  = daily.get("precipitation_sum") or []
    d_tmax  = daily.get("temperature_2m_max") or []
    d_tmin  = daily.get("temperature_2m_min") or []
    d_wmax  = daily.get("wind_speed_10m_max") or []
    n_d = min(thresholds["lookahead_days"],
              len(d_times), len(d_prec), len(d_tmax), len(d_tmin), len(d_wmax))

    for i in range(n_d):
        when = str(d_times[i])
        try:
            p   = float(d_prec[i])
            sev = _severity(p, thresholds["precip_mm_daily_warning"], thresholds["precip_mm_daily_critical"])
            if sev:
                alerts.append(SevereAlert(sev, "precip", when,
                    f"High daily precip: {p:.1f} mm", {"precip_mm": p, "window": "day", "source": "daily"}))
        except Exception:
            pass
        try:
            w   = float(d_wmax[i])
            sev = _severity(w, thresholds["wind_kph_warning"], thresholds["wind_kph_critical"])
            if sev:
                alerts.append(SevereAlert(sev, "wind", when,
                    f"High max wind: {w:.1f} km/h", {"wind_kph": w, "source": "daily"}))
        except Exception:
            pass
        try:
            tmax = float(d_tmax[i])
            sev  = _severity(tmax, thresholds["temp_c_high_warning"], thresholds["temp_c_high_critical"])
            if sev:
                alerts.append(SevereAlert(sev, "heat", when,
                    f"High daily max temp: {tmax:.1f}°C",
                    {"temp_c": tmax, "metric": "daily_max", "source": "daily"}))
        except Exception:
            pass
        try:
            tmin = float(d_tmin[i])
            if tmin <= thresholds["temp_c_low_critical"]:
                alerts.append(SevereAlert("critical", "cold", when,
                    f"Extreme daily min temp: {tmin:.1f}°C",
                    {"temp_c": tmin, "metric": "daily_min", "source": "daily"}))
            elif tmin <= thresholds["temp_c_low_warning"]:
                alerts.append(SevereAlert("warning", "cold", when,
                    f"Low daily min temp: {tmin:.1f}°C",
                    {"temp_c": tmin, "metric": "daily_min", "source": "daily"}))
        except Exception:
            pass

    # De-duplicate: keep highest severity per (kind, when, message-prefix)
    rank = {"warning": 1, "critical": 2}
    dedup: Dict[Tuple, SevereAlert] = {}
    for a in alerts:
        key  = (a.kind, a.when, a.message.split(":")[0])
        prev = dedup.get(key)
        if prev is None or rank.get(a.severity, 0) > rank.get(prev.severity, 0):
            dedup[key] = a

    result = list(dedup.values())
    result.sort(key=lambda a: (-rank.get(a.severity, 0), a.when, a.kind))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. TOOL DISPATCHER  ─  called by the agent loop
# ─────────────────────────────────────────────────────────────────────────────

def get_weather_and_alerts(city: str, state: str, country: str = "USA") -> str:
    """
    Full pipeline: geocode → forecast → evaluate → return JSON string.
    This is what the Gemini function-call invokes.
    """
    try:
        lat, lon = _get_lat_long(city, state, country)
    except Exception as e:
        return json.dumps({"error": f"Geocoding failed: {e}"})

    try:
        forecast = _get_extended_weather_forecast(lat, lon, forecast_days=7)
    except Exception as e:
        return json.dumps({"error": f"Forecast fetch failed: {e}"})

    current = forecast.get("current_weather") or {}
    daily   = forecast.get("daily") or {}

    # Build a compact daily summary (next 7 days)
    daily_summary = []
    times = daily.get("time") or []
    tmax  = daily.get("temperature_2m_max") or []
    tmin  = daily.get("temperature_2m_min") or []
    psum  = daily.get("precipitation_sum") or []
    wmax  = daily.get("wind_speed_10m_max") or []

    for i in range(min(7, len(times))):
        daily_summary.append({
            "date":          times[i],
            "temp_max_c":    tmax[i] if i < len(tmax) else None,
            "temp_min_c":    tmin[i] if i < len(tmin) else None,
            "precip_mm":     psum[i] if i < len(psum) else None,
            "wind_max_kph":  wmax[i] if i < len(wmax) else None,
        })

    # Evaluate severe weather
    alerts = _evaluate_severe_weather(forecast, THRESHOLDS)
    alerts_list = [asdict(a) for a in alerts]

    result = {
        "location": {"city": city, "state": state, "country": country,
                     "lat": lat, "lon": lon},
        "current_weather": current,
        "daily_forecast":  daily_summary,
        "severe_alerts":   alerts_list,
        "alert_count":     len(alerts_list),
        "thresholds_used": THRESHOLDS,
    }
    return json.dumps(result, indent=2)


def get_location_coordinates(city: str, state: str, country: str = "USA") -> str:
    """
    Lightweight geocoding-only tool. Useful when the agent just needs
    coordinates before deciding whether to fetch a full forecast.
    """
    try:
        lat, lon = _get_lat_long(city, state, country)
        return json.dumps({"city": city, "state": state, "country": country,
                           "lat": lat, "lon": lon})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# 5. GEMINI TOOL SCHEMAS  ─  what we register with the model
# ─────────────────────────────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="get_weather_and_alerts",
            description=(
                "Fetches the real-time weather forecast and evaluates it against "
                "severe-weather thresholds for a given city. "
                "Returns current conditions, a 7-day daily summary, and a list of "
                "any active severe-weather alerts (wind, precipitation, heat, cold)."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "city":    types.Schema(type=types.Type.STRING,
                                           description="City name, e.g. 'Miami'"),
                    "state":   types.Schema(type=types.Type.STRING,
                                           description="State or province, e.g. 'Florida'"),
                    "country": types.Schema(type=types.Type.STRING,
                                           description="Country, default 'USA'"),
                },
                required=["city", "state"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_location_coordinates",
            description=(
                "Geocodes a city/state to latitude and longitude coordinates. "
                "Use this if you only need the location without a full forecast."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "city":    types.Schema(type=types.Type.STRING,
                                           description="City name"),
                    "state":   types.Schema(type=types.Type.STRING,
                                           description="State or province"),
                    "country": types.Schema(type=types.Type.STRING,
                                           description="Country, default 'USA'"),
                },
                required=["city", "state"],
            ),
        ),
    ])
]


# ─────────────────────────────────────────────────────────────────────────────
# 6. TOOL REGISTRY  ─  maps function name → Python callable
# ─────────────────────────────────────────────────────────────────────────────

TOOL_REGISTRY: Dict[str, Any] = {
    "get_weather_and_alerts":   get_weather_and_alerts,
    "get_location_coordinates": get_location_coordinates,
}


# ─────────────────────────────────────────────────────────────────────────────
# 7. THE REACT AGENT LOOP
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Real_Time_Weather_Alerts_Agent, an expert meteorological assistant.

Your job:
1. THINK about what the user is asking.
2. ACT by calling a tool to get real weather data.
3. OBSERVE the tool result.
4. Repeat steps 1-3 if you need more information.
5. ANSWER with a clear, helpful final response once you have all the data.

Always use real weather data — never make up forecasts or conditions.
When severe alerts exist, highlight them clearly with their severity level.
Format temperatures in both Celsius and Fahrenheit for US audiences.
"""

def celsius_to_fahrenheit(c: float) -> float:
    return round(c * 9 / 5 + 32, 1)


def run_agent(user_query: str) -> str:
    """
    The ReAct loop:
      Thought → Action (tool call) → Observation → … → Final Answer
    """
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Conversation history: list of types.Content objects
    conversation: List[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=user_query)])
    ]

    print(f"\n{'='*70}")
    print(f"  USER: {user_query}")
    print(f"{'='*70}")

    for iteration in range(1, MAX_REACT_ITERATIONS + 1):
        print(f"\n── Iteration {iteration} ──────────────────────────────────────────────")

        # ── THINK / ACT: send conversation to Gemini ─────────────────────
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=conversation,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                temperature=0.1,   # low temperature for factual tasks
            ),
        )

        candidate = response.candidates[0]
        finish    = candidate.finish_reason

        # Collect all parts from this turn
        text_parts:     List[str]                  = []
        function_calls: List[types.FunctionCall]   = []

        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)
            if hasattr(part, "function_call") and part.function_call:
                function_calls.append(part.function_call)

        # Echo any reasoning text the model produced
        if text_parts:
            reasoning = "\n".join(text_parts).strip()
            print(f"\n  [THOUGHT]\n{reasoning}")

        # ── No tool call → model has produced its final answer ─────────
        if not function_calls:
            final_answer = "\n".join(text_parts).strip()
            print(f"\n{'='*70}")
            print(f"  AGENT FINAL ANSWER:\n{final_answer}")
            print(f"{'='*70}\n")
            return final_answer

        # Append model's turn (including the function-call part) to history
        conversation.append(
            types.Content(role="model", parts=candidate.content.parts)
        )

        # ── OBSERVE: execute every tool the model requested ──────────────
        tool_result_parts: List[types.Part] = []

        for fc in function_calls:
            fn_name = fc.name
            fn_args = dict(fc.args) if fc.args else {}

            print(f"\n  [ACTION] Calling tool: {fn_name}")
            print(f"           Args: {json.dumps(fn_args, indent=11)}")

            if fn_name not in TOOL_REGISTRY:
                observation = json.dumps({"error": f"Unknown tool: {fn_name}"})
            else:
                try:
                    observation = TOOL_REGISTRY[fn_name](**fn_args)
                except Exception as exc:
                    observation = json.dumps({"error": str(exc)})

            # Pretty-print a truncated observation for the terminal
            try:
                obs_data = json.loads(observation)
                obs_preview = json.dumps(obs_data, indent=2)
                # Truncate long outputs so the terminal stays readable
                if len(obs_preview) > 1200:
                    obs_preview = obs_preview[:1200] + "\n  ... [truncated for display]"
            except Exception:
                obs_preview = observation[:1200]

            print(f"\n  [OBSERVATION]\n{obs_preview}")

            tool_result_parts.append(
                types.Part.from_function_response(
                    name=fn_name,
                    response={"result": observation},
                )
            )

        # Append the tool results as a "user" turn (Gemini convention)
        conversation.append(
            types.Content(role="user", parts=tool_result_parts)
        )

    # Safety net: max iterations reached
    fallback = "I was unable to complete the analysis within the allowed steps."
    print(f"\n  [AGENT] {fallback}")
    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# 8. INTERACTIVE CLI
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLE_QUERIES = [
    "Are there any severe weather alerts for Miami, Florida?",
    "What is the 7-day weather forecast for Chicago, Illinois?",
    "Should I be worried about the weather in Phoenix, Arizona this week?",
    "Check weather alerts for Seattle, Washington",
    "Is there any extreme cold warning for Minneapolis, Minnesota?",
]


def main() -> None:
    print("\n" + "="*70)
    print("   Real_Time_Weather_Alerts_Agent")
    print("   Powered by: Gemini 2.0 Flash  |  Open-Meteo  |  Nominatim")
    print("   ReAct Loop: Thought → Action → Observation → Answer")
    print("="*70)
    print("\nExample questions you can ask:")
    for i, q in enumerate(EXAMPLE_QUERIES, 1):
        print(f"  {i}. {q}")
    print("\nType 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Allow user to pick an example by number
        if user_input.isdigit() and 1 <= int(user_input) <= len(EXAMPLE_QUERIES):
            user_input = EXAMPLE_QUERIES[int(user_input) - 1]
            print(f"Running: {user_input}")

        try:
            run_agent(user_input)
        except Exception as exc:
            print(f"\n[ERROR] {exc}")


if __name__ == "__main__":
    main()
