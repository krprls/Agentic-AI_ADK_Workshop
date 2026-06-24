# Real-Time Weather Alerts Agent
 
ReAct Loop  |  Google GenAI SDK  |  Gemini 2.5 Flash
  No paid API needed for weather data (Nominatim + Open-Meteo are free)

A ReAct-style AI agent that answers natural-language weather questions using **real, live data** — no mocked or hallucinated forecasts. Built with the Google GenAI SDK (Gemini 2.5 Flash) and two free, no-key APIs: [Nominatim](https://nominatim.openstreetmap.org/) (geocoding) and [Open-Meteo](https://open-meteo.com/) (forecasts).

The agent follows a classic **Thought → Action → Observation → Answer** loop: it reasons about the user's question, calls a weather tool, reads the result, and either calls another tool or produces a final answer — for up to 10 iterations.

## Features

- **ReAct agent loop** built directly on the Gemini function-calling API (no agent framework dependency)
- **Free data sources** — geocoding via Nominatim, forecasts via Open-Meteo (no API key, no cost)
- **Severe weather detection** across four categories: wind, precipitation, heat, and cold
- **Configurable thresholds** for warning vs. critical severity, with lookahead over the next 24 hours and 3 days
- **7-day forecast summary** alongside current conditions
- **Two tools** exposed to the model: a full weather+alerts pipeline, and a lightweight geocoding-only lookup
- **Interactive CLI** with example questions you can run by typing their number

## How it works

```
User question
   │
   ▼
Gemini (Thought) ──calls──▶ get_weather_and_alerts(city, state, country)
   │                              │
   │                              ▼
   │                     Nominatim (geocode) → Open-Meteo (forecast)
   │                              │
   │                              ▼
   │                     Evaluate against severity thresholds
   │                              │
   ◀──────── Observation (JSON) ──┘
   │
   ▼
Gemini (Thought) → Final Answer
```

The agent registers two tools with Gemini:

| Tool | Purpose |
|---|---|
| `get_weather_and_alerts` | Full pipeline: geocode → 7-day forecast → severe-weather evaluation |
| `get_location_coordinates` | Geocoding only, for when the model just needs lat/lon |

Severe weather is flagged when forecast values cross these thresholds (editable in `THRESHOLDS`):

| Condition | Warning | Critical |
|---|---|---|
| Wind speed | 50 km/h | 80 km/h |
| Precipitation (next hour) | 10 mm | 25 mm |
| Precipitation (daily) | 30 mm | 60 mm |
| High temperature | 35°C | 38°C |
| Low temperature | -5°C | -10°C |

## Setup

**Requirements:** Python 3.9+

```bash
pip install google-genai python-dotenv
```

Create a `.env` file in this directory:

```
GEMINI_API_KEY=your_key_here
```

Get a free Gemini API key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).

> **Note:** Update `CONTACT_EMAIL` in the script to your real email address — Nominatim's usage policy requires a valid contact for its free tier.

## Usage

```bash
python real_time_weather_alerts_agent.py
```

Then ask a question, or type a number to run one of the built-in examples:

```
Are there any severe weather alerts for Miami, Florida?
What is the 7-day weather forecast for Chicago, Illinois?
Should I be worried about the weather in Phoenix, Arizona this week?
Check weather alerts for Seattle, Washington
Is there any extreme cold warning for Minneapolis, Minnesota?
```

Type `quit` or `exit` to stop.

## Sample output

![Example run](homework1image.png)

## Project structure

```
real_time_weather_alerts_agent.py   # agent loop, tools, and CLI
.env                                 # GEMINI_API_KEY (not committed)
```

## Notes

- Nominatim requests are rate-limited client-side (≥1.1s between calls) to comply with its usage policy.
- HTTP requests retry on transient failures (429/500/502/503/504) with backoff.
- `MAX_REACT_ITERATIONS` (default 10) caps the agent loop as a safety net against runaway tool calls.
