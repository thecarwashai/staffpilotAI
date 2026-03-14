# staffpilot.py
# StaffPilot AI - Instant Forecast Mode
# Architected by Gopi Chand
#
# Install:
#   pip install streamlit requests pandas
#
# Primary weather: Open-Meteo (free, no key required).
# Fallback weather: WeatherAPI.com (free tier, requires WEATHERAPI_KEY in secrets).
#   Get a free key at https://www.weatherapi.com/signup.aspx
#   Add to .streamlit/secrets.toml:  WEATHERAPI_KEY = "your_key_here"

import json
import math
from datetime import datetime, date, timedelta
from typing import Dict, Any, Tuple, List, Optional

import pandas as pd
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="StaffPilot AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark operational dashboard, monospace + clean sans, amber accent
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

/* ── Root tokens ── */
:root {
    --bg:        #0a0c0f;
    --surface:   #111418;
    --surface2:  #181c22;
    --border:    #232830;
    --accent:    #f5a623;
    --accent2:   #e8870a;
    --green:     #22c55e;
    --red:       #ef4444;
    --blue:      #3b82f6;
    --text:      #e8eaf0;
    --muted:     #6b7280;
    --mono:      'IBM Plex Mono', monospace;
    --sans:      'IBM Plex Sans', sans-serif;
}

/* ── Global resets ── */
html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* ── Top nav bar ── */
.sp-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 48px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 100;
}
.sp-logo {
    display: flex;
    align-items: center;
    gap: 12px;
}
.sp-logo-mark {
    width: 36px; height: 36px;
    background: var(--accent);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; font-family: var(--mono); font-weight: 600;
    color: #000;
}
.sp-logo-text {
    font-family: var(--mono);
    font-size: 16px;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: var(--text);
}
.sp-logo-sub {
    font-family: var(--sans);
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 2px;
}
.sp-badge {
    font-family: var(--mono);
    font-size: 11px;
    background: rgba(245,166,35,0.12);
    color: var(--accent);
    border: 1px solid rgba(245,166,35,0.30);
    border-radius: 20px;
    padding: 4px 14px;
    letter-spacing: 0.06em;
}

/* ── Page wrapper ── */
.sp-page {
    padding: 40px 48px 80px;
    max-width: 1280px;
    margin: 0 auto;
}

/* ── Section heading ── */
.sp-section-title {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 6px;
}
.sp-section-h2 {
    font-family: var(--sans);
    font-size: 26px;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 4px;
    line-height: 1.2;
}
.sp-section-desc {
    font-size: 14px;
    color: var(--muted);
    line-height: 1.6;
    max-width: 640px;
    margin-bottom: 32px;
}

/* ── Input card ── */
.sp-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 20px;
}
.sp-card-title {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sp-card-title::before {
    content: '';
    display: inline-block;
    width: 3px; height: 14px;
    background: var(--accent);
    border-radius: 2px;
}

/* ── Nuke every white/light surface Streamlit injects ── */
.stApp, .stApp > div, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"], [data-testid="block-container"],
[data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"],
section.main, .main .block-container,
div[data-testid="stForm"], div[data-testid="stColumn"],
div[class*="css"] { background-color: transparent !important; }

/* Force the outermost shell dark */
.stApp { background-color: var(--bg) !important; }

/* ── All text globally ── */
p, span, div, label, h1, h2, h3, h4, li {
    color: var(--text) !important;
}

/* ── Text input ── */
div[data-testid="stTextInput"] > div,
div[data-testid="stTextInput"] > div > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
}
div[data-testid="stTextInput"] input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
    caret-color: var(--accent) !important;
}
div[data-testid="stTextInput"] input::placeholder { color: var(--muted) !important; opacity: 1 !important; }
div[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(245,166,35,0.12) !important;
    outline: none !important;
}

/* ── Number input ── */
div[data-testid="stNumberInput"] > div,
div[data-testid="stNumberInput"] > div > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
}
div[data-testid="stNumberInput"] input {
    background: var(--surface2) !important;
    border: none !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 14px !important;
    caret-color: var(--accent) !important;
}
div[data-testid="stNumberInput"] input:focus { outline: none !important; box-shadow: none !important; }
/* stepper +/- buttons */
div[data-testid="stNumberInput"] button {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
}
div[data-testid="stNumberInput"] button:hover { background: var(--border) !important; }
div[data-testid="stNumberInput"] button svg { fill: var(--text) !important; stroke: var(--text) !important; }

/* ── Widget labels ── */
label[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] span {
    font-family: var(--sans) !important;
    font-size: 13px !important;
    color: var(--muted) !important;
    margin-bottom: 4px !important;
}

/* ── Spinner / status ── */
[data-testid="stStatusWidget"], .stSpinner > div {
    background: transparent !important;
    color: var(--muted) !important;
}
.stSpinner svg { stroke: var(--accent) !important; }

/* ── Selectbox / dropdown ── */
div[data-testid="stSelectbox"] > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
div[data-testid="stSelectbox"] svg { fill: var(--muted) !important; }

/* ── Alert / info / warning boxes ── */
[data-testid="stAlert"], div[role="alert"],
.stAlert > div, .element-container .stAlert {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

/* ── Expander ── */
[data-testid="stExpander"] { background: var(--surface) !important; border-color: var(--border) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* ── Generate button ── */
div.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: var(--mono) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 32px !important;
    cursor: pointer !important;
    transition: background 0.15s, transform 0.1s !important;
    width: 100% !important;
    margin-top: 8px !important;
}
div.stButton > button:hover {
    background: var(--accent2) !important;
    transform: translateY(-1px) !important;
}

/* ── Metric strip ── */
.sp-metrics-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 28px;
}
.sp-metric {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
}
.sp-metric::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent);
}
.sp-metric.green::after { background: var(--green); }
.sp-metric.red::after   { background: var(--red); }
.sp-metric.blue::after  { background: var(--blue); }

.sp-metric-label {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.sp-metric-value {
    font-family: var(--mono);
    font-size: 28px;
    font-weight: 600;
    color: var(--text);
    line-height: 1;
}
.sp-metric-sub {
    font-size: 12px;
    color: var(--muted);
    margin-top: 4px;
}

/* ── Forecast table ── */
.sp-table-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 24px;
}
.sp-table-header {
    padding: 18px 24px 14px;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sp-table-header::before {
    content: '';
    display: inline-block;
    width: 3px; height: 14px;
    background: var(--accent);
    border-radius: 2px;
}
table.sp-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}
table.sp-table th {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 10px 20px;
    text-align: left;
    background: var(--surface2);
    border-bottom: 1px solid var(--border);
}
table.sp-table td {
    padding: 14px 20px;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 13px;
    color: var(--text);
    vertical-align: middle;
}
table.sp-table tr:last-child td { border-bottom: none; }
table.sp-table tr:hover td { background: var(--surface2); }

/* Day name */
.day-name { font-weight: 600; color: var(--text); }
.day-date { font-size: 11px; color: var(--muted); margin-top: 2px; }

/* Cars bar */
.cars-bar-wrap { display: flex; align-items: center; gap: 10px; }
.cars-bar-bg {
    flex: 1; height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
    max-width: 100px;
}
.cars-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: var(--accent);
}
.cars-val { min-width: 42px; text-align: right; }

/* Staff pips */
.staff-pips { display: flex; gap: 4px; }
.pip {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: var(--accent);
}
.pip.empty {
    background: transparent;
    border: 1px solid var(--border);
}

/* Weather pill */
.wx-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4px 10px;
    font-size: 11px;
    color: var(--text);
    white-space: nowrap;
}
.wx-pill.rain   { border-color: #3b82f6; color: #93c5fd; background: rgba(59,130,246,0.08); }
.wx-pill.clear  { border-color: #22c55e; color: #86efac; background: rgba(34,197,94,0.08); }
.wx-pill.cold   { border-color: #818cf8; color: #c7d2fe; background: rgba(129,140,248,0.08); }
.wx-pill.heat   { border-color: #f97316; color: #fdba74; background: rgba(249,115,22,0.08); }

/* Impact badge */
.impact-neg { color: var(--red); font-weight: 600; }
.impact-pos { color: var(--green); font-weight: 600; }
.impact-neu { color: var(--muted); }

/* ── Summary panel ── */
.sp-summary {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 24px;
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 24px;
}
.sp-summary-item {}
.sp-summary-item-label {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.sp-summary-item-val {
    font-family: var(--sans);
    font-size: 16px;
    font-weight: 600;
    color: var(--text);
}

/* ── Confidence banner ── */
.sp-confidence {
    display: flex;
    align-items: center;
    gap: 14px;
    background: rgba(245,166,35,0.07);
    border: 1px solid rgba(245,166,35,0.22);
    border-radius: 10px;
    padding: 14px 20px;
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 28px;
}
.sp-confidence strong { color: var(--accent); }
.sp-conf-icon { font-size: 18px; }

/* ── Disclaimer ── */
.sp-disclaimer {
    font-size: 12px;
    color: var(--muted);
    line-height: 1.6;
    padding: 14px 20px;
    background: var(--surface2);
    border-left: 3px solid var(--border);
    border-radius: 0 8px 8px 0;
    margin-top: 12px;
}

/* ── Error ── */
.sp-error {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 10px;
    padding: 16px 20px;
    color: #fca5a5;
    font-size: 13px;
    font-family: var(--mono);
}

/* ── Hide native Streamlit chart (we use custom SVG) ── */
[data-testid="stVegaLiteChart"] { display: none !important; }

/* ── Column gaps ── */
[data-testid="stHorizontalBlock"] { gap: 16px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MAX_STAFF = 5

DOW_MULT = {
    "Monday":    0.72,
    "Tuesday":   0.78,
    "Wednesday": 0.85,
    "Thursday":  0.92,
    "Friday":    1.05,
    "Saturday":  1.28,
    "Sunday":    1.15,
}

QUICK_CPSH = 9.5

# ─────────────────────────────────────────────────────────────────────────────
# WEATHER HELPERS  (Primary: Open-Meteo  |  Fallback: WeatherAPI.com)
# ─────────────────────────────────────────────────────────────────────────────

HEADERS = {"User-Agent": "StaffPilotAI/1.0 (streamlit-app; contact@staffpilot.ai)"}

# WeatherAPI key — optional; enables fallback if Open-Meteo fails
WEATHERAPI_KEY: str = st.secrets.get("WEATHERAPI_KEY", "")


# ── Geocoding ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_lat_lon(location: str) -> Tuple[float, float]:
    """Resolve ZIP / city to (lat, lon) via Open-Meteo geocoding."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    try:
        res = requests.get(url, params={"name": location, "count": 1},
                           headers=HEADERS, timeout=15)
    except requests.exceptions.Timeout:
        raise RuntimeError("Geocoding request timed out. Check your network.")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot reach geocoding service. Check network settings.")
    if res.status_code != 200:
        raise RuntimeError(f"Geocoding API returned HTTP {res.status_code}.")
    results = res.json().get("results") or []
    if not results:
        raise RuntimeError("Location not found. Try a ZIP code or 'City, State'.")
    return float(results[0]["latitude"]), float(results[0]["longitude"])


# ── Primary: Open-Meteo ───────────────────────────────────────────────────────

def _normalize_daily(daily: Dict[str, Any]) -> Dict[str, Any]:
    """Fill nulls and validate shape. Returns cleaned daily dict."""
    n = len(daily["time"])
    prob_raw = daily.get("precipitation_probability_max") or [None] * n
    daily["precipitation_probability_max"] = [
        float(v) if v is not None else 0.0 for v in prob_raw
    ]
    daily["precipitation_sum"] = [
        float(v) if v is not None else 0.0 for v in daily["precipitation_sum"]
    ]
    daily["temperature_2m_max"] = [
        float(v) if v is not None else 65.0 for v in daily["temperature_2m_max"]
    ]
    return daily


def _fetch_open_meteo(lat: float, lon: float) -> Dict[str, Any]:
    """Fetch 7-day forecast from Open-Meteo with up to 2 retries."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_probability_max,precipitation_sum,temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "forecast_days": 7,
        "timezone": "auto",
    }
    last_err = "Unknown error"
    for attempt in range(2):           # try twice before giving up
        try:
            res = requests.get(url, params=params, headers=HEADERS, timeout=20)
        except requests.exceptions.Timeout:
            last_err = f"Timed out (attempt {attempt + 1})"
            continue
        except requests.exceptions.ConnectionError as e:
            last_err = f"Connection error: {e}"
            break                      # connection refused won't fix itself on retry

        if res.status_code != 200:
            try:
                detail = res.json().get("reason", res.text[:120])
            except Exception:
                detail = res.text[:120]
            last_err = f"HTTP {res.status_code}: {detail}"
            break

        daily = res.json().get("daily")
        if not daily:
            last_err = "No daily data in response"
            break

        for k in ["time", "precipitation_sum", "temperature_2m_max"]:
            if k not in daily or not daily[k]:
                last_err = f"Missing field: {k}"
                break
        else:
            if len(daily["time"]) < 7:
                last_err = f"Only returned {len(daily['time'])} days"
                break
            return _normalize_daily(daily)

    raise RuntimeError(f"Open-Meteo: {last_err}")


# ── Fallback 1: wttr.in (free, no key required) ───────────────────────────────

def _fetch_wttr(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetch 7-day forecast from wttr.in JSON API.
    Completely free, no key required — good Streamlit Cloud fallback.
    Normalises to the same dict shape as Open-Meteo.
    """
    url = f"https://wttr.in/{lat},{lon}"
    params = {"format": "j1"}
    try:
        res = requests.get(url, params=params, headers=HEADERS, timeout=20)
    except requests.exceptions.Timeout:
        raise RuntimeError("wttr.in request timed out.")
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Cannot reach wttr.in: {e}")

    if res.status_code != 200:
        raise RuntimeError(f"wttr.in HTTP {res.status_code}")

    try:
        data = res.json()
    except Exception:
        raise RuntimeError("wttr.in returned non-JSON response.")

    weather_days = data.get("weather", [])
    if len(weather_days) < 3:
        raise RuntimeError(f"wttr.in only returned {len(weather_days)} days.")

    # wttr.in returns 3 days max on free tier — extend to 7 by cycling the pattern
    # (good enough for staffing: day 4-7 use day 1-3 weather as an estimate)
    base = weather_days[:]
    while len(base) < 7:
        base.append(base[len(base) % 3])

    times, probs, rains, temps = [], [], [], []
    base_date = datetime.utcnow().date()
    for i, day in enumerate(base[:7]):
        times.append((base_date + timedelta(days=i)).isoformat())
        # hourly entries → derive max temp (F) and total precip (mm→in)
        hourly = day.get("hourly", [])
        temp_max = max((float(h.get("tempF", 65)) for h in hourly), default=65.0)
        precip_mm = sum(float(h.get("precipMM", 0)) for h in hourly)
        chance = max((float(h.get("chanceofrain", 0)) for h in hourly), default=0.0)
        temps.append(temp_max)
        rains.append(round(precip_mm / 25.4, 3))
        probs.append(chance)

    return {
        "time":                           times,
        "precipitation_probability_max":  probs,
        "precipitation_sum":              rains,
        "temperature_2m_max":             temps,
    }


# ── Fallback 2: WeatherAPI.com (requires key) ────────────────────────────────

def _fetch_weatherapi(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetch 7-day forecast from WeatherAPI.com.
    Skipped silently if WEATHERAPI_KEY is not set.
    """
    if not WEATHERAPI_KEY:
        raise RuntimeError("WeatherAPI.com key not configured (WEATHERAPI_KEY missing from secrets).")

    url = "https://api.weatherapi.com/v1/forecast.json"
    params = {"key": WEATHERAPI_KEY, "q": f"{lat},{lon}", "days": 7, "aqi": "no", "alerts": "no"}
    try:
        res = requests.get(url, params=params, headers=HEADERS, timeout=20)
    except requests.exceptions.Timeout:
        raise RuntimeError("WeatherAPI.com timed out.")
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Cannot reach WeatherAPI.com: {e}")

    if res.status_code != 200:
        try:
            detail = res.json().get("error", {}).get("message", res.text[:120])
        except Exception:
            detail = res.text[:120]
        raise RuntimeError(f"WeatherAPI.com HTTP {res.status_code}: {detail}")

    forecast_days = res.json().get("forecast", {}).get("forecastday", [])
    if len(forecast_days) < 7:
        raise RuntimeError(f"WeatherAPI.com only returned {len(forecast_days)} days.")

    times, probs, rains, temps = [], [], [], []
    for day in forecast_days:
        times.append(day["date"])
        d = day.get("day", {})
        probs.append(float(d.get("daily_chance_of_rain", 0)))
        rains.append(round(float(d.get("totalprecip_mm", 0.0)) / 25.4, 3))
        temps.append(float(d.get("maxtemp_f", 65.0)))

    return {
        "time":                           times,
        "precipitation_probability_max":  probs,
        "precipitation_sum":              rains,
        "temperature_2m_max":             temps,
    }


# ── Public interface — tries all three sources in order ──────────────────────

@st.cache_data(ttl=1800)
def get_weather_7d(lat: float, lon: float) -> Dict[str, Any]:
    """
    Tries Open-Meteo → wttr.in → WeatherAPI.com in order.
    Returns the first successful result. Raises only if all three fail.
    """
    errors: Dict[str, str] = {}

    sources = [
        ("Open-Meteo",      lambda: _fetch_open_meteo(lat, lon)),
        ("wttr.in",         lambda: _fetch_wttr(lat, lon)),
        ("WeatherAPI.com",  lambda: _fetch_weatherapi(lat, lon)),
    ]

    for name, fetch in sources:
        try:
            data = fetch()
            st.session_state["_wx_source"] = name if name == "Open-Meteo" else f"{name} (fallback)"
            return data
        except RuntimeError as e:
            errors[name] = str(e)

    bullet = "\n".join(f"• {k}: {v}" for k, v in errors.items())
    raise RuntimeError(f"All weather sources failed:\n{bullet}")


# ─────────────────────────────────────────────────────────────────────────────
# FORECAST LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def weather_impact_percent(prob: float, rain_in: float, temp_f: float) -> Tuple[int, str, str]:
    impact = 0
    reasons = []
    category = "clear"

    if prob >= 80 or rain_in >= 0.25:
        impact -= 45; reasons.append("Heavy Rain"); category = "rain"
    elif prob >= 60 or rain_in >= 0.10:
        impact -= 25; reasons.append("Moderate Rain"); category = "rain"
    elif prob >= 40:
        impact -= 12; reasons.append("Light Rain"); category = "rain"
    else:
        reasons.append("Clear")

    if temp_f < 40:
        impact -= 20; reasons.append("Cold"); category = "cold" if category == "clear" else category
    elif temp_f < 50:
        impact -= 10; reasons.append("Cool")
    elif 55 <= temp_f <= 82:
        impact += 6; reasons.append("Ideal Temps")
    elif temp_f > 90:
        impact -= 5; reasons.append("Extreme Heat"); category = "heat" if category == "clear" else category

    return int(impact), " + ".join(reasons), category


def weather_factor(prob: float, rain_in: float, temp_f: float) -> float:
    factor = 1.0
    if prob >= 80 or rain_in >= 0.25:    factor *= 0.55
    elif prob >= 60 or rain_in >= 0.10:  factor *= 0.75
    elif prob >= 40:                     factor *= 0.90
    if temp_f < 40:    factor *= 0.80
    elif temp_f < 50:  factor *= 0.90
    elif 55 <= temp_f <= 82: factor *= 1.06
    elif temp_f > 90:  factor *= 0.97
    return factor


def rebound_boost(prev_prob: float, prev_rain: float) -> float:
    if prev_prob >= 80 or prev_rain >= 0.25: return 0.30
    if prev_prob >= 60 or prev_rain >= 0.10: return 0.20
    if prev_prob >= 40: return 0.10
    return 0.0


def forecast_week_quick(location: str, min_cars: int, avg_cars: int, max_cars: int) -> List[Dict[str, Any]]:
    lat, lon = get_lat_lon(location)
    w = get_weather_7d(lat, lon)
    out = []
    prev_prob = 0.0
    prev_rain = 0.0

    for i in range(7):
        day_iso = w["time"][i]
        prob     = float(w["precipitation_probability_max"][i])
        rain_in  = float(w["precipitation_sum"][i])
        temp_f   = float(w["temperature_2m_max"][i])

        dow = datetime.fromisoformat(day_iso).strftime("%A")
        dow_factor = DOW_MULT.get(dow, 1.0)

        impact_pct, reason, wx_cat = weather_impact_percent(prob, rain_in, temp_f)
        wf = weather_factor(prob, rain_in, temp_f)

        if prob < 30 and temp_f > 50:
            wf *= (1 + rebound_boost(prev_prob, prev_rain))

        raw  = avg_cars * dow_factor * wf
        cars = int(max(min_cars, min(raw, max_cars)))

        peak_hour = cars * 0.12
        staff = math.ceil(peak_hour / QUICK_CPSH) if QUICK_CPSH > 0 else 0
        staff = min(MAX_STAFF, max(0, staff))

        out.append({
            "dow":      dow,
            "date":     day_iso,
            "cars":     cars,
            "staff":    staff,
            "impact":   impact_pct,
            "reason":   reason,
            "wx_cat":   wx_cat,
            "rain_pct": int(prob),
            "rain_in":  round(rain_in, 2),
            "temp_f":   round(temp_f, 1),
        })
        prev_prob = prob
        prev_rain = rain_in

    return out


def confidence_label(rows: List[Dict]) -> str:
    avg_rain = sum(r["rain_pct"] for r in rows) / len(rows) if rows else 0
    if avg_rain >= 60: return "MEDIUM — weather-heavy week"
    if avg_rain >= 35: return "HIGH — some variability"
    return "HIGH — stable conditions"


# ─────────────────────────────────────────────────────────────────────────────
# HTML BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def nav_html() -> str:
    return """
<div class="sp-nav">
  <div class="sp-logo">
    <div class="sp-logo-mark">SP</div>
    <div>
      <div class="sp-logo-text">StaffPilot AI</div>
      <div class="sp-logo-sub">Architected by Gopi Chand</div>
    </div>
  </div>
  <div class="sp-badge">⚡ Instant Forecast</div>
</div>
"""


def metrics_html(rows: List[Dict], min_cars: int, max_cars: int) -> str:
    busiest  = max(rows, key=lambda r: r["cars"])
    slowest  = min(rows, key=lambda r: r["cars"])
    rain_days = sum(1 for r in rows if r["rain_pct"] >= 60)
    peak_staff = max(r["staff"] for r in rows)
    avg_cars   = int(sum(r["cars"] for r in rows) / len(rows))
    avg_impact = int(sum(r["impact"] for r in rows) / len(rows))
    impact_cls = "red" if avg_impact < 0 else ("green" if avg_impact > 0 else "")
    impact_str = f"{avg_impact:+d}%"
    conf = confidence_label(rows)

    return f"""
<div class="sp-metrics-row">
  <div class="sp-metric">
    <div class="sp-metric-label">Avg Daily Cars</div>
    <div class="sp-metric-value">{avg_cars}</div>
    <div class="sp-metric-sub">7-day average</div>
  </div>
  <div class="sp-metric green">
    <div class="sp-metric-label">Busiest Day</div>
    <div class="sp-metric-value">{busiest['cars']}</div>
    <div class="sp-metric-sub">{busiest['dow']}</div>
  </div>
  <div class="sp-metric red">
    <div class="sp-metric-label">Slowest Day</div>
    <div class="sp-metric-value">{slowest['cars']}</div>
    <div class="sp-metric-sub">{slowest['dow']}</div>
  </div>
  <div class="sp-metric blue">
    <div class="sp-metric-label">Rain-Impacted Days</div>
    <div class="sp-metric-value">{rain_days}</div>
    <div class="sp-metric-sub">≥60% precip probability</div>
  </div>
</div>
<div class="sp-confidence">
  <span class="sp-conf-icon">📡</span>
  <span>Forecast confidence: <strong>{conf}</strong> &nbsp;·&nbsp; Peak staff cap: <strong>{peak_staff}/{MAX_STAFF}</strong> &nbsp;·&nbsp; Avg weather impact: <strong>{impact_str}</strong></span>
</div>
"""


def table_html(rows: List[Dict], max_cars: int) -> str:
    rows_html = ""
    for r in rows:
        # Cars bar
        bar_pct = int((r["cars"] / max(max_cars, 1)) * 100)
        cars_cell = f"""
<div class="cars-bar-wrap">
  <span class="cars-val">{r['cars']}</span>
  <div class="cars-bar-bg"><div class="cars-bar-fill" style="width:{bar_pct}%"></div></div>
</div>"""

        # Staff pips
        pips = "".join(
            f'<div class="pip{"" if j < r["staff"] else " empty"}"></div>'
            for j in range(MAX_STAFF)
        )
        staff_cell = f'<div class="staff-pips">{pips}</div><span style="font-size:11px;color:var(--muted);margin-left:8px">{r["staff"]}</span>'

        # Weather pill
        wx_icons = {"rain": "🌧", "cold": "🥶", "heat": "🌡", "clear": "☀️"}
        icon = wx_icons.get(r["wx_cat"], "☀️")
        wx_cell = f'<span class="wx-pill {r["wx_cat"]}">{icon} {r["reason"]}</span>'

        # Impact
        if r["impact"] < 0:
            imp_cell = f'<span class="impact-neg">{r["impact"]:+d}%</span>'
        elif r["impact"] > 0:
            imp_cell = f'<span class="impact-pos">{r["impact"]:+d}%</span>'
        else:
            imp_cell = f'<span class="impact-neu">{r["impact"]:+d}%</span>'

        rows_html += f"""
<tr>
  <td>
    <div class="day-name">{r['dow']}</div>
    <div class="day-date">{r['date']}</div>
  </td>
  <td>{cars_cell}</td>
  <td style="display:flex;align-items:center">{staff_cell}</td>
  <td>{imp_cell}</td>
  <td>{wx_cell}</td>
  <td style="font-family:var(--mono);font-size:12px;color:var(--muted)">{r['temp_f']}°F</td>
  <td style="font-family:var(--mono);font-size:12px;color:var(--muted)">{r['rain_pct']}% · {r['rain_in']}"</td>
</tr>"""

    return f"""
<div class="sp-table-wrap">
  <div class="sp-table-header">7-Day Staffing Forecast</div>
  <table class="sp-table">
    <thead>
      <tr>
        <th>Day</th>
        <th>Forecast Cars</th>
        <th>Peak Staff</th>
        <th>Weather Impact</th>
        <th>Conditions</th>
        <th>High Temp</th>
        <th>Precip</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
"""


def mini_chart_html(rows: List[Dict], max_cars: int) -> str:
    """SVG sparkline bar chart."""
    W, H = 900, 120
    pad_l, pad_r, pad_t, pad_b = 10, 10, 10, 24
    inner_w = W - pad_l - pad_r
    inner_h = H - pad_t - pad_b
    n = len(rows)
    bar_w = inner_w / n
    bar_gap = bar_w * 0.28
    rect_w = bar_w - bar_gap

    bars = ""
    labels = ""
    for i, r in enumerate(rows):
        h = max(4, int((r["cars"] / max(max_cars, 1)) * inner_h))
        x = pad_l + i * bar_w + bar_gap / 2
        y = pad_t + inner_h - h
        color = "#ef4444" if r["rain_pct"] >= 60 else ("#f5a623" if r["cars"] == max(rr["cars"] for rr in rows) else "#374151")
        bars += f'<rect x="{x:.1f}" y="{y}" width="{rect_w:.1f}" height="{h}" rx="3" fill="{color}" opacity="0.85"/>'
        cx = x + rect_w / 2
        labels += f'<text x="{cx:.1f}" y="{H - 4}" text-anchor="middle" font-size="10" fill="#6b7280" font-family="IBM Plex Mono, monospace">{r["dow"][:3]}</text>'

    return f"""
<div class="sp-table-wrap" style="padding:20px 24px 8px">
  <div class="sp-table-header">Daily Volume Trend</div>
  <svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;display:block">
    {bars}{labels}
  </svg>
  <div style="display:flex;gap:20px;padding:10px 0 6px;font-size:11px;color:var(--muted);font-family:var(--mono)">
    <span style="display:flex;align-items:center;gap:6px"><span style="width:10px;height:10px;background:#f5a623;border-radius:2px;display:inline-block"></span>Busiest</span>
    <span style="display:flex;align-items:center;gap:6px"><span style="width:10px;height:10px;background:#ef4444;border-radius:2px;display:inline-block"></span>Rain Day (≥60%)</span>
    <span style="display:flex;align-items:center;gap:6px"><span style="width:10px;height:10px;background:#374151;border-radius:2px;display:inline-block"></span>Normal</span>
  </div>
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(nav_html(), unsafe_allow_html=True)

st.markdown('<div class="sp-page">', unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:36px;padding-top:8px">
  <div class="sp-section-title">Weather Intelligence</div>
  <h2 class="sp-section-h2">7-Day Staffing Forecast</h2>
  <p class="sp-section-desc">
    Enter your location and demand range. StaffPilot combines weather data, day-of-week patterns,
    and bounce-back modeling to generate an instant staffing plan.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Input card ────────────────────────────────────────────────────────────────
st.markdown('<div class="sp-card"><div class="sp-card-title">Location &amp; Demand Parameters</div>', unsafe_allow_html=True)

col_loc, col_blank = st.columns([2, 2])
with col_loc:
    location = st.text_input("ZIP Code or City", placeholder="e.g. 30301 or Atlanta, GA", label_visibility="visible")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    min_cars = st.number_input("Lowest Cars / Day", min_value=0, max_value=8000, value=80, step=10)
with c2:
    avg_cars = st.number_input("Average Cars / Day", min_value=10, max_value=8000, value=150, step=10)
with c3:
    max_cars = st.number_input("Highest Cars / Day", min_value=20, max_value=12000, value=280, step=10)

if not (min_cars <= avg_cars <= max_cars):
    st.markdown('<div class="sp-error">⚠ Guardrails must satisfy: Lowest ≤ Average ≤ Highest</div>', unsafe_allow_html=True)

st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    run = st.button("⚡ Generate Forecast")

st.markdown("</div>", unsafe_allow_html=True)  # close sp-card

# ── Disclaimer ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="sp-disclaimer">
  <strong style="color:var(--text)">Accuracy note:</strong>
  This forecast uses weather data + industry day-of-week curves. It does <em>not</em> account for
  active promotions, local road closures, nearby competition, or event-driven spikes.
  Treat <strong>Expected Change %</strong> as directional guidance, not a hard prediction.
</div>
""", unsafe_allow_html=True)

# ── Results ───────────────────────────────────────────────────────────────────
if run:
    if not location.strip():
        st.markdown('<div class="sp-error">⚠ Please enter a ZIP code or city before generating.</div>', unsafe_allow_html=True)
    elif not (min_cars <= avg_cars <= max_cars):
        st.markdown('<div class="sp-error">⚠ Fix guardrail values before generating.</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Fetching weather data…"):
            try:
                rows = forecast_week_quick(location.strip(), int(min_cars), int(avg_cars), int(max_cars))
            except Exception as e:
                st.markdown(f'<div class="sp-error">⚠ {e}</div>', unsafe_allow_html=True)
                rows = []

        if rows:
            st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

            # Show which weather source was used (fallback makes this visible)
            wx_source = st.session_state.get("_wx_source", "Open-Meteo")
            is_fallback = "fallback" in wx_source.lower()
            src_color  = "#f97316" if is_fallback else "#22c55e"
            src_icon   = "⚠️" if is_fallback else "✅"
            src_note   = " — Open-Meteo was unavailable" if is_fallback else ""
            st.markdown(
                f'<div style="font-family:var(--mono);font-size:11px;color:{src_color};'
                f'margin-bottom:16px;letter-spacing:0.08em">'
                f'{src_icon} Weather source: <strong>{wx_source}</strong>{src_note}</div>',
                unsafe_allow_html=True,
            )

            st.markdown(metrics_html(rows, int(min_cars), int(max_cars)), unsafe_allow_html=True)
            st.markdown(mini_chart_html(rows, int(max_cars)), unsafe_allow_html=True)
            st.markdown(table_html(rows, int(max_cars)), unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # close sp-page
