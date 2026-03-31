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
    --bg:      #0a0c0f;
    --surface: #111418;
    --surf2:   #1a1f27;
    --border:  #2a3040;
    --accent:  #f5a623;
    --accent2: #e8870a;
    --green:   #22c55e;
    --red:     #ef4444;
    --blue:    #60a5fa;
    --canada:  #d52b1e;
    --text:    #f0f2f7;
    --sub:     #c4c9d4;
    --muted:   #9ca3af;
    --mono:    'IBM Plex Mono', monospace;
    --sans:    'IBM Plex Sans', sans-serif;
    --max-w:   860px;
}

/* ── Kill every white/light surface Streamlit injects ── */
.stApp,
.stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="block-container"],
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="stColumn"],
section.main,
div[data-testid="stForm"],
div[class*="css"] {
    background-color: transparent !important;
}
.stApp { background-color: var(--bg) !important; }

/* ── Center & constrain content column ── */
.block-container,
[data-testid="block-container"],
[data-testid="stAppViewBlockContainer"] {
    max-width: var(--max-w) !important;
    padding: 0 16px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* ── Global text ── */
html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
p, span, div, label, li { color: var(--text) !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { display: none !important; }

/* ── Nav bar ── */
.sp-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 100;
    padding: 14px max(16px, calc(50% - var(--max-w)/2 + 16px));
}
.sp-logo { display: flex; align-items: center; gap: 12px; }
.sp-logo-mark {
    width: 32px; height: 32px;
    background: var(--accent);
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-family: var(--mono); font-weight: 700;
    color: #000; flex-shrink: 0;
}
.sp-logo-text {
    font-family: var(--mono); font-size: 15px; font-weight: 600;
    letter-spacing: 0.08em; color: var(--text);
}
.sp-logo-sub {
    font-size: 10px; color: var(--muted);
    letter-spacing: 0.10em; text-transform: uppercase; margin-top: 1px;
}
.sp-badge {
    font-family: var(--mono); font-size: 11px;
    background: rgba(245,166,35,0.12); color: var(--accent);
    border: 1px solid rgba(245,166,35,0.28);
    border-radius: 20px; padding: 4px 12px;
    letter-spacing: 0.06em; white-space: nowrap;
}

/* ── Country toggle ── */
.sp-country-toggle {
    display: flex;
    gap: 0;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 20px;
    width: fit-content;
}
.sp-country-btn {
    padding: 10px 24px;
    font-family: var(--mono);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    cursor: pointer;
    border: none;
    background: var(--surf2);
    color: var(--muted);
    transition: all 0.15s;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sp-country-btn.active-usa {
    background: rgba(245,166,35,0.15);
    color: var(--accent);
    border-right: 1px solid var(--border);
}
.sp-country-btn.active-canada {
    background: rgba(213,43,30,0.15);
    color: #f87171;
}
.sp-canada-note {
    display: flex; align-items: flex-start; gap: 10px;
    background: rgba(213,43,30,0.07);
    border: 1px solid rgba(213,43,30,0.25);
    border-radius: 8px; padding: 12px 16px;
    font-size: 12px; color: var(--sub);
    margin-bottom: 16px; line-height: 1.6;
}
.sp-canada-note strong { color: #fca5a5; }

/* ── Page wrapper ── */
.sp-page { padding: 32px 0 80px; }

/* ── Section headings ── */
.sp-section-title {
    font-family: var(--mono); font-size: 11px;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--accent); margin-bottom: 6px;
}
.sp-section-h2 {
    font-family: var(--sans); font-size: 24px; font-weight: 700;
    color: var(--text); margin: 0 0 4px; line-height: 1.25;
}
.sp-section-desc {
    font-size: 14px; color: var(--sub); line-height: 1.65;
    max-width: 600px; margin-bottom: 28px;
}

/* ── Input card ── */
.sp-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px 24px 20px;
    margin-bottom: 16px;
}
.sp-card-title {
    font-family: var(--mono); font-size: 11px; color: var(--muted);
    letter-spacing: 0.14em; text-transform: uppercase;
    margin-bottom: 18px;
    display: flex; align-items: center; gap: 8px;
}
.sp-card-title::before {
    content: ''; display: inline-block;
    width: 3px; height: 13px;
    background: var(--accent); border-radius: 2px;
}

/* ── Text input ── */
div[data-testid="stTextInput"] > div,
div[data-testid="stTextInput"] > div > div {
    background: var(--surf2) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
}
div[data-testid="stTextInput"] input {
    background: var(--surf2) !important;
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
    background: var(--surf2) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
}
div[data-testid="stNumberInput"] input {
    background: var(--surf2) !important;
    border: none !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 14px !important;
    caret-color: var(--accent) !important;
}
div[data-testid="stNumberInput"] input:focus { outline: none !important; box-shadow: none !important; }
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
    color: var(--sub) !important;
    margin-bottom: 4px !important;
}

/* ── Radio buttons (country selector) ── */
div[data-testid="stRadio"] > label {
    font-family: var(--mono) !important;
    font-size: 13px !important;
    color: var(--sub) !important;
}
div[data-testid="stRadio"] [data-testid="stWidgetLabel"] p {
    font-size: 13px !important; color: var(--sub) !important;
}
div[data-testid="stRadio"] > div {
    display: flex !important; gap: 0 !important;
    background: var(--surf2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    width: fit-content !important;
    padding: 0 !important;
}
div[data-testid="stRadio"] > div > label {
    padding: 10px 20px !important;
    font-family: var(--mono) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    border-right: 1px solid var(--border) !important;
    color: var(--muted) !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
}
div[data-testid="stRadio"] > div > label:last-child { border-right: none !important; }
div[data-testid="stRadio"] > div > label:has(input:checked) {
    background: rgba(245,166,35,0.15) !important;
    color: var(--accent) !important;
}

/* ── Button ── */
div.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: var(--mono) !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 32px !important;
    width: 100% !important;
    margin-top: 8px !important;
    transition: background 0.15s, transform 0.1s !important;
}
div.stButton > button:hover {
    background: var(--accent2) !important;
    transform: translateY(-1px) !important;
}

/* ── Spinner ── */
[data-testid="stStatusWidget"], .stSpinner > div {
    background: transparent !important; color: var(--muted) !important;
}
.stSpinner svg { stroke: var(--accent) !important; }

/* ── Alerts ── */
[data-testid="stAlert"], div[role="alert"], .stAlert > div {
    background: var(--surf2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

/* ── Metric strip ── */
.sp-metrics-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}
.sp-metric {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
    position: relative; overflow: hidden;
}
.sp-metric::after {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: var(--accent);
}
.sp-metric.green::after { background: var(--green); }
.sp-metric.red::after   { background: var(--red); }
.sp-metric.blue::after  { background: var(--blue); }
.sp-metric-label {
    font-family: var(--mono); font-size: 10px; color: var(--muted);
    letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 8px;
}
.sp-metric-value {
    font-family: var(--mono); font-size: 26px; font-weight: 600;
    color: var(--text); line-height: 1;
}
.sp-metric-sub { font-size: 12px; color: var(--sub); margin-top: 4px; }

/* ── Confidence banner ── */
.sp-confidence {
    display: flex; align-items: center; gap: 12px;
    background: rgba(245,166,35,0.07);
    border: 1px solid rgba(245,166,35,0.22);
    border-radius: 10px; padding: 12px 18px;
    font-size: 13px; color: var(--sub); margin-bottom: 20px;
}
.sp-confidence strong { color: var(--accent); }
.sp-conf-icon { font-size: 16px; }

/* ── Forecast table ── */
.sp-table-wrap {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden; margin-bottom: 20px;
}
.sp-table-header {
    padding: 16px 20px 12px; border-bottom: 1px solid var(--border);
    font-family: var(--mono); font-size: 11px; color: var(--sub);
    letter-spacing: 0.14em; text-transform: uppercase;
    display: flex; align-items: center; gap: 8px;
}
.sp-table-header::before {
    content: ''; display: inline-block;
    width: 3px; height: 13px; background: var(--accent); border-radius: 2px;
}
table.sp-table { width: 100%; border-collapse: collapse; font-size: 13px; }
table.sp-table th {
    font-family: var(--mono); font-size: 10px;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--muted); padding: 10px 18px; text-align: left;
    background: var(--surf2); border-bottom: 1px solid var(--border);
}
table.sp-table td {
    padding: 13px 18px; border-bottom: 1px solid var(--border);
    font-family: var(--mono); font-size: 13px; color: var(--text);
    vertical-align: middle;
}
table.sp-table tr:last-child td { border-bottom: none; }
table.sp-table tr:hover td { background: rgba(255,255,255,0.03); }

.day-name  { font-weight: 600; color: var(--text); }
.day-date  { font-size: 11px; color: var(--muted); margin-top: 2px; }

.cars-bar-wrap { display: flex; align-items: center; gap: 10px; }
.cars-bar-bg {
    flex: 1; height: 5px; background: var(--border);
    border-radius: 3px; overflow: hidden; max-width: 80px;
}
.cars-bar-fill { height: 100%; border-radius: 3px; background: var(--accent); }
.cars-val { min-width: 38px; text-align: right; color: var(--text); }

.staff-pips { display: flex; gap: 4px; }
.pip { width: 9px; height: 9px; border-radius: 50%; background: var(--accent); }
.pip.empty { background: transparent; border: 1px solid var(--border); }

.wx-pill {
    display: inline-flex; align-items: center; gap: 5px;
    background: var(--surf2); border: 1px solid var(--border);
    border-radius: 20px; padding: 3px 9px;
    font-size: 11px; color: var(--sub); white-space: nowrap;
}
.wx-pill.rain  { border-color:#3b82f6; color:#93c5fd; background:rgba(59,130,246,0.08); }
.wx-pill.clear { border-color:#22c55e; color:#86efac; background:rgba(34,197,94,0.08); }
.wx-pill.cold  { border-color:#818cf8; color:#c7d2fe; background:rgba(129,140,248,0.08); }
.wx-pill.heat  { border-color:#f97316; color:#fdba74; background:rgba(249,115,22,0.08); }
.wx-pill.snow  { border-color:#a5b4fc; color:#c7d2fe; background:rgba(165,180,252,0.10); }
.wx-pill.blizzard { border-color:#ef4444; color:#fca5a5; background:rgba(239,68,68,0.10); }

.impact-neg { color: #f87171; font-weight: 600; }
.impact-pos { color: #4ade80; font-weight: 600; }
.impact-neu { color: var(--muted); }

/* ── Canada flag accent ── */
.sp-canada-accent {
    display: inline-flex; align-items: center; gap: 6px;
    font-family: var(--mono); font-size: 11px;
    color: #f87171; letter-spacing: 0.08em;
}

/* ── Disclaimer ── */
.sp-disclaimer {
    font-size: 12px; color: var(--sub); line-height: 1.65;
    padding: 14px 18px;
    background: var(--surf2);
    border-left: 3px solid var(--border);
    border-radius: 0 8px 8px 0; margin-top: 10px;
}

/* ── Error box ── */
.sp-error {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 10px; padding: 14px 18px;
    color: #fca5a5; font-size: 13px; font-family: var(--mono);
    white-space: pre-wrap;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Hide native chart / column gaps ── */
[data-testid="stVegaLiteChart"] { display: none !important; }
[data-testid="stHorizontalBlock"] { gap: 12px !important; }
hr { border-color: var(--border) !important; }

/* ── Chart legend ── */
.sp-chart-legend {
    display: flex;
    gap: 16px;
    padding: 10px 0 6px;
    font-size: 11px;
    color: var(--muted);
    font-family: var(--mono);
    flex-wrap: wrap;
    row-gap: 8px;
}

/* ═══════════════════════════════════════════════════════
   MOBILE RESPONSIVE  ( ≤ 640px )
   ═══════════════════════════════════════════════════════ */
@media (max-width: 640px) {

  /* Tighter page padding */
  .block-container,
  [data-testid="block-container"],
  [data-testid="stAppViewBlockContainer"] {
    padding: 0 10px !important;
  }

  /* Nav — stack logo + badge vertically */
  .sp-nav {
    padding: 12px 12px !important;
    flex-wrap: wrap;
    gap: 8px;
  }
  .sp-logo-text { font-size: 13px !important; }
  .sp-badge { font-size: 10px !important; padding: 3px 9px !important; }

  /* Hero text */
  .sp-section-h2 { font-size: 19px !important; }
  .sp-section-desc { font-size: 13px !important; }

  /* Card tighter */
  .sp-card { padding: 16px 14px 14px !important; }

  /* Metrics: 2×2 grid on mobile instead of 4×1 */
  .sp-metrics-row {
    grid-template-columns: repeat(2, 1fr) !important;
    gap: 8px !important;
  }
  .sp-metric { padding: 14px 14px !important; }
  .sp-metric-value { font-size: 21px !important; }
  .sp-metric-label { font-size: 9px !important; }
  .sp-metric-sub { font-size: 11px !important; }

  /* Confidence banner wraps */
  .sp-confidence { flex-wrap: wrap; font-size: 12px !important; padding: 10px 14px !important; }

  /* Table: hide less-critical columns, make text smaller */
  table.sp-table th,
  table.sp-table td { padding: 10px 10px !important; font-size: 11px !important; }

  /* Hide High Temp + Precip columns on very small screens */
  table.sp-table th:nth-child(6),
  table.sp-table td:nth-child(6),
  table.sp-table th:nth-child(7),
  table.sp-table td:nth-child(7) { display: none !important; }

  /* Cars bar narrower */
  .cars-bar-bg { max-width: 48px !important; }
  .cars-val { min-width: 28px !important; font-size: 11px !important; }

  /* Staff pips smaller */
  .pip { width: 7px !important; height: 7px !important; }

  /* Weather pill smaller */
  .wx-pill { font-size: 10px !important; padding: 2px 7px !important; }

  /* Chart legend wraps tighter */
  .sp-chart-legend { gap: 10px !important; font-size: 10px !important; }

  /* Table wrap horizontal scroll as last resort */
  .sp-table-wrap { overflow-x: auto !important; }

  /* Canada note smaller */
  .sp-canada-note { font-size: 11px !important; padding: 10px 12px !important; }

  /* Disclaimer */
  .sp-disclaimer { font-size: 11px !important; }
}

/* Tablet ( 641px – 900px ) */
@media (min-width: 641px) and (max-width: 900px) {
  .sp-metrics-row {
    grid-template-columns: repeat(2, 1fr) !important;
    gap: 10px !important;
  }
  table.sp-table th,
  table.sp-table td { padding: 11px 12px !important; font-size: 12px !important; }
  .sp-card { padding: 20px 18px !important; }
}

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
# CANADA POSTAL CODE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# FSA → (lat, lon, city_name) for all 18 Canadian province letter prefixes.
# These are approximate centroids — accurate enough for weather (±50 km is fine).
# Used as a hard fallback when Open-Meteo geocoding can't resolve a postal code.
CANADA_FSA_FALLBACK: Dict[str, Tuple[float, float, str]] = {
    # British Columbia
    "V": (49.2827, -123.1207, "Vancouver, BC"),
    # Alberta
    "T": (51.0447, -114.0719, "Calgary, AB"),
    # Saskatchewan
    "S": (52.1332, -106.6700, "Saskatoon, SK"),
    # Manitoba
    "R": (49.8951, -97.1384,  "Winnipeg, MB"),
    # Ontario
    "K": (44.2312, -76.4860,  "Kingston, ON"),
    "L": (43.7315, -79.7624,  "Brampton, ON"),
    "M": (43.6532, -79.3832,  "Toronto, ON"),
    "N": (42.9849, -81.2453,  "London, ON"),
    "P": (46.4924, -80.9931,  "Sudbury, ON"),
    # Quebec
    "G": (46.8139, -71.2080,  "Quebec City, QC"),
    "H": (45.5017, -73.5673,  "Montreal, QC"),
    "J": (45.3874, -75.6919,  "Gatineau, QC"),
    # New Brunswick
    "E": (45.9636, -66.6431,  "Fredericton, NB"),
    # Nova Scotia
    "B": (44.6488, -63.5752,  "Halifax, NS"),
    # Prince Edward Island
    "C": (46.2382, -63.1311,  "Charlottetown, PE"),
    # Newfoundland
    "A": (47.5615, -52.7126,  "St. John's, NL"),
    # Northwest Territories / Nunavut / Yukon
    "X": (62.4540, -114.3718, "Yellowknife, NT"),
    "Y": (60.7212, -135.0568, "Whitehorse, YT"),
}


CANADA_PROVINCES = {
    "AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE", "QC", "SK", "YT",
}


def _extract_canadian_postal(raw: str) -> str:
    """
    Extract postal code from any common Canadian input, including:
      'T8N 5E1'      -> 'T8N 5E1'
      'T8N5E1'       -> 'T8N 5E1'
      'AB T8N 5E1'   -> 'T8N 5E1'   (province prefix)
      'AB, T8N 5E1'  -> 'T8N 5E1'
      'T8N'          -> 'T8N'        (FSA only)
      'Calgary, AB'  -> 'Calgary, AB' (city name, returned unchanged)
    """
    text = raw.strip().upper()

    # Strip leading province abbreviation (e.g. "AB ", "AB, ", "AB- ")
    for prov in CANADA_PROVINCES:
        for sep in (prov + " ", prov + ", ", prov + "- ", prov + "-"):
            if text.startswith(sep):
                text = text[len(sep):].strip()
                break

    # Strip trailing province (e.g. "T8N 5E1, AB")
    for prov in CANADA_PROVINCES:
        for sep in (", " + prov, " " + prov):
            if text.endswith(sep):
                text = text[: -len(sep)].strip()
                break

    # Collapse spaces/hyphens between the two halves
    code = text.replace(" ", "").replace("-", "")

    # Full 6-char postal code A1A1A1
    if (len(code) == 6
            and code[0].isalpha() and code[1].isdigit() and code[2].isalpha()
            and code[3].isdigit() and code[4].isalpha() and code[5].isdigit()):
        return code[:3] + " " + code[3:]

    # 3-char FSA A1A
    if (len(code) == 3
            and code[0].isalpha() and code[1].isdigit() and code[2].isalpha()):
        return code

    # Nothing matched — probably a city name, return original
    return raw.strip()


def normalize_canadian_postal(raw: str) -> str:
    """Public wrapper — always returns the best cleaned representation."""
    return _extract_canadian_postal(raw)


def _looks_like_canadian_postal(s: str) -> bool:
    """True if string looks like a full postal code (A1A 1A1) or FSA (A1A)."""
    code = s.strip().upper().replace(" ", "")
    if len(code) == 6:
        return (code[0].isalpha() and code[1].isdigit() and code[2].isalpha()
                and code[3].isdigit() and code[4].isalpha() and code[5].isdigit())
    if len(code) == 3:
        return code[0].isalpha() and code[1].isdigit() and code[2].isalpha()
    return False


def _open_meteo_geocode(query: str) -> Optional[Tuple[float, float]]:
    """Single Open-Meteo geocoding attempt. Returns (lat, lon) or None."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    try:
        res = requests.get(url, params={"name": query, "count": 5},
                           headers=HEADERS, timeout=15)
        if res.status_code != 200:
            return None
        results = res.json().get("results") or []
        # Prefer results tagged as Canada when we're searching for Canada
        canada_results = [r for r in results if r.get("country_code") == "CA"]
        best = canada_results[0] if canada_results else (results[0] if results else None)
        if best:
            return float(best["latitude"]), float(best["longitude"])
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# WEATHER HELPERS  (Primary: Open-Meteo  |  Fallback: WeatherAPI.com)
# ─────────────────────────────────────────────────────────────────────────────

HEADERS = {"User-Agent": "StaffPilotAI/1.0 (streamlit-app; contact@staffpilot.ai)"}

# WeatherAPI key — optional; enables fallback if Open-Meteo fails
WEATHERAPI_KEY: str = st.secrets.get("WEATHERAPI_KEY", "")


# ── Geocoding ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_lat_lon(location: str, country: str = "USA") -> Tuple[float, float]:
    """
    Resolve ZIP / postal code / city to (lat, lon).

    Canada strategy (tried in order):
      1. Query Open-Meteo with just the city/FSA string (e.g. "M5V")
      2. Query Open-Meteo with "M5V, Canada"  
      3. Query Open-Meteo with just the FSA prefix letter's known city
         (e.g. "M" → "Toronto, Canada")
      4. Hard-coded FSA-prefix centroid table (always works for valid CA codes)

    USA strategy: single Open-Meteo query.
    """
    if country == "Canada":
        clean = location.strip().upper().replace(" ", "")

        queries_to_try: List[str] = []

        if _looks_like_canadian_postal(clean):
            fsa = clean[:3]  # Forward Sortation Area
            queries_to_try = [
                fsa,                        # bare FSA
                f"{fsa}, Canada",           # FSA + country
                f"{fsa} Canada",            # no comma variant
            ]
            # Also try the FSA prefix city name
            prefix = fsa[0]
            if prefix in CANADA_FSA_FALLBACK:
                city_name = CANADA_FSA_FALLBACK[prefix][2]
                queries_to_try.append(city_name)
        else:
            # Looks like a city name — try as-is and with ", Canada"
            queries_to_try = [
                location.strip(),
                f"{location.strip()}, Canada",
            ]

        # Try each query variant
        for q in queries_to_try:
            result = _open_meteo_geocode(q)
            if result:
                return result

        # Hard fallback: FSA prefix table
        if _looks_like_canadian_postal(clean):
            prefix = clean[0]
            if prefix in CANADA_FSA_FALLBACK:
                lat, lon, city = CANADA_FSA_FALLBACK[prefix]
                # Store a note so the UI can warn the user
                st.session_state["_geo_fallback"] = city
                return lat, lon

        raise RuntimeError(
            f"Could not locate '{location}' in Canada. "
            "Try a city name like 'Toronto' or 'Calgary, AB'."
        )

    # ── USA ──────────────────────────────────────────────────────────────────
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
    # snowfall_sum may not always exist; default to zeros
    snow_raw = daily.get("snowfall_sum") or [None] * n
    daily["snowfall_sum"] = [
        float(v) if v is not None else 0.0 for v in snow_raw
    ]
    return daily


def _fetch_open_meteo(lat: float, lon: float) -> Dict[str, Any]:
    """Fetch 7-day forecast from Open-Meteo with up to 2 retries. Also requests snowfall."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_probability_max,precipitation_sum,temperature_2m_max,snowfall_sum",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "forecast_days": 7,
        "timezone": "auto",
    }
    last_err = "Unknown error"
    for attempt in range(2):
        try:
            res = requests.get(url, params=params, headers=HEADERS, timeout=20)
        except requests.exceptions.Timeout:
            last_err = f"Timed out (attempt {attempt + 1})"
            continue
        except requests.exceptions.ConnectionError as e:
            last_err = f"Connection error: {e}"
            break

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


# ── Fallback 1: wttr.in ───────────────────────────────────────────────────────

def _fetch_wttr(lat: float, lon: float) -> Dict[str, Any]:
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

    base = weather_days[:]
    while len(base) < 7:
        base.append(base[len(base) % 3])

    times, probs, rains, temps, snows = [], [], [], [], []
    base_date = datetime.utcnow().date()
    for i, day in enumerate(base[:7]):
        times.append((base_date + timedelta(days=i)).isoformat())
        hourly = day.get("hourly", [])
        temp_max = max((float(h.get("tempF", 65)) for h in hourly), default=65.0)
        precip_mm = sum(float(h.get("precipMM", 0)) for h in hourly)
        snow_cm = sum(float(h.get("snowDepthCM", 0)) for h in hourly)
        chance = max((float(h.get("chanceofrain", 0)) for h in hourly), default=0.0)
        temps.append(temp_max)
        rains.append(round(precip_mm / 25.4, 3))
        snows.append(round(snow_cm / 2.54, 3))  # cm → inches
        probs.append(chance)

    return {
        "time":                           times,
        "precipitation_probability_max":  probs,
        "precipitation_sum":              rains,
        "temperature_2m_max":             temps,
        "snowfall_sum":                   snows,
    }


# ── Fallback 2: WeatherAPI.com ────────────────────────────────────────────────

def _fetch_weatherapi(lat: float, lon: float) -> Dict[str, Any]:
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

    times, probs, rains, temps, snows = [], [], [], [], []
    for day in forecast_days:
        times.append(day["date"])
        d = day.get("day", {})
        probs.append(float(d.get("daily_chance_of_rain", 0)))
        rains.append(round(float(d.get("totalprecip_mm", 0.0)) / 25.4, 3))
        snows.append(round(float(d.get("totalsnow_cm", 0.0)) / 2.54, 3))
        temps.append(float(d.get("maxtemp_f", 65.0)))

    return {
        "time":                           times,
        "precipitation_probability_max":  probs,
        "precipitation_sum":              rains,
        "temperature_2m_max":             temps,
        "snowfall_sum":                   snows,
    }


# ── Fallback 3: Climate-average estimator (no network required) ──────────────
#
# Uses lat/lon + current month to produce plausible 7-day weather estimates
# from built-in climate normals. No API calls, no rate limits, always works.
# Accuracy: seasonal directional guidance, not day-specific forecast.
#
# Climate zones are determined by latitude band + longitude (continent):
#   - Arctic    (lat > 60)
#   - Subarctic (lat 55-60, e.g. northern Canada)
#   - Cold      (lat 45-55, e.g. southern Canada, northern US)
#   - Temperate (lat 35-45, e.g. mid US)
#   - Warm      (lat 25-35, e.g. southern US)
#   - Hot       (lat < 25)
#
# Monthly normals per zone: (temp_f_max, precip_prob_pct, rain_in_day, snow_in_day)

_CLIMATE_NORMALS: Dict[str, List[Tuple[float, float, float, float]]] = {
    # zone: [(temp_f, precip_prob, rain_in, snow_in)] * 12 months Jan-Dec
    "arctic": [
        (-4,25,0.0,0.6),(-4,22,0.0,0.5),(10,20,0.0,0.4),(28,18,0.0,0.2),
        (44,25,0.1,0.0),(58,35,0.1,0.0),(65,40,0.1,0.0),(62,38,0.1,0.0),
        (48,30,0.1,0.1),(28,25,0.0,0.3),(10,25,0.0,0.5),(-2,25,0.0,0.6),
    ],
    "subarctic": [
        (5,20,0.0,0.5),(10,18,0.0,0.4),(25,20,0.0,0.3),(42,22,0.0,0.1),
        (58,30,0.1,0.0),(70,38,0.1,0.0),(76,35,0.1,0.0),(72,33,0.1,0.0),
        (57,30,0.1,0.0),(40,25,0.0,0.1),(22,22,0.0,0.3),(8,20,0.0,0.5),
    ],
    "cold": [
        (28,30,0.0,0.3),(32,28,0.0,0.3),(44,32,0.1,0.1),(58,35,0.1,0.0),
        (68,38,0.1,0.0),(78,40,0.1,0.0),(83,35,0.1,0.0),(80,32,0.1,0.0),
        (68,35,0.1,0.0),(54,30,0.1,0.0),(40,30,0.0,0.1),(30,30,0.0,0.2),
    ],
    "temperate": [
        (45,35,0.1,0.0),(50,33,0.1,0.0),(60,38,0.1,0.0),(68,40,0.1,0.0),
        (76,38,0.1,0.0),(84,35,0.1,0.0),(89,30,0.1,0.0),(87,28,0.1,0.0),
        (78,32,0.1,0.0),(66,32,0.1,0.0),(54,35,0.1,0.0),(46,35,0.1,0.0),
    ],
    "warm": [
        (62,35,0.1,0.0),(65,33,0.1,0.0),(72,38,0.1,0.0),(78,35,0.1,0.0),
        (85,35,0.1,0.0),(92,38,0.1,0.0),(95,45,0.1,0.0),(94,42,0.1,0.0),
        (88,38,0.1,0.0),(78,32,0.1,0.0),(68,30,0.1,0.0),(62,32,0.1,0.0),
    ],
    "hot": [
        (75,30,0.1,0.0),(78,28,0.1,0.0),(84,30,0.1,0.0),(90,28,0.0,0.0),
        (95,25,0.0,0.0),(100,20,0.0,0.0),(98,25,0.1,0.0),(97,30,0.1,0.0),
        (92,30,0.1,0.0),(85,25,0.0,0.0),(79,25,0.0,0.0),(75,28,0.0,0.0),
    ],
}

# Day-to-day variability seeds — slight variation across the 7 days
_DAY_OFFSETS: List[Tuple[float, float, float]] = [
    (0.0,  0,    0.0),
    (2.0,  5,    0.02),
    (-3.0, -8,   0.0),
    (4.0,  10,   0.03),
    (-1.0, -5,   0.0),
    (3.0,  8,    0.01),
    (-2.0, 3,    0.0),
]


def _climate_zone(lat: float, lon: float) -> str:
    """Map lat/lon to a climate zone string."""
    a = abs(lat)
    if a > 60:   return "arctic"
    if a > 55:   return "subarctic"
    if a > 45:   return "cold"
    if a > 35:   return "temperate"
    if a > 25:   return "warm"
    return "hot"


def _fetch_climate_estimate(lat: float, lon: float) -> Dict[str, Any]:
    """
    Generate a 7-day weather estimate from built-in climate normals.
    No network required. Always succeeds.
    """
    zone = _climate_zone(lat, lon)
    normals = _CLIMATE_NORMALS[zone]
    month_idx = datetime.utcnow().month - 1  # 0-based
    base_temp, base_prob, base_rain, base_snow = normals[month_idx]

    times, probs, rains, temps, snows = [], [], [], [], []
    base_date = datetime.utcnow().date()

    for i in range(7):
        t_off, p_off, r_off = _DAY_OFFSETS[i]
        times.append((base_date + timedelta(days=i)).isoformat())
        temp  = max(-40.0, base_temp + t_off)
        prob  = max(0.0, min(100.0, base_prob + p_off))
        rain  = max(0.0, base_rain + r_off)
        snow  = base_snow if temp < 34 else 0.0
        temps.append(round(temp, 1))
        probs.append(round(prob, 1))
        rains.append(round(rain, 3))
        snows.append(round(snow, 2))

    return {
        "time":                           times,
        "precipitation_probability_max":  probs,
        "precipitation_sum":              rains,
        "temperature_2m_max":             temps,
        "snowfall_sum":                   snows,
    }


@st.cache_data(ttl=1800)
def get_weather_7d(lat: float, lon: float) -> Dict[str, Any]:
    errors: Dict[str, str] = {}

    live_sources = [
        ("Open-Meteo",      lambda: _fetch_open_meteo(lat, lon)),
        ("wttr.in",         lambda: _fetch_wttr(lat, lon)),
        ("WeatherAPI.com",  lambda: _fetch_weatherapi(lat, lon)),
    ]

    for name, fetch in live_sources:
        try:
            data = fetch()
            st.session_state["_wx_source"] = name if name == "Open-Meteo" else f"{name} (fallback)"
            st.session_state["_wx_estimated"] = False
            return data
        except RuntimeError as e:
            errors[name] = str(e)

    # All live sources failed — use built-in climate estimate
    try:
        data = _fetch_climate_estimate(lat, lon)
        zone = _climate_zone(lat, lon)
        st.session_state["_wx_source"] = f"Climate Estimate ({zone} zone)"
        st.session_state["_wx_estimated"] = True
        st.session_state["_wx_errors"] = errors
        return data
    except Exception as e:
        errors["Climate Estimate"] = str(e)

    bullet = "\n".join(f"• {k}: {v}" for k, v in errors.items())
    raise RuntimeError(f"All weather sources failed:\n{bullet}")


# ─────────────────────────────────────────────────────────────────────────────
# FORECAST LOGIC — USA vs CANADA ALGORITHMS
# ─────────────────────────────────────────────────────────────────────────────

# ── USA algorithm (original) ─────────────────────────────────────────────────

def weather_impact_percent_usa(prob: float, rain_in: float, temp_f: float) -> Tuple[int, str, str]:
    """US-tuned: rain and cold both significantly dampen car wash demand."""
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


def weather_factor_usa(prob: float, rain_in: float, temp_f: float) -> float:
    factor = 1.0
    if prob >= 80 or rain_in >= 0.25:    factor *= 0.55
    elif prob >= 60 or rain_in >= 0.10:  factor *= 0.75
    elif prob >= 40:                     factor *= 0.90
    if temp_f < 40:    factor *= 0.80
    elif temp_f < 50:  factor *= 0.90
    elif 55 <= temp_f <= 82: factor *= 1.06
    elif temp_f > 90:  factor *= 0.97
    return factor


# ── Canada algorithm (snow-aware) ────────────────────────────────────────────
#
# Key differences vs USA:
#  1. Snow tolerance: Canadians are accustomed to winter driving; light–moderate
#     snow doesn't deter car wash visits as strongly. People actually wash their
#     cars MORE after road salt / slush exposure.
#  2. Cold tolerance: Sub-zero temps are normal. Operational concern shifts to
#     "is the wash physically open?" (freeze risk) rather than customer reluctance.
#  3. "Ideal temp" window is shifted colder (10°C / 50°F–25°C / 77°F).
#  4. Post-snow rebound is STRONG — salty slush roads drive a surge the next clear day.
#  5. Blizzard / whiteout conditions (heavy snow + very cold) do suppress volume.

def _snow_category(snow_in: float, temp_f: float) -> str:
    """Classify snow severity for Canada."""
    if snow_in >= 6 and temp_f < 25:
        return "blizzard"
    if snow_in >= 3:
        return "heavy_snow"
    if snow_in >= 0.5:
        return "moderate_snow"
    if snow_in > 0:
        return "light_snow"
    return "none"


def weather_impact_percent_canada(
    prob: float, rain_in: float, temp_f: float, snow_in: float
) -> Tuple[int, str, str]:
    """
    Canada-tuned impact.
    Snow: mild suppressor (customers used to it) UNLESS blizzard.
    Cold: tolerated down to ~14°F / -10°C before meaningful suppression.
    Post-snow salt-rebound is handled in forecast_week_quick_canada().
    Rain: similar to USA but slightly less impact (rain still dirty = wash demand).
    """
    impact = 0
    reasons = []
    category = "clear"

    snow_cat = _snow_category(snow_in, temp_f)

    # ── Snow effects ──────────────────────────────────────────────────────────
    if snow_cat == "blizzard":
        # Roads may close; customers stay home
        impact -= 50
        reasons.append("Blizzard")
        category = "blizzard"
    elif snow_cat == "heavy_snow":
        # Significant but Canadians push through; salt demand creates wash motive
        impact -= 20
        reasons.append("Heavy Snow")
        category = "snow"
    elif snow_cat == "moderate_snow":
        # Light suppression; also a wash trigger (salt / slush)
        impact -= 8
        reasons.append("Moderate Snow")
        category = "snow"
    elif snow_cat == "light_snow":
        # Minimal impact; salt residue actually encourages washing
        impact -= 2
        reasons.append("Light Snow")
        category = "snow"

    # ── Rain effects (only if no snow dominant) ───────────────────────────────
    if snow_cat == "none":
        if prob >= 80 or rain_in >= 0.25:
            impact -= 38; reasons.append("Heavy Rain"); category = "rain"
        elif prob >= 60 or rain_in >= 0.10:
            impact -= 20; reasons.append("Moderate Rain"); category = "rain"
        elif prob >= 40:
            impact -= 10; reasons.append("Light Rain"); category = "rain"
        else:
            reasons.append("Clear")

    # ── Temperature effects ────────────────────────────────────────────────────
    # Canadians accept cold; only suppress when genuinely dangerous
    if temp_f < 14:        # < -10°C — operational freeze risk
        impact -= 22
        reasons.append("Extreme Cold")
        if category == "clear": category = "cold"
    elif temp_f < 25:      # -3.9°C to -10°C — cold but functional
        impact -= 10
        reasons.append("Very Cold")
        if category == "clear": category = "cold"
    elif temp_f < 37:      # Just below freezing — common; minimal impact
        impact -= 4
        reasons.append("Below Freezing")
    elif 50 <= temp_f <= 77:  # Ideal Canadian wash weather
        impact += 7
        reasons.append("Ideal Temps")
    elif temp_f > 86:
        impact -= 4
        reasons.append("Hot")

    if not reasons:
        reasons.append("Normal")

    return int(impact), " + ".join(reasons), category


def weather_factor_canada(prob: float, rain_in: float, temp_f: float, snow_in: float) -> float:
    """Canada weather multiplier."""
    factor = 1.0
    snow_cat = _snow_category(snow_in, temp_f)

    # Snow
    if snow_cat == "blizzard":        factor *= 0.50
    elif snow_cat == "heavy_snow":    factor *= 0.80
    elif snow_cat == "moderate_snow": factor *= 0.92
    elif snow_cat == "light_snow":    factor *= 0.98

    # Rain (only when no snow)
    if snow_cat == "none":
        if prob >= 80 or rain_in >= 0.25:    factor *= 0.62
        elif prob >= 60 or rain_in >= 0.10:  factor *= 0.80
        elif prob >= 40:                     factor *= 0.92

    # Temperature
    if temp_f < 14:          factor *= 0.78
    elif temp_f < 25:        factor *= 0.90
    elif temp_f < 37:        factor *= 0.96
    elif 50 <= temp_f <= 77: factor *= 1.07
    elif temp_f > 86:        factor *= 0.97

    return factor


# ── Rebound logic ─────────────────────────────────────────────────────────────

def rebound_boost_usa(prev_prob: float, prev_rain: float) -> float:
    if prev_prob >= 80 or prev_rain >= 0.25: return 0.30
    if prev_prob >= 60 or prev_rain >= 0.10: return 0.20
    if prev_prob >= 40: return 0.10
    return 0.0


def rebound_boost_canada(prev_snow_in: float, prev_rain: float, prev_prob: float) -> float:
    """
    Canada post-event rebound.
    After snow (especially heavy snow with road salt), rebound is STRONGER than rain.
    After blizzard, rebound is very strong on the first clear day.
    """
    if prev_snow_in >= 6:    return 0.45  # post-blizzard surge
    if prev_snow_in >= 3:    return 0.35  # heavy snow → salt wash-off demand
    if prev_snow_in >= 0.5:  return 0.20  # moderate snow
    if prev_snow_in > 0:     return 0.10  # light snow
    # Rain rebound (similar to USA but slightly less)
    if prev_prob >= 80 or prev_rain >= 0.25: return 0.25
    if prev_prob >= 60 or prev_rain >= 0.10: return 0.15
    if prev_prob >= 40: return 0.07
    return 0.0


# ── Main forecast builders ────────────────────────────────────────────────────

def forecast_week_quick(location: str, min_cars: int, avg_cars: int, max_cars: int,
                        country: str = "USA") -> List[Dict[str, Any]]:
    lat, lon = get_lat_lon(location, country)
    w = get_weather_7d(lat, lon)
    out = []
    prev_prob  = 0.0
    prev_rain  = 0.0
    prev_snow  = 0.0

    for i in range(7):
        day_iso = w["time"][i]
        prob     = float(w["precipitation_probability_max"][i])
        rain_in  = float(w["precipitation_sum"][i])
        temp_f   = float(w["temperature_2m_max"][i])
        snow_in  = float(w.get("snowfall_sum", [0]*7)[i])

        dow = datetime.fromisoformat(day_iso).strftime("%A")
        dow_factor = DOW_MULT.get(dow, 1.0)

        if country == "Canada":
            impact_pct, reason, wx_cat = weather_impact_percent_canada(prob, rain_in, temp_f, snow_in)
            wf = weather_factor_canada(prob, rain_in, temp_f, snow_in)
            # Apply Canada rebound on clear days
            if (snow_in < 0.5) and (prob < 30) and (temp_f > 25):
                wf *= (1 + rebound_boost_canada(prev_snow, prev_rain, prev_prob))
        else:
            impact_pct, reason, wx_cat = weather_impact_percent_usa(prob, rain_in, temp_f)
            wf = weather_factor_usa(prob, rain_in, temp_f)
            if prob < 30 and temp_f > 50:
                wf *= (1 + rebound_boost_usa(prev_prob, prev_rain))

        raw  = avg_cars * dow_factor * wf
        cars = int(max(min_cars, min(raw, max_cars)))

        peak_hour = cars * 0.12
        staff = math.ceil(peak_hour / QUICK_CPSH) if QUICK_CPSH > 0 else 0
        staff = min(MAX_STAFF, max(0, staff))

        # Celsius for Canadian display
        temp_c = round((temp_f - 32) * 5 / 9, 1)

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
            "temp_c":   temp_c,
            "snow_in":  round(snow_in, 2),
            "country":  country,
        })
        prev_prob = prob
        prev_rain = rain_in
        prev_snow = snow_in

    return out


def confidence_label(rows: List[Dict]) -> str:
    avg_rain = sum(r["rain_pct"] for r in rows) / len(rows) if rows else 0
    has_snow = any(r.get("snow_in", 0) > 0 for r in rows)
    if has_snow and avg_rain >= 50: return "MEDIUM — mixed precip week"
    if has_snow: return "HIGH — snow expected; salt-rebound modeled"
    if avg_rain >= 60: return "MEDIUM — weather-heavy week"
    if avg_rain >= 35: return "HIGH — some variability"
    return "HIGH — stable conditions"


# ─────────────────────────────────────────────────────────────────────────────
# HTML BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def nav_html() -> str:
    return """
<div class="sp-nav">
  <div style="max-width:860px;margin:0 auto;width:100%;display:flex;align-items:center;justify-content:space-between">
    <div class="sp-logo">
      <div class="sp-logo-mark">SP</div>
      <div>
        <div class="sp-logo-text">StaffPilot AI</div>
        <div class="sp-logo-sub">Architected by Gopi Chand</div>
      </div>
    </div>
    <div class="sp-badge">⚡ Instant Forecast</div>
  </div>
</div>
"""


def metrics_html(rows: List[Dict], min_cars: int, max_cars: int) -> str:
    busiest  = max(rows, key=lambda r: r["cars"])
    slowest  = min(rows, key=lambda r: r["cars"])
    rain_days = sum(1 for r in rows if r["rain_pct"] >= 60)
    snow_days = sum(1 for r in rows if r.get("snow_in", 0) >= 0.5)
    peak_staff = max(r["staff"] for r in rows)
    avg_cars   = int(sum(r["cars"] for r in rows) / len(rows))
    avg_impact = int(sum(r["impact"] for r in rows) / len(rows))
    impact_cls = "red" if avg_impact < 0 else ("green" if avg_impact > 0 else "")
    impact_str = f"{avg_impact:+d}%"
    conf = confidence_label(rows)
    is_canada = rows[0].get("country") == "Canada"

    # For Canada, show snow days instead of pure rain days
    if is_canada and snow_days > 0:
        precip_label = "Snow + Rain Days"
        precip_val   = f"{snow_days}❄ {rain_days}🌧"
        precip_sub   = "≥0.5\" snow or ≥60% rain"
    else:
        precip_label = "Rain-Impacted Days"
        precip_val   = str(rain_days)
        precip_sub   = "≥60% precip probability"

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
    <div class="sp-metric-label">{precip_label}</div>
    <div class="sp-metric-value">{precip_val}</div>
    <div class="sp-metric-sub">{precip_sub}</div>
  </div>
</div>
<div class="sp-confidence">
  <span class="sp-conf-icon">📡</span>
  <span>Forecast confidence: <strong>{conf}</strong> &nbsp;·&nbsp; Peak staff cap: <strong>{peak_staff}/{MAX_STAFF}</strong> &nbsp;·&nbsp; Avg weather impact: <strong>{impact_str}</strong></span>
</div>
"""


def table_html(rows: List[Dict], max_cars: int) -> str:
    is_canada = rows[0].get("country") == "Canada"
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
        wx_icons = {"rain": "🌧", "cold": "🥶", "heat": "🌡", "clear": "☀️",
                    "snow": "❄️", "blizzard": "🌨"}
        icon = wx_icons.get(r["wx_cat"], "☀️")
        wx_cell = f'<span class="wx-pill {r["wx_cat"]}">{icon} {r["reason"]}</span>'

        # Impact
        if r["impact"] < 0:
            imp_cell = f'<span class="impact-neg">{r["impact"]:+d}%</span>'
        elif r["impact"] > 0:
            imp_cell = f'<span class="impact-pos">{r["impact"]:+d}%</span>'
        else:
            imp_cell = f'<span class="impact-neu">{r["impact"]:+d}%</span>'

        # Temperature display — Canada shows °C primary, °F secondary
        if is_canada:
            temp_display = f'{r["temp_c"]}°C <span style="color:var(--muted);font-size:11px">({r["temp_f"]}°F)</span>'
        else:
            temp_display = f'{r["temp_f"]}°F'

        # Precip display — Canada adds snow column
        if is_canada:
            snow_str = f' · {r["snow_in"]}" ❄' if r.get("snow_in", 0) > 0 else ""
            precip_display = f'{r["rain_pct"]}% · {r["rain_in"]}"{snow_str}'
        else:
            precip_display = f'{r["rain_pct"]}% · {r["rain_in"]}"'

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
  <td style="font-family:var(--mono);font-size:12px;color:var(--muted)">{temp_display}</td>
  <td style="font-family:var(--mono);font-size:12px;color:var(--muted)">{precip_display}</td>
</tr>"""

    temp_col_label = "High Temp (°C)" if is_canada else "High Temp"
    precip_col_label = "Precip / Snow" if is_canada else "Precip"

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
        <th>{temp_col_label}</th>
        <th>{precip_col_label}</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
"""


def canada_algorithm_note_html() -> str:
    return """
<div class="sp-canada-note">
  <span style="font-size:18px">🍁</span>
  <div>
    <strong>Canadian Weather Model Active</strong> — Snow tolerance adjusted for Canadian driving habits.
    Light–moderate snow applies minimal suppression (customers are accustomed to winter roads and road salt
    actively <em>increases</em> wash demand). Heavy snow reduces volume moderately; blizzard conditions
    apply strong suppression. Post-snow rebound is modeled as stronger than rain rebound.
    Temperature thresholds shifted: extreme cold penalty only applies below −10°C (14°F).
    Temperatures display in °C.
  </div>
</div>
"""


def _legend_dot(color, label):
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px">'
        f'<span style="width:10px;height:10px;background:{color};border-radius:2px;display:inline-block"></span>'
        f'{label}</span>'
    )


def mini_chart_html(rows, max_cars):
    """SVG sparkline bar chart — mobile-friendly, legend bug fixed."""
    W, H = 900, 120
    pad_l, pad_r, pad_t, pad_b = 10, 10, 10, 24
    inner_w = W - pad_l - pad_r
    inner_h = H - pad_t - pad_b
    n = len(rows)
    bar_w = inner_w / n
    bar_gap = bar_w * 0.28
    rect_w = bar_w - bar_gap

    max_row_cars = max(r["cars"] for r in rows)
    bars = ""
    labels = ""
    for i, r in enumerate(rows):
        h = max(4, int((r["cars"] / max(max_cars, 1)) * inner_h))
        x = pad_l + i * bar_w + bar_gap / 2
        y = pad_t + inner_h - h
        wx = r.get("wx_cat", "clear")
        if wx == "blizzard":
            color = "#a78bfa"
        elif wx == "snow":
            color = "#818cf8"
        elif r["rain_pct"] >= 60:
            color = "#ef4444"
        elif r["cars"] == max_row_cars:
            color = "#f5a623"
        else:
            color = "#374151"
        bars += f'<rect x="{x:.1f}" y="{y}" width="{rect_w:.1f}" height="{h}" rx="3" fill="{color}" opacity="0.85"/>'
        cx = x + rect_w / 2
        labels += f'<text x="{cx:.1f}" y="{H - 4}" text-anchor="middle" font-size="10" fill="#6b7280" font-family="IBM Plex Mono, monospace">{r["dow"][:3]}</text>'

    is_canada = rows[0].get("country") == "Canada"
    legend_items = [
        _legend_dot("#f5a623", "Busiest"),
        _legend_dot("#ef4444", "Rain (≥60%)"),
    ]
    if is_canada:
        legend_items.append(_legend_dot("#818cf8", "Snow Day"))
        legend_items.append(_legend_dot("#a78bfa", "Blizzard"))
    legend_items.append(_legend_dot("#374151", "Normal"))
    legend_html = " ".join(legend_items)

    svg = (
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;height:auto;display:block">{bars}{labels}</svg>'
    )
    return (
        '<div class="sp-table-wrap" style="padding:20px 24px 8px">'
        '<div class="sp-table-header">Daily Volume Trend</div>'
        + svg
        + f'<div class="sp-chart-legend">{legend_html}</div>'
        + '</div>'
    )





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

# Country selector
country = st.radio(
    "Country",
    options=["🇺🇸  USA", "🍁  Canada"],
    horizontal=True,
    label_visibility="collapsed",
)
is_canada = "Canada" in country
country_code = "Canada" if is_canada else "USA"

# Location input — placeholder adapts to country
if is_canada:
    loc_placeholder = "e.g. M5V or Toronto, ON"
    loc_label = "Postal Code or City"
    st.markdown(canada_algorithm_note_html(), unsafe_allow_html=True)
else:
    loc_placeholder = "e.g. 30301 or Atlanta, GA"
    loc_label = "ZIP Code or City"

col_loc, col_blank = st.columns([2, 2])
with col_loc:
    location_raw = st.text_input(loc_label, placeholder=loc_placeholder, label_visibility="visible")

# Normalize Canadian postal code
if is_canada and location_raw:
    location = normalize_canadian_postal(location_raw)
else:
    location = location_raw.strip() if location_raw else ""

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
canada_extra = " Snow rebound modeling accounts for post-salt demand surges." if is_canada else ""
st.markdown(f"""
<div class="sp-disclaimer">
  <strong style="color:var(--text)">Accuracy note:</strong>
  This forecast uses weather data + industry day-of-week curves. It does <em>not</em> account for
  active promotions, local road closures, nearby competition, or event-driven spikes.
  Treat <strong>Expected Change %</strong> as directional guidance, not a hard prediction.{canada_extra}
</div>
""", unsafe_allow_html=True)

# ── Results ───────────────────────────────────────────────────────────────────
if run:
    if not location.strip():
        st.markdown(
            f'<div class="sp-error">⚠ Please enter a {"postal code or city" if is_canada else "ZIP code or city"} before generating.</div>',
            unsafe_allow_html=True
        )
    elif not (min_cars <= avg_cars <= max_cars):
        st.markdown('<div class="sp-error">⚠ Fix guardrail values before generating.</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Fetching weather data…"):
            try:
                rows = forecast_week_quick(
                    location.strip(), int(min_cars), int(avg_cars), int(max_cars),
                    country=country_code
                )
            except Exception as e:
                st.markdown(f'<div class="sp-error">⚠ {e}</div>', unsafe_allow_html=True)
                rows = []

        if rows:
            st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

            # Geo fallback warning (shown when postal code resolved via FSA prefix table)
            geo_fallback = st.session_state.pop("_geo_fallback", None)
            if geo_fallback:
                st.markdown(
                    f'<div style="font-family:var(--mono);font-size:11px;color:#f97316;'
                    f'margin-bottom:10px;letter-spacing:0.08em">'
                    f'📍 Postal code resolved to nearest city: <strong>{geo_fallback}</strong> '
                    f'— weather accuracy within ~50 km</div>',
                    unsafe_allow_html=True,
                )

            # Climate-estimate warning banner
            wx_estimated = st.session_state.get("_wx_estimated", False)
            if wx_estimated:
                wx_errors = st.session_state.get("_wx_errors", {})
                err_lines = " &nbsp;·&nbsp; ".join(
                    f"{k}: {v.split(chr(10))[0][:80]}" for k, v in wx_errors.items()
                )
                st.markdown(
                    f'''<div style="background:rgba(249,115,22,0.08);border:1px solid rgba(249,115,22,0.35);'
                    f'border-radius:10px;padding:14px 18px;margin-bottom:16px;font-size:13px;color:#fdba74;line-height:1.6">'
                    f'<strong style="color:#fb923c">⚠ Live weather unavailable — using seasonal climate estimates</strong><br>'
                    f'Staffing direction is still reliable, but day-specific rain/snow accuracy is reduced. '
                    f'Add a free <a href="https://www.weatherapi.com/signup.aspx" target="_blank" '
                    f'style="color:#f97316">WeatherAPI.com key</a> to your Streamlit secrets for live data.<br>'
                    f'<span style="font-size:11px;color:#9ca3af;font-family:var(--mono)">{err_lines}</span>'
                    f'</div>''',
                    unsafe_allow_html=True,
                )

            wx_source = st.session_state.get("_wx_source", "Open-Meteo")
            is_estimated = st.session_state.get("_wx_estimated", False)
            is_fallback = "fallback" in wx_source.lower() or is_estimated
            src_color  = "#f97316" if is_fallback else "#22c55e"
            src_icon   = "🌡" if is_estimated else ("⚠️" if is_fallback else "✅")
            src_note   = " — Open-Meteo was unavailable" if ("fallback" in wx_source.lower() and not is_estimated) else ""
            canada_tag = ' &nbsp;·&nbsp; <span style="color:#f87171">🍁 Canadian algorithm active</span>' if is_canada else ""
            st.markdown(
                f'<div style="font-family:var(--mono);font-size:11px;color:{src_color};'
                f'margin-bottom:16px;letter-spacing:0.08em">'
                f'{src_icon} Weather source: <strong>{wx_source}</strong>{src_note}{canada_tag}</div>',
                unsafe_allow_html=True,
            )

            st.markdown(metrics_html(rows, int(min_cars), int(max_cars)), unsafe_allow_html=True)
            st.markdown(mini_chart_html(rows, int(max_cars)), unsafe_allow_html=True)
            st.markdown(table_html(rows, int(max_cars)), unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # close sp-page
