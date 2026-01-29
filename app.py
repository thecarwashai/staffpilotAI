import streamlit as st
from supabase import create_client
import requests
import pandas as pd
from datetime import datetime
import math
from typing import Tuple, Dict, Any, List

# ---------------- CONFIG ----------------

MAX_STAFF = 5

TRAFFIC_MULT = {
    "Low": 0.9,
    "Medium": 1.0,
    "High": 1.1,
    "Very High": 1.2
}

# ---------------- SUPABASE ----------------
# Requires SUPABASE_URL and SUPABASE_KEY in Streamlit secrets
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Supabase credentials not found. Add SUPABASE_URL and SUPABASE_KEY to Streamlit secrets.")
    st.stop()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- WEATHER ----------------

@st.cache_data(ttl=60 * 60)  # cache for 1 hour
def get_lat_lon(location: str) -> Tuple[float, float]:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    try:
        res = requests.get(url, params={"name": location, "count": 1}, timeout=10).json()
        results = res.get("results")
        if not results:
            raise ValueError("No geocoding results")
        return results[0]["latitude"], results[0]["longitude"]
    except Exception as e:
        raise RuntimeError(f"Failed to geocode '{location}': {e}")

@st.cache_data(ttl=60 * 30)  # cache for 30 minutes
def get_weather(lat: float, lon: float) -> Dict[str, Any]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_probability_max,precipitation_sum,temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "auto"
    }
    try:
        res = requests.get(url, params=params, timeout=10).json()
        daily = res.get("daily")
        if not daily:
            raise ValueError("No daily data returned from weather API")
        return daily
    except Exception as e:
        raise RuntimeError(f"Failed to fetch weather: {e}")

# ---------------- FORECAST ENGINE ----------------

def weather_factor(prob: float, rain: float, temp: float) -> float:
    factor = 1.0

    # Rain suppression
    if prob >= 80 or rain >= 0.25:
        factor *= 0.55
    elif prob >= 60 or rain >= 0.10:
        factor *= 0.75
    elif prob >= 40:
        factor *= 0.9

    # Temperature
    if temp < 40:
        factor *= 0.8
    elif temp < 50:
        factor *= 0.9
    elif 50 <= temp <= 85:
        factor *= 1.05

    return factor

def rebound_boost(prev_prob: float, prev_rain: float) -> float:
    if prev_prob >= 80 or prev_rain >= 0.25:
        return 0.30
    elif prev_prob >= 60 or prev_rain >= 0.10:
        return 0.20
    elif prev_prob >= 40:
        return 0.10
    return 0.0


def forecast_week(site: Dict[str, Any]) -> List[Dict[str, Any]]:
    weather = get_weather(site["lat"], site["lon"])

    baseline = site["baseline"]
    member_floor = site.get("member_washes", 0)
    traffic = TRAFFIC_MULT.get(site.get("traffic", "Medium"), 1.0)

    forecasts = []

    prev_prob = 0.0
    prev_rain = 0.0

    # Safeguard: ensure weather arrays are present and have 7 entries
    times = weather.get("time", [])
    probs = weather.get("precipitation_probability_max", [])
    rains = weather.get("precipitation_sum", [])
    temps = weather.get("temperature_2m_max", [])

    days_count = min(7, len(times), len(probs), len(rains), len(temps))

    for i in range(days_count):
        day = times[i]
        prob = float(probs[i])
        rain = float(rains[i])
        temp = float(temps[i])

        dow = datetime.fromisoformat(day).strftime("%A")
        base = baseline.get(dow, 0)

        retail = max(base - member_floor, 0)

        factor = weather_factor(prob, rain, temp)

        # bounce-back
        if prob < 30 and temp > 50:
            factor *= (1 + rebound_boost(prev_prob, prev_rain))

        cars = int(member_floor + retail * traffic * factor)

        forecasts.append({
            "day": dow,
            "cars": cars,
            "prob": prob,
            "temp": temp
        })

        prev_prob = prob
        prev_rain = rain

    return forecasts

# ---------------- STAFFING ----------------

def staff_needed(cars: int, hours: int, prep: bool) -> int:
    peak_hour_cars = cars * 0.12  # assume peak hour is ~12%
    cpsh = 8 if prep else 11
    staff = math.ceil(peak_hour_cars / cpsh) if cpsh > 0 else 0
    return min(MAX_STAFF, max(0, staff))

# ---------------- UI ----------------

st.set_page_config(page_title="StaffPilot AI", layout="centered")
st.title("🚀 StaffPilot AI")

menu = st.sidebar.selectbox(
    "Menu",
    ["Create Site", "View Forecast"]
)

# ---------------- CREATE SITE ----------------

if menu == "Create Site":

    st.header("New Operator Setup")

    alias = st.text_input("Site Name")

    location = st.text_input("City / ZIP")

    col1, col2 = st.columns(2)
    with col1:
        open_hour = st.slider("Opening Hour", 0, 23, 8)
    with col2:
        close_hour = st.slider("Closing Hour", 1, 24, 20)

    traffic = st.selectbox("Street Traffic", list(TRAFFIC_MULT.keys()))

    member_washes = st.number_input("Estimated Member Washes / Day", 0, 500, 60, step=1)

    min_staff = st.slider("Minimum Staff", 1, 10, 3)

    prep = st.checkbox("Prep Every Car?")
    one_loader = st.checkbox("Can operate with one loader?")

    st.subheader("Normal Week Cars")

    baseline: Dict[str, int] = {}
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    for d in days:
        baseline[d] = st.number_input(d, 0, 5000, 120, key=f"baseline_{d}")

    if st.button("Create Site"):
        if not alias.strip():
            st.warning("Please enter a Site Name.")
        elif not location.strip():
            st.warning("Please enter a City / ZIP location.")
        else:
            try:
                lat, lon = get_lat_lon(location)
            except Exception as e:
                st.error(f"Geocoding error: {e}")
            else:
                payload = {
                    "alias": alias,
                    "location": location,
                    "lat": lat,
                    "lon": lon,
                    "open_hour": int(open_hour),
                    "close_hour": int(close_hour),
                    "traffic": traffic,
                    "min_staff": int(min_staff),
                    "prep": bool(prep),
                    "one_loader": bool(one_loader),
                    "member_washes": int(member_washes),
                    "baseline": baseline
                }

                try:
                    res = supabase.table("sites").insert(payload).execute()
                    # supabase client returns .data or .json depending on version
                    st.success("Site Created!")
                    st.json(payload)
                except Exception as e:
                    st.error(f"Failed to create site in Supabase: {e}")

# ---------------- FORECAST ----------------

if menu == "View Forecast":

    try:
        res = supabase.table("sites").select("*").execute()
        sites = res.data if hasattr(res, "data") else res
    except Exception as e:
        st.error(f"Failed to fetch sites from Supabase: {e}")
        st.stop()

    if not sites:
        st.warning("Create a site first.")
        st.stop()

    site = st.selectbox("Choose Site", sites, format_func=lambda x: x.get("alias", "Unknown"))

    if st.button("Generate 7-Day Forecast"):

        try:
            forecasts = forecast_week(site)
        except Exception as e:
            st.error(f"Forecast generation failed: {e}")
            st.stop()

        hours = max(1, int(site.get("close_hour", 20)) - int(site.get("open_hour", 8)))

        rows = []

        for f in forecasts:
            staff = staff_needed(
                f["cars"],
                hours,
                site.get("prep", False)
            )

            rows.append({
                "Day": f["day"],
                "Forecast Cars": f["cars"],
                "Peak Staff": max(int(site.get("min_staff", 1)), staff),
                "Rain %": f["prob"],
                "Temp (F)": f["temp"]
            })

        df = pd.DataFrame(rows)

        st.dataframe(df, use_container_width=True)
        st.success("AI Forecast Generated ✅")
