import streamlit as st
from supabase import create_client
import requests
import pandas as pd
from datetime import datetime
import math
import json

# ---------------- CONFIG ----------------

st.set_page_config(page_title="StaffPilot AI", layout="centered")

MAX_STAFF = 5

TRAFFIC_MULT = {
    "Low": 0.9,
    "Medium": 1.0,
    "High": 1.1,
    "Very High": 1.2
}

# ---------------- SUPABASE ----------------

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- WEATHER ----------------

@st.cache_data(ttl=1800)
def get_lat_lon(location):
    url = "https://geocoding-api.open-meteo.com/v1/search"

    res = requests.get(
        url,
        params={"name": location, "count": 1},
        timeout=10
    )

    if res.status_code != 200:
        raise RuntimeError("Geocoding API failed")

    data = res.json()

    if "results" not in data or not data["results"]:
        raise RuntimeError("Invalid location. Try City + State or ZIP.")

    return (
        data["results"][0]["latitude"],
        data["results"][0]["longitude"]
    )


@st.cache_data(ttl=1800)
def get_weather(lat, lon):

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_probability_max,precipitation_sum,temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "forecast_days": 7,  # 🔥 CRITICAL FIX
        "timezone": "auto"
    }

    res = requests.get(url, params=params, timeout=10)

    if res.status_code != 200:
        raise RuntimeError("Weather API failed")

    data = res.json()

    if "daily" not in data:
        raise RuntimeError("Weather returned no daily data")

    daily = data["daily"]

    required = [
        "time",
        "precipitation_probability_max",
        "precipitation_sum",
        "temperature_2m_max"
    ]

    for r in required:
        if r not in daily or not daily[r]:
            raise RuntimeError(f"Weather missing field: {r}")

    return daily


# ---------------- FORECAST ENGINE ----------------

def weather_factor(prob, rain, temp):

    factor = 1.0

    if prob >= 80 or rain >= 0.25:
        factor *= 0.55
    elif prob >= 60 or rain >= 0.10:
        factor *= 0.75
    elif prob >= 40:
        factor *= 0.9

    if temp < 40:
        factor *= 0.8
    elif temp < 50:
        factor *= 0.9
    elif 50 <= temp <= 85:
        factor *= 1.05

    return factor


def rebound_boost(prev_prob, prev_rain):

    if prev_prob >= 80 or prev_rain >= 0.25:
        return 0.30
    elif prev_prob >= 60 or prev_rain >= 0.10:
        return 0.20
    elif prev_prob >= 40:
        return 0.10
    return 0.0


def forecast_week(site):

    weather = get_weather(site["lat"], site["lon"])

    baseline = site["baseline"]

    # 🔥 Normalize JSON
    if isinstance(baseline, str):
        baseline = json.loads(baseline)

    baseline = {k.lower(): v for k, v in baseline.items()}

    member_floor = site.get("member_washes", 0)
    traffic = TRAFFIC_MULT.get(site.get("traffic", "Medium"), 1.0)

    forecasts = []

    prev_prob = 0
    prev_rain = 0

    times = weather["time"]
    probs = weather["precipitation_probability_max"]
    rains = weather["precipitation_sum"]
    temps = weather["temperature_2m_max"]

    for i in range(7):

        day_iso = times[i]
        prob = float(probs[i])
        rain = float(rains[i])
        temp = float(temps[i])

        dow = datetime.fromisoformat(day_iso).strftime("%A").lower()

        base = baseline.get(dow, 120)  # fallback default

        retail = max(base - member_floor, 0)

        factor = weather_factor(prob, rain, temp)

        # Bounce-back logic
        if prob < 30 and temp > 50:
            factor *= (1 + rebound_boost(prev_prob, prev_rain))

        cars = int(member_floor + retail * traffic * factor)

        forecasts.append({
            "Day": dow.title(),
            "Forecast Cars": cars,
            "Rain %": prob,
            "Temp": temp
        })

        prev_prob = prob
        prev_rain = rain

    return forecasts


# ---------------- STAFFING ----------------

def staff_needed(cars, hours, prep):

    peak_hour = cars * 0.12

    cpsh = 8 if prep else 11

    staff = math.ceil(peak_hour / cpsh)

    return min(MAX_STAFF, staff)


# ---------------- UI ----------------

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

    open_hour = st.slider("Opening Hour", 0, 23, 8)
    close_hour = st.slider("Closing Hour", 1, 24, 20)

    traffic = st.selectbox("Street Traffic", list(TRAFFIC_MULT.keys()))
    member_washes = st.number_input("Member Washes / Day", 0, 500, 60)

    min_staff = st.slider("Minimum Staff", 1, 5, 3)

    prep = st.checkbox("Prep Every Car?")
    one_loader = st.checkbox("Can operate with one loader?")

    st.subheader("Normal Week Cars")

    baseline = {}

    days = [
        "Monday","Tuesday","Wednesday",
        "Thursday","Friday","Saturday","Sunday"
    ]

    for d in days:
        baseline[d] = st.number_input(d, 0, 5000, 120)

    if st.button("Create Site"):

        lat, lon = get_lat_lon(location)

        supabase.table("sites").insert({
            "alias": alias,
            "location": location,
            "lat": lat,
            "lon": lon,
            "open_hour": open_hour,
            "close_hour": close_hour,
            "traffic": traffic,
            "min_staff": min_staff,
            "prep": prep,
            "one_loader": one_loader,
            "member_washes": member_washes,
            "baseline": baseline
        }).execute()

        st.success("Site Created!")

# ---------------- FORECAST ----------------

if menu == "View Forecast":

    res = supabase.table("sites").select("*").execute()
    sites = res.data

    if not sites:
        st.warning("Create a site first.")
        st.stop()

    site = st.selectbox(
        "Choose Site",
        sites,
        format_func=lambda x: x["alias"]
    )

    if st.button("Generate AI Staffing Plan"):

        try:
            forecasts = forecast_week(site)
        except Exception as e:
            st.error(f"Forecast failed: {e}")
            st.stop()

        if not forecasts:
            st.error("No forecast generated.")
            st.stop()

        hours = max(1, site["close_hour"] - site["open_hour"])

        rows = []

        for f in forecasts:

            staff = staff_needed(
                f["Forecast Cars"],
                hours,
                site["prep"]
            )

            peak_staff = min(
                MAX_STAFF,
                max(site["min_staff"], staff)
            )

            f["Peak Staff"] = peak_staff

            rows.append(f)

        df = pd.DataFrame(rows)

        st.dataframe(df, use_container_width=True)

        # 🔥 Operators LOVE this
        st.line_chart(df.set_index("Day")["Forecast Cars"])

        st.success("AI Staffing Plan Generated ✅")
